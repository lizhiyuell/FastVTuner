import sys 
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from systems.base import SystemBase, TuningRecord
from systems.vdb_engine import VDBEngine
from common import *
import numpy as np
import json
import time
import subprocess as sp
import random
import os
import re
import torch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement, ConstrainedExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels import ProductKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.priors.torch_priors import GammaPrior


REF_POINT = torch.tensor([0.5, 0.5], dtype=torch.float64)
NR_SEARCH_PER_BUILD = 5
SEARCH_RECALL_STOP_DELTA = 0.01
MAX_SEARCH_ONLY_STEPS = 32
MIN_SAMPLED_RECALL_FOR_FULL_TEST = 0.7
MAX_SAMPLED_RECALL_FOR_FULL_TEST = 1.0
SAMPLED_RECALL_BLEND_INITIAL_WEIGHT = 0.7
SAMPLED_RECALL_BLEND_MIN_WEIGHT = 0.05
SAMPLED_RECALL_BLEND_DECAY_STEPS = 30.0


class BlendedRecallObjective(MCMultiOutputObjective):
    def __init__(self, sampled_recall_weight):
        super().__init__()
        self.sampled_recall_weight = sampled_recall_weight

    def forward(self, samples, X=None):
        tput = samples[..., 0:1]
        full_recall = samples[..., 1:2]
        sampled_recall = samples[..., 2:3]
        recall = (
            (1.0 - self.sampled_recall_weight) * full_recall
            + self.sampled_recall_weight * sampled_recall
        )
        return torch.cat([tput, recall], dim=-1)

def hypervolume_calcu(all_sol, ref_point=[0,0], opt_max=True):
    rank, f = fast_non_dominated_sort(all_sol)
    pareto_sol = np.array(all_sol)[f[0]]
    if opt_max:
        pareto_sol = pareto_sol[pareto_sol[:,0].argsort(kind="mergesort")]
    else:
        pareto_sol = pareto_sol[pareto_sol[:,0].argsort(kind="mergesort")[::-1]]

    volume = 0
    for sol in pareto_sol.tolist():
        sol[0] = max(sol[0], ref_point[0])
        sol[1] = max(sol[1], ref_point[1])
        volume += (sol[0] - ref_point[0]) * (sol[1] - ref_point[1])
        ref_point[0] = sol[0]
    return volume

def fast_non_dominated_sort(P):
    def compare(p1, p2):
        D = len(p1)
        p1_dominate_p2 = True 
        p2_dominate_p1 = True
        for i in range(D):
            if p1[i] < p2[i]:
                p1_dominate_p2 = False
            if p1[i] > p2[i]:
                p2_dominate_p1 = False

        if p1_dominate_p2 == p2_dominate_p1:
            return 0
        return 1 if p1_dominate_p2 else -1
    P_size = len(P)
    n = np.full(shape=P_size, fill_value=0)
    S = []
    f = [] 
    rank = np.full(shape=P_size, fill_value=-1)

    f_0 = []
    for p in range(P_size):
        n_p = 0
        S_p = []
        for q in range(P_size):
            if p == q:
                continue
            cmp = compare(P[p], P[q])
            if cmp == 1:
                S_p.append(q)
            elif cmp == -1:
                n_p += 1
        S.append(S_p)
        n[p] = n_p
        if n_p == 0:
            rank[p] = 0
            f_0.append(p)
    f.append(f_0)
    i = 0
    while len(f[i]) != 0: 
        Q = []
        for p in f[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        f.append(Q)
    return rank, f


class EHVIBO:
    def __init__(self, knob_num, seed) -> None:
        self.knob_num = knob_num
        self.bounds = torch.tensor(
            [[0.0] * self.knob_num, [1.0] * self.knob_num],
            dtype=torch.float64,
        )
        self.seed = seed
        self.X_init = None
        self.Y_init = None
    
    def make_kernel(self,):
        covar_module1 = MaternKernel(
                nu=2.5,
                active_dims=(0),
                lengthscale_prior=GammaPrior(3.0, 6.0),
            )
        covar_module2 = MaternKernel(
                nu=2.5,
                # active_dims=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),
                active_dims=tuple(i for i in range(1, self.knob_num)),
                lengthscale_prior=GammaPrior(3.0, 6.0),
            )
        
        product_covar_module = ProductKernel(covar_module1, covar_module2)

        return ScaleKernel(
            product_covar_module,
            outputscale_prior=GammaPrior(2.0, 0.15),
            )
    
    def recommend(self, fixed_features, q, inequality_constraints=None, sampled_recall_weight=0.0):
        # assume 2-dim output: [fitness, recall]
        
        qehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        objective = None
        if self.has_sampled_recall_model:
            objective = BlendedRecallObjective(sampled_recall_weight)

        with torch.no_grad():
            pred = self.model.posterior(self.X_init).mean
            if objective is not None:
                pred = objective(pred)

        partitioning = FastNondominatedPartitioning(ref_point=REF_POINT, Y=pred,)
        
        acq_func = qLogExpectedHypervolumeImprovement(
            model=self.model,
            ref_point=REF_POINT,
            partitioning=partitioning,
            sampler=qehvi_sampler,
            objective=objective,
        )

        candidate, ei = optimize_acqf(
            acq_func, bounds=self.bounds, q=q, num_restarts=10, raw_samples=100, 
            fixed_features=fixed_features, 
            inequality_constraints=inequality_constraints,
            options={'seed':self.seed}
            )
        new_x = candidate.detach()
        
        with torch.no_grad():
            new_x_mean = self.model.posterior(new_x).mean
            new_x_std = torch.sqrt(self.model.posterior(new_x).variance)

        return new_x.numpy(), ei.item(), new_x_mean.numpy(), new_x_std.numpy()
    
    def update_samples(self, X, Y, sampled_X=None, sampled_Y=None):
        self.X_init = torch.tensor(X,dtype=torch.float64)
        self.Y_init = torch.tensor(Y,dtype=torch.float64)
        self.has_sampled_recall_model = sampled_X is not None and sampled_Y is not None and len(sampled_X) > 0
        models = []
        self.stands = []

        for i in range(self.Y_init.shape[-1]):
            train_y = self.Y_init[..., i : i + 1]
            models.append(SingleTaskGP(
                self.X_init, train_y,
                # covar_module=self.make_kernel(),
                outcome_transform = Standardize(m=1)
                ))

        if self.has_sampled_recall_model:
            sampled_X_init = torch.tensor(sampled_X,dtype=torch.float64)
            sampled_Y_init = torch.tensor(sampled_Y,dtype=torch.float64)
            models.append(SingleTaskGP(
                sampled_X_init, sampled_Y_init,
                outcome_transform = Standardize(m=1)
                ))
            
        self.model = ModelListGP(*models)
        self.mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll)


# The implementation of the FastVTunerSystem
# adapted from PollingBayesianOptimization
class FastVTunerSystem(SystemBase):
    def __init__(
        self,
        vdb_name,
        dataset_name,
        top_k = 10,
        single_tune_query_ratio=1.0,
        single_test_query_ratio=1.0,
        sampled_dataset_name=None,
        seed=1206,
    ):
        super().__init__(
            vdb_name=vdb_name,
            dataset_name=dataset_name,
            top_k=top_k,
            single_tune_query_ratio=single_tune_query_ratio,
            single_test_query_ratio=single_test_query_ratio,
        )

        self.knob_num = len(self.vdb_config.param_names)
        self.default_conf = self.vdb_config.default_config
        self.vbo = EHVIBO(self.knob_num, seed=seed)
        self.seed = seed
        self.sampled_vdb_engine = None
        self.current_sampled_record = None
        self.current_skip_full_test = False
        self.skip_build = False
        self.search_param_directions = {}

        if sampled_dataset_name is not None:
            self.init_sampled_vdb_engine(sampled_dataset_name)
        torch.manual_seed(seed)
        random.seed(seed)

        self.polling_sys, self.polling_index = self.vdb_config.get_polling_params()

        self.X = {key: [] for key in self.polling_index.keys()}
        self.Y = {key: [] for key in self.polling_index.keys()}
        self.sampled_recall_records = []

        self.remain_types = list(self.polling_index.keys())
        self.polling_round_num = 0
        self.worst_type_record = []

        # the key of index type
        if self.vdb_name=="milvus":
            index_type_name = "index_type"
        else:
            raise NotImplementedError

        # the original init_sample function
        for k in self.remain_types:
            # get the default configurations, but change the index into target one
            param_original = self.vdb_config.get_original_param()
            # change the index type
            index_pos = self.vdb_config.get_param_index(index_type_name)
            param_original[index_pos] = k

            self.vdb_config.set_original_param(param_original)

            # Initial points must be real full tests and do not use sampled filtering.
            self.current_sampled_record = None
            self.current_skip_full_test = False
            self.vdb_engine.start()
            res_record = self.single_tune()
            self.single_test()
            self._append_tuning_result(k, res_record)
            self._probe_initial_search_params(k)
            self.vdb_engine.stop()

        self.current_skip_full_test = False
        self.update_model()

    def init_sampled_vdb_engine(self, sampled_dataset_name):
        self.sampled_vdb_engine = VDBEngine(self.vdb_name)
        self.sampled_vdb_engine.load_dataset(sampled_dataset_name)

    def step(self,):
        if len(self.remain_types) > 1:
            self.successive_abandon()

        polling_k, new_x = self.rr_polling()

        # the new_x is an array of parameter array, we detach it
        self.vdb_config.set_normalized_param(new_x[0])
        self._run_sampled_test()
        self._append_sampled_recall_result(polling_k)
        self.current_skip_full_test = self._should_skip_full_test()
        if self.current_skip_full_test:
            self.single_tune()
            self.single_test()
            self.update_model()
        else:
            self.vdb_engine.start()
            try:
                res_record = self.single_tune()
                self.single_test()
                self._append_tuning_result(polling_k, res_record)
                self.update_model()

                if self._has_search_params(polling_k):
                    self._run_monotonic_search_only(polling_k)
            finally:
                self.skip_build = False
                self.vdb_engine.stop()

        self.current_skip_full_test = False
        self.skip_build = False

    def _append_tuning_result(self, index_type, record):
        self.X[index_type].append(self.vdb_config.get_normalized_param())
        self.Y[index_type].append([
            record.query_throughput,
            record.recall,
            record.query_latency
        ])

    def _append_sampled_recall_result(self, index_type):
        if self.current_sampled_record is None:
            return
        self.sampled_recall_records.append(
            {
                "index_type": index_type,
                "x": self.vdb_config.get_normalized_param(),
                "recall": self.current_sampled_record["recall"],
            }
        )

    def _has_search_params(self, index_type):
        return any(
            self.vdb_config.param_meta[idx]["class"] == "searching"
            for idx in self.polling_index[index_type]
        )

    def _get_search_param_idxs(self, index_type):
        return [
            idx for idx in self.polling_index[index_type]
            if self.vdb_config.param_meta[idx]["class"] == "searching"
        ]

    def _get_effective_search_range(self, idx):
        meta = self.vdb_config.param_meta[idx]
        if meta["type"] == "enum":
            return list(meta["enum_values"])

        param_min = meta["min"]
        param_max = meta["max"]
        constrain = meta.get("constrain")
        if constrain is not None:
            op, other_name = self.vdb_config._parse_constrain(meta["name"], constrain)
            other_idx = self.vdb_config.get_param_index(other_name)
            other_value = self.vdb_config.get_original_param()[other_idx]
            if op == "<=":
                param_max = min(param_max, other_value)
            elif op == ">=":
                param_min = max(param_min, other_value)

        return param_min, param_max

    def _make_search_probe_values(self, idx):
        meta = self.vdb_config.param_meta[idx]
        value_range = self._get_effective_search_range(idx)

        if meta["type"] == "enum":
            values = value_range
            if len(values) <= NR_SEARCH_PER_BUILD:
                return values
            chosen = np.linspace(0, len(values) - 1, NR_SEARCH_PER_BUILD)
            return [values[int(round(pos))] for pos in chosen]

        param_min, param_max = value_range
        if param_max < param_min:
            return []

        if param_min > 0:
            raw_values = np.geomspace(float(param_min), float(param_max), NR_SEARCH_PER_BUILD)
        else:
            raw_values = np.linspace(float(param_min), float(param_max), NR_SEARCH_PER_BUILD)

        if meta["type"] == "integer":
            values = [int(round(value)) for value in raw_values]
            values[0] = int(param_min)
            values[-1] = int(param_max)
            return sorted(set(values))

        values = [float(value) for value in raw_values]
        values[0] = float(param_min)
        values[-1] = float(param_max)
        return values

    def _infer_search_direction(self, records):
        records = [record for record in records if record[1] is not None]
        if len(records) < 2:
            return "increasing"

        records.sort(key=lambda record: record[0])
        recall_delta = records[-1][1] - records[0][1]
        if recall_delta < 0:
            return "decreasing"
        return "increasing"

    def _run_search_only_tune(self, index_type):
        self.current_sampled_record = None
        self.skip_build = True
        record = self.single_tune()
        self._append_tuning_result(index_type, record)
        return record

    def _probe_initial_search_params(self, index_type):
        search_idxs = self._get_search_param_idxs(index_type)
        if not search_idxs:
            return

        self.search_param_directions[index_type] = {}
        base_params = self.vdb_config.get_original_param()
        old_skip_build = self.skip_build
        try:
            for idx in search_idxs:
                param_name = self.vdb_config.param_names[idx]
                records = []
                for value in self._make_search_probe_values(idx):
                    params = list(base_params)
                    params[idx] = value
                    self.vdb_config.set_original_param(params)
                    record = self._run_search_only_tune(index_type)
                    records.append((value, record.recall))

                direction = self._infer_search_direction(records)
                self.search_param_directions[index_type][param_name] = direction
                print(
                    f"[FastVTuner] initial probe: index_type={index_type}, "
                    f"param={param_name}, direction={direction}",
                    flush=True,
                )
        finally:
            self.skip_build = old_skip_build
            self.vdb_config.set_original_param(base_params)

    def _next_search_value(self, idx, value, direction):
        meta = self.vdb_config.param_meta[idx]
        value_range = self._get_effective_search_range(idx)

        if meta["type"] == "enum":
            values = value_range
            if value not in values:
                return values[0] if direction == "increasing" else values[-1]
            pos = values.index(value)
            if direction == "increasing":
                return values[min(pos + 1, len(values) - 1)]
            return values[max(pos - 1, 0)]

        param_min, param_max = value_range
        if direction == "increasing":
            if meta["type"] == "integer":
                return min(int(param_max), max(int(value) + 1, int(value) * 2))
            return min(float(param_max), float(value) * 2)

        if meta["type"] == "integer":
            return max(int(param_min), int(value) // 2)
        return max(float(param_min), float(value) / 2)

    def _run_monotonic_search_only(self, index_type):
        search_idxs = self._get_search_param_idxs(index_type)
        if not search_idxs:
            return

        base_params = self.vdb_config.get_original_param()
        current_values = {}
        directions = {}
        for idx in search_idxs:
            param_name = self.vdb_config.param_names[idx]
            direction = self.search_param_directions.get(index_type).get(param_name)
            directions[idx] = direction
            value_range = self._get_effective_search_range(idx)
            if self.vdb_config.param_meta[idx]["type"] == "enum":
                current_values[idx] = value_range[0] if direction == "increasing" else value_range[-1]
            else:
                param_min, param_max = value_range
                current_values[idx] = param_min if direction == "increasing" else param_max

        prev_recall = None
        old_skip_build = self.skip_build
        try:
            for _ in range(MAX_SEARCH_ONLY_STEPS):
                params = list(base_params)
                for idx, value in current_values.items():
                    params[idx] = value

                self.vdb_config.set_original_param(params)

                record = self._run_search_only_tune(index_type)
                self.update_model()
                if prev_recall is not None and abs(record.recall - prev_recall) < SEARCH_RECALL_STOP_DELTA:
                    break

                prev_recall = record.recall
                next_values = {}
                for idx, value in current_values.items():
                    next_values[idx] = self._next_search_value(idx, value, directions[idx])

                if next_values == current_values:
                    break
                current_values = next_values
        finally:
            self.skip_build = old_skip_build
            self.vdb_config.set_original_param(base_params)

    def reward_transform(self,):
        # to calculate within each index type set
        Y = []
        self.chosen_ref_k = dict.fromkeys(self.polling_index.keys(), None)
        for k, Y_k in self.Y.items():
            Y_k_arr = np.array(Y_k, dtype=float)[:,:2]
            _, popu = fast_non_dominated_sort(Y_k_arr)

            max_ref = np.max(Y_k_arr, axis=0)
            max_ref[~np.isfinite(max_ref) | (max_ref == 0)] = 1.0
            fitness = -1 / (np.abs(Y_k_arr[:,0] / max_ref[0] - Y_k_arr[:,1] / max_ref[1]) + 1e-6)
            fitness[popu[0]] = - fitness[popu[0]]

            chosen_idx = np.argmax(fitness)
            chosen_ref = Y_k_arr[chosen_idx,:]
            self.chosen_ref_k[k] = chosen_ref.tolist()

            norm_ref = chosen_ref.copy()
            norm_ref[~np.isfinite(norm_ref) | (norm_ref == 0)] = 1.0
            Y_k_arr[:,0] /= norm_ref[0]
            Y_k_arr[:,1] /= norm_ref[1]

            Y += Y_k_arr.tolist()

        self.norm_X = [j for item in self.X.values() for j in item]
        self.norm_Y = Y

    def get_normalized_sampled_recall_records(self):
        sampled_X = []
        sampled_Y = []
        for record in self.sampled_recall_records:
            recall = record["recall"]
            if recall is None:
                continue

            index_type = record["index_type"]
            chosen_ref = self.chosen_ref_k.get(index_type)
            if chosen_ref is None:
                continue

            recall_ref = chosen_ref[1]
            if not np.isfinite(recall_ref) or recall_ref == 0:
                recall_ref = 1.0

            sampled_X.append(record["x"])
            sampled_Y.append([float(recall) / recall_ref])
        return sampled_X, sampled_Y

    def get_sampled_recall_blend_weight(self):
        step = max(self._step_id, 0)
        decay = np.exp(-step / SAMPLED_RECALL_BLEND_DECAY_STEPS)
        return (
            SAMPLED_RECALL_BLEND_MIN_WEIGHT
            + (SAMPLED_RECALL_BLEND_INITIAL_WEIGHT - SAMPLED_RECALL_BLEND_MIN_WEIGHT) * decay
        )

    def update_model(self,):
        self.reward_transform()
        sampled_X, sampled_Y = self.get_normalized_sampled_recall_records()
        self.vbo.update_samples(self.norm_X, self.norm_Y, sampled_X, sampled_Y)

    def _build_inequality_constraints(self, fixed_features):
        constraints = []
        for indices, weights, rhs in self.vdb_config.get_inequality_constraints(fixed_features):
            constraints.append(
                (
                    torch.tensor(indices, dtype=torch.long),
                    torch.tensor(weights, dtype=torch.float64),
                    float(rhs),
                )
            )
        return constraints

    def rr_polling(self, search_only=False):
        if search_only:
            raise NotImplementedError("search_only polling uses monotonic real probing")

        polling_idx = self.polling_round_num % len(self.remain_types)
        polling_k = self.remain_types[polling_idx]
        fixed_idxs = [i for i in range(self.knob_num) if i not in self.polling_sys+self.polling_index[polling_k]]
        fixed_features = dict(zip(fixed_idxs, np.array(self.default_conf)[fixed_idxs]))
        fixed_features[0] = self.vdb_config.get_normalized('index_type', polling_k)
        inequality_constraints = self._build_inequality_constraints(fixed_features)
        new_x, ei, new_mean, new_std = self.vbo.recommend(
            fixed_features,
            1,
            inequality_constraints=inequality_constraints,
            sampled_recall_weight=self.get_sampled_recall_blend_weight(),
        )

        self.polling_round_num += 1

        return polling_k, new_x
    
    def successive_abandon(self,):
        self.index_type_score() # update record worst type

        window = 10

        if self.worst_type_record[-window:] == [self.worst_type_record[-1]] * window and len(self.remain_types) > 1:
            self.remain_types.remove(self.worst_type_record[-1])
            self.polling_round_num = 0
 
    def index_type_score(self, ):
        # to calculate within the whole set
        Y = [j for item in self.Y.values() for j in item]
        Y_arr = np.array(Y, dtype=float)[:,:2]
        _, popu = fast_non_dominated_sort(Y_arr)

        max_ref = np.max(Y_arr, axis=0)
        max_ref[~np.isfinite(max_ref) | (max_ref == 0)] = 1.0
        fitness = -1 / (np.abs(Y_arr[:,0] / max_ref[0] - Y_arr[:,1] / max_ref[1]) + 1e-6)
        fitness[popu[0]] = - fitness[popu[0]]

        chosen_idx = np.argmax(fitness)
        self.chosen_ref_whole = Y_arr[chosen_idx,:]
        norm_ref = self.chosen_ref_whole.copy()
        norm_ref[~np.isfinite(norm_ref) | (norm_ref == 0)] = 1.0

        self.delta_hv = dict.fromkeys(self.remain_types, -9999)

        for k in self.remain_types:
            Y_nok = [j for i,item in self.Y.items() if i != k for j in item]

            Y_nok_arr = np.array(Y_nok, dtype=float)[:,:2] / norm_ref
            _, popu_nok = fast_non_dominated_sort(Y_nok_arr)
            popu0_nok = Y_nok_arr[popu_nok[0],:]

            self.delta_hv[k] = hypervolume_calcu(popu0_nok, ref_point=[0.5,0.5])

        self.worst_type_record.append(max(self.delta_hv, key=lambda k: self.delta_hv[k]))

    def _run_sampled_test(self):
        if self.sampled_vdb_engine is None:
            self.current_sampled_record = None
            return

        self.current_sampled_record = self._test_on_sampled_dataset()

    def _should_skip_full_test(self):
        if self.current_sampled_record is None:
            return False

        sampled_recall = self.current_sampled_record["recall"]
        if (
            sampled_recall is None
            or sampled_recall < MIN_SAMPLED_RECALL_FOR_FULL_TEST
            or sampled_recall > MAX_SAMPLED_RECALL_FOR_FULL_TEST
        ):
            return True

        # an alturnative simple implementation
        return False

    def _test_on_sampled_dataset(self):
        sampled_record = {
            "sampled_step_id": self._step_id + 1,
            "params": dict(
                zip(
                    self.vdb_config.param_names,
                    self.vdb_config.get_original_param(),
                )
            ),
            "index_time": 0,
            "query_time": 0,
            "query_throughput": 0,
            "recall": 0,
            "record_nr": 0,
            "query_latency": 0,
        }

        self.sampled_vdb_engine.start()
        try:
            try:
                sampled_record["index_time"] = self.sampled_vdb_engine.build()
                query_time, recall, query_count = self.sampled_vdb_engine.query(
                    self._top_k,
                    test=True,
                    ratio=self._single_test_query_ratio,
                )
                sampled_record["query_time"] = query_time
                sampled_record["query_throughput"] = query_count / query_time if query_time > 0 else 0.0
                sampled_record["recall"] = recall
                sampled_record["record_nr"] = query_count
                sampled_record["query_latency"] = query_time / query_count if query_count > 0 else 0.0
            except:
                pass
        finally:
            self.sampled_vdb_engine.stop()

        return sampled_record

    def _build_extra_record(self, search_only=False):
        extra = {}

        if self.current_sampled_record is not None:
            extra.update({
                "sampled_index_time": self.current_sampled_record["index_time"],
                "sampled_query_time": self.current_sampled_record["query_time"],
                "sampled_query_throughput": self.current_sampled_record["query_throughput"],
                "sampled_recall": self.current_sampled_record["recall"],
                "sampled_record_nr": self.current_sampled_record["record_nr"],
                "sampled_query_latency": self.current_sampled_record["query_latency"],
            })

        extra["search_only"] = search_only
        return extra

    def _single_tune_impl(self):
        self._step_id = self._step_id + 1
        if self.current_skip_full_test:
            return TuningRecord(
                step_id=self._step_id,
                phase="tune",
                dataset_name=self.dataset_name,
                build_parallel=BUILD_PARALLEL,
                search_parallel=SEARCH_PARALLEL,
                params=dict(
                    zip(
                        self.vdb_config.param_names,
                        self.vdb_config.get_original_param(),
                    )
                ),
                index_time=0.0,
                query_time=0.0,
                recall=0,
                record_nr=0,
                query_throughput=0.0,
                query_latency=0.0,
                skip=True,
                extra=self._build_extra_record(search_only=self.skip_build),
            )

        print(f"[FastVTuner] round {self._step_id}: start tune", flush=True)
        try:
            if self.skip_build:
                build_time = 0
            else:
                build_time = self.vdb_engine.build()
            query_time, recall, query_count = self.vdb_engine.query(
                self._top_k,
                test=False,
                ratio=self._single_tune_query_ratio,
            )
            query_throughput = query_count / query_time if query_time > 0 else 0.0
            query_latency = query_time / query_count if query_count > 0 else 0.0
        except:
            build_time = 0
            query_time, recall, query_count = 0, 0, 0
            query_throughput = 0
            query_latency = 0

        return TuningRecord(
            step_id=self._step_id,
            phase="tune",
            dataset_name=self.dataset_name,
            build_parallel=BUILD_PARALLEL,
            search_parallel=SEARCH_PARALLEL,
            params=dict(
                zip(
                    self.vdb_config.param_names,
                    self.vdb_config.get_original_param(),
                )
            ),
            index_time=build_time,
            query_time=query_time,
            recall=recall if self.current_sampled_record is None else self.current_sampled_record["recall"],
            record_nr=query_count,
            query_throughput=query_throughput,
            query_latency=query_latency,
            extra=self._build_extra_record(search_only=self.skip_build),
        )

    # in case of failed building
    def _single_test_impl(self):
        if self.current_skip_full_test:
            return TuningRecord(
                step_id=self._step_id,
                phase="test",
                dataset_name=self.dataset_name,
                build_parallel=BUILD_PARALLEL,
                search_parallel=SEARCH_PARALLEL,
                params=dict(
                    zip(
                        self.vdb_config.param_names,
                        self.vdb_config.get_original_param(),
                    )
                ),
                index_time=0.0,
                query_time=0.0,
                recall=0,
                record_nr=0,
                query_throughput=0.0,
                query_latency=0.0,
                skip=True,
                extra=self._build_extra_record(),
            )

        print(f"[FastVTuner] round {self._step_id}: start test", flush=True)
        try:
            query_time, recall, query_count = self.vdb_engine.query(
                self._top_k,
                test=True,
                ratio=self._single_test_query_ratio,
            )
            query_throughput = query_count / query_time if query_time > 0 else 0.0
            query_latency = query_time / query_count if query_count > 0 else 0.0
        except:
            query_time, recall, query_count = 0, 0, 0
            query_throughput = 0
            query_latency = 0
        return TuningRecord(
            step_id=self._step_id,
            phase="test",
            dataset_name=self.dataset_name,
            build_parallel=BUILD_PARALLEL,
            search_parallel=SEARCH_PARALLEL,
            params=dict(
                zip(
                    self.vdb_config.param_names,
                    self.vdb_config.get_original_param(),
                )
            ),
            index_time=0.0,
            query_time=query_time,
            recall=recall if self.current_sampled_record is None else self.current_sampled_record["recall"],
            record_nr=query_count,
            query_throughput=query_throughput,
            query_latency=query_latency,
            extra=self._build_extra_record(),
        )

def main():
    system = FastVTunerSystem(
        vdb_name="milvus",
        dataset_name="gist",
        sampled_dataset_name="gist-p-10",
        # dataset_name="gist-p-1",
        # sampled_dataset_name="gist-p-1",
    )
    
    # for i in range(200):
    for i in range(65):
        system.step()


if __name__ == "__main__":
    main()
