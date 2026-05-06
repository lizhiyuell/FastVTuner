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
from scipy.stats import qmc


REF_POINT = torch.tensor([0.5, 0.5], dtype=torch.float64)
NR_SEARCH_PER_BUILD = 5
SEARCH_ONLY_SAMPLE_NUM = 256
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
        self.search_only_candidates = []

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
            self.vdb_engine.stop()

            self.X[k].append(self.vdb_config.get_normalized_param())
            self.Y[k].append([
                res_record.query_throughput,
                res_record.recall,
                res_record.query_latency
            ])

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
                    self.search_only_candidates = []
                    for _ in range(NR_SEARCH_PER_BUILD):
                        _, new_x = self.rr_polling(search_only=True)
                        if new_x is None:
                            break
                        self.vdb_config.set_normalized_param(new_x[0])
                        self.current_sampled_record = None
                        self.skip_build = True
                        search_record = self.single_tune()
                        self._append_tuning_result(polling_k, search_record)
                        self.update_model()
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

    def _valid_normalized_param(self, params):
        for indices, weights, rhs in self.vdb_config.get_inequality_constraints({}):
            total = 0.0
            for idx, weight in zip(indices, weights):
                total += float(params[idx]) * float(weight)
            if total < float(rhs) - 1e-9:
                return False
        return True

    def _search_param_key(self, params, allowed_idxs):
        key = []
        for idx in allowed_idxs:
            name = self.vdb_config.param_names[idx]
            key.append(self.vdb_config.get_original(name, params[idx]))
        return tuple(key)

    def _predict_search_candidates(self, candidates):
        X = torch.tensor(candidates, dtype=torch.float64)
        with torch.no_grad():
            recall = self.vbo.model.models[1].posterior(X).mean.squeeze(-1).numpy()
            tput = self.vbo.model.models[0].posterior(X).mean.squeeze(-1).numpy()
        return recall, tput

    def _make_search_only_candidates(self, polling_k, allowed_idxs):
        current = np.array(self.vdb_config.get_normalized_param(), dtype=float)
        search_dim = len(allowed_idxs)
        if search_dim == 0:
            return [current]

        sampler = qmc.Sobol(
            d=search_dim,
            scramble=True,
            seed=self.seed + self._step_id,
        )
        sampled = sampler.random(SEARCH_ONLY_SAMPLE_NUM)

        candidates = []
        seen_keys = set()
        for x in self.X[polling_k]:
            seen_keys.add(self._search_param_key(x, allowed_idxs))

        for sample in sampled:
            candidate = current.copy()
            for idx, value in zip(allowed_idxs, sample):
                candidate[idx] = value

            if not self._valid_normalized_param(candidate):
                continue

            key = self._search_param_key(candidate, allowed_idxs)
            if key in seen_keys:
                continue

            seen_keys.add(key)
            candidates.append(candidate)

        if len(candidates) <= NR_SEARCH_PER_BUILD:
            return candidates

        recall, tput = self._predict_search_candidates(candidates)
        finite_idxs = np.where(np.isfinite(recall))[0]
        if len(finite_idxs) <= NR_SEARCH_PER_BUILD:
            return [candidates[idx] for idx in finite_idxs]

        recall_min = float(np.min(recall[finite_idxs]))
        recall_max = float(np.max(recall[finite_idxs]))
        print(f"step_id {self._step_id}, min_recall={recall_min}, max_recall={recall_max}")
        if recall_max <= recall_min:
            chosen_idxs = finite_idxs[np.argsort(-tput[finite_idxs])[:NR_SEARCH_PER_BUILD]]
            return [candidates[idx] for idx in chosen_idxs]

        target_recalls = np.linspace(recall_min, recall_max, NR_SEARCH_PER_BUILD + 2)[1:-1]
        chosen_idxs = []
        remaining = set(finite_idxs.tolist())
        for target in target_recalls:
            if not remaining:
                break
            best_idx = min(
                remaining,
                key=lambda idx: (abs(float(recall[idx]) - target), -float(tput[idx])),
            )
            chosen_idxs.append(best_idx)
            remaining.remove(best_idx)

        return [candidates[idx] for idx in chosen_idxs]

    def _rr_polling_search_only(self):
        index_type_idx = self.vdb_config.get_param_index("index_type")
        polling_k = self.vdb_config.get_original_param()[index_type_idx]
        allowed_idxs = [
            idx for idx in self.polling_index[polling_k]
            if self.vdb_config.param_meta[idx]["class"] == "searching"
        ]

        if not self.search_only_candidates:
            self.search_only_candidates = self._make_search_only_candidates(polling_k, allowed_idxs)

        if self.search_only_candidates:
            return polling_k, np.array([self.search_only_candidates.pop(0)])

        return polling_k, None

    def rr_polling(self, search_only=False):
        if search_only:
            return self._rr_polling_search_only()

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
    
    for i in range(200):
    # for i in range(65):
        system.step()


if __name__ == "__main__":
    main()
