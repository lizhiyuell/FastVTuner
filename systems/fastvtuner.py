import sys 
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from systems.base import SystemBase, TuningRecord
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
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels import ProductKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.priors.torch_priors import GammaPrior
from scipy.stats import qmc


REF_POINT = torch.tensor([0.5,0.5])

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
        self.bounds = torch.tensor([[0.0] * self.knob_num, [1.0] * self.knob_num])
        self.seed = seed
        self.X_init = None
        self.Y_init = None

        self.kernel_init()
    
    def kernel_init(self,):
        covar_module1 = MaternKernel(
                nu=2.5,
                active_dims=(0),
                lengthscale_prior=GammaPrior(3.0, 6.0),
            )
        covar_module2 = MaternKernel(
                nu=2.5,
                active_dims=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),
                lengthscale_prior=GammaPrior(3.0, 6.0),
            )
        
        product_covar_module = ProductKernel(covar_module1, covar_module2)

        self.covar_module = ScaleKernel(
            product_covar_module,
            outputscale_prior=GammaPrior(2.0, 0.15),
            )
    
    def recommend(self, fixed_features, q):
        # assume 2-dim output: [fitness, recall]
        
        qehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        with torch.no_grad():
            pred = self.model.posterior(self.X_init).mean

        partitioning = FastNondominatedPartitioning(ref_point=REF_POINT, Y=pred,)
        
        acq_func = qLogExpectedHypervolumeImprovement(
            model=self.model,
            ref_point=REF_POINT,
            partitioning=partitioning,
            sampler=qehvi_sampler,
        )

        candidate, ei = optimize_acqf(
            acq_func, bounds=self.bounds, q=q, num_restarts=10, raw_samples=100, 
            fixed_features=fixed_features, 
            options={'seed':self.seed}
            )
        new_x = candidate.detach()
        
        with torch.no_grad():
            new_x_mean = self.model.posterior(new_x).mean
            new_x_std = torch.sqrt(self.model.posterior(new_x).variance)

        return new_x.numpy(), ei.item(), new_x_mean.numpy(), new_x_std.numpy()
    
    def update_samples(self, X, Y,):
        self.X_init = torch.tensor(X,dtype=torch.float64)
        self.Y_init = torch.tensor(Y,dtype=torch.float64)

        models = []
        self.stands = []

        for i in range(self.Y_init.shape[-1]):
            train_y = self.Y_init[..., i : i + 1]
            models.append(SingleTaskGP(
                self.X_init, train_y,
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
        seed=1206
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
        torch.manual_seed(seed)
        random.seed(seed)

        self.polling_sys = [0] + [9,10,11,12,13,14,15]
        self.polling_index = {
            'FLAT': [],
            'IVF_FLAT': [1,2],
            'IVF_SQ8': [1,2],
            'IVF_PQ': [1,2,3,4],
            'HNSW': [5,6,7],
            'SCANN': [1,2,8],
        }

        self.X = {key: [] for key in self.polling_index.keys()}
        self.Y = {key: [] for key in self.polling_index.keys()}

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

            # run the exp with default configurations
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

        self.update_model()

    def step(self,):
        if len(self.remain_types) > 1:
            self.successive_abandon()

        polling_k, new_x = self.rr_polling()

        # the new_x is an array of parameter array, we detach it
        self.vdb_config.set_normalized_param(new_x[0])
        self.vdb_engine.start()
        res_record = self.single_tune()
        self.single_test()
        self.vdb_engine.stop()

        self.X[polling_k].append(self.vdb_config.get_normalized_param())
        self.Y[polling_k].append([
            res_record.query_throughput,
            res_record.recall,
            res_record.query_latency
        ])
        
        self.update_model()

    def reward_transform(self,):
        # to calculate within each index type set
        Y = []
        self.chosen_ref_k = dict.fromkeys(self.polling_index.keys(), None)
        for k, Y_k in self.Y.items():
            Y_k_arr = np.array(Y_k)[:,:2]
            _, popu = fast_non_dominated_sort(Y_k_arr)

            fitness = -1 / (np.abs(Y_k_arr[:,0] / np.max(Y_k_arr[:,0]) - Y_k_arr[:,1] / np.max(Y_k_arr[:,1])) + 1e-6)
            fitness[popu[0]] = - fitness[popu[0]]

            chosen_idx = np.argmax(fitness)
            chosen_ref = Y_k_arr[chosen_idx,:]
            self.chosen_ref_k[k] = chosen_ref.tolist()

            Y_k_arr[:,0] /= chosen_ref[0]
            Y_k_arr[:,1] /= chosen_ref[1]

            Y += Y_k_arr.tolist()

        self.norm_X = [j for item in self.X.values() for j in item]
        self.norm_Y = Y

    def update_model(self,):
        self.reward_transform()
        self.vbo.update_samples(self.norm_X, self.norm_Y)

    def rr_polling(self,):
        polling_idx = self.polling_round_num % len(self.remain_types)
        polling_k = self.remain_types[polling_idx]

        fixed_idxs = [i for i in range(self.knob_num) if i not in self.polling_sys+self.polling_index[polling_k]]
        fixed_features = dict(zip(fixed_idxs, np.array(self.default_conf)[fixed_idxs]))
        fixed_features[0] = self.vdb_config.get_normalized('index_type', polling_k)
        new_x, ei, new_mean, new_std = self.vbo.recommend(fixed_features, 1)

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
        Y_arr = np.array(Y)[:,:2]
        _, popu = fast_non_dominated_sort(Y_arr)

        fitness = -1 / (np.abs(Y_arr[:,0] / np.max(Y_arr[:,0]) - Y_arr[:,1] / np.max(Y_arr[:,1])) + 1e-6)
        fitness[popu[0]] = - fitness[popu[0]]

        chosen_idx = np.argmax(fitness)
        self.chosen_ref_whole = Y_arr[chosen_idx,:]

        self.delta_hv = dict.fromkeys(self.remain_types, -9999)

        for k in self.remain_types:
            Y_nok = [j for i,item in self.Y.items() if i != k for j in item]

            Y_nok_arr = np.array(Y_nok)[:,:2] / self.chosen_ref_whole
            _, popu_nok = fast_non_dominated_sort(Y_nok_arr)
            popu0_nok = Y_nok_arr[popu_nok[0],:]

            self.delta_hv[k] = hypervolume_calcu(popu0_nok, ref_point=[0.5,0.5])

        self.worst_type_record.append(max(self.delta_hv, key=lambda k: self.delta_hv[k]))

    def _single_tune_impl(self):
        self._step_id = self._step_id + 1
        print(f"[FastVTuner] round {self._step_id}: start build", flush=True)
        build_time = self.vdb_engine.build()
        query_time, recall, query_count = self.vdb_engine.query(
            self._top_k,
            test=False,
            ratio=self._single_tune_query_ratio,
        )
        query_throughput = query_count / query_time if query_time > 0 else 0.0
        query_latency = query_time / query_count if query_count > 0 else 0.0

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
            recall=recall,
            record_nr=query_count,
            query_throughput=query_throughput,
            query_latency=query_latency,
        )

    def _single_test_impl(self):
        print(f"[FastVTuner] round {self._step_id}: start test", flush=True)
        query_time, recall, query_count = self.vdb_engine.query(
            self._top_k,
            test=True,
            ratio=self._single_test_query_ratio,
        )
        query_throughput = query_count / query_time if query_time > 0 else 0.0
        query_latency = query_time / query_count if query_count > 0 else 0.0

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
            recall=recall,
            record_nr=query_count,
            query_throughput=query_throughput,
            query_latency=query_latency,
        )

def main():
    system = FastVTunerSystem(
        vdb_name="milvus",
        # dataset_name="gist",
        # dataset_name="gist-p-10",
        dataset_name="gist-p-1",
    )
    
    for i in range(100):
        system.step()


if __name__ == "__main__":
    main()
