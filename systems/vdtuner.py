from __future__ import annotations

import random
import re
import subprocess as sp
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

# from botorch.acquisition import (
#     ConstrainedExpectedImprovement,
#     ExpectedImprovement,
#     LogExpectedImprovement,
# )
# from botorch.acquisition.multi_objective.logei import (
#     qLogExpectedHypervolumeImprovement as qHypervolumeImprovement,
# )
# from botorch.exceptions.warnings import InputDataWarning


from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement, ConstrainedExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement

from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels import ProductKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.priors.torch_priors import GammaPrior

from systems.base import SystemBase, TuningRecord
from systems.common import *

# 这里的bench_path是不是也需要改一下了？现在这个应该没用了
class RealEnv:
    def __init__(self, bench_path: str | Path = RUN_PY, knob_path: str | Path = KNOB_PATH) -> None:
        self.bench_path = Path(bench_path)
        self.knob_stand = KnobStand(knob_path)
        self.names = list(self.knob_stand.knobs_detail.keys())
        self.t1 = time.time()
        self.t2 = time.time()
        self.sampled_times = 0
        self.X_record = []
        self.Y1_record = []
        self.Y2_record = []
        self.Y_record = []

    def get_state(self, knob_vals_arr, run_params, log_file):
        dataset_name = _dataset_token(run_params)
        if not dataset_name:
            raise ValueError("Unable to infer dataset name from run_params")

        Y1, Y2, Y3 = [], [], []
        for record in knob_vals_arr:
            conf_value = [
                self.knob_stand.scale_back(self.names[j], knob_val)[1]
                for j, knob_val in enumerate(record)
            ]

            # 这一块也得改一下？？不同数据库肯定不能这样搞，不统一
            index_value, system_value = conf_value[:9], conf_value[9:]
            index_name, system_name = self.names[:9], self.names[9:]
            index_conf = dict(zip(index_name, index_value))
            system_conf = dict(zip(system_name, system_value))

            configure_index(*filter_index_rule(index_conf))
            configure_system(filter_system_rule(system_conf))

            # 这一块运行的代码，后面再对对？？？目前这个写法肯定是有问题的

            try:
                print("----------")
                print(index_conf)
                print(system_conf)
                start_time = time.time()
                # 这里需要按照修改？？？而且从这里固定query_ratio来看，我们这一块代码应该是根本没有跑的！
                result_bundle = _run_benchmark(
                    dataset_name,
                    tune_query_ratio=0.5,
                    test_query_ratio=0.5,
                    split_seed=42,
                )
                end_time = time.time()
                print(f"[One round] {end_time - start_time}s")

                tune_result = result_bundle["tune"]
                query_time = float(tune_result.get("query_time", 0.0))
                record_nr = int(tune_result.get("record_nr", 0))
                y1 = record_nr / query_time if query_time > 0 else 0.0
                y2 = float(tune_result["recall"])
                y3 = query_time / record_nr if record_nr > 0 else 0.0
                upload_time = float(tune_result["index_time"])
                total_load_time = float(tune_result["upload_total"])
                search_time = query_time
                self.Y1_record.append(y1)
                self.Y2_record.append(y2)
            except Exception:
                y1 = min(self.Y1_record) if self.Y1_record else 0.0
                y2 = min(self.Y2_record) if self.Y2_record else 0.0
                upload_time = 0.0
                total_load_time = 0.0
                search_time = 0.0
                y3 = 0.0

            self.sampled_times += 1
            self.t2 = time.time()
            print(f"[{self.sampled_times}] {int(self.t2 - self.t1)} {y1} {y2} {y3}")
            sp.run(
                [
                    "bash",
                    "-lc",
                    f"echo [{self.sampled_times}] {int(self.t2 - self.t1)} {index_conf} {system_conf} {upload_time} {total_load_time} {search_time} {y1} {y2} {y3} >> {log_file}",
                ],
                stdout=sp.PIPE,
                stderr=sp.PIPE,
                check=False,
            )

            Y1.append(y1)
            Y2.append(y2)
            Y3.append(y3)

        return np.array([Y1, Y2, Y3]).T

    # 将参数配置到系统中
    def config_system_with_params(self, knob_vals_arr):
        conf_value = [
            self.knob_stand.scale_back(self.names[j], knob_val)[1]
            for j, knob_val in enumerate(knob_vals_arr)
        ]

        # 这里肯定也需要对应修改？？？适配不同的vdb？？
        index_value, system_value = conf_value[:9], conf_value[9:]
        index_name, system_name = self.names[:9], self.names[9:]
        index_conf = dict(zip(index_name, index_value))
        system_conf = dict(zip(system_name, system_value))
        configure_index(*filter_index_rule(index_conf))
        configure_system(filter_system_rule(system_conf))

    def config_system_with_params_original(self, conf_value):
        # 同理？？？
        index_value, system_value = conf_value[:9], conf_value[9:]
        index_name, system_name = self.names[:9], self.names[9:]
        index_conf = dict(zip(index_name, index_value))
        system_conf = dict(zip(system_name, system_value))
        configure_index(*filter_index_rule(index_conf))
        configure_system(filter_system_rule(system_conf))

    def default_conf(self):
        return [
            self.knob_stand.scale_forward(k, v["default"])
            for k, v in self.knob_stand.knobs_detail.items()
        ]

# 根据dimension，配置m的值，这个函数应该只是针对milvus是需要的？？
def update_m_with_dimension(dimension):
    factors = []
    for i in range(1, dimension + 1):
        if dimension % i == 0:
            factors.append(i)
    factors.sort()

    default_value = min(factors, key=lambda x: abs(x - 10))
    for filename in [INDEX_PARAM_PATH, KNOB_PATH]:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["m"]["default"] = default_value
        data["m"]["enum_values"] = factors
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# 这一块的代码，应该也得放在common里面？？
def cast_value(s: str):
    s = s.strip()
    if s == "True":
        return True
    if s == "False":
        return False
    if NUM_RE.match(s):
        temp_num = float(s)
        return temp_num if temp_num < 1 else int(temp_num)
    return s

def extract_line_params(line):
    env = RealEnv()
    knob_names = env.names
    knob_num = len(knob_names)
    x = [0 for _ in range(knob_num)]

    cur_idx = 0
    values = []
    for m in BRACE_RE.finditer(line):
        inner = m.group(1)
        for part in inner.split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            k, v = part.split(":", 1)
            original_val = cast_value(v)
            if k == "dataCoord*segment*sealProportion":
                original_val = original_val * 100
            values.append(original_val)
            cur_idx += 1
    return values


# 这个显然也不应该在这个文件？？
def get_Pareto_result(filename):
    res = []
    with open(filename, "r", encoding="utf-8") as ifs:
        for line in ifs:
            content = line.split()
            tput = float(content[-3])
            recall = float(content[-2])
            res.append([tput, recall, extract_line_params(line)])

    res = sorted(res, key=lambda x: x[0])
    final = [[], [], []]
    max_recall = 0
    pos = len(res) - 1
    while pos >= 0:
        t, r, p = res[pos][0], res[pos][1], res[pos][2]
        if r > max_recall:
            final[0].append(t)
            final[1].append(r)
            final[2].append(p)
        max_recall = max(max_recall, r)
        pos -= 1

    final[0].reverse()
    final[1].reverse()
    final[2].reverse()
    return final


REF_POINT = torch.tensor([0.5, 0.5])

def hypervolume_calcu(all_sol, ref_point=[0, 0], opt_max=True):
    rank, f = fast_non_dominated_sort(all_sol)
    pareto_sol = np.array(all_sol)[f[0]]
    if opt_max:
        pareto_sol = pareto_sol[pareto_sol[:, 0].argsort(kind="mergesort")]
    else:
        pareto_sol = pareto_sol[pareto_sol[:, 0].argsort(kind="mergesort")[::-1]]

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

    def kernel_init(self):
        covar_module1 = MaternKernel(
            nu=2.5,
            active_dims=(0,),
            lengthscale_prior=GammaPrior(3.0, 6.0),
        )
        covar_module2 = MaternKernel(
            nu=2.5,
            active_dims=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            lengthscale_prior=GammaPrior(3.0, 6.0),
        )
        product_covar_module = ProductKernel(covar_module1, covar_module2)
        self.covar_module = ScaleKernel(
            product_covar_module,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )

    def recommend(self, fixed_features, q, rr_cons):
        qehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        with torch.no_grad():
            pred = self.model.posterior(self.X_init).mean

        partitioning = FastNondominatedPartitioning(ref_point=REF_POINT, Y=pred)
        acq_func = qExpectedHypervolumeImprovement(
            model=self.model,
            ref_point=REF_POINT,
            partitioning=partitioning,
            sampler=qehvi_sampler,
        )

        candidate, ei = optimize_acqf(
            acq_func,
            bounds=self.bounds,
            q=q,
            num_restarts=10,
            raw_samples=100,
            fixed_features=fixed_features,
            options={"seed": self.seed},
        )
        new_x = candidate.detach()

        with torch.no_grad():
            new_x_mean = self.model.posterior(new_x).mean
            new_x_std = torch.sqrt(self.model.posterior(new_x).variance)

        return new_x.numpy(), ei.item(), new_x_mean.numpy(), new_x_std.numpy()

    def update_samples(self, X, Y):
        self.X_init = torch.tensor(X, dtype=torch.float64)
        self.Y_init = torch.tensor(Y, dtype=torch.float64)

        models = []
        self.stands = []
        for i in range(self.Y_init.shape[-1]):
            train_y = self.Y_init[..., i : i + 1]
            models.append(
                SingleTaskGP(
                    self.X_init,
                    train_y,
                    outcome_transform=Standardize(m=1),
                )
            )

        self.model = ModelListGP(*models)
        self.mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(self.mll)


class PollingBayesianOptimization:
    def __init__(
        self,
        env,
        seed=1206,
        threshold=None,
        run_params='"" "" glove-100-angular', #MNC
        log_file="record.log",
        pobo_file="pobo_record.log",
    ) -> None:
        self.env = env
        self.knob_num = len(env.names)
        self.default_conf = self.env.default_conf()
        self.vbo = EHVIBO(self.knob_num, seed=seed)

        self.seed = seed
        torch.manual_seed(seed)
        random.seed(seed)

        #MNC
        self.polling_sys = [0] + [9, 10, 11, 12, 13, 14, 15]
        self.polling_index = {
            "FLAT": [],
            "IVF_FLAT": [1, 2],
            "IVF_SQ8": [1, 2],
            "IVF_PQ": [1, 2, 3, 4],
            "HNSW": [5, 6, 7],
            "SCANN": [1, 2, 8],
            "AUTOINDEX": [],
        }

        self.threshold = threshold
        self.X = {k: [] for k in self.polling_index.keys()}
        self.Y = {k: [] for k in self.polling_index.keys()}

        self.remain_types = list(self.polling_index.keys())
        self.polling_round_num = 0
        self.worst_type_record = []

        self.run_params = run_params
        self.log_file = log_file
        self.pobo_file = pobo_file

    def init_sample(self):
        for k in self.remain_types:
            x = [self.default_conf[:]]
            x[0][0] = self.env.knob_stand.scale_forward("index_type", k)
            y = self.env.get_state(x, self.run_params, self.log_file)

            self.X[k] = self.X[k] + x
            self.Y[k] = self.Y[k] + y.tolist()

        self.update_model()


    # -------------- ??
    def total_steps(self) -> int:
        return sum(len(items) for items in self.X.values())

    def _initial_types_pending(self) -> list[str]:
        return [k for k in self.polling_index if len(self.X[k]) == 0]

    def propose(self) -> tuple[str, list[float]]:
        pending_types = self._initial_types_pending()
        if pending_types:
            index_type = pending_types[0]
            x = self.default_conf[:]
            x[0] = self.env.knob_stand.scale_forward("index_type", index_type)
            return index_type, x

        if len(self.remain_types) > 1:
            self.successive_abandon()

        with open(self.pobo_file, "a", encoding="utf-8") as f:
            f.write(
                f"{list(self.chosen_ref_whole)} {list(self.chosen_ref_k.values())} "
                f"{list(self.delta_hv.values())} {self.worst_type_record[-1]} {self.remain_types}\n"
            )

        polling_k, new_x = self.rr_polling()
        return polling_k, new_x[0].tolist()

    def observe(self, polling_k: str, x: Sequence[float], y: Sequence[float]) -> None:
        self.X[polling_k] = self.X[polling_k] + [list(x)]
        self.Y[polling_k] = self.Y[polling_k] + [list(y)]

        if self._initial_types_pending():
            return

        self.update_model()
        self._model_ready = True
    # -------------- ??


    # -------------- 这个step函数真的还在起作用吗？
    def step(self):
        if len(self.remain_types) > 1:
            self.successive_abandon()

        # ------- 这一块代码至少不要了
        sp.run(
            f'echo {list(self.chosen_ref_whole)} {list(self.chosen_ref_k.values())}  {list(self.delta_hv.values())} {self.worst_type_record[-1]} {self.remain_types} >> {self.pobo_file}',
            shell=True,
            stdout=sp.PIPE,
        )
        # ------- 这一块代码至少不要了

        polling_k, new_x = self.rr_polling()
        new_y = self.env.get_state(new_x, self.run_params, self.log_file)

        self.X[polling_k] = self.X[polling_k] + new_x.tolist()
        self.Y[polling_k] = self.Y[polling_k] + new_y.tolist()
        self.update_model()

    def reward_transform(self):
        Y = []
        self.chosen_ref_k = dict.fromkeys(self.polling_index.keys(), None)
        for k, Y_k in self.Y.items():
            Y_k_arr = np.array(Y_k)[:, :2]
            _, popu = fast_non_dominated_sort(Y_k_arr)

            fitness = -1 / (
                np.abs(
                    Y_k_arr[:, 0] / np.max(Y_k_arr[:, 0])
                    - Y_k_arr[:, 1] / np.max(Y_k_arr[:, 1])
                )
                + 1e-6
            )
            fitness[popu[0]] = -fitness[popu[0]]

            chosen_idx = np.argmax(fitness)
            chosen_ref = Y_k_arr[chosen_idx, :]
            self.chosen_ref_k[k] = chosen_ref.tolist()

            Y_k_arr[:, 0] /= chosen_ref[0]
            Y_k_arr[:, 1] /= chosen_ref[1]
            Y += Y_k_arr.tolist()

        self.norm_X = [j for item in self.X.values() for j in item]
        self.norm_Y = Y

    def update_model(self):
        self.reward_transform()
        self.vbo.update_samples(self.norm_X, self.norm_Y)

    def rr_polling(self):
        polling_idx = self.polling_round_num % len(self.remain_types)
        polling_k = self.remain_types[polling_idx]

        fixed_idxs = [
            i
            for i in range(self.knob_num)
            if i not in self.polling_sys + self.polling_index[polling_k]
        ]
        fixed_features = dict(zip(fixed_idxs, np.array(self.default_conf)[fixed_idxs]))
        fixed_features[0] = self.env.knob_stand.scale_forward("index_type", polling_k)
        new_x, ei, new_mean, new_std = self.vbo.recommend(fixed_features, 1, self.threshold)

        self.polling_round_num += 1
        return polling_k, new_x

    def successive_abandon(self):
        self.index_type_score()
        window = 10
        if (
            self.worst_type_record[-window:]
            == [self.worst_type_record[-1]] * window
            and len(self.remain_types) > 1
        ):
            self.remain_types.remove(self.worst_type_record[-1])
            self.polling_round_num = 0

    def index_type_score(self):
        Y = [j for item in self.Y.values() for j in item]
        Y_arr = np.array(Y)[:, :2]
        _, popu = fast_non_dominated_sort(Y_arr)

        fitness = -1 / (
            np.abs(
                Y_arr[:, 0] / np.max(Y_arr[:, 0]) - Y_arr[:, 1] / np.max(Y_arr[:, 1])
            )
            + 1e-6
        )
        fitness[popu[0]] = -fitness[popu[0]]

        chosen_idx = np.argmax(fitness)
        self.chosen_ref_whole = Y_arr[chosen_idx, :]

        self.delta_hv = dict.fromkeys(self.remain_types, -9999)
        for k in self.remain_types:
            Y_nok = [j for i, item in self.Y.items() if i != k for j in item]
            Y_nok_arr = np.array(Y_nok)[:, :2] / self.chosen_ref_whole
            _, popu_nok = fast_non_dominated_sort(Y_nok_arr)
            popu0_nok = Y_nok_arr[popu_nok[0], :]
            self.delta_hv[k] = hypervolume_calcu(popu0_nok, ref_point=[0.5, 0.5])

        self.worst_type_record.append(max(self.delta_hv, key=lambda k: self.delta_hv[k]))


# -------------检查完成-------------

class VDTunerSystem(SystemBase):
    # 兼容 SystemBase 接口的 VDTuner 包装器，对接旧版 VDTuner 实现。

    def __init__(
        self,
        single_tune_query_ratio: float = 0.5,
        single_test_query_ratio: float = 0.5,
        seed: int = 42,
        engine_name: str = DEFAULT_ENGINE_NAME,
        results_dir: str | Path = RESULTS_DIR,
        benchmark_timeout: float = 86400.0,
        **fixed_params: Any,
    ) -> None:
        super().__init__(
            single_tune_query_ratio=single_tune_query_ratio,
            single_test_query_ratio=single_test_query_ratio,
            seed=seed,
            **fixed_params,
        )
        self.engine_name = engine_name
        self.results_dir = Path(results_dir)
        self.benchmark_timeout = benchmark_timeout
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.knob_stand = KnobStand(KNOB_PATH)
        self.names = list(self.knob_stand.knobs_detail.keys())
        self.dimension: int | None = None
        self._env = RealEnv()
        self._optimizer: PollingBayesianOptimization | None = None
        self._dataset_run_params: str | None = None
        self._cached_test_record: TuningRecord | None = None
        self._cached_test_params: tuple[Any, ...] | None = None

    def _on_dataset_loaded(self, dataset_name: str) -> None:
        self.dimension = infer_dimension_from_dataset_name(dataset_name) or self.fixed_params.get(
            "dimension"
        )
        if self.dimension:
            update_m_with_dimension(int(self.dimension))
        self._dataset_run_params = dataset_name
        self._optimizer = None
        self._cached_test_record = None
        self._cached_test_params = None

    def _ensure_optimizer(self) -> PollingBayesianOptimization:
        if self.dataset_name is None:
            raise RuntimeError("No dataset is bound. Call load_dataset() first.")
        if self._optimizer is None:
            run_params = self._dataset_run_params or self.dataset_name
            record_file = self.results_dir / f"record_{self.dataset_name}.log"
            pobo_file = self.results_dir / f"pobo_{self.dataset_name}.log"
            self._optimizer = PollingBayesianOptimization(
                self._env,
                seed=self.seed,
                run_params=run_params,
                log_file=str(record_file),
                pobo_file=str(pobo_file),
            )
        return self._optimizer

    def _resolve_dataset_name(self, dataset_name: str | None) -> str:
        self._require_dataset()
        return dataset_name or self.dataset_name

    def _split_real_conf(
        self,
        real_conf: Sequence[Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        index_conf = dict(zip(self.names[:9], real_conf[:9]))
        system_conf = dict(zip(self.names[9:], real_conf[9:]))
        return index_conf, system_conf

    def _apply_real_conf(self, real_conf: Sequence[Any]) -> None:
        index_conf, system_conf = self._split_real_conf(real_conf)
        configure_index(*filter_index_rule(index_conf), engine_name=self.engine_name)
        configure_system(filter_system_rule(system_conf))

    def _params_to_real_conf(self, params: Sequence[Any] | None) -> list[Any]:
        if params is None:
            return [v["default"] for v in self.knob_stand.knobs_detail.values()]
        if len(params) != len(self.names):
            raise ValueError(
                f"Expected {len(self.names)} parameters, got {len(params)}"
            )

        real_vals = []
        for j, knob_name in enumerate(self.names):
            knob = self.knob_stand.knobs_detail[knob_name]
            value = params[j]
            if knob["type"] == "enum":
                if value in knob["enum_values"]:
                    real_vals.append(value)
                elif isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
                    real_vals.append(self.knob_stand.scale_back(knob_name, float(value))[1])
                elif isinstance(value, int) and 0 <= value < len(knob["enum_values"]):
                    real_vals.append(knob["enum_values"][value])
                else:
                    real_vals.append(value)
            elif knob["type"] == "integer":
                if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
                    real_vals.append(self.knob_stand.scale_back(knob_name, float(value))[1])
                else:
                    real_vals.append(int(value))
            else:
                real_vals.append(value)
        return real_vals

    def _build_record(
        self,
        *,
        params: Sequence[Any] | None,
        real_conf: Sequence[Any],
        dataset_name: str,
        phase: str,
        result: dict[str, Any],
    ) -> TuningRecord:
        index_conf, system_conf = self._split_real_conf(real_conf)
        return TuningRecord(
            step_id=0,
            phase=phase,
            dataset_name=dataset_name,
            params={
                "index": index_conf,
                "system": system_conf,
                "normalized": list(params) if params is not None else None,
            },
            index_time=float(result["index_time"]),
            query_time=float(result.get("query_time", 0.0)),
            recall=float(result["recall"]),
            record_nr=int(result.get("record_nr", 0)),
        )

    def _run_configured_pair(
        self,
        params: Sequence[Any] | None = None,
        dataset_name: str | None = None,
    ) -> tuple[TuningRecord, TuningRecord | None]:
        dataset_name = self._resolve_dataset_name(dataset_name)
        real_conf = self._params_to_real_conf(params)
        self._apply_real_conf(real_conf)
        result_bundle = _run_benchmark(
            dataset_name,
            engine_name=self.engine_name,
            timeout=self.benchmark_timeout,
            tune_query_ratio=self.single_tune_query_ratio,
            test_query_ratio=self.single_test_query_ratio,
            split_seed=self.seed,
        )
        tune_record = self._build_record(
            params=params,
            real_conf=real_conf,
            dataset_name=dataset_name,
            phase="tune",
            result=result_bundle["tune"],
        )
        test_result = result_bundle.get("test")
        test_record = None
        if test_result is not None:
            test_record = self._build_record(
                params=params,
                real_conf=real_conf,
                dataset_name=dataset_name,
                phase="test",
                result=test_result,
            )
        return tune_record, test_record

    def _run_test_only(
        self,
        params: Sequence[Any] | None = None,
        dataset_name: str | None = None,
    ) -> TuningRecord:
        dataset_name = self._resolve_dataset_name(dataset_name)
        real_conf = self._params_to_real_conf(params)
        self._apply_real_conf(real_conf)
        _, stop_cmd = _start_server(self.engine_name)
        try:
            test_result = _run_benchmark_phase(
                dataset_name,
                engine_name=self.engine_name,
                timeout=self.benchmark_timeout,
                query_phase="test",
                tune_query_ratio=self.single_tune_query_ratio,
                test_query_ratio=self.single_test_query_ratio,
                split_seed=self.seed,
                skip_upload=False,
            )
        finally:
            _stop_server(stop_cmd)

        return self._build_record(
            params=params,
            real_conf=real_conf,
            dataset_name=dataset_name,
            phase="test",
            result=test_result,
        )

    @staticmethod
    def _params_cache_key(params: Sequence[Any] | None) -> tuple[Any, ...] | None:
        return None if params is None else tuple(params)

    @staticmethod
    def _record_normalized_params(record: TuningRecord) -> tuple[Any, ...] | None:
        normalized = record.params.get("normalized")
        if normalized is None:
            return None
        return tuple(normalized)

    def _invalidate_cached_test_record(self) -> None:
        self._cached_test_record = None
        self._cached_test_params = None

    def _store_cached_test_record(
        self,
        params: Sequence[Any] | None,
        test_record: TuningRecord | None,
    ) -> None:
        self._cached_test_record = test_record
        self._cached_test_params = self._params_cache_key(params) if test_record else None

    def _consume_cached_test_record(
        self,
        params: Sequence[Any] | None,
    ) -> TuningRecord | None:
        requested_key = self._params_cache_key(params)
        if self._cached_test_record is None:
            return None
        cached_key = self._cached_test_params
        if cached_key != requested_key:
            record_key = self._record_normalized_params(self._cached_test_record)
            if record_key != requested_key:
                return None
        record = self._cached_test_record
        self._invalidate_cached_test_record()
        return record

    def _single_tune_impl(self, **kwargs: Any) -> TuningRecord:
        params = kwargs.get("params")
        dataset_name = kwargs.get("dataset_name")
        polling_k = None
        if params is None:
            optimizer = self._ensure_optimizer()
            polling_k, params = optimizer.propose()
        record, test_record = self._run_configured_pair(
            params=params,
            dataset_name=dataset_name,
        )
        self._store_cached_test_record(params, test_record)
        if polling_k is not None:
            avg_query_time = record.query_time / record.record_nr if record.record_nr > 0 else 0.0
            throughput = record.record_nr / record.query_time if record.query_time > 0 else 0.0
            self._ensure_optimizer().observe(
                polling_k,
                params,
                [
                    float(throughput),
                    float(record.recall),
                    float(avg_query_time),
                ],
            )
        return record

    def _single_test_impl(self, **kwargs: Any) -> TuningRecord:
        params = kwargs.get("params")
        dataset_name = kwargs.get("dataset_name")
        cached_record = self._consume_cached_test_record(params)
        if cached_record is not None:
            return cached_record
        return self._run_test_only(params=params, dataset_name=dataset_name)

    # 复现旧目录里 `main_tuner.py` 的入口行为。
    def run_legacy_optimization(
        self,
        datasets: Iterable[str],
        total_nr_step: int = 300,
        result_dir: str | Path | None = None,
    ) -> list[Path]:
        result_root = Path(result_dir) if result_dir else self.results_dir
        result_root.mkdir(parents=True, exist_ok=True)
        produced_logs: list[Path] = []

        for dataset_name in datasets:
            self.load_dataset(dataset_name)
            record_file_name = result_root / f"record_{dataset_name}.log"
            pobo_file_name = result_root / f"pobo_record_{dataset_name}.log"
            if record_file_name.exists():
                record_file_name.unlink()
            if pobo_file_name.exists():
                pobo_file_name.unlink()

            optimizer = PollingBayesianOptimization(
                self._env,
                seed=self.seed,
                run_params=dataset_name,
                log_file=str(record_file_name),
                pobo_file=str(pobo_file_name),
            )
            optimizer.init_sample()
            for _ in range(max(0, total_nr_step - 7)):
                optimizer.step()
            produced_logs.extend([record_file_name, pobo_file_name])

        return produced_logs


DEFAULT_DATASET_RUNS = [
    "glove-100-angular-p-10",
    "glove-100-angular-p-30",
    "glove-100-angular-p-50",
    "glove-100-angular-p-70",
    "glove-100-angular-p-90",
    "glove-100-angular-p-100",
    "gist-960-euclidean-p-10",
    "gist-960-euclidean-p-30",
    "gist-960-euclidean-p-50",
    "gist-960-euclidean-p-70",
    "gist-960-euclidean-p-90",
    "gist-960-euclidean-p-100",
    "deep-image-96-angular-p-10",
    "deep-image-96-angular-p-30",
    "deep-image-96-angular-p-50",
    "deep-image-96-angular-p-70",
    "deep-image-96-angular-p-90",
    "deep-image-96-angular-p-100",
]


def run_main(total_nr_step: int = 300, datasets: Iterable[str] = DEFAULT_DATASET_RUNS):
    system = VDTunerSystem(seed=1)
    system.run_legacy_optimization(datasets=datasets, total_nr_step=total_nr_step)


if __name__ == "__main__":
    run_main()
