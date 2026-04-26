from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from random import Random
from pathlib import Path
import json
import sys
from typing import Any, Literal

from systems.vdb_config import VDBConfig
from systems.vdb_engine import VDBEngine

from systems.common import *

Phase = Literal["tune", "test"]


@dataclass(slots=True)
# 表示一次调优或测试步骤的结构化结果，是所有query的统计结果。
class TuningRecord:
    step_id: int # 步骤ID
    phase: Phase # tune/test的结果
    dataset_name: str # 调优数据集名称
    build_parallel: int # 索引构建的并发数
    search_parallel: int # 索引检索的并发数
    params: dict[str, Any] # 数据集构造/搜索参数列表
    index_time: float # 总索引构建时间
    query_time: float # 本轮所有查询的总耗时
    recall: float # 平均召回率
    record_nr: int # 本轮参与统计的查询总条目数
    query_throughput: float = 0.0 # query吞吐率
    query_latency: float = 0.0 # 平均query延迟
    skip: bool = False # 是否跳过全量数据集调优
    extra: dict[str, Any] = field(default_factory=dict) # 系统特有的扩展字段

    # 将调优记录转换为普通字典，便于序列化或写日志。
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        extra = data.pop("extra", {}) or {}
        for key, value in extra.items():
            if key not in data:
                data[key] = value
        return data


# 调优系统的基类，每一种调优系统都要集成
class SystemBase(ABC):
    # 初始化系统基类状态，包括随机种子、固定参数和历史记录容器。
    def __init__(
        self,
        vdb_name: str,
        dataset_name: str,
        top_k: int = 10,
        single_tune_query_ratio: float = 0.5,
        single_test_query_ratio: float = 0.5,
    ) -> None:
        self.seed = 42
        self.vdb_name = vdb_name
        self.vdb_engine = VDBEngine(vdb_name)

        self._set_query_ratios(
            single_tune_query_ratio=single_tune_query_ratio,
            single_test_query_ratio=single_test_query_ratio,
        )
        self.dataset_name = dataset_name
        self.vdb_engine.load_dataset(dataset_name)

        # we need to build the config with dimension
        self.vdb_config = VDBConfig(vdb_name, self.vdb_engine.dimension)

        self._history: list[TuningRecord] = []
        self._step_id = 0
        self._rng = Random(self.seed)
        workload_name = Path(getattr(sys.modules.get("__main__"), "__file__", "interactive")).stem
        log_dir = RESULT_ROOT / workload_name / self.vdb_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_paths = {
            "tune": log_dir / f"{self.dataset_name}_tune.txt",
            "test": log_dir / f"{self.dataset_name}_test.txt",
        }
        self._log_files = {
            phase: path.open("w", encoding="utf-8")
            for phase, path in self._log_paths.items()
        }

        self.set_top_k(top_k)

    def __del__(self) -> None:
        try:
            log_files = getattr(self, "_log_files", None)
            if log_files is not None:
                for log_file in log_files.values():
                    if log_file is not None and not log_file.closed:
                        log_file.close()
        except Exception:
            pass

    # 返回只读形式的历史记录，避免外部直接修改内部列表。
    @property
    def history(self) -> tuple[TuningRecord, ...]:
        return tuple(self._history)

    # 统一设置并检查 tune/test 默认 query ratio。
    def _set_query_ratios(
        self,
        single_tune_query_ratio: float,
        single_test_query_ratio: float,
    ) -> None:
        assert 0 < single_tune_query_ratio <= 1
        assert 0 < single_test_query_ratio <= 1
        self._single_tune_query_ratio = single_tune_query_ratio
        self._single_test_query_ratio = single_test_query_ratio

    def set_top_k(self, top_k):
        if int(top_k) <= 0:
            raise ValueError("top_k must be positive")
        self._top_k = int(top_k)

    # 执行一次调优步骤，并将结果追加到历史记录中。。
    def single_tune(self) -> TuningRecord:
        return self._run_phase("tune", self._single_tune_impl)

    # 执行一次测试步骤，并将结果追加到历史记录中。
    def single_test(self) -> TuningRecord:
        return self._run_phase("test", self._single_test_impl)

    # 执行指定阶段并统一维护结果校验与历史记录。
    def _run_phase(self, phase: Phase, runner) -> TuningRecord:
        self._require_dataset()
        record = runner()
        self._append_record(record, expected_phase=phase)
        log_file = self._log_files[phase]
        log_file.write(json.dumps(record.to_dict(), ensure_ascii=False))
        log_file.write("\n")
        log_file.flush()
        return record

    # 校验记录合法性，并维护统一递增的 step_id 与历史列表。
    def _append_record(self, record: TuningRecord, expected_phase: Phase) -> None:
        if record.phase != expected_phase:
            raise ValueError(
                f"record phase mismatch: expected {expected_phase}, got {record.phase}"
            )
        if record.dataset_name != self.dataset_name:
            raise ValueError(
                "record dataset_name does not match the currently bound dataset"
            )
        if record.step_id <= 0:
            raise ValueError("record step_id must be positive")
        self._history.append(record)

    # 确保当前系统已经绑定数据集，否则拒绝执行 tune/test。
    def _require_dataset(self) -> None:
        if self.dataset_name is None:
            raise RuntimeError("No dataset is bound. Call load_dataset() first.")

    @abstractmethod
    # 子类需要实现：执行一次具体的调优逻辑（包含一次index构建和一次index检索）。
    def _single_tune_impl(self) -> TuningRecord:
        raise NotImplementedError

    @abstractmethod
    # 子类需要实现：执行一次具体的测试逻辑。
    def _single_test_impl(self) -> TuningRecord:
        raise NotImplementedError
