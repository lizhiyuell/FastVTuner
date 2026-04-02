from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from random import Random
from typing import Any, Literal


Phase = Literal["tune", "test"]


@dataclass(slots=True)
# 表示一次调优或测试步骤的结构化结果，是所有query的统计结果。
class TuningRecord:
    step_id: int # 步骤ID
    phase: Phase # tune/test的结果
    dataset_name: str # 调优数据集名称
    params: dict[str, Any] # 数据集构造/搜索参数列表
    index_time: float # 总索引构建时间
    query_time: float # 本轮所有查询的总耗时
    recall: float # 平均召回率
    record_nr: int # 本轮参与统计的查询总条目数

    # 将调优记录转换为普通字典，便于序列化或写日志。
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# 调优系统的基类，每一种调优系统都要集成
class SystemBase(ABC):
    # 初始化系统基类状态，包括随机种子、固定参数和历史记录容器。
    def __init__(
        self,
        single_tune_query_ratio: float = 0.5,
        single_test_query_ratio: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.seed = seed

        # 设置tune和test的query的比例（在原来的query数据集中切分）
        assert single_tune_query_ratio > 0 and single_tune_query_ratio < 1
        assert single_test_query_ratio > 0 and single_test_query_ratio < 1
        assert single_tune_query_ratio + single_test_query_ratio <= 1
        self._single_tune_query_ratio = single_tune_query_ratio
        self._single_test_query_ratio = single_test_query_ratio

        self.dataset_name: str | None = None
        self._history: list[TuningRecord] = []
        self._step_id = 0
        self._rng = Random(seed)

    # 返回只读形式的历史记录，避免外部直接修改内部列表。
    @property
    def history(self) -> tuple[TuningRecord, ...]:
        return tuple(self._history)

    # 绑定当前实验使用的数据集名称，并触发子类的数据集切换钩子。
    # 这里本身不做底层原始数据装载，只负责维护系统层面的状态切换。
    def load_dataset(self, dataset_name: str) -> None:
        if not dataset_name:
            raise ValueError("dataset_name must be a non-empty string")

        if self.dataset_name != dataset_name:
            self._history.clear()
            self._step_id = 0

        self.dataset_name = dataset_name
        self._on_dataset_loaded(dataset_name)

    # 按当前比例将 query 下标，随机拆分为互不重叠的 tune/test 两部分。
    def split_query_indices(self, total_queries: int) -> tuple[list[int], list[int]]:
        if total_queries <= 0:
            raise ValueError("total_queries must be positive")

        indices = list(range(total_queries))
        self._rng.shuffle(indices) # 这里的seed是固定的

        tune_count = int(total_queries * self._single_tune_query_ratio)
        if self._single_tune_query_ratio > 0.0 and tune_count == 0:
            tune_count = 1
        tune_count = min(total_queries, tune_count)

        remaining = total_queries - tune_count
        test_count = int(total_queries * self._single_test_query_ratio)
        if self._single_test_query_ratio > 0.0 and test_count == 0 and remaining > 0:
            test_count = 1
        test_count = min(remaining, test_count)

        tune_indices = indices[:tune_count]
        test_indices = indices[tune_count : tune_count + test_count]
        return tune_indices, test_indices

    # 执行一次调优步骤，并将结果追加到历史记录中。
    def single_tune(self, **kwargs: Any) -> TuningRecord:
        return self._run_phase("tune", self._single_tune_impl, **kwargs)

    # 执行一次测试步骤，并将结果追加到历史记录中。
    def single_test(self, **kwargs: Any) -> TuningRecord:
        return self._run_phase("test", self._single_test_impl, **kwargs)

    # 执行指定阶段并统一维护结果校验与历史记录。
    def _run_phase(self, phase: Phase, runner, **kwargs: Any) -> TuningRecord:
        self._require_dataset()
        record = runner(**kwargs)
        self._append_record(record, expected_phase=phase)
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
        self._step_id += 1
        if record.step_id != self._step_id:
            record.step_id = self._step_id
        self._history.append(record)

    # 确保当前系统已经绑定数据集，否则拒绝执行 tune/test。
    def _require_dataset(self) -> None:
        if self.dataset_name is None:
            raise RuntimeError("No dataset is bound. Call load_dataset() first.")

    # 子类可选钩子：在切换数据集后准备特定状态。
    def _on_dataset_loaded(self, dataset_name: str) -> None:
        pass

    @abstractmethod
    # 子类需要实现：执行一次具体的调优逻辑。
    def _single_tune_impl(self, **kwargs: Any) -> TuningRecord:
        raise NotImplementedError

    @abstractmethod
    # 子类需要实现：执行一次具体的测试逻辑。
    def _single_test_impl(self, **kwargs: Any) -> TuningRecord:
        raise NotImplementedError
