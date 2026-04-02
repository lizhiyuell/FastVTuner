# 用于评估 query ratio 敏感性的实验脚本。
# 这个脚本会固定 tune_query_ratio，然后枚举多个 test_query_ratio，
# 分别执行完整一轮调优，并记录每一步 tune/test 的结果。

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from systems.vdtuner import VDTunerSystem  # noqa: E402


# 要测试的数据集列表。
DATASETS = [
    "glove-100-angular-p-10",
    "glove-100-angular-p-50",
    "glove-100-angular-p-100",
]
# 对同一个数据集，分别使用这些 test_query_ratio 反复跑完整实验。
TEST_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5]
# 调优阶段固定使用的 query ratio。
TUNE_QUERY_RATIO = 0.5
# 每组实验执行的调优步数。
TUNE_STEPS = 50
SYSTEM_NAME = VDTunerSystem.__name__.replace("System", "").lower()
BENCHMARK_NAME = Path(__file__).stem
OUTPUT_DIR = REPO_ROOT / "results" / BENCHMARK_NAME


# 例如 "milvus-default" -> "milvus"
def _derive_vdb_name(engine_name: str) -> str:
    return engine_name.split("-", 1)[0]


VDB_NAME = _derive_vdb_name(VDTunerSystem(seed=1).engine_name)
PERF_OUTPUT_FILE = OUTPUT_DIR / f"{SYSTEM_NAME}_{VDB_NAME}_perf.txt"
PARAM_OUTPUT_FILE = OUTPUT_DIR / f"{SYSTEM_NAME}_{VDB_NAME}_param.txt"
PERF_HEADER = (
    "dataset\ttune_query_ratio\ttest_query_ratio\ttune_step\t"
    "index_build_time\ttuning_query_time\ttuning_qps\ttuning_recall\t"
    "testing_query_time\ttesting_qps\ttesting_recall\n"
)
PARAM_HEADER = (
    "dataset\ttune_query_ratio\ttest_query_ratio\ttune_step\t"
    "index_params...\tsystem_params...\n"
)


# 将参数字典转成便于写入文本日志的 key=value 形式。
def _format_param_items(params: dict) -> str:
    return "\t".join(f"{key}={value}" for key, value in params.items())


def _qps(record) -> float:
    return record.record_nr / record.query_time if record.query_time > 0 else 0.0


def _write_section_header(writer, dataset_name: str, test_ratio: float, header: str) -> None:
    writer.write(
        f"# dataset={dataset_name}\ttune_query_ratio={TUNE_QUERY_RATIO}"
        f"\ttest_query_ratio={test_ratio}\ttune_steps={TUNE_STEPS}\n"
    )
    writer.write(header)
    writer.flush()


def _run_single_step(system: VDTunerSystem):
    tune_record = system.single_tune()
    test_record = system.single_test(params=tune_record.params.get("normalized"))
    return tune_record, test_record


def _emit_row(dataset_name: str, test_ratio: float, step_idx: int, tune_record, test_record, perf_writer, param_writer) -> None:
    tuning_query_time = float(tune_record.query_time)
    tuning_qps = _qps(tune_record)
    testing_query_time = float(test_record.query_time)
    testing_qps = _qps(test_record)
    perf_writer.write(
        f"{dataset_name}\t{TUNE_QUERY_RATIO:.2f}\t{test_ratio:.2f}\t{step_idx}\t"
        f"{float(tune_record.index_time):.6f}\t{tuning_query_time:.6f}\t{tuning_qps:.6f}\t"
        f"{float(tune_record.recall):.6f}\t{testing_query_time:.6f}\t{testing_qps:.6f}\t"
        f"{float(test_record.recall):.6f}\n"
    )
    perf_writer.flush()

    param_items = [
        part
        for part in (
            _format_param_items(tune_record.params.get("index", {})),
            _format_param_items(tune_record.params.get("system", {})),
        )
        if part
    ]
    param_writer.write(
        f"{dataset_name}\t{TUNE_QUERY_RATIO:.2f}\t{test_ratio:.2f}\t{step_idx}"
        f"{''.join(f'\t{part}' for part in param_items)}\n"
    )
    param_writer.flush()

    print(
        f"[result] finished step={step_idx}/{TUNE_STEPS} dataset={dataset_name} "
        f"test_query_ratio={test_ratio:.2f} index_time={float(tune_record.index_time):.4f}s "
        f"tuning_query_time={tuning_query_time:.4f}s tuning_qps={tuning_qps:.4f} "
        f"tuning_recall={float(tune_record.recall):.4f} testing_query_time={testing_query_time:.4f}s "
        f"testing_qps={testing_qps:.4f} testing_recall={float(test_record.recall):.4f}",
        flush=True,
    )


def _run_single_setting(dataset_name: str, test_ratio: float, perf_writer, param_writer) -> None:
    print(
        f"[dataset] start dataset={dataset_name} "
        f"tune_query_ratio={TUNE_QUERY_RATIO:.2f} test_query_ratio={test_ratio:.2f}",
        flush=True,
    )
    system = VDTunerSystem(
        single_tune_query_ratio=TUNE_QUERY_RATIO,
        single_test_query_ratio=test_ratio,
        seed=1,
    )
    print(f"[dataset] loading dataset={dataset_name}", flush=True)
    system.load_dataset(dataset_name)
    _write_section_header(perf_writer, dataset_name, test_ratio, PERF_HEADER)
    _write_section_header(param_writer, dataset_name, test_ratio, PARAM_HEADER)

    for step_idx in range(1, TUNE_STEPS + 1):
        progress = f"{step_idx}/{TUNE_STEPS}" if step_idx > 1 else "1"
        print(f"[progress] running tune step {progress}", flush=True)
        _emit_row(
            dataset_name,
            test_ratio,
            step_idx,
            *_run_single_step(system),
            perf_writer,
            param_writer,
        )

    print(f"[dataset] completed dataset={dataset_name} test_query_ratio={test_ratio:.2f}", flush=True)


# 输出目录不存在时自动创建。
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[output] perf log: {PERF_OUTPUT_FILE}", flush=True)
    print(f"[output] param log: {PARAM_OUTPUT_FILE}", flush=True)
    with PERF_OUTPUT_FILE.open("w", encoding="utf-8") as perf_writer, PARAM_OUTPUT_FILE.open(
        "w", encoding="utf-8"
    ) as param_writer:
        for dataset_name in DATASETS:
            for test_ratio in TEST_RATIOS:
                _run_single_setting(dataset_name, test_ratio, perf_writer, param_writer)
    print("[summary] all settings completed", flush=True)


if __name__ == "__main__":
    main()
