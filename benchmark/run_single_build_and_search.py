from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from systems.vdtuner import VDTunerSystem  # noqa: E402


# Manual experiment settings. Edit these values directly before running.
DATASET_NAME = "glove-100-angular-p-10"
SEED = 1
SYSTEM_NAME = VDTunerSystem.__name__.replace("System", "").lower()
BENCHMARK_NAME = Path(__file__).stem
OUTPUT_DIR = REPO_ROOT / "results" / BENCHMARK_NAME

# Full-build + full-search:
# - build always uses the full dataset
# - search uses the full query split by setting test_query_ratio=1.0
TUNE_QUERY_RATIO = 0.0
TEST_QUERY_RATIO = 1.0

# Override only the parameters you want to change.
# Unspecified parameters fall back to the current defaults in config.
INDEX_PARAMS: dict[str, Any] = {
    "index_type": "FLAT",
    "nlist": 128,
    "nprobe": 7,
    "m": 10,
    "nbits": 8,
    "M": 32,
    "efConstruction": 256,
    "ef": 500,
    "reorder_k": 500
}

SYSTEM_PARAMS: dict[str, Any] = {
    "dataCoord*segment*maxSize": 10000,
    "dataCoord*segment*sealProportion": 99,
    "queryCoord*autoHandoff": 1.0,
    "queryCoord*autoBalance": 1.0,
    "common*gracefulTime": 100000,
    "dataNode*segment*insertBufSize": 10000000000,
    "rootCoord*minSegmentSizeToEnableIndex": 10000,
}


def _derive_vdb_name(engine_name: str) -> str:
    return engine_name.split("-", 1)[0]


VDB_NAME = _derive_vdb_name(VDTunerSystem(seed=SEED).engine_name)
PERF_OUTPUT_FILE = OUTPUT_DIR / f"{SYSTEM_NAME}_{VDB_NAME}_perf.txt"
PARAM_OUTPUT_FILE = OUTPUT_DIR / f"{SYSTEM_NAME}_{VDB_NAME}_param.txt"


def _format_param_items(params: dict[str, Any]) -> str:
    return "\t".join(f"{key}={value}" for key, value in params.items())


def _qps(record) -> float:
    return record.record_nr / record.query_time if record.query_time > 0 else 0.0


def _build_normalized_params(
    system: VDTunerSystem,
    *,
    index_params: dict[str, Any],
    system_params: dict[str, Any],
) -> list[float]:
    knob_details = system.knob_stand.knobs_detail
    overrides = dict(index_params)
    overrides.update(system_params)

    unknown_keys = sorted(set(overrides) - set(knob_details))
    if unknown_keys:
        raise KeyError(f"Unknown knob names: {unknown_keys}")

    normalized_params: list[float] = []
    for knob_name in system.names:
        knob = knob_details[knob_name]
        real_value = overrides.get(knob_name, knob["default"])
        normalized_params.append(system.knob_stand.scale_forward(knob_name, real_value))
    return normalized_params


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[output] perf log: {PERF_OUTPUT_FILE}", flush=True)
    print(f"[output] param log: {PARAM_OUTPUT_FILE}", flush=True)
    print(
        f"[dataset] start dataset={DATASET_NAME} "
        f"tune_query_ratio={TUNE_QUERY_RATIO:.2f} test_query_ratio={TEST_QUERY_RATIO:.2f}",
        flush=True,
    )

    system = VDTunerSystem(
        single_tune_query_ratio=TUNE_QUERY_RATIO,
        single_test_query_ratio=TEST_QUERY_RATIO,
        seed=SEED,
    )
    system.load_dataset(DATASET_NAME)

    params = _build_normalized_params(
        system,
        index_params=INDEX_PARAMS,
        system_params=SYSTEM_PARAMS,
    )

    print("[progress] running single build + search", flush=True)
    record = system.single_test(params=params)

    testing_query_time = float(record.query_time)
    testing_qps = _qps(record)
    testing_recall = float(record.recall)
    index_params = record.params.get("index", {})
    system_params = record.params.get("system", {})

    with PERF_OUTPUT_FILE.open("w", encoding="utf-8") as perf_writer:
        perf_writer.write(
            f"# dataset={DATASET_NAME}\ttune_query_ratio={TUNE_QUERY_RATIO}"
            f"\ttest_query_ratio={TEST_QUERY_RATIO}\ttune_steps=1\n"
        )
        perf_writer.write(
            "dataset\ttune_query_ratio\ttest_query_ratio\ttune_step\t"
            "index_build_time\ttuning_query_time\ttuning_qps\ttuning_recall\t"
            "testing_query_time\ttesting_qps\ttesting_recall\n"
        )
        perf_writer.write(
            f"{DATASET_NAME}\t{TUNE_QUERY_RATIO:.2f}\t{TEST_QUERY_RATIO:.2f}\t1\t"
            f"{float(record.index_time):.6f}\t0.000000\t0.000000\t0.000000\t"
            f"{testing_query_time:.6f}\t{testing_qps:.6f}\t{testing_recall:.6f}\n"
        )

    with PARAM_OUTPUT_FILE.open("w", encoding="utf-8") as param_writer:
        param_writer.write(
            f"# dataset={DATASET_NAME}\ttune_query_ratio={TUNE_QUERY_RATIO}"
            f"\ttest_query_ratio={TEST_QUERY_RATIO}\ttune_steps=1\n"
        )
        param_writer.write(
            "dataset\ttune_query_ratio\ttest_query_ratio\ttune_step\t"
            "index_params...\tsystem_params...\n"
        )
        param_writer.write(
            f"{DATASET_NAME}\t{TUNE_QUERY_RATIO:.2f}\t{TEST_QUERY_RATIO:.2f}\t1"
        )
        if index_params:
            param_writer.write(f"\t{_format_param_items(index_params)}")
        if system_params:
            param_writer.write(f"\t{_format_param_items(system_params)}")
        param_writer.write("\n")

    print(
        f"[result] finished step=1/1 dataset={DATASET_NAME} "
        f"index_time={float(record.index_time):.4f}s "
        f"testing_query_time={testing_query_time:.4f}s "
        f"testing_qps={testing_qps:.4f} testing_recall={testing_recall:.4f}",
        flush=True,
    )
    print("[summary] single build + search completed", flush=True)


if __name__ == "__main__":
    main()
