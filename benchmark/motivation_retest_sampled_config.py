import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEMS_ROOT = PROJECT_ROOT / "systems"
for path in (PROJECT_ROOT, SYSTEMS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from systems.common import BUILD_PARALLEL, RESULT_ROOT, SEARCH_PARALLEL, load_result_config
from systems.vdb_config import VDBConfig
from systems.vdb_engine import VDBEngine


VDB_NAME = "milvus"
DATASET_NAME = "gist"
RESULT_FILE = PROJECT_ROOT / "results" / "vdtuner" / VDB_NAME / "gist-p-10_tune.txt"
TOP_K = 10
OUTPUT_DIR = RESULT_ROOT / Path(__file__).stem / VDB_NAME
TUNE_OUTPUT_FILE = OUTPUT_DIR / f"{DATASET_NAME}_tune.txt"
TEST_OUTPUT_FILE = OUTPUT_DIR / f"{DATASET_NAME}_test.txt"


def _build_record(step_id, phase, param_names, params, index_time, query_time, recall, query_count):
    query_throughput = query_count / query_time if query_time > 0 else 0.0
    query_latency = query_time / query_count if query_count > 0 else 0.0
    return {
        "step_id": step_id,
        "phase": phase,
        "dataset_name": DATASET_NAME,
        "build_parallel": BUILD_PARALLEL,
        "search_parallel": SEARCH_PARALLEL,
        "params": dict(zip(param_names, params)),
        "index_time": index_time,
        "query_time": query_time,
        "recall": recall,
        "record_nr": query_count,
        "query_throughput": query_throughput,
        "query_latency": query_latency,
    }


def main():
    configs = load_result_config(RESULT_FILE)
    if not configs:
        raise ValueError(f"No configs found in result file: {RESULT_FILE}")

    vdb_engine = VDBEngine(VDB_NAME)
    vdb_engine.load_dataset(DATASET_NAME)
    vdb_config = VDBConfig(VDB_NAME, vdb_engine.dimension)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[input] dataset={DATASET_NAME}", flush=True)
    print(f"[input] result_file={RESULT_FILE}", flush=True)
    print(f"[output] tune_file={TUNE_OUTPUT_FILE}", flush=True)
    print(f"[output] test_file={TEST_OUTPUT_FILE}", flush=True)

    vdb_engine.start()
    try:
        with TUNE_OUTPUT_FILE.open("w", encoding="utf-8") as tune_f, TEST_OUTPUT_FILE.open("w", encoding="utf-8") as test_f:
            for step_id, params in enumerate(configs, start=1):
                original_params = [params[name] for name in vdb_config.param_names]
                vdb_config.set_original_param(original_params)

                print(f"[progress] retest {step_id}/{len(configs)}", flush=True)

                index_time = vdb_engine.build()
                tune_query_time, tune_recall, tune_query_count = vdb_engine.query(TOP_K, test=False, ratio=1.0)
                tune_record = _build_record(
                    step_id,
                    "tune",
                    vdb_config.param_names,
                    vdb_config.get_original_param(),
                    index_time,
                    tune_query_time,
                    tune_recall,
                    tune_query_count,
                )
                tune_f.write(json.dumps(tune_record, ensure_ascii=False))
                tune_f.write("\n")
                tune_f.flush()

                test_query_time, test_recall, test_query_count = vdb_engine.query(TOP_K, test=True, ratio=1.0)
                test_record = _build_record(
                    step_id,
                    "test",
                    vdb_config.param_names,
                    vdb_config.get_original_param(),
                    0.0,
                    test_query_time,
                    test_recall,
                    test_query_count,
                )
                test_f.write(json.dumps(test_record, ensure_ascii=False))
                test_f.write("\n")
                test_f.flush()
    finally:
        vdb_engine.stop()


if __name__ == "__main__":
    main()
