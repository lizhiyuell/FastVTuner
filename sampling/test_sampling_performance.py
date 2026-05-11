import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEMS_ROOT = PROJECT_ROOT / "systems"
for path in (PROJECT_ROOT, SYSTEMS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from systems.common import BUILD_PARALLEL, SEARCH_PARALLEL
from systems.vdb_config import VDBConfig
from systems.vdb_engine import VDBEngine


DATASET_NAMES = [
    # "gist-random-p1",
    # "gist-random-p10",
    "gist",
]
CONFIG_PATH = PROJECT_ROOT / "sampling" / "milvus_random_sampling.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "results" / "sampling"
VDB_TYPE = "milvus"
TOP_K = 10
TUNE_QUERY_RATIO = 1.0
TEST_QUERY_RATIO = 1.0


def run_sampling(dataset_name):
    records = []
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        for line_nr, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_line_nr"] = line_nr
            records.append(record)

    engine = VDBEngine(records[0].get("vdb_type", VDB_TYPE))
    engine.load_dataset(dataset_name)
    vdb_config = VDBConfig(engine.db_type, engine.dimension)

    groups = []
    group_by_key = {}
    for record in records:
        parameter = record["parameter"]
        key = json.dumps(
            {
                "index": parameter["index"],
                "build": parameter.get("build") or {},
            },
            sort_keys=True,
        )
        if key not in group_by_key:
            group_by_key[key] = []
            groups.append(group_by_key[key])
        group_by_key[key].append(record)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tune_path = OUTPUT_DIR / f"{dataset_name}_tune.txt"
    test_path = OUTPUT_DIR / f"{dataset_name}_test.txt"

    tune_step = 0
    test_step = 0
    print(
        f"[sampling] dataset={dataset_name}, configs={len(records)}, build_groups={len(groups)}",
        flush=True,
    )

    with tune_path.open("w", encoding="utf-8") as tune_f, test_path.open(
        "w", encoding="utf-8"
    ) as test_f:
        for group_id, group in enumerate(groups, 1):
            print(
                f"[sampling] build group {group_id}/{len(groups)}, configs={len(group)}",
                flush=True,
            )

            first_parameter = group[0]["parameter"]
            vdb_config.set_params_to_default_values(apply=False)
            params = dict(zip(vdb_config.param_names, vdb_config.get_original_param()))
            params["index_type"] = first_parameter["index"]
            params.update(first_parameter.get("build") or {})
            params.update(first_parameter.get("search") or {})
            vdb_config.set_original_param([params[name] for name in vdb_config.param_names])

            engine.start()
            try:
                try:
                    build_time = engine.build()
                except Exception:
                    build_time = 0.0
                    print("[sampling] build failed", flush=True)

                for item_id, record in enumerate(group):
                    parameter = record["parameter"]
                    vdb_config.set_params_to_default_values(apply=False)
                    params = dict(zip(vdb_config.param_names, vdb_config.get_original_param()))
                    params["index_type"] = parameter["index"]
                    params.update(parameter.get("build") or {})
                    params.update(parameter.get("search") or {})
                    vdb_config.set_original_param(
                        [params[name] for name in vdb_config.param_names]
                    )

                    search_only = item_id > 0
                    for phase, is_test, ratio, out_f in (
                        ("tune", False, TUNE_QUERY_RATIO, tune_f),
                        ("test", True, TEST_QUERY_RATIO, test_f),
                    ):
                        if phase == "tune":
                            tune_step += 1
                            step_id = tune_step
                            index_time = build_time if item_id == 0 else 0.0
                        else:
                            test_step += 1
                            step_id = test_step
                            index_time = 0.0

                        try:
                            query_time, recall, query_count = engine.query(
                                TOP_K,
                                test=is_test,
                                ratio=ratio,
                            )
                            query_throughput = query_count / query_time if query_time > 0 else 0.0
                            query_latency = query_time / query_count if query_count > 0 else 0.0
                        except Exception:
                            query_time = 0.0
                            recall = 0.0
                            query_count = 0
                            query_throughput = 0.0
                            query_latency = 0.0

                        result = {
                            "step_id": step_id,
                            "phase": phase,
                            "dataset_name": dataset_name,
                            "build_parallel": BUILD_PARALLEL,
                            "search_parallel": SEARCH_PARALLEL,
                            "params": params,
                            "index_time": index_time,
                            "query_time": query_time,
                            "recall": recall,
                            "record_nr": query_count,
                            "query_throughput": query_throughput,
                            "query_latency": query_latency,
                            "skip": False,
                            "search_only": search_only,
                            "build_sample_id": record.get("build_sample_id"),
                            "search_sample_id": record.get("search_sample_id"),
                            "config_line_nr": record.get("_line_nr"),
                        }
                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out_f.flush()
            finally:
                engine.stop()

    print(f"[sampling] wrote {tune_path}", flush=True)
    print(f"[sampling] wrote {test_path}", flush=True)


def main():
    for dataset_name in DATASET_NAMES:
        run_sampling(dataset_name)


if __name__ == "__main__":
    main()
