import argparse
import json
import random
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
    # "gist",
    # "gist-random-p1",
    # "gist-random-p10",
    "gist_KClose"
]
FULL_DATASET_NAME = "gist"
CONFIG_META_PATH = PROJECT_ROOT / "config" / "milvus.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "sampling"
VDB_TYPE = "milvus"
TOP_K = 10
TUNE_QUERY_RATIO = 1.0
TEST_QUERY_RATIO = 1.0
BUILD_SAMPLES = 32
RECALL_STOP_DELTA = 0.01


def load_meta(config_path):
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sample_value(detail):
    knob_type = detail["type"]
    if knob_type == "integer":
        return random.randint(int(detail["min"]), int(detail["max"]))
    if knob_type == "float":
        return random.uniform(float(detail["min"]), float(detail["max"]))
    if knob_type == "enum":
        return random.choice(detail["enum_values"])
    raise ValueError(f"Unsupported parameter type: {knob_type}")


def parse_constraint(constrain):
    constrain = str(constrain).strip()
    for op in ("<=", ">="):
        if constrain.startswith(op):
            return op, constrain[len(op):].strip()
    raise ValueError(f"Unsupported constrain: {constrain}")


def apply_constraint(param_name, detail, params, meta):
    min_value = detail.get("min")
    max_value = detail.get("max")

    if "constrain" not in detail:
        return min_value, max_value

    op, other_name = parse_constraint(detail["constrain"])
    if other_name not in params:
        raise ValueError(f"Constraint of `{param_name}` depends on missing `{other_name}`")

    other_value = params[other_name]
    other_detail = meta[other_name]
    if other_detail["type"] not in ("integer", "float"):
        raise ValueError(f"Constraint target `{other_name}` must be numeric")

    if op == "<=":
        max_value = min(max_value, other_value)
    else:
        min_value = max(min_value, other_value)

    return min_value, max_value


def get_related_params(meta, index_type, knob_class):
    params = []
    for name, detail in meta.items():
        if detail.get("class") != knob_class:
            continue
        if index_type in detail.get("related_index", []):
            params.append(name)
    return params


def make_build_sample(meta, index_type):
    build = {}
    for name in get_related_params(meta, index_type, "building"):
        build[name] = sample_value(meta[name])
    return index_type, build


def build_key(index_type, build):
    return json.dumps({"index": index_type, "build": build}, sort_keys=True)


def make_build_samples(meta, build_samples):
    index_types = meta["index_type"]["enum_values"]
    if build_samples < len(index_types):
        raise ValueError(
            f"build_samples must be at least {len(index_types)} to sample every index once"
        )

    samples = []
    seen_builds = set()
    max_attempts = build_samples * 100
    attempts = 0

    for index_type in index_types:
        index_type, build = make_build_sample(meta, index_type)
        key = build_key(index_type, build)
        seen_builds.add(key)
        samples.append((index_type, build, len(samples)))

    while len(samples) < build_samples and attempts < max_attempts:
        attempts += 1
        index_type = random.choice(index_types)
        index_type, build = make_build_sample(meta, index_type)
        key = build_key(index_type, build)
        if key in seen_builds:
            continue
        seen_builds.add(key)
        samples.append((index_type, build, len(samples)))

    if len(samples) != build_samples:
        raise RuntimeError(
            f"Only generated {len(samples)} unique build samples, expected {build_samples}"
        )

    return samples


def get_min_search(meta, index_type, build):
    params = {"index_type": index_type}
    params.update(build)
    search = {}

    for name in get_related_params(meta, index_type, "searching"):
        detail = meta[name]
        if detail["type"] == "enum":
            value = detail["enum_values"][0]
        else:
            value, _ = apply_constraint(name, detail, params, meta)
        search[name] = value
        params[name] = value

    return search


def next_search(meta, index_type, build, search):
    params = {"index_type": index_type}
    params.update(build)
    params.update(search)
    next_values = dict(search)
    changed = False

    for name in get_related_params(meta, index_type, "searching"):
        detail = meta[name]
        if detail["type"] == "enum":
            continue

        _, max_value = apply_constraint(name, detail, params, meta)
        value = search[name] * 2
        if detail["type"] == "integer":
            value = int(value)
        else:
            value = float(value)
        value = min(value, max_value)
        if value != search[name]:
            changed = True
        next_values[name] = value
        params[name] = value

    if not changed:
        return None
    return next_values


def make_record(vdb_type, index_type, build, search, build_sample_id, search_sample_id):
    return {
        "vdb_type": vdb_type,
        "parameter": {
            "index": index_type,
            "build": build,
            "search": search,
        },
        "build_sample_id": build_sample_id,
        "search_sample_id": search_sample_id,
    }


def split_params(meta, params):
    index_type = params["index_type"]
    build = {}
    search = {}
    for name in get_related_params(meta, index_type, "building"):
        build[name] = params[name]
    for name in get_related_params(meta, index_type, "searching"):
        search[name] = params[name]
    return index_type, build, search


def read_full_result_configs(meta, full_dataset_name):
    input_path = OUTPUT_DIR / f"{full_dataset_name}_tune.txt"
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_nr, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            result = json.loads(line)
            params = result["params"]
            index_type, build, search = split_params(meta, params)
            record = make_record(
                VDB_TYPE,
                index_type,
                build,
                search,
                result.get("build_sample_id"),
                result.get("search_sample_id"),
            )
            record["_line_nr"] = line_nr
            records.append(record)
    return records


def set_config_params(vdb_config, parameter):
    vdb_config.set_params_to_default_values(apply=False)
    params = dict(zip(vdb_config.param_names, vdb_config.get_original_param()))
    params["index_type"] = parameter["index"]
    params.update(parameter.get("build") or {})
    params.update(parameter.get("search") or {})
    vdb_config.set_original_param([params[name] for name in vdb_config.param_names])
    return params


def run_query(engine, params, record, dataset_name, phase, step_id, index_time, search_only, out_f):
    is_test = phase == "test"
    ratio = TEST_QUERY_RATIO if is_test else TUNE_QUERY_RATIO

    try:
        query_time, recall, query_count, latency_list, recall_list = engine.query(
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
        latency_list = []
        recall_list = []
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
        "latency_list": latency_list,
        "recall_list": recall_list,
        "skip": False,
        "search_only": search_only,
        "build_sample_id": record.get("build_sample_id"),
        "search_sample_id": record.get("search_sample_id"),
        "config_line_nr": record.get("_line_nr"),
    }
    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
    out_f.flush()
    return recall


def output_paths(dataset_name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return (
        OUTPUT_DIR / f"{dataset_name}_tune.txt",
        OUTPUT_DIR / f"{dataset_name}_test.txt",
    )


def run_full_sampling(dataset_name, meta, args):
    engine = VDBEngine(args.vdb_type)
    engine.load_dataset(dataset_name)
    vdb_config = VDBConfig(engine.db_type, engine.dimension)
    build_samples = make_build_samples(meta, args.build_samples)
    tune_path, test_path = output_paths(dataset_name)
    tune_step = 0
    test_step = 0

    print(
        f"[sampling] full dataset={dataset_name}, build_groups={len(build_samples)}",
        flush=True,
    )

    with tune_path.open("w", encoding="utf-8") as tune_f, test_path.open(
        "w", encoding="utf-8"
    ) as test_f:
        for group_id, (index_type, build, build_sample_id) in enumerate(build_samples, 1):
            print(
                f"[sampling] build group {group_id}/{len(build_samples)}, index={index_type}",
                flush=True,
            )

            search = get_min_search(meta, index_type, build)
            parameter = {"index": index_type, "build": build, "search": search}
            set_config_params(vdb_config, parameter)

            engine.start()
            try:
                try:
                    build_time = engine.build()
                except Exception:
                    build_time = 0.0
                    print("[sampling] build failed", flush=True)

                last_recall = None
                search_sample_id = 0
                while search is not None:
                    record = make_record(
                        args.vdb_type,
                        index_type,
                        build,
                        search,
                        build_sample_id,
                        search_sample_id,
                    )
                    record["_line_nr"] = tune_step + 1

                    params = set_config_params(vdb_config, record["parameter"])
                    tune_step += 1
                    recall = run_query(
                        engine,
                        params,
                        record,
                        dataset_name,
                        "tune",
                        tune_step,
                        build_time if search_sample_id == 0 else 0.0,
                        search_sample_id > 0,
                        tune_f,
                    )
                    test_step += 1
                    run_query(
                        engine,
                        params,
                        record,
                        dataset_name,
                        "test",
                        test_step,
                        0.0,
                        search_sample_id > 0,
                        test_f,
                    )

                    if last_recall is not None and abs(recall - last_recall) < args.recall_stop_delta:
                        break
                    last_recall = recall
                    search = next_search(meta, index_type, build, search)
                    search_sample_id += 1
            finally:
                engine.stop()

    print(f"[sampling] wrote {tune_path}", flush=True)
    print(f"[sampling] wrote {test_path}", flush=True)


def group_records(records):
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
    return groups


def run_configured_sampling(dataset_name, meta, args):
    records = read_full_result_configs(meta, args.full_dataset_name)
    engine = VDBEngine(records[0].get("vdb_type", args.vdb_type))
    engine.load_dataset(dataset_name)
    vdb_config = VDBConfig(engine.db_type, engine.dimension)
    groups = group_records(records)
    tune_path, test_path = output_paths(dataset_name)
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

            params = set_config_params(vdb_config, group[0]["parameter"])
            engine.start()
            try:
                try:
                    build_time = engine.build()
                except Exception:
                    build_time = 0.0
                    print("[sampling] build failed", flush=True)

                for item_id, record in enumerate(group):
                    params = set_config_params(vdb_config, record["parameter"])
                    search_only = item_id > 0
                    tune_step += 1
                    run_query(
                        engine,
                        params,
                        record,
                        dataset_name,
                        "tune",
                        tune_step,
                        build_time if item_id == 0 else 0.0,
                        search_only,
                        tune_f,
                    )
                    test_step += 1
                    run_query(
                        engine,
                        params,
                        record,
                        dataset_name,
                        "test",
                        test_step,
                        0.0,
                        search_only,
                        test_f,
                    )
            finally:
                engine.stop()

    print(f"[sampling] wrote {tune_path}", flush=True)
    print(f"[sampling] wrote {test_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run dynamic sampling performance tests.")
    parser.add_argument("--vdb-type", default=VDB_TYPE)
    parser.add_argument("--config", default=str(CONFIG_META_PATH))
    parser.add_argument("--build-samples", type=int, default=BUILD_SAMPLES)
    parser.add_argument("--full-dataset-name", default=FULL_DATASET_NAME)
    parser.add_argument("--recall-stop-delta", type=float, default=RECALL_STOP_DELTA)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.config = Path(args.config)

    if args.seed is not None:
        random.seed(args.seed)

    meta = load_meta(args.config)
    for dataset_name in DATASET_NAMES:
        if dataset_name == args.full_dataset_name:
            run_full_sampling(dataset_name, meta, args)
        else:
            run_configured_sampling(dataset_name, meta, args)


if __name__ == "__main__":
    main()
