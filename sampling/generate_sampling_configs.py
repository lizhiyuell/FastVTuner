import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "milvus.json"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "milvus_random_sampling.jsonl"

DEFAULT_VDB_TYPE = "milvus"
DEFAULT_BUILD_SAMPLES = 32
DEFAULT_SEARCH_SAMPLES_PER_BUILD = 16


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


def sample_search_value(param_name, detail, params, meta):
    knob_type = detail["type"]
    if knob_type == "enum":
        return random.choice(detail["enum_values"])

    min_value, max_value = apply_constraint(param_name, detail, params, meta)
    if min_value > max_value:
        raise ValueError(
            f"Invalid range for `{param_name}` after constraints: [{min_value}, {max_value}]"
        )

    if knob_type == "integer":
        return random.randint(int(min_value), int(max_value))
    if knob_type == "float":
        return random.uniform(float(min_value), float(max_value))

    raise ValueError(f"Unsupported parameter type: {knob_type}")


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


def make_search_sample(meta, index_type, build):
    params = {"index_type": index_type}
    params.update(build)

    search = {}
    for name in get_related_params(meta, index_type, "searching"):
        value = sample_search_value(name, meta[name], params, meta)
        search[name] = value
        params[name] = value

    return search


def build_key(index_type, build):
    return json.dumps({"index": index_type, "build": build}, sort_keys=True)


def record_key(index_type, build, search):
    return json.dumps(
        {"index": index_type, "build": build, "search": search},
        sort_keys=True,
    )


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


def add_build_records(
    records,
    seen_records,
    meta,
    vdb_type,
    index_type,
    build,
    build_sample_id,
    search_samples_per_build,
):
    build_params = get_related_params(meta, index_type, "building")
    search_params = get_related_params(meta, index_type, "searching")
    target_search_samples = search_samples_per_build
    if not build_params or not search_params:
        target_search_samples = 1

    seen_searches = set()
    attempts = 0
    max_attempts = max(target_search_samples * 100, 100)

    while len(seen_searches) < target_search_samples and attempts < max_attempts:
        attempts += 1
        search = make_search_sample(meta, index_type, build)
        key = record_key(index_type, build, search)
        if key in seen_records:
            continue
        seen_records.add(key)
        seen_searches.add(key)
        records.append(
            make_record(
                vdb_type,
                index_type,
                build,
                search,
                build_sample_id,
                len(seen_searches) - 1,
            )
        )

    if len(seen_searches) != target_search_samples:
        raise RuntimeError(
            f"Only generated {len(seen_searches)} unique search samples for "
            f"index={index_type}, build={build}, expected {target_search_samples}"
        )


def generate_records(meta, vdb_type, build_samples, search_samples_per_build):
    index_types = meta["index_type"]["enum_values"]
    if build_samples < len(index_types):
        raise ValueError(
            f"build_samples must be at least {len(index_types)} to sample every index once"
        )

    records = []
    seen_builds = set()
    seen_records = set()
    max_attempts = build_samples * 100
    attempts = 0

    for index_type in index_types:
        index_type, build = make_build_sample(meta, index_type)
        key = build_key(index_type, build)
        seen_builds.add(key)
        add_build_records(
            records,
            seen_records,
            meta,
            vdb_type,
            index_type,
            build,
            len(seen_builds) - 1,
            search_samples_per_build,
        )

    while len(seen_builds) < build_samples and attempts < max_attempts:
        attempts += 1
        index_type = random.choice(index_types)
        index_type, build = make_build_sample(meta, index_type)
        key = build_key(index_type, build)
        if key in seen_builds:
            continue
        seen_builds.add(key)
        add_build_records(
            records,
            seen_records,
            meta,
            vdb_type,
            index_type,
            build,
            len(seen_builds) - 1,
            search_samples_per_build,
        )

    if len(seen_builds) != build_samples:
        raise RuntimeError(
            f"Only generated {len(seen_builds)} unique build samples, expected {build_samples}"
        )

    return records


def write_jsonl(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly generate Milvus VDB build/search sampling configs."
    )
    parser.add_argument("--vdb-type", default=DEFAULT_VDB_TYPE)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--build-samples", type=int, default=DEFAULT_BUILD_SAMPLES)
    parser.add_argument(
        "--search-samples-per-build",
        type=int,
        default=DEFAULT_SEARCH_SAMPLES_PER_BUILD,
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    meta = load_meta(Path(args.config))
    records = generate_records(
        meta,
        args.vdb_type,
        args.build_samples,
        args.search_samples_per_build,
    )
    write_jsonl(records, Path(args.output))
    print(f"Wrote {len(records)} configs to {args.output}")


if __name__ == "__main__":
    main()
