#!/usr/bin/env python3
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
if str(DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_ROOT))

from make_fastvtuner_dataset import canonical_metric, exact_topk

INPUT_FILE = "gist.npz"
TOP_K = 100
GROUND_TRUTH_TOP_K = 10
BATCH_SIZE = 256
SECOND_SAMPLE_RATE = 0.1
OUTPUT_FILE = None


def resolve_input_path(input_file):
    path = Path(input_file).expanduser()
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    if path.is_absolute():
        return path

    if path.exists():
        return path.resolve()

    return (DATASET_ROOT / path).resolve()


def default_output_path(input_path):
    parts = input_path.stem.split("-")
    parts[0] = f"{parts[0]}_KClose"
    return DATASET_ROOT / ("-".join(parts) + input_path.suffix)


def collect_keep_ids(data, top_k):
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    id_arrays = []
    for key in ("search_top100", "test_top100"):
        if key not in data:
            raise KeyError(f"Missing required field: {key}")
        golden = np.asarray(data[key])
        if golden.ndim != 2:
            raise ValueError(f"{key} must be a 2D array.")
        if golden.shape[1] < top_k:
            raise ValueError(f"{key} only has {golden.shape[1]} columns, but top_k={top_k}.")
        id_arrays.append(golden[:, :top_k].reshape(-1))

    keep_ids = np.unique(np.concatenate(id_arrays)).astype(np.int64, copy=False)
    return keep_ids


def second_sample_keep_ids(keep_ids, sample_rate):
    if sample_rate <= 0:
        raise ValueError("SECOND_SAMPLE_RATE must be positive.")
    if sample_rate >= 1 or len(keep_ids) == 0:
        return keep_ids

    sample_size = max(1, int(len(keep_ids) * sample_rate))
    sampled_pos = np.random.choice(len(keep_ids), size=sample_size, replace=False)
    sampled_pos.sort()
    return keep_ids[sampled_pos]


def write_sampled_dataset(input_path, output_path, top_k, second_sample_rate):
    with np.load(input_path) as data:
        if "train" not in data:
            raise KeyError("Missing required field: train")
        for key in ("distance_metric", "search", "test"):
            if key not in data:
                raise KeyError(f"Missing required field: {key}")

        train = np.asarray(data["train"], dtype=np.float32)
        search = np.asarray(data["search"], dtype=np.float32)
        test = np.asarray(data["test"], dtype=np.float32)
        distance_metric = str(np.asarray(data["distance_metric"]).item())
        metric = canonical_metric(distance_metric)

        keep_ids = collect_keep_ids(data, top_k)
        keep_ids = keep_ids[(0 <= keep_ids) & (keep_ids < len(train))]
        first_sampled_size = len(keep_ids)
        keep_ids = second_sample_keep_ids(keep_ids, second_sample_rate)
        sampled_train = train[keep_ids]
        if len(sampled_train) < GROUND_TRUTH_TOP_K:
            raise ValueError(
                f"Sampled train size is {len(sampled_train)}, which is smaller than top-{GROUND_TRUTH_TOP_K}."
            )

        print("[gt] computing test top-100")
        test_top100 = exact_topk(
            test,
            sampled_train,
            metric=metric,
            topk=GROUND_TRUTH_TOP_K,
            batch_size=BATCH_SIZE,
        )
        print("[gt] computing search top-100")
        search_top100 = exact_topk(
            search,
            sampled_train,
            metric=metric,
            topk=GROUND_TRUTH_TOP_K,
            batch_size=BATCH_SIZE,
        )

        sampling_rate = len(sampled_train) / len(train)

        output_data = {}
        for key in data.files:
            output_data[key] = np.asarray(data[key])
        output_data["train"] = sampled_train
        output_data["search_top100"] = search_top100
        output_data["test_top100"] = test_top100
        output_data["sampling_rate"] = np.array(sampling_rate, dtype=np.float64)
        output_data["second_sample_rate"] = np.array(second_sample_rate, dtype=np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **output_data)
    return len(train), first_sampled_size, len(sampled_train), sampling_rate


def main():
    input_path = resolve_input_path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE).expanduser() if OUTPUT_FILE else default_output_path(input_path)
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    original_size, first_sampled_size, sampled_size, sampling_rate = write_sampled_dataset(
        input_path,
        output_path,
        TOP_K,
        SECOND_SAMPLE_RATE,
    )

    print(f"[load] input={input_path}")
    print(f"[save] output={output_path}")
    print(f"[sample] train={original_size} -> {first_sampled_size} -> {sampled_size}")
    print(f"second_sample_rate={SECOND_SAMPLE_RATE:.8f}")
    print(f"sampling_rate={sampling_rate:.8f}")


if __name__ == "__main__":
    main()
