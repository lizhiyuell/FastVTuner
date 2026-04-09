import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
RESULT_ROOT = PROJECT_ROOT / "results"

# 在这里维护项目中关心的数据集名称
DATASET_NAMES = [
    "gist",
    "glove",
]


def _to_scalar(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
    return value


def _load_dataset_info(dataset_name):
    dataset_file = DATASET_ROOT / dataset_name
    dataset = np.load(dataset_file)

    train = dataset["train"]
    search = dataset["search"]
    test = dataset["test"]
    dimension = _to_scalar(dataset["dimension"])
    distance_metric = _to_scalar(dataset["distance_metric"])

    if isinstance(distance_metric, bytes):
        distance_metric = distance_metric.decode("utf-8")

    return {
        "name": dataset_name,
        "dimension": int(dimension),
        "distance_metric": str(distance_metric),
        "train_size": int(len(train)),
        "search_size": int(len(search)),
        "test_size": int(len(test)),
        "total_size": int(len(train) + len(search) + len(test)),
    }
def _format_result(info):
    lines = [
        f"dataset: {info['name']}",
        f"dimension: {info['dimension']}",
        f"distance_metric: {info['distance_metric']}",
        f"train_size: {info['train_size']}",
        f"search_size: {info['search_size']}",
        f"test_size: {info['test_size']}",
        f"total_size: {info['total_size']}",
    ]
    return "\n".join(lines)


def main():
    output_dir = RESULT_ROOT / "print_workload_info"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "result.txt"

    dataset_names = [f"{dataset_name}.npz" for dataset_name in DATASET_NAMES]
    results = [_load_dataset_info(dataset_name) for dataset_name in dataset_names]

    with output_file.open("w", encoding="utf-8") as f:
        for idx, info in enumerate(results):
            if idx > 0:
                f.write("\n\n")
            f.write(_format_result(info))

    print(output_file)


if __name__ == "__main__":
    main()
