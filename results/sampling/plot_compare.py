import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent


def load_records(path):
    records = []
    with path.open() as f:
        for line_nr, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_nr} is not valid JSON") from exc

            if "step_id" not in item or "recall" not in item or "query_throughput" not in item:
                continue

            records.append(
                {
                    "step_id": item["step_id"],
                    "recall": item["recall"],
                    "tput": item["query_throughput"],
                }
            )
    return records


def collect_results(suffix):
    results = {}
    pattern = f"*_{suffix}.txt"
    ending = f"_{suffix}.txt"
    for path in sorted(BASE_DIR.glob(pattern)):
        method = path.name[: -len(ending)]
        records = load_records(path)
        if records:
            results[method] = records
    return results


def plot_results(results, title, output_path):
    if not results:
        print(f"skip {output_path}: no data")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    metrics = [("recall", "Recall"), ("tput", "Tput")]

    for ax, (key, label) in zip(axes, metrics):
        for method, records in results.items():
            xs = [item["step_id"] for item in records]
            ys = [item[key] for item in records]
            ax.scatter(xs, ys, s=1, alpha=0.75, label=method)
        ax.set_ylabel(label)
        ax.set_xlabel("step_id")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        if key == "recall":
            ax.set_ylim(0, 1)

    fig.suptitle(title)
    fig.legend(loc="upper center", ncol=min(4, len(results)), bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"wrote {output_path}")


def main():
    tune_results = collect_results("tune")
    test_results = collect_results("test")

    plot_results(tune_results, "Tune Compare", BASE_DIR / "compare_tune.png")
    plot_results(test_results, "Test Compare", BASE_DIR / "compare_test.png")


if __name__ == "__main__":
    main()
