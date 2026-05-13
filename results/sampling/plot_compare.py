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


def is_base_dataset(name):
    return "-" not in name and "_" not in name


def group_results(results):
    base_names = sorted(name for name in results if is_base_dataset(name))
    grouped = {}

    for base_name in base_names:
        grouped[base_name] = {}
        for name, records in results.items():
            if name == base_name or name.startswith(f"{base_name}-") or name.startswith(f"{base_name}_"):
                grouped[base_name][name] = records

    return grouped


def build_relative_records(records, base_by_step):
    relative_records = []
    for item in records:
        base_item = base_by_step.get(item["step_id"])
        if not base_item:
            continue

        relative_item = {"step_id": item["step_id"]}
        for key in ("recall", "tput"):
            base_value = base_item[key]
            relative_item[key] = item[key] / base_value if base_value else None
        relative_records.append(relative_item)
    return relative_records


def plot_metric(ax, results, key, label, relative=False):
    for method, records in results.items():
        xs = []
        ys = []
        for item in records:
            value = item[key]
            if value is None:
                continue
            xs.append(item["step_id"])
            ys.append(value)
        ax.scatter(xs, ys, s=1, alpha=0.75, label=method)

    ax.set_ylabel(label)
    ax.set_xlabel("step_id")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    if key == "recall" and not relative:
        ax.set_ylim(0, 1)
    if relative:
        ax.axhline(1, color="black", linestyle=":", linewidth=0.8, alpha=0.7)
    if results:
        ax.legend(loc="best", markerscale=4)


def plot_results(results, base_name, title, output_path):
    if not results:
        print(f"skip {output_path}: no data")
        return

    base_records = results.get(base_name)
    if not base_records:
        print(f"skip {output_path}: missing base dataset {base_name}")
        return

    base_by_step = {item["step_id"]: item for item in base_records}
    relative_results = {
        method: build_relative_records(records, base_by_step)
        for method, records in results.items()
        if method != base_name
    }

    fig, axes = plt.subplots(4, 1, figsize=(12, 14))
    plot_metric(axes[0], results, "recall", "Recall")
    plot_metric(axes[1], results, "tput", "Tput")
    plot_metric(axes[2], relative_results, "recall", "Relative Recall", relative=True)
    plot_metric(axes[3], relative_results, "tput", "Relative Tput", relative=True)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"wrote {output_path}")


def main():
    tune_results = collect_results("tune")
    test_results = collect_results("test")

    for suffix, results in (("tune", tune_results), ("test", test_results)):
        grouped = group_results(results)
        base_names = sorted(grouped)
        print(f"{suffix} base datasets: {', '.join(base_names) if base_names else 'none'}")
        for base_name, base_results in grouped.items():
            output_path = BASE_DIR / f"compare_{base_name}_{suffix}.png"
            plot_results(
                base_results,
                base_name,
                f"{suffix.title()} Compare: {base_name}",
                output_path,
            )


if __name__ == "__main__":
    main()
