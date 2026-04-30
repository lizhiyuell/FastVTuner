import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt


def parse_number(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_stats(values):
    if not values:
        return None

    avg = sum(values) / len(values)
    var = sum((value - avg) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": avg,
        "var": var,
    }


def format_value(value):
    return f"{value:.8g}"


def collect_results(result_file):
    gaps = {
        "throughput": {
            "abs": [],
            "rel": [],
        },
        "recall": {
            "abs": [],
            "rel": [],
        },
    }

    fields = {
        "throughput": ("query_throughput", "sampled_query_throughput"),
        "recall": ("recall", "sampled_recall"),
    }

    points = []
    with open(result_file, "r", encoding="utf-8") as f:
        for line_nr, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            point = {
                "test_id": data.get("step_id", line_nr),
                "throughput": parse_number(data.get("query_throughput")),
                "sampled_throughput": parse_number(data.get("sampled_query_throughput")),
                "recall": parse_number(data.get("recall")),
                "sampled_recall": parse_number(data.get("sampled_recall")),
            }
            points.append(point)

            for metric, names in fields.items():
                full = parse_number(data.get(names[0]))
                sampled = parse_number(data.get(names[1]))
                if full is None or sampled is None:
                    continue

                abs_gap = abs(full - sampled)
                gaps[metric]["abs"].append(abs_gap)
                if full != 0:
                    gaps[metric]["rel"].append(abs_gap / abs(full))

    return gaps, points


def print_stats(name, stats):
    if stats is None:
        print(f"{name}\tcount=0")
        return

    values = [
        f"count={stats['count']}",
        f"min={format_value(stats['min'])}",
        f"max={format_value(stats['max'])}",
        f"avg={format_value(stats['avg'])}",
        f"var={format_value(stats['var'])}",
    ]
    print(f"{name}\t" + "\t".join(values))


def get_plot_values(points, full_key, sampled_key):
    full_x = []
    full_y = []
    sampled_x = []
    sampled_y = []
    for point in points:
        if point[full_key] is not None:
            full_x.append(point["test_id"])
            full_y.append(point[full_key])
        if point[sampled_key] is not None:
            sampled_x.append(point["test_id"])
            sampled_y.append(point[sampled_key])
    return full_x, full_y, sampled_x, sampled_y


def plot_axis(ax, points, full_key, sampled_key, title, ylabel):
    full_x, full_y, sampled_x, sampled_y = get_plot_values(points, full_key, sampled_key)
    ax.scatter(full_x, full_y, s=28, alpha=0.8, color="tab:blue", label="full")
    ax.scatter(
        sampled_x,
        sampled_y,
        s=42,
        facecolors="none",
        edgecolors="tab:orange",
        linewidths=1.4,
        label="sample",
    )
    ax.set_title(title)
    ax.set_xlabel("test ID")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()


def plot_results(points, output_file):
    if not points:
        raise ValueError("No valid records found for plotting.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_axis(
        axes[0],
        points,
        "throughput",
        "sampled_throughput",
        "Throughput",
        "throughput",
    )
    plot_axis(
        axes[1],
        points,
        "recall",
        "sampled_recall",
        "Recall",
        "recall",
    )
    fig.tight_layout()
    fig.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze full/sample throughput and recall gaps.")
    parser.add_argument("result_file", help="Result file, e.g. results/fastvtuner/milvus/gist_tune.txt")
    args = parser.parse_args()

    result_file = Path(args.result_file)
    if not result_file.exists():
        raise FileNotFoundError(f"File not found: {result_file}")

    gaps, points = collect_results(result_file)
    print(f"file\t{result_file}")
    print("relative_gap\tabs(full - sampled) / abs(full)")
    for metric in ("throughput", "recall"):
        print_stats(f"{metric}_abs_gap", get_stats(gaps[metric]["abs"]))
        print_stats(f"{metric}_rel_gap", get_stats(gaps[metric]["rel"]))

    output_file = Path.cwd() / f"{result_file.stem}_sample_gap.png"
    plot_results(points, output_file)
    print(f"plot\t{output_file}")


if __name__ == "__main__":
    main()
