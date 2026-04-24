import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(file_path):
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            step_id = data.get("step_id")
            if step_id is None:
                continue

            params = data.get("params") or {}
            results.append(
                {
                    "step_id": int(step_id),
                    "index_type": str(params.get("index_type", "UNKNOWN")),
                    "full_tput": float(data.get("query_throughput", 0.0) or 0.0),
                    "full_recall": float(data.get("recall", 0.0) or 0.0),
                    "sampled_tput": float(data.get("sampled_query_throughput", 0.0) or 0.0),
                    "sampled_recall": float(data.get("sampled_recall", 0.0) or 0.0),
                }
            )

    if not results:
        raise ValueError(f"No valid records found in {file_path}")
    return results


def get_pareto_points(points, tput_key, recall_key):
    sorted_points = sorted(
        points,
        key=lambda item: (item[tput_key], item[recall_key], item["step_id"]),
    )

    pareto = []
    max_recall = float("-inf")
    for item in reversed(sorted_points):
        if item[recall_key] > max_recall:
            pareto.append(item)
            max_recall = item[recall_key]

    pareto.reverse()
    return pareto


def get_axis_limits(points, x_key, y_key):
    x_values = [item[x_key] for item in points]
    y_values = [item[y_key] for item in points]

    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)

    x_pad = max((x_max - x_min) * 0.08, 1e-4)
    y_pad = max((y_max - y_min) * 0.08, 1e-4)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def build_color_map(results):
    index_types = sorted({item["index_type"] for item in results})
    cmap = plt.get_cmap("tab10")
    if len(index_types) > 10:
        cmap = plt.get_cmap("tab20")

    color_map = {}
    for idx, index_type in enumerate(index_types):
        color_map[index_type] = cmap(idx % cmap.N)
    return color_map


def create_annotations(ax, points, x_key, y_key):
    x_offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.004
    y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
    for item in points:
        ax.annotate(
            str(item["step_id"]),
            xy=(item[x_key], item[y_key]),
            xytext=(item[x_key] + x_offset, item[y_key] + y_offset),
            textcoords="data",
            fontsize=7,
            ha="left",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.15",
                "fc": "white",
                "ec": "none",
                "alpha": 0.8,
            },
        )


def plot_one_axis(ax, points, x_key, y_key, title, color_map):
    pareto = get_pareto_points(points, y_key, x_key)
    xlim, ylim = get_axis_limits(points, x_key, y_key)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plotted_types = []
    for index_type in sorted(color_map):
        group = [item for item in points if item["index_type"] == index_type]
        if not group:
            continue
        plotted_types.append(index_type)
        ax.scatter(
            [item[x_key] for item in group],
            [item[y_key] for item in group],
            s=26,
            alpha=0.8,
            color=color_map[index_type],
            label=index_type,
        )

    ax.plot(
        [item[x_key] for item in pareto],
        [item[y_key] for item in pareto],
        "-o",
        linewidth=2,
        markersize=4,
        color="black",
        label="Pareto",
        zorder=3,
    )

    create_annotations(ax, points, x_key, y_key)

    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Throughput")
    ax.grid(True, linestyle="--", alpha=0.3)


def plot_results(results, output_file):
    color_map = build_color_map(results)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    plot_one_axis(
        axes[0],
        results,
        "full_recall",
        "full_tput",
        "Full Dataset",
        color_map,
    )
    plot_one_axis(
        axes[1],
        results,
        "sampled_recall",
        "sampled_tput",
        "Sampled Dataset",
        color_map,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot full/sample approximation results with Pareto frontiers.")
    parser.add_argument("result_file", help="Path to a result txt file, e.g. results/fastvtuner/milvus/gist_tune.txt")
    args = parser.parse_args()

    result_file = Path(args.result_file).resolve()
    if not result_file.exists():
        raise FileNotFoundError(f"File not found: {result_file}")

    results = load_results(result_file)
    output_file = result_file.parent / "sample_approximation.png"
    plot_results(results, output_file)
    print(f"Saved plot to {output_file}")


if __name__ == "__main__":
    main()
