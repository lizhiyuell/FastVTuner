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


def format_value(value):
    if value is None:
        return ""
    return f"{value:.6g}"


def load_results(result_file):
    results = []
    with open(result_file, "r", encoding="utf-8") as f:
        for line_nr, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            params = data.get("params") or {}
            index_type = params.get("index_type", "UNKNOWN")
            full_tput = parse_number(data.get("query_throughput"))
            sampled_tput = parse_number(data.get("sampled_query_throughput"))
            ratio = None
            if full_tput is not None and sampled_tput:
                ratio = full_tput / sampled_tput

            results.append(
                {
                    "line_nr": line_nr,
                    "step_id": data.get("step_id", ""),
                    "index_type": str(index_type),
                    "params": params,
                    "full_tput": full_tput,
                    "full_recall": parse_number(data.get("recall")),
                    "sampled_tput": sampled_tput,
                    "sampled_recall": parse_number(data.get("sampled_recall")),
                    "full_over_sampled_tput": ratio,
                }
            )

    return results


def sort_key(item):
    sampled_recall = item["sampled_recall"]
    if sampled_recall is None:
        sampled_recall = float("inf")
    return sampled_recall, item["line_nr"]


def build_log(results):
    by_index_type = {}
    for item in results:
        by_index_type.setdefault(item["index_type"], []).append(item)

    columns = [
        "step_id",
        "full_tput",
        "full_recall",
        "sampled_tput",
        "sampled_recall",
        "full_over_sampled_tput",
    ]

    blocks = []
    for index_type in sorted(by_index_type):
        rows = sorted(by_index_type[index_type], key=sort_key)
        block = [f"index_type: {index_type}", "\t".join(columns)]
        for item in rows:
            values = [
                str(item["step_id"]),
                format_value(item["full_tput"]),
                format_value(item["full_recall"]),
                format_value(item["sampled_tput"]),
                format_value(item["sampled_recall"]),
                format_value(item["full_over_sampled_tput"]),
            ]
            block.append("\t".join(values))
        blocks.append("\n".join(block))

    return "\n\n".join(blocks) + "\n"


def get_plot_results(results):
    points = []
    for item in results:
        if item["full_recall"] is None or item["full_over_sampled_tput"] is None:
            continue
        if item["full_over_sampled_tput"] <= 0:
            continue
        points.append(item)
    return points


def build_color_map(results):
    index_types = sorted({item["index_type"] for item in results})
    cmap = plt.get_cmap("tab10")
    if len(index_types) > 10:
        cmap = plt.get_cmap("tab20")

    color_map = {}
    for idx, index_type in enumerate(index_types):
        color_map[index_type] = cmap(idx % cmap.N)
    return color_map


def plot_axis(ax, results, color_map, title, log_scale=False):
    for index_type in sorted(color_map):
        group = [item for item in results if item["index_type"] == index_type]
        if not group:
            continue
        ax.scatter(
            [item["full_recall"] for item in group],
            [item["full_over_sampled_tput"] for item in group],
            s=28,
            alpha=0.8,
            color=color_map[index_type],
            label=index_type,
        )

    if log_scale:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("full_tput / sampled_tput")
    ax.grid(True, linestyle="--", alpha=0.3)


def plot_results(results, output_file):
    points = get_plot_results(results)
    if not points:
        raise ValueError("No valid points found for plotting.")

    color_map = build_color_map(points)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    plot_axis(axes[0], points, color_map, "Linear y-axis")
    plot_axis(axes[1], points, color_map, "Log y-axis", log_scale=True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Summarize result records and plot throughput ratio.")
    parser.add_argument("result_file", help="Result file, e.g. results/fastvtuner/milvus/gist_fastvtuner_tune.txt")
    args = parser.parse_args()

    result_file = Path(args.result_file)
    if not result_file.exists():
        raise FileNotFoundError(f"File not found: {result_file}")

    results = load_results(result_file)
    if not results:
        raise ValueError(f"No valid records found in {result_file}")

    output_file = result_file.with_suffix(".log")
    output_file.write_text(build_log(results), encoding="utf-8")
    print(f"Saved log to {output_file}")

    plot_file = result_file.with_suffix(".png")
    plot_results(results, plot_file)
    print(f"Saved plot to {plot_file}")


if __name__ == "__main__":
    main()
