import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, StrMethodFormatter


RESULT_FILES = [
    ("fastvtuner", Path("../results/fastvtuner/milvus/gist_fastvtuner_tune.txt")),
    ("vdtuner", Path("../results/vdtuner/milvus/gist_vdtuner_tune.txt")),
]

RECALL_TARGETS = [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
OUTPUT_FILE = "target_recall_progress.png"


def parse_number(value, default=0.0):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def record_time(record):
    return (
        parse_number(record.get("index_time"))
        + parse_number(record.get("query_time"))
        + parse_number(record.get("sampled_index_time"))
        + parse_number(record.get("sampled_query_time"))
    )


def load_records(result_file):
    records = []
    with open(result_file, "r", encoding="utf-8") as f:
        for line_nr, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                record["_line_nr"] = line_nr
                records.append(record)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {result_file}:{line_nr}") from exc
    return records


def build_curve(records, target):
    xs = []
    ys = []
    turns = []
    total_time = 0.0
    best_tput = None

    for record in records:
        total_time += record_time(record)

        recall = parse_number(record.get("recall"), None)
        tput = parse_number(record.get("query_throughput"), None)
        if recall is not None and tput is not None and recall >= target:
            if best_tput is None or tput > best_tput:
                best_tput = tput
                item_id = record.get("step_id", record.get("_line_nr"))
                turns.append((total_time / 3600, best_tput, item_id))

        if best_tput is not None:
            xs.append(total_time / 3600)
            ys.append(best_tput)

    return xs, ys, turns


def annotate_turns(ax, turns, color):
    for x, y, item_id in turns:
        ax.annotate(
            str(item_id),
            xy=(x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color=color,
        )


def plot_results(all_records, output_file):
    fig, axes = plt.subplots(
        len(RECALL_TARGETS),
        1,
        figsize=(10, 2.4 * len(RECALL_TARGETS)),
        sharex=True,
    )

    if len(RECALL_TARGETS) == 1:
        axes = [axes]

    for ax, target in zip(axes, RECALL_TARGETS):
        for label, records in all_records:
            xs, ys, turns = build_curve(records, target)
            if not xs:
                continue
            line = ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.4, label=label)[0]
            annotate_turns(ax, turns, line.get_color())

        ax.set_ylabel("Throughput")
        ax.set_title(f"Recall >= {target:g}")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Cumulative time (h)")
    for ax in axes:
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))

    fig.tight_layout()
    fig.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    all_records = []
    for label, result_file in RESULT_FILES:
        if not result_file.exists():
            raise FileNotFoundError(f"File not found: {result_file}")
        all_records.append((label, load_records(result_file)))

    output_file = Path.cwd() / OUTPUT_FILE
    plot_results(all_records, output_file)
    print(f"Saved plot to {output_file}")


if __name__ == "__main__":
    main()
