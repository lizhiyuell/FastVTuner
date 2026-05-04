import argparse
import json
import os

import matplotlib.pyplot as plt


def load_results(filename, max_step_id=None):
    phase_results = {}

    with open(filename, "r") as ifs:
        for line in ifs:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            step_id = data.get("step_id")
            if max_step_id is not None and (step_id is None or int(step_id) > max_step_id):
                continue

            phase = data.get("phase", "unknown")
            tput = data.get("query_throughput")
            recall = data.get("recall")

            if tput is None or recall is None:
                continue

            phase_results.setdefault(phase, []).append(
                {
                    "tput": float(tput),
                    "recall": float(recall),
                    "params": data.get("params", {}),
                }
            )

    return phase_results


def get_pareto_result(results):
    res = sorted(results, key=lambda x: x["tput"])

    final = []
    max_recall = float("-inf")
    pos = len(res) - 1
    while pos >= 0:
        item = res[pos]
        if item["recall"] > max_recall:
            final.append(item)
        max_recall = max(max_recall, item["recall"])
        pos -= 1

    final.reverse()
    return final


def get_axis_limits(all_results):
    x_values = []
    y_values = []

    for results in all_results.values():
        for item in results:
            x_values.append(item["recall"])
            y_values.append(item["tput"])

    if not x_values or not y_values:
        raise ValueError("No valid result points found.")

    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)

    x_pad = max((x_max - x_min) * 0.05, 1e-4)
    y_pad = max((y_max - y_min) * 0.05, 1e-4)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def plot_results(all_results, output_file, title):
    xlim, ylim = get_axis_limits(all_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        "tune": "tab:blue",
        "test": "tab:orange",
    }

    for ax, phase in zip(axes, ("tune", "test")):
        results = all_results.get(phase, [])
        if not results:
            raise ValueError(f"No valid result points found for phase: {phase}")

        pareto = get_pareto_result(results)
        x_all = [item["recall"] for item in results]
        y_all = [item["tput"] for item in results]
        x_pareto = [item["recall"] for item in pareto]
        y_pareto = [item["tput"] for item in pareto]

        color = colors.get(phase, "tab:green")
        ax.scatter(x_all, y_all, s=8, alpha=0.65, color=color, label=f"{phase} all")
        ax.plot(x_pareto, y_pareto, "-o", linewidth=2, markersize=4, color="tab:red", label=f"{phase} Pareto")
        ax.set_title(f"{phase} ({len(results)} points)")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Throughput")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot throughput-recall results and Pareto frontier.")
    parser.add_argument("name", help="Dataset prefix, e.g. gist")
    parser.add_argument("n", nargs="?", type=int, help="Only plot points with step_id <= n")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    all_results = {}
    for phase in ("tune", "test"):
        input_file = os.path.join(base_dir, f"{args.name}_{phase}.txt")
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"File not found: {input_file}")

        phase_results = load_results(input_file, args.n)
        all_results[phase] = phase_results.get(phase, [])

    output_file = os.path.join(base_dir, f"{args.name}_plot.png")
    plot_results(all_results, output_file, args.name)
    print(f"Saved plot to {output_file}")


if __name__ == "__main__":
    main()
