import json
import os

import matplotlib.pyplot as plt


VDTUNER_TUNE_FILE = "/root/lzy/FastVTuner/results/vdtuner/milvus/gist_tune.txt"
VDTUNER_TEST_FILE = "/root/lzy/FastVTuner/results/vdtuner/milvus/gist_test.txt"
MOTIVATION_TUNE_FILE = "/root/lzy/FastVTuner/results/motivation_retest_sampled_config/milvus/gist-p-10_tune.txt"
MOTIVATION_TEST_FILE = "/root/lzy/FastVTuner/results/motivation_retest_sampled_config/milvus/gist-p-10_test.txt"

TUNE_PLOT_FILE = "/root/lzy/FastVTuner/results/vdtuner/milvus/gist_tune_pareto_line_compare.png"
TEST_PLOT_FILE = "/root/lzy/FastVTuner/results/vdtuner/milvus/gist_test_pareto_line_compare.png"


def load_results(filename, expected_phase):
    results = []

    with open(filename, "r") as ifs:
        for line in ifs:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            if data.get("phase") != expected_phase:
                continue

            step_id = data.get("step_id")
            tput = data.get("query_throughput")
            recall = data.get("recall")
            if step_id is None or tput is None or recall is None:
                continue

            results.append(
                {
                    "step_id": int(step_id),
                    "tput": float(tput),
                    "recall": float(recall),
                }
            )

    return results


def get_pareto_points(results):
    ordered = sorted(results, key=lambda item: item["tput"])

    pareto = []
    max_recall = float("-inf")
    pos = len(ordered) - 1
    while pos >= 0:
        item = ordered[pos]
        if item["recall"] > max_recall:
            pareto.append(item)
        max_recall = max(max_recall, item["recall"])
        pos -= 1

    pareto.reverse()
    return pareto


def is_dominated(point, frontier):
    for other in frontier:
        better_or_equal = other["recall"] >= point["recall"] and other["tput"] >= point["tput"]
        strictly_better = other["recall"] > point["recall"] or other["tput"] > point["tput"]
        if better_or_equal and strictly_better:
            return True
    return False


def count_dominated(points, frontier):
    count = 0
    ids = []
    for point in points:
        if is_dominated(point, frontier):
            count += 1
            ids.append(point["step_id"])
    return count, ids


def plot_phase(phase, vdtuner_results, motivation_results, vdtuner_pareto, motivation_pareto, output_file):
    plt.figure(figsize=(8, 6))

    vdtuner_x = [item["recall"] for item in vdtuner_results]
    vdtuner_y = [item["tput"] for item in vdtuner_results]
    motivation_x = [item["recall"] for item in motivation_results]
    motivation_y = [item["tput"] for item in motivation_results]

    vdtuner_pareto_x = [item["recall"] for item in vdtuner_pareto]
    vdtuner_pareto_y = [item["tput"] for item in vdtuner_pareto]
    motivation_pareto_x = [item["recall"] for item in motivation_pareto]
    motivation_pareto_y = [item["tput"] for item in motivation_pareto]

    plt.scatter(vdtuner_x, vdtuner_y, s=10, alpha=0.18, color="#9ecae1", label="vdtuner all points")
    plt.scatter(motivation_x, motivation_y, s=10, alpha=0.18, color="#fcbba1", label="motivation all points")
    plt.plot(vdtuner_pareto_x, vdtuner_pareto_y, "-o", linewidth=2, markersize=4, color="tab:blue", label="vdtuner Pareto")
    plt.plot(motivation_pareto_x, motivation_pareto_y, "-o", linewidth=2, markersize=4, color="tab:red", label="motivation Pareto")

    plt.xlabel("Recall")
    plt.ylabel("Throughput")
    plt.title(f"gist {phase}: Pareto line comparison")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()


def analyze_phase(phase, vdtuner_file, motivation_file, output_file):
    vdtuner_results = load_results(vdtuner_file, phase)
    motivation_results = load_results(motivation_file, phase)

    vdtuner_pareto = get_pareto_points(vdtuner_results)
    motivation_pareto = get_pareto_points(motivation_results)

    vdtuner_ids = [item["step_id"] for item in vdtuner_pareto]
    motivation_ids = [item["step_id"] for item in motivation_pareto]
    id_intersection = sorted(set(vdtuner_ids) & set(motivation_ids))

    vdtuner_dominated_count, vdtuner_dominated_ids = count_dominated(vdtuner_pareto, motivation_pareto)
    motivation_dominated_count, motivation_dominated_ids = count_dominated(motivation_pareto, vdtuner_pareto)

    print(f"[{phase}]")
    print(f"vdtuner_pareto_count: {len(vdtuner_pareto)}")
    print(f"motivation_pareto_count: {len(motivation_pareto)}")
    print(f"vdtuner_pareto_ids: {vdtuner_ids}")
    print(f"motivation_pareto_ids: {motivation_ids}")
    print(f"id_intersection_count: {len(id_intersection)}")
    print(f"id_intersection: {id_intersection}")
    print(f"vdtuner_points_dominated_by_motivation: {vdtuner_dominated_count}")
    print(f"vdtuner_dominated_ids: {vdtuner_dominated_ids}")
    print(f"motivation_points_dominated_by_vdtuner: {motivation_dominated_count}")
    print(f"motivation_dominated_ids: {motivation_dominated_ids}")
    print()

    plot_phase(phase, vdtuner_results, motivation_results, vdtuner_pareto, motivation_pareto, output_file)
    print(f"saved_plot: {os.path.basename(output_file)}")
    print()


def main():
    analyze_phase("tune", VDTUNER_TUNE_FILE, MOTIVATION_TUNE_FILE, TUNE_PLOT_FILE)
    analyze_phase("test", VDTUNER_TEST_FILE, MOTIVATION_TEST_FILE, TEST_PLOT_FILE)


if __name__ == "__main__":
    main()
