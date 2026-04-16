import json
import os

import matplotlib.pyplot as plt


SAMPLED_TUNE_FILE = "/root/lzy/FastVTuner/results/vdtuner/milvus/gist-p-10_tune.txt"
SAMPLED_TEST_FILE = "/root/lzy/FastVTuner/results/vdtuner/milvus/gist-p-10_test.txt"
FULL_TUNE_FILE = "/root/lzy/FastVTuner/results/motivation_retest_sampled_config/milvus/gist-p-10_tune.txt"
FULL_TEST_FILE = "/root/lzy/FastVTuner/results/motivation_retest_sampled_config/milvus/gist-p-10_test.txt"
TUNE_PLOT_FILE = "/root/lzy/FastVTuner/results/motivation_retest_sampled_config/milvus/gist_tune_pareto_compare.png"
TEST_PLOT_FILE = "/root/lzy/FastVTuner/results/motivation_retest_sampled_config/milvus/gist_test_pareto_compare.png"


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

            tput = data.get("query_throughput")
            recall = data.get("recall")
            step_id = data.get("step_id")
            if tput is None or recall is None or step_id is None:
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


def get_points_by_ids(results, ids):
    id_set = set(ids)
    selected = []
    for item in results:
        if item["step_id"] in id_set:
            selected.append(item)
    return selected


def plot_phase(phase, full_results, sampled_ids, full_pareto, output_file):
    plt.figure(figsize=(8, 6))

    full_x = [item["recall"] for item in full_results]
    full_y = [item["tput"] for item in full_results]
    sampled_points_on_full = get_points_by_ids(full_results, sampled_ids)
    sampled_x = [item["recall"] for item in sampled_points_on_full]
    sampled_y = [item["tput"] for item in sampled_points_on_full]
    full_pareto_x = [item["recall"] for item in full_pareto]
    full_pareto_y = [item["tput"] for item in full_pareto]

    plt.scatter(
        full_x,
        full_y,
        s=28,
        alpha=0.55,
        facecolors="none",
        edgecolors="#b3b3b3",
        linewidths=0.9,
        label="full all points",
    )
    plt.plot(
        full_pareto_x,
        full_pareto_y,
        color="tab:red",
        linewidth=1.4,
        alpha=0.9,
        label="full Pareto line",
    )
    plt.scatter(
        full_pareto_x,
        full_pareto_y,
        s=60,
        alpha=0.95,
        facecolors="none",
        edgecolors="tab:red",
        linewidths=1.4,
        label="full Pareto points",
    )
    plt.scatter(
        sampled_x,
        sampled_y,
        s=30,
        alpha=0.95,
        color="tab:blue",
        label="sampled Pareto IDs on full",
    )

    plt.xlabel("Recall")
    plt.ylabel("Throughput")
    plt.title(f"gist {phase}: full Pareto and sampled Pareto IDs on full")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()


def analyze_phase(phase, sampled_file, full_file):
    sampled_results = load_results(sampled_file, phase)
    full_results = load_results(full_file, phase)

    sampled_pareto = get_pareto_points(sampled_results)
    full_pareto = get_pareto_points(full_results)

    sampled_ids = {item["step_id"] for item in sampled_pareto}
    full_ids = {item["step_id"] for item in full_pareto}
    intersection_ids = sampled_ids & full_ids

    sampled_count = len(sampled_ids)
    full_count = len(full_ids)
    intersection_count = len(intersection_ids)

    precision = intersection_count / sampled_count if sampled_count else 0.0
    recall = intersection_count / full_count if full_count else 0.0

    print(f"[{phase}]")
    print(f"sampled_file: {os.path.basename(sampled_file)}")
    print(f"full_file: {os.path.basename(full_file)}")
    print(f"sampled_pareto_count: {sampled_count}")
    print(f"full_pareto_count: {full_count}")
    print(f"intersection_count: {intersection_count}")
    print(f"precision: {precision:.6f}")
    print(f"recall: {recall:.6f}")
    print(f"sampled_pareto_ids: {sorted(sampled_ids)}")
    print(f"full_pareto_ids: {sorted(full_ids)}")
    print(f"intersection_ids: {sorted(intersection_ids)}")
    print()

    return full_results, sorted(sampled_ids), full_pareto


def main():
    tune_full_results, tune_sampled_ids, tune_full_pareto = analyze_phase("tune", SAMPLED_TUNE_FILE, FULL_TUNE_FILE)
    test_full_results, test_sampled_ids, test_full_pareto = analyze_phase("test", SAMPLED_TEST_FILE, FULL_TEST_FILE)

    plot_phase("tune", tune_full_results, tune_sampled_ids, tune_full_pareto, TUNE_PLOT_FILE)
    plot_phase("test", test_full_results, test_sampled_ids, test_full_pareto, TEST_PLOT_FILE)

    print(f"saved_plot: {os.path.basename(TUNE_PLOT_FILE)}")
    print(f"saved_plot: {os.path.basename(TEST_PLOT_FILE)}")


if __name__ == "__main__":
    main()
