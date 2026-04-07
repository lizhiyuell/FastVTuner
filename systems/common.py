from __future__ import annotations

import json
import os
from pathlib import Path
from scipy.stats import qmc


# 所有要用到的参数
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
BENCHMARK_ROOT = PROJECT_ROOT / "benchmark"
CONFIG_ROOT = PROJECT_ROOT / "config"
RESULT_ROOT = PROJECT_ROOT / "results"
DOCKER_CONFIG_ROOT = PROJECT_ROOT / "docker_config"
DOCKER_VOLUME_DIR = Path("/extend/volume")


def get_physical_cpu_count():
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        physical_cores = set()
        physical_id = None
        core_id = None

        with cpuinfo_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    if physical_id is not None and core_id is not None:
                        physical_cores.add((physical_id, core_id))
                    physical_id = None
                    core_id = None
                    continue

                if ":" not in line:
                    continue

                key, value = [part.strip() for part in line.split(":", 1)]
                if key == "physical id":
                    physical_id = value
                elif key == "core id":
                    core_id = value

            if physical_id is not None and core_id is not None:
                physical_cores.add((physical_id, core_id))

        if physical_cores:
            return len(physical_cores)

    return os.cpu_count() or 1


# specially designed for Milvus
def update_m_with_dimension(config_template_path, dimension):
    # 1. find all factors of 'dimension' and sort them
    factors = []
    for i in range(1, dimension + 1):
        if dimension % i == 0:
            factors.append(i)
    factors.sort()

    # 2. find the one that is nearest to '10' as the default value
    default_value = min(factors, key=lambda x: abs(x - 10))

    # 3. read the index_param.json and whole_param.json, updating m.default and m.enum_values
    with open(config_template_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["m"]["default"] = default_value
    data["m"]["enum_values"] = factors

    with open(config_template_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def LHS_sample(dimension, num_points, seed):
    sampler = qmc.LatinHypercube(d=dimension, seed=seed)
    latin_samples = sampler.random(n=num_points)
    return latin_samples
