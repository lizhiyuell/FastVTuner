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

BUILD_PARALLEL = 20
SEARCH_PARALLEL = 20

# specially designed for Milvus
def update_m_with_dimension(config_template_path, dimension):
    # 1. find all factors of 'dimension' and sort them
    factors = []
    for i in range(1, dimension + 1):
        if dimension % i == 0:
            factors.append(i)
    factors.sort()

    lower_bound = (dimension + 7) // 8
    default_candidates = [factor for factor in factors if factor >= lower_bound]
    if not default_candidates:
        raise ValueError(f"Cannot find valid m >= {lower_bound} for dimension={dimension}")

    # 2. read the index_param.json and whole_param.json, updating m.default and m.enum_values
    with open(config_template_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["m"]["default"] = int(default_candidates[0])
    data["m"]["enum_values"] = factors

    with open(config_template_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def LHS_sample(dimension, num_points, seed):
    sampler = qmc.LatinHypercube(d=dimension, seed=seed)
    latin_samples = sampler.random(n=num_points)
    return latin_samples
