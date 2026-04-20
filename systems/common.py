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
INDEX_BUILD_TIMEOUT_SECONDS = 30 * 60
# INDEX_BUILD_TIMEOUT_SECONDS = 10

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

# Given a previous result file, load the configuration (dict) and return as a list
def load_result_config(result_file_path):
    configs = []
    with open(result_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            params = data.get("params")
            if params is not None:
                configs.append(params)
    return configs
