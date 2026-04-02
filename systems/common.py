from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import socket
import subprocess as sp
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Sequence
from scipy.stats import qmc

import numpy as np
import torch
import yaml

# 所有要用到的参数
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
BENCHMARK_ROOT = PROJECT_ROOT / "benchmark"
CONFIG_ROOT = PROJECT_ROOT / "config"
RESULT_ROOT = PROJECT_ROOT / "results"
DOCKER_CONFIG_ROOT = PROJECT_ROOT / "docker_config"
DOCKER_VOLUME_DIR = Path("/extend/volume")


# 所有要用到的参数

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = Path(
    os.getenv("VDTUNER_BENCHMARK_ROOT", PROJECT_ROOT / "vector-db-benchmark")
)
CONFIG_DIR = Path(os.getenv("VDTUNER_CONFIG_DIR", PROJECT_ROOT / "config" / "milvus"))
INDEX_PARAM_PATH = Path(
    os.getenv("VDTUNER_INDEX_PARAM_PATH", CONFIG_DIR / "index_param.json")
)
KNOB_PATH = Path(os.getenv("VDTUNER_KNOB_PATH", CONFIG_DIR / "whole_param.json"))
BENCHMARK_CONFIG_PATH = Path(
    os.getenv(
        "VDTUNER_BENCHMARK_CONFIG_PATH",
        BENCHMARK_ROOT / "experiments" / "configurations" / "milvus-single-node.json",
    )
)
MILVUS_YAML_PATH = Path(
    os.getenv(
        "VDTUNER_MILVUS_YAML_PATH",
        BENCHMARK_ROOT / "engine" / "servers" / "milvus-single-node" / "milvus.yaml",
    )
)
RESULTS_DIR = Path(os.getenv("VDTUNER_RESULTS_DIR", BENCHMARK_ROOT / "results"))
RUN_PY = BENCHMARK_ROOT / "run.py"
DEFAULT_ENGINE_NAME = os.getenv("VDTUNER_ENGINE_NAME", "milvus-default")
DEFAULT_DATASET_PATTERN = os.getenv("VDTUNER_DATASET_PATTERN", "*")
DEFAULT_READY_TIMEOUT_SECONDS = float(os.getenv("VDTUNER_READY_TIMEOUT", "180"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BRACE_RE = re.compile(r"\{([^{}]*)\}")
NUM_RE = re.compile(r"^\d+(\.\d+)?$")
DATASET_DIM_RE = re.compile(r"-(\d+)(?:-p-\d+)?(?:$|-)")


def _physical_core_count() -> int:
    cpu_count = os.cpu_count() or 1
    try:
        result = sp.run(
            ["lscpu"],
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            cores_per_socket = sockets = None
            for line in result.stdout.splitlines():
                if line.startswith("Core(s) per socket:"):
                    cores_per_socket = int(line.split(":", 1)[1].strip())
                elif line.startswith("Socket(s):"):
                    sockets = int(line.split(":", 1)[1].strip())
            if cores_per_socket and sockets:
                return max(1, cores_per_socket * sockets)
    except Exception:
        pass
    return max(1, cpu_count // 2 if cpu_count > 1 else 1)


DEFAULT_PARALLELISM = int(os.getenv("VDTUNER_PARALLELISM", str(_physical_core_count())))


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str | Path, data: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _vdb_name(engine_name: str) -> str:
    return engine_name.split("-", 1)[0]


def _dataset_token(run_params: Any) -> str:
    if run_params is None:
        return ""
    if isinstance(run_params, str):
        tokens = [t.strip('"\'') for t in run_params.split() if t.strip('"\'')]
        for token in tokens:
            if DATASET_DIM_RE.search(token):
                return token
        return tokens[-1] if tokens else run_params
    if isinstance(run_params, Sequence) and run_params:
        for item in run_params:
            if isinstance(item, str) and DATASET_DIM_RE.search(item):
                return item
        return str(run_params[-1])
    return str(run_params)


def infer_dimension_from_dataset_name(dataset_name: str) -> int | None:
    match = DATASET_DIM_RE.search(dataset_name)
    return int(match.group(1)) if match else None


def LHS_sample(dimension: int, num_points: int, seed: int) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=dimension, seed=seed)
    latin_samples = sampler.random(n=num_points)
    return latin_samples

# 这个参数就是加载所有可能的参数，然后将它们映射到[0, 1]，或者反向映射。我们规定不同数据库的可调优数据类型都统一为json格式
class KnobStand:
    def __init__(self, path: str | Path) -> None:
        with Path(path).open("r", encoding="utf-8") as f:
            self.knobs_detail = json.load(f)

    def scale_back(self, knob_name: str, zero_one_val: float) -> tuple[int, Any]:
        knob = self.knobs_detail[knob_name]
        if knob["type"] == "integer":
            real_val = int(zero_one_val * (knob["max"] - knob["min"]) + knob["min"])
            return real_val, real_val
        elif knob['type'] == 'enum':
            enum_size = len(knob['enum_values'])
            enum_index = int(enum_size * zero_one_val)
            enum_index = min(enum_size - 1, enum_index)
            real_val = knob['enum_values'][enum_index]
            return enum_index, real_val
        raise ValueError(f"Unsupported knob type: {knob['type']}")

    def scale_forward(self, knob_name: str, real_val: Any) -> float:
        knob = self.knobs_detail[knob_name]
        if knob['type'] == 'integer':
            zero_one_val = (real_val - knob['min']) / (knob['max'] - knob['min'])
            return zero_one_val
        elif knob['type'] == 'enum':
            enum_size = len(knob['enum_values'])
            zero_one_val = knob['enum_values'].index(real_val) / enum_size
            return zero_one_val
        raise ValueError(f"Unsupported knob type: {knob['type']}")


# 这几个关于configure的函数，我们后面再统一调节了，至少得适配不同VDB？？？
def filter_index_rule(conf: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    conf = {item: detail["default"] for item, detail in _load_json(INDEX_PARAM_PATH).items()} | dict(conf)
    conf["nprobe"] = min(conf["nlist"], max(1, int(conf["nprobe"])))
    index_type = conf["index_type"]
    if index_type in {"AUTOINDEX", "FLAT"}:
        return index_type, {}, {}
    if index_type in {"IVF_FLAT", "IVF_SQ8"}:
        return index_type, {"nlist": conf["nlist"]}, {"nprobe": conf["nprobe"]}
    if index_type == "IVF_PQ":
        return index_type, {
            "nlist": conf["nlist"],
            "m": conf["m"],
            "nbits": conf["nbits"],
        }, {"nprobe": conf["nprobe"]}
    if index_type == "HNSW":
        return index_type, {
            "M": conf["M"],
            "efConstruction": conf["efConstruction"],
        }, {"ef": conf["ef"]}
    if index_type == "SCANN":
        return index_type, {"nlist": conf["nlist"]}, {
            "nprobe": conf["nprobe"],
            "reorder_k": conf["reorder_k"],
        }
    raise ValueError(f"Unsupported index type: {index_type}")

# 同上？？
def configure_index(
    index_type: str,
    building_params: dict[str, Any],
    searching_params: dict[str, Any],
    engine_name: str = DEFAULT_ENGINE_NAME,
) -> None:
    conf = _load_json(BENCHMARK_CONFIG_PATH)
    if not isinstance(conf, list) or not conf:
        raise ValueError(f"Unexpected benchmark config shape: {BENCHMARK_CONFIG_PATH}")
    target = next((item for item in conf if item.get("name") == engine_name), None)
    if target is None:
        raise ValueError(
            f"Engine config `{engine_name}` not found in benchmark config: {BENCHMARK_CONFIG_PATH}"
        )
    target.setdefault("upload_params", {})
    target["upload_params"].update(
        index_type=index_type,
        index_params=building_params,
        parallel=DEFAULT_PARALLELISM,
    )
    target["search_params"] = [{"parallel": DEFAULT_PARALLELISM, "config": dict(searching_params)}]
    _write_json(BENCHMARK_CONFIG_PATH, conf)

# 同上？？
def filter_system_rule(conf: dict[str, Any]) -> dict[str, Any]:
    conf = dict(conf)
    if "dataCoord*segment*sealProportion" in conf:
        conf["dataCoord*segment*sealProportion"] /= 100
    return conf


def _set_nested_value(tree: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split("*")
    node = tree
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


# 同上？？
def configure_system(params: dict[str, Any]) -> None:
    if not MILVUS_YAML_PATH.exists():
        raise FileNotFoundError(MILVUS_YAML_PATH)
    with MILVUS_YAML_PATH.open("r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    for key, value in params.items():
        _set_nested_value(conf, key, value)
    with MILVUS_YAML_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(conf, f, sort_keys=False)


# 后面都是helper函数，可能不会被用到？？？后面再过滤一下
def _result_paths(dataset_name: str, suffix: str, engine_name: str = DEFAULT_ENGINE_NAME) -> list[Path]:
    return sorted(
        RESULTS_DIR.glob(f"{engine_name}-{dataset_name}-{suffix}-*.json"),
        key=lambda path: path.stat().st_mtime,
    )


def _parse_benchmark_results(
    dataset_name: str,
    engine_name: str = DEFAULT_ENGINE_NAME,
    since_time: float | None = None,
    query_phase: str | None = None,
) -> dict[str, Any]:
    upload_files = _result_paths(dataset_name, "upload", engine_name)
    search_files = _result_paths(dataset_name, "search", engine_name)
    if since_time is not None:
        upload_files = [path for path in upload_files if path.stat().st_mtime >= since_time]
        search_files = [path for path in search_files if path.stat().st_mtime >= since_time]
    if not upload_files and not search_files:
        raise RuntimeError(f"No benchmark result files found for dataset: {dataset_name}")

    upload_stats = _load_json(upload_files[-1]) if upload_files else {}
    search_stats = [_load_json(path) for path in search_files]
    if query_phase is not None:
        search_stats = [
            item for item in search_stats if item.get("params", {}).get("query_phase") == query_phase
        ]
        if not search_stats:
            raise RuntimeError(
                f"No benchmark search results found for dataset={dataset_name}, query_phase={query_phase}"
            )

    search_results = [item.get("results", {}) for item in search_stats]
    record_nr = sum(int(item.get("params", {}).get("selected_query_count", 0)) for item in search_stats)
    precisions = [float(item.get("mean_precisions", 0.0)) for item in search_results] or [0.0]
    total_times = [float(item.get("total_time", 0.0)) for item in search_results] or [0.0]

    upload_results = upload_stats.get("results", {})
    index_time = float(upload_results.get("upload_time", upload_results.get("total_time", 0.0)))
    return {
        "dataset_name": dataset_name,
        "query_phase": query_phase,
        "upload": upload_stats,
        "search": search_stats,
        "recall": float(np.mean(precisions)),
        "index_time": index_time,
        "query_time": float(np.mean(total_times)),
        "upload_total": float(upload_results.get("total_time", index_time)),
        "record_nr": record_nr,
    }


def _docker_compose_command() -> str:
    if shutil.which("docker"):
        return "docker compose"
    if shutil.which("docker-compose"):
        return "docker-compose"
    raise RuntimeError(
        "Neither `docker compose` nor `docker-compose` is available in PATH. "
        "Run FastVTuner inside the `fastvtuner` conda environment with Docker installed."
    )


def _clear_directory_contents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _purge_milvus_persistent_state() -> None:
    for path in (
        Path(os.getenv("VDTUNER_ETCD_DATA_DIR", "/extend/index/etcd")),
        Path(os.getenv("VDTUNER_MINIO_DATA_DIR", "/extend/index/minio")),
        Path(os.getenv("VDTUNER_MILVUS_DATA_DIR", "/extend/index/milvus")),
    ):
        _clear_directory_contents(path)


def _server_dir_for_vdb(vdb_name: str) -> Path:
    return BENCHMARK_ROOT / "engine" / "servers" / f"{vdb_name}-single-node"


def _tcp_port_ready(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_ready(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError, urllib.error.HTTPError):
        return False


def _redis_ready(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout) as conn:
            conn.sendall(b"*1\r\n$4\r\nPING\r\n")
            return b"PONG" in conn.recv(64)
    except OSError:
        return False


def _wait_until_ready(
    check: Callable[[], bool],
    *,
    timeout_seconds: float,
    poll_interval_seconds: float = 2.0,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(poll_interval_seconds)
    return check()


def _wait_ready_milvus(host: str = "localhost") -> bool:
    return _wait_until_ready(
        lambda: _tcp_port_ready(host, 19530),
        timeout_seconds=DEFAULT_READY_TIMEOUT_SECONDS,
    )


def _wait_ready_qdrant(host: str = "localhost") -> bool:
    return _wait_until_ready(
        lambda: _http_ready(f"http://{host}:6333/readyz"),
        timeout_seconds=DEFAULT_READY_TIMEOUT_SECONDS,
    )


def _wait_ready_weaviate(host: str = "localhost") -> bool:
    return _wait_until_ready(
        lambda: _http_ready(f"http://{host}:8090/.well-known/ready"),
        timeout_seconds=DEFAULT_READY_TIMEOUT_SECONDS,
    )


def _wait_ready_elasticsearch(host: str = "localhost") -> bool:
    return _wait_until_ready(
        lambda: _http_ready(f"http://{host}:9200/_cluster/health"),
        timeout_seconds=DEFAULT_READY_TIMEOUT_SECONDS,
    )


def _wait_ready_redis(host: str = "localhost") -> bool:
    return _wait_until_ready(
        lambda: _redis_ready(host, 6379),
        timeout_seconds=DEFAULT_READY_TIMEOUT_SECONDS,
    )


VDB_READY_WAITERS = {
    "milvus": _wait_ready_milvus,
    "qdrant": _wait_ready_qdrant,
    "weaviate": _wait_ready_weaviate,
    "elasticsearch": _wait_ready_elasticsearch,
    "redis": _wait_ready_redis,
}


def _compose_status_snapshot(server_dir: Path) -> str:
    compose_cmd = _docker_compose_command()
    result = sp.run(
        ["bash", "-lc", f"cd {shlex.quote(str(server_dir))} ; {compose_cmd} ps"],
        cwd=str(BENCHMARK_ROOT),
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        text=True,
        check=False,
    )
    return "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part) or (
        "<no compose status available>"
    )


def _compose_logs_snapshot(server_dir: Path, tail: int = 200) -> str:
    compose_cmd = _docker_compose_command()
    result = sp.run(
        ["bash", "-lc", f"cd {shlex.quote(str(server_dir))} ; {compose_cmd} logs --tail {tail}"],
        cwd=str(BENCHMARK_ROOT),
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        text=True,
        check=False,
    )
    return "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part) or (
        "<no compose logs available>"
    )


def _wait_server_ready(vdb_name: str, server_dir: Path, host: str = "localhost") -> None:
    waiter = VDB_READY_WAITERS.get(vdb_name)
    if waiter is None:
        raise ValueError(f"No ready waiter registered for VDB `{vdb_name}`")
    if waiter(host):
        return
    raise RuntimeError(
        f"{vdb_name} did not become ready within {DEFAULT_READY_TIMEOUT_SECONDS:.0f}s.\n"
        f"Compose status:\n{_compose_status_snapshot(server_dir)}\n\n"
        f"Recent container logs:\n{_compose_logs_snapshot(server_dir)}"
    )


def _start_server(engine_name: str) -> tuple[str, str]:
    vdb_name = _vdb_name(engine_name)
    server_dir = _server_dir_for_vdb(vdb_name)
    if not server_dir.exists():
        raise FileNotFoundError(server_dir)

    compose_cmd = _docker_compose_command()
    quoted_server_dir = shlex.quote(str(server_dir))
    if vdb_name == "milvus":
        _purge_milvus_persistent_state()
    start_cmd = f"cd {quoted_server_dir} ; {compose_cmd} down ; {compose_cmd} up -d"
    stop_cmd = f"cd {quoted_server_dir} ; {compose_cmd} down"
    start_result = sp.run(
        ["bash", "-lc", start_cmd],
        cwd=str(BENCHMARK_ROOT),
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        text=True,
        check=False,
    )
    if start_result.returncode not in (0, None):
        raise RuntimeError(
            f"{vdb_name} startup failed:\nstdout:\n{start_result.stdout}\nstderr:\n{start_result.stderr}"
        )
    _wait_server_ready(vdb_name, server_dir)
    return str(server_dir), stop_cmd


def _stop_server(stop_cmd: str) -> None:
    sp.run(
        ["bash", "-lc", stop_cmd],
        cwd=str(BENCHMARK_ROOT),
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        text=True,
        check=False,
    )


def _run_benchmark_phase(
    dataset_name: str,
    engine_name: str = DEFAULT_ENGINE_NAME,
    timeout: float = 86400.0,
    *,
    query_phase: str,
    tune_query_ratio: float,
    test_query_ratio: float,
    split_seed: int,
    skip_upload: bool = False,
) -> dict[str, Any]:
    if not RUN_PY.exists():
        raise FileNotFoundError(RUN_PY)

    cmd = [
        sys.executable,
        str(RUN_PY),
        "--engines",
        engine_name,
        "--datasets",
        dataset_name,
        "--host",
        "localhost",
        "--timeout",
        str(timeout),
    ]
    if skip_upload:
        cmd.append("--skip-upload")

    env = os.environ.copy()
    env.update(
        FASTVTUNER_QUERY_PHASE=query_phase,
        FASTVTUNER_TUNE_QUERY_RATIO=str(tune_query_ratio),
        FASTVTUNER_TEST_QUERY_RATIO=str(test_query_ratio),
        FASTVTUNER_QUERY_SPLIT_SEED=str(split_seed),
    )
    marker_time = time.time()
    result = sp.run(
        cmd,
        cwd=str(BENCHMARK_ROOT),
        env=env,
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        text=True,
        check=False,
    )
    if result.returncode not in (0, None):
        raise RuntimeError(
            "Benchmark execution failed:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return _parse_benchmark_results(
        dataset_name,
        engine_name=engine_name,
        since_time=marker_time,
        query_phase=query_phase,
    )


def _run_benchmark(
    dataset_name: str,
    engine_name: str = DEFAULT_ENGINE_NAME,
    timeout: float = 86400.0,
    *,
    tune_query_ratio: float,
    test_query_ratio: float,
    split_seed: int,
) -> dict[str, Any]:
    _, stop_cmd = _start_server(engine_name)
    try:
        result = {
            "tune": _run_benchmark_phase(
                dataset_name,
                engine_name=engine_name,
                timeout=timeout,
                query_phase="tune",
                tune_query_ratio=tune_query_ratio,
                test_query_ratio=test_query_ratio,
                split_seed=split_seed,
            ),
            "test": None,
        }
        if test_query_ratio > 0.0:
            result["test"] = _run_benchmark_phase(
                dataset_name,
                engine_name=engine_name,
                timeout=timeout,
                query_phase="test",
                tune_query_ratio=tune_query_ratio,
                test_query_ratio=test_query_ratio,
                split_seed=split_seed,
                skip_upload=True,
            )
        return result
    finally:
        _stop_server(stop_cmd)
