from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import subprocess as sp
import time
from urllib import error, request
from typing import Any
import shutil
import json
import numpy as np
from common import *
import os
import pymilvus

env = os.environ.copy()
env["DOCKER_VOLUME_DIRECTORY"] = DOCKER_VOLUME_DIR
DEVNULL = sp.DEVNULL


SERVER_PATH_MAP = {
    "milvus": "milvus-single-node",
    "qdrant": "qdrant-single-node",
}

COLLECTION_NAME = "tuning_benchmark"
BATCH_SIZE = 64
CURRENT_CONFIG_PATH = CONFIG_ROOT / "current.json"
_BUILD_COLLECTION = None


def _get_mp_start_method():
    if "forkserver" in mp.get_all_start_methods():
        return "forkserver"
    return "spawn"


def _init_milvus_build_worker():
    global _BUILD_COLLECTION
    pymilvus.connections.connect(alias="default", host="localhost", port="19530")
    _BUILD_COLLECTION = pymilvus.Collection(COLLECTION_NAME, using="default")


def _insert_milvus_batch(batch):
    start, batch_vectors = batch
    end = start + len(batch_vectors)
    _BUILD_COLLECTION.insert([list(range(start, end)), batch_vectors])

class VDBEngine:
    def __init__(
        self,
        db_type: str, # which DB is used for
    ) -> None:
        if db_type not in SERVER_PATH_MAP:
            raise ValueError(f"Unsupported db_type: {db_type}")

        self.db_type = db_type
        # the base dir for the docker config files
        self.server_path = DOCKER_CONFIG_ROOT / db_type
        self.datasets_root = DATASET_ROOT

        self.dataset_name: str | None = None
        self.vec_dataset: Any | None = None
        self.vec_search: Any | None = None
        self.vec_test: Any | None = None
        self.search_params = {}

    def load_dataset(self, dataset_name: str) -> None:
        if not dataset_name:
            raise ValueError("dataset_name must be a non-empty string")

        self.dataset_name = dataset_name
        self.vec_dataset = None
        self.vec_search = None
        self.vec_test = None
        self.dimension = None
        self.distance_metric = None

        dataset_file = self.datasets_root / f"{dataset_name}.npz"

        if dataset_file.exists():
            dataset = np.load(dataset_file)
            self.dimension = dataset["dimension"]
            self.distance_metric = dataset["distance_metric"]
            self.vec_dataset = dataset["train"]
            self.vec_search = dataset["search"]
            self.vec_test = dataset["test"]
            self.vec_search_top100 = dataset["search_top100"]
            self.vec_test_top100 = dataset["test_top100"]
        else:
            raise FileNotFoundError(
                f"Cannot find dataset npy for '{dataset_name}' under {self.datasets_root}"
            )

    def start(self) -> None:
        # 1. remove the previous results
        self._remove_previous()

        # 2. start the container
        # stop the previous one
        sp.run(
            ["docker", "compose", "down"],
            cwd=self.server_path,
            env=env,
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
        sp.run(
            ["docker", "compose", "up", "-d"],
            cwd=self.server_path,
            env=env,
            stdout=DEVNULL,
            stderr=DEVNULL,
        )

        # 3. wait for connection built
        while True:
            try:
                with request.urlopen("http://localhost:9091/healthz", timeout=10):
                    break
            except (error.URLError, error.HTTPError):
                time.sleep(1)

        # print("[VDB] VDB started")

    def stop(self) -> None:
        sp.run(
            ["docker", "compose", "down"],
            cwd=self.server_path,
            env=env,
            stdout=DEVNULL,
            stderr=DEVNULL,
        )

        # print("[VDB] VDB ended")

    def _load_current_config(self):
        with CURRENT_CONFIG_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)

    # build the index with dataset vectors
    def build(self):
        if self.vec_dataset is None:
            raise RuntimeError("Dataset is not loaded")

        if self.db_type == "milvus":
            current_config = self._load_current_config()
            upload_params = dict(current_config.get("upload_params") or {})
            index_type = current_config.get("index")
            index_params = dict(upload_params.get("index_params") or {})
            parallel = int(upload_params.get("parallel", 1))
            if not index_type:
                raise ValueError(f"Missing `index` in {CURRENT_CONFIG_PATH}")

            # we only use several fixed distance names
            should_norm = False
            if self.distance_metric=="angular":
                metric_type = "IP"
                should_norm = True
            elif self.distance_metric=="euclidean":
                metric_type = "L2"
            elif self.distance_metric=="innter-product":
                metric_type = "IP"
            else:
                raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

            vectors = np.asarray(self.vec_dataset, dtype=np.float32)
            if should_norm:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vectors = vectors / norms

            pymilvus.connections.connect(alias="default", host="localhost", port="19530")
            collection_name = COLLECTION_NAME
            try:
                pymilvus.utility.drop_collection(collection_name, using="default")
            except pymilvus.MilvusException:
                pass

            schema = pymilvus.CollectionSchema(
                [
                    pymilvus.FieldSchema(name="id", dtype=pymilvus.DataType.INT64, is_primary=True),
                    pymilvus.FieldSchema(
                        name="vector",
                        dtype=pymilvus.DataType.FLOAT_VECTOR,
                        dim=int(np.asarray(self.dimension).reshape(-1)[0]),
                    ),
                ],
                description=collection_name,
            )
            collection = pymilvus.Collection(name=collection_name, schema=schema, using="default")

            total_start = time.perf_counter()
            batch_size = BATCH_SIZE

            if int(parallel) <= 1:
                for start in range(0, len(vectors), batch_size):
                    end = min(start + batch_size, len(vectors))
                    collection.insert([list(range(start, end)), vectors[start:end].tolist()])
            else:
                ctx = mp.get_context(_get_mp_start_method())
                batches = (
                    (start, vectors[start : start + batch_size].tolist())
                    for start in range(0, len(vectors), batch_size)
                )
                with ctx.Pool(
                    processes=int(parallel),
                    initializer=_init_milvus_build_worker,
                ) as pool:
                    list(pool.imap(_insert_milvus_batch, batches))

            collection.flush()
            collection.create_index(
                field_name="vector",
                index_params={
                    "metric_type": metric_type,
                    "index_type": index_type,
                    "params": index_params,
                },
            )
            for index in collection.indexes:
                pymilvus.wait_for_index_building_complete(
                    collection_name,
                    index_name=index.index_name,
                    using="default",
                )
            collection.load()
            return  time.perf_counter() - total_start
        else:
            raise NotImplementedError(f"build is not implemented for {self.db_type}")

    # one search step with search_vecs
    # test: whether use the search/test vectors
    # ratio: how many query vecs are used, in (0, 1]
    def query(self, top_k, test=False, ratio=1.0):
        if self.vec_dataset is None:
            raise RuntimeError("Dataset is not loaded")
        if ratio <= 0 or ratio > 1:
            raise ValueError("ratio must be in (0, 1]")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if self.db_type == "milvus":
            current_config = self._load_current_config()
            search_section = dict(current_config.get("search_params") or {})
            search_params = dict(search_section.get("config") or {})
            parallel = int(search_section.get("parallel", 1))

            should_norm = False
            if self.distance_metric == "angular":
                metric_type = "IP"
                should_norm = True
            elif self.distance_metric == "euclidean":
                metric_type = "L2"
            elif self.distance_metric == "innter-product":
                metric_type = "IP"
            else:
                raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

            query_vecs = self.vec_test if test else self.vec_search
            query_top100 = self.vec_test_top100 if test else self.vec_search_top100
            use_count = max(1, int(len(query_vecs) * ratio))

            vectors = np.asarray(query_vecs[:use_count], dtype=np.float32)
            if should_norm:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vectors = vectors / norms

            pymilvus.connections.connect(alias="default", host="localhost", port="19530")
            collection = pymilvus.Collection(COLLECTION_NAME, using="default")
            search_conf = {"metric_type": metric_type, "params": dict(search_params or {})}

            total_start = time.perf_counter()
            if int(parallel) <= 1:
                recalls = []
                for i in range(use_count):
                    res = collection.search(
                        data=[vectors[i].tolist()],
                        anns_field="vector",
                        param=search_conf,
                        limit=top_k,
                    )
                    ids = set(res[0].ids)
                    gt = set(np.asarray(query_top100[i]).tolist()[:top_k])
                    recalls.append(len(ids.intersection(gt)) / top_k)
            else:
                with ThreadPoolExecutor(max_workers=int(parallel)) as executor:
                    results = list(
                        executor.map(
                            lambda i: collection.search(
                                data=[vectors[i].tolist()],
                                anns_field="vector",
                                param=search_conf,
                                limit=top_k,
                            ),
                            range(use_count),
                        )
                    )
                recalls = []
                for i, res in enumerate(results):
                    ids = set(res[0].ids)
                    gt = set(np.asarray(query_top100[i]).tolist()[:top_k])
                    recalls.append(len(ids.intersection(gt)) / top_k)

            total_query_time = time.perf_counter() - total_start
            return total_query_time, float(np.mean(recalls)), use_count
        else:
            raise NotImplementedError(f"query is not implemented for {self.db_type}")

    def _remove_previous(self):
        if not DOCKER_VOLUME_DIR.exists():
            return

        for entry in os.scandir(DOCKER_VOLUME_DIR):
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path)
            else:
                os.unlink(entry.path)


if __name__=="__main__":
    
    # dataset_name = "gist-p-10"
    dataset_name = "gist"

    vdbengine = VDBEngine("milvus")
    vdbengine.load_dataset(dataset_name)

    vdbengine.start()

    ts = vdbengine.build()

    print(f"Build finish in {ts}")

    ts, recall, query_count = vdbengine.query(10, False, 1.0)

    print(f"Search time {ts}, recall {recall}%, query_count {query_count}")

    vdbengine.stop()
