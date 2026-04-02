#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read train/test vectors from an HDF5 ANN dataset, split test into "
            "search/test, compute exact ground truth, and save everything into "
            "an NPZ data file."
        )
    )
    parser.add_argument("--input-hdf5", required=True, help="Path to the input HDF5 file.")
    parser.add_argument(
        "--output-file",
        required=True,
        help="Output NPZ file name or path. If a relative path is given, it is resolved from the current directory.",
    )
    parser.add_argument("--dimension", required=True, type=int, help="Vector dimension to store in the output file.")
    parser.add_argument(
        "--distance-metric",
        required=True,
        help="Distance metric for exact ground truth: l2/euclidean, cosine/angular, dot/ip.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to split the original test set into search and test.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size used by exact top-k computation.",
    )
    return parser.parse_args()


def canonical_metric(metric: str) -> str:
    normalized = metric.strip().lower()
    aliases = {
        "l2": "l2",
        "euclidean": "l2",
        "cosine": "cosine",
        "angular": "cosine",
        "dot": "dot",
        "ip": "dot",
        "inner_product": "dot",
        "inner-product": "dot",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported distance metric: {metric}")
    return aliases[normalized]


def load_hdf5_vectors(hdf5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(hdf5_path, "r") as data:
        train = np.asarray(data["train"], dtype=np.float32)
        test = np.asarray(data["test"], dtype=np.float32)
    return train, test


def split_test_vectors(test: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if test.shape[0] < 2:
        raise ValueError("The HDF5 test split must contain at least 2 vectors.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(test.shape[0])
    midpoint = test.shape[0] // 2
    search_indices = indices[:midpoint]
    test_indices = indices[midpoint:]
    return test[search_indices], test[test_indices]


def maybe_normalize(vectors: np.ndarray, metric: str) -> np.ndarray:
    if metric != "cosine":
        return np.asarray(vectors, dtype=np.float32)
    vectors = np.asarray(vectors, dtype=np.float32).copy()
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    vectors /= norms
    return vectors


def topk_indices_from_scores(scores: np.ndarray, topk: int, largest: bool) -> np.ndarray:
    if largest:
        partial = np.argpartition(-scores, kth=topk - 1, axis=1)[:, :topk]
        partial_scores = np.take_along_axis(scores, partial, axis=1)
        order = np.argsort(-partial_scores, axis=1)
    else:
        partial = np.argpartition(scores, kth=topk - 1, axis=1)[:, :topk]
        partial_scores = np.take_along_axis(scores, partial, axis=1)
        order = np.argsort(partial_scores, axis=1)
    return np.take_along_axis(partial, order, axis=1).astype(np.int64, copy=False)


def exact_topk(
    queries: np.ndarray,
    base: np.ndarray,
    metric: str,
    topk: int,
    batch_size: int,
    exclude_self: bool = False,
) -> np.ndarray:
    if topk <= 0:
        raise ValueError("topk must be positive.")
    if base.shape[0] < topk + int(exclude_self):
        raise ValueError("Not enough base vectors to compute the requested top-k.")

    metric = canonical_metric(metric)
    queries_work = maybe_normalize(queries, metric)
    base_work = maybe_normalize(base, metric)

    if metric == "l2":
        base_sq = np.sum(base_work * base_work, axis=1, dtype=np.float32)

    results = []
    for start in range(0, queries_work.shape[0], batch_size):
        end = min(start + batch_size, queries_work.shape[0])
        query_batch = queries_work[start:end]
        scores = query_batch @ base_work.T

        if metric == "l2":
            query_sq = np.sum(query_batch * query_batch, axis=1, dtype=np.float32)
            distances = query_sq[:, None] + base_sq[None, :] - 2.0 * scores
            np.maximum(distances, 0.0, out=distances)
            if exclude_self:
                local_rows = np.arange(end - start)
                global_rows = np.arange(start, end)
                distances[local_rows, global_rows] = np.inf
            result = topk_indices_from_scores(distances, topk=topk, largest=False)
        else:
            if exclude_self:
                local_rows = np.arange(end - start)
                global_rows = np.arange(start, end)
                scores[local_rows, global_rows] = -np.inf
            result = topk_indices_from_scores(scores, topk=topk, largest=True)
        results.append(result)

    return np.vstack(results)


def write_npz_dataset(
    output_path: Path,
    *,
    dimension: int,
    distance_metric: str,
    train: np.ndarray,
    search: np.ndarray,
    test: np.ndarray,
    train_top10: np.ndarray,
    test_top100: np.ndarray,
    search_top100: np.ndarray,
) -> None:
    np.savez(
        output_path,
        dimension=np.array(dimension, dtype=np.int64),
        distance_metric=np.array(distance_metric),
        train=np.asarray(train, dtype=np.float32),
        search=np.asarray(search, dtype=np.float32),
        test=np.asarray(test, dtype=np.float32),
        train_top10=np.asarray(train_top10, dtype=np.int64),
        test_top100=np.asarray(test_top100, dtype=np.int64),
        search_top100=np.asarray(search_top100, dtype=np.int64),
    )


def main() -> None:
    args = parse_args()
    metric = canonical_metric(args.distance_metric)

    input_hdf5 = Path(args.input_hdf5).expanduser().resolve()
    output_path = Path(args.output_file).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    train, raw_test = load_hdf5_vectors(input_hdf5)

    if train.ndim != 2 or raw_test.ndim != 2:
        raise ValueError("Only 2D train/test vector datasets are supported.")
    if train.shape[1] != args.dimension or raw_test.shape[1] != args.dimension:
        raise ValueError(
            f"Dimension mismatch: expected {args.dimension}, "
            f"got train={train.shape[1]}, test={raw_test.shape[1]}."
        )

    search, test = split_test_vectors(raw_test, args.seed)

    print(f"[load] input={input_hdf5}")
    print(f"[shape] train={train.shape}, search={search.shape}, test={test.shape}")
    print(f"[metric] {args.distance_metric} -> {metric}")
    print(f"[gt] computing train top-10")
    train_top10 = exact_topk(
        train,
        train,
        metric=metric,
        topk=10,
        batch_size=args.batch_size,
        exclude_self=True,
    )
    print(f"[gt] computing test top-100")
    test_top100 = exact_topk(
        test,
        train,
        metric=metric,
        topk=100,
        batch_size=args.batch_size,
    )
    print(f"[gt] computing search top-100")
    search_top100 = exact_topk(
        search,
        train,
        metric=metric,
        topk=100,
        batch_size=args.batch_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_npz_dataset(
        output_path,
        dimension=args.dimension,
        distance_metric=args.distance_metric,
        train=train,
        search=search,
        test=test,
        train_top10=train_top10,
        test_top100=test_top100,
        search_top100=search_top100,
    )
    print(f"[save] wrote dataset file to {output_path}")


if __name__ == "__main__":
    main()
