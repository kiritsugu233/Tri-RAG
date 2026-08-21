"""Real-scale retrieval-only latency benchmark with auditable scan reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import scipy

from .indexes import StreamingExactSquaredL2Index, StreamingSearchResult
from .lid import LIDEstimate, estimate_lid_from_squared_distances
from .policies import TriPredictPolicy
from .projection import dense_gaussian_projection, projection_metadata
from .utils import array_fingerprint, fingerprint, write_json


class RetrievalBenchmarkConfigError(ValueError):
    pass


@dataclass(frozen=True)
class BenchmarkSeeds:
    corpus: int
    queries: int
    projection: int


@dataclass(frozen=True)
class BenchmarkDataset:
    corpus_size: int
    query_count: int
    warmup_query_count: int
    dimension: int
    dtype: str
    generation_batch_rows: int


@dataclass(frozen=True)
class BenchmarkProjection:
    m_prime: int


@dataclass(frozen=True)
class BenchmarkSearch:
    k_gt: int
    m_pilot: int
    s_lid: int
    min_lid_neighbors: int
    m_grid: List[int]
    fixed_budget: int
    corpus_block_rows: int


@dataclass(frozen=True)
class BenchmarkLID:
    clip_min: float
    clip_max: float
    duplicate_tolerance: float
    fallback: float


@dataclass(frozen=True)
class BenchmarkTriPredict:
    target: float
    max_rank_samples: int


@dataclass(frozen=True)
class RetrievalBenchmarkConfig:
    schema_version: int
    run_name: str
    seeds: BenchmarkSeeds
    dataset: BenchmarkDataset
    projection: BenchmarkProjection
    search: BenchmarkSearch
    lid: BenchmarkLID
    tri_predict: BenchmarkTriPredict
    raw: Dict[str, Any]
    config_fingerprint: str


@dataclass(frozen=True)
class GeneratedEmbeddings:
    corpus_path: Path
    corpus_norms_path: Path
    projected_path: Path
    projected_norms_path: Path
    queries_path: Path
    corpus_hash: str
    projected_hash: str
    query_hash: str
    corpus_generation_seconds: float
    projection_generation_seconds: float


METHOD_ORIGINAL_FIXED = "original_space_fixed_m"
METHOD_PROJECTED_FIXED = "projected_space_fixed_m"
METHOD_TRI_REUSE = "projected_tri_predict_reuse"
METHOD_TRI_DOUBLE = "projected_tri_predict_double_scan"
METHODS = (
    METHOD_ORIGINAL_FIXED,
    METHOD_PROJECTED_FIXED,
    METHOD_TRI_REUSE,
    METHOD_TRI_DOUBLE,
)
LATENCY_FIELDS = (
    "query_projection_ms",
    "original_search_ms",
    "pilot_search_ms",
    "pilot_original_distance_ms",
    "lid_estimation_ms",
    "lid_total_ms",
    "tri_predict_ms",
    "expansion_ms",
    "original_rerank_ms",
    "total_ms",
)


def _section(raw: Mapping[str, Any], key: str, expected: set[str]) -> Dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise RetrievalBenchmarkConfigError(f"{key} must be an object")
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown or missing:
        raise RetrievalBenchmarkConfigError(
            f"invalid {key} keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return dict(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RetrievalBenchmarkConfigError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RetrievalBenchmarkConfigError(f"{name} must be a nonnegative integer")
    return value


def load_retrieval_benchmark_config(
    path: Union[str, Path]
) -> RetrievalBenchmarkConfig:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalBenchmarkConfigError(
            f"cannot load benchmark config {config_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise RetrievalBenchmarkConfigError("config root must be an object")
    expected_root = {
        "schema_version",
        "run_name",
        "seeds",
        "dataset",
        "projection",
        "search",
        "lid",
        "tri_predict",
    }
    if set(raw) != expected_root:
        raise RetrievalBenchmarkConfigError(
            f"invalid root keys; missing={sorted(expected_root-set(raw))}, "
            f"unknown={sorted(set(raw)-expected_root)}"
        )
    seeds = _section(raw, "seeds", {"corpus", "queries", "projection"})
    dataset = _section(
        raw,
        "dataset",
        {
            "corpus_size",
            "query_count",
            "warmup_query_count",
            "dimension",
            "dtype",
            "generation_batch_rows",
        },
    )
    projection = _section(raw, "projection", {"m_prime"})
    search = _section(
        raw,
        "search",
        {
            "k_gt",
            "m_pilot",
            "s_lid",
            "min_lid_neighbors",
            "m_grid",
            "fixed_budget",
            "corpus_block_rows",
        },
    )
    lid = _section(
        raw, "lid", {"clip_min", "clip_max", "duplicate_tolerance", "fallback"}
    )
    tri_predict = _section(raw, "tri_predict", {"target", "max_rank_samples"})
    if raw["schema_version"] != 1:
        raise RetrievalBenchmarkConfigError("schema_version must be 1")
    if not isinstance(raw["run_name"], str) or not raw["run_name"]:
        raise RetrievalBenchmarkConfigError("run_name must be a nonempty string")
    for key, value in seeds.items():
        _nonnegative_int(value, f"seeds.{key}")
    for key in (
        "corpus_size",
        "query_count",
        "dimension",
        "generation_batch_rows",
    ):
        _positive_int(dataset[key], f"dataset.{key}")
    _nonnegative_int(dataset["warmup_query_count"], "dataset.warmup_query_count")
    if dataset["dtype"] != "float32":
        raise RetrievalBenchmarkConfigError("dataset.dtype must be float32")
    _positive_int(projection["m_prime"], "projection.m_prime")
    for key in (
        "k_gt",
        "m_pilot",
        "s_lid",
        "min_lid_neighbors",
        "fixed_budget",
        "corpus_block_rows",
    ):
        _positive_int(search[key], f"search.{key}")
    grid = search["m_grid"]
    if not isinstance(grid, list) or not grid:
        raise RetrievalBenchmarkConfigError("search.m_grid must be a nonempty list")
    for value in grid:
        _positive_int(value, "search.m_grid item")
    if grid != sorted(set(grid)):
        raise RetrievalBenchmarkConfigError("search.m_grid must be strictly increasing")
    minimum_budget = max(search["k_gt"], search["m_pilot"])
    if grid[0] < minimum_budget:
        raise RetrievalBenchmarkConfigError(
            f"all search budgets must be at least {minimum_budget}"
        )
    if grid[-1] > dataset["corpus_size"]:
        raise RetrievalBenchmarkConfigError("maximum budget exceeds corpus size")
    if search["fixed_budget"] not in grid:
        raise RetrievalBenchmarkConfigError("search.fixed_budget must be in m_grid")
    if search["s_lid"] > search["m_pilot"]:
        raise RetrievalBenchmarkConfigError("s_lid cannot exceed m_pilot")
    if search["min_lid_neighbors"] > search["s_lid"]:
        raise RetrievalBenchmarkConfigError(
            "min_lid_neighbors cannot exceed s_lid"
        )
    for key, value in lid.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise RetrievalBenchmarkConfigError(f"lid.{key} must be positive")
    if lid["clip_min"] >= lid["clip_max"]:
        raise RetrievalBenchmarkConfigError("lid.clip_min must be below clip_max")
    if not 0 < tri_predict["target"] <= 1:
        raise RetrievalBenchmarkConfigError("tri_predict.target must lie in (0,1]")
    _positive_int(tri_predict["max_rank_samples"], "tri_predict.max_rank_samples")
    return RetrievalBenchmarkConfig(
        schema_version=1,
        run_name=raw["run_name"],
        seeds=BenchmarkSeeds(**seeds),
        dataset=BenchmarkDataset(**dataset),
        projection=BenchmarkProjection(**projection),
        search=BenchmarkSearch(**search),
        lid=BenchmarkLID(**lid),
        tri_predict=BenchmarkTriPredict(**tri_predict),
        raw=raw,
        config_fingerprint=fingerprint(raw),
    )


def _current_rss_bytes() -> int:
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1024)


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _normalized_random_rows(
    rng: np.random.Generator, rows: int, dimension: int
) -> np.ndarray:
    values = rng.standard_normal(size=(rows, dimension)).astype(np.float32)
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0.0):
        raise FloatingPointError("random generator produced a zero vector")
    values /= norms[:, None]
    return values


def _generate_embeddings(
    config: RetrievalBenchmarkConfig, data_dir: Path
) -> GeneratedEmbeddings:
    data_dir.mkdir(parents=True, exist_ok=False)
    dataset = config.dataset
    corpus_path = data_dir / "corpus.float32.mmap"
    corpus_norms_path = data_dir / "corpus_squared_norms.npy"
    projected_path = data_dir / "projected_corpus.float32.mmap"
    projected_norms_path = data_dir / "projected_squared_norms.npy"
    queries_path = data_dir / "queries.npy"
    corpus = np.memmap(
        corpus_path,
        mode="w+",
        dtype=np.float32,
        shape=(dataset.corpus_size, dataset.dimension),
    )
    corpus_norms = np.empty(dataset.corpus_size, dtype=np.float32)
    corpus_digest = hashlib.sha256()
    rng = np.random.default_rng(config.seeds.corpus)
    started = perf_counter()
    for start in range(0, dataset.corpus_size, dataset.generation_batch_rows):
        stop = min(start + dataset.generation_batch_rows, dataset.corpus_size)
        batch = _normalized_random_rows(rng, stop - start, dataset.dimension)
        corpus[start:stop] = batch
        corpus_norms[start:stop] = np.einsum("ij,ij->i", batch, batch)
        corpus_digest.update(batch.tobytes(order="C"))
    corpus.flush()
    np.save(corpus_norms_path, corpus_norms, allow_pickle=False)
    corpus_seconds = perf_counter() - started

    matrix = dense_gaussian_projection(
        config.projection.m_prime, dataset.dimension, config.seeds.projection
    ).astype(np.float32)
    projected = np.memmap(
        projected_path,
        mode="w+",
        dtype=np.float32,
        shape=(dataset.corpus_size, config.projection.m_prime),
    )
    projected_norms = np.empty(dataset.corpus_size, dtype=np.float32)
    projected_digest = hashlib.sha256()
    started = perf_counter()
    for start in range(0, dataset.corpus_size, dataset.generation_batch_rows):
        stop = min(start + dataset.generation_batch_rows, dataset.corpus_size)
        batch = np.asarray(corpus[start:stop]) @ matrix.T
        batch = np.asarray(batch, dtype=np.float32)
        projected[start:stop] = batch
        projected_norms[start:stop] = np.einsum("ij,ij->i", batch, batch)
        projected_digest.update(batch.tobytes(order="C"))
    projected.flush()
    np.save(projected_norms_path, projected_norms, allow_pickle=False)
    projection_seconds = perf_counter() - started

    query_rng = np.random.default_rng(config.seeds.queries)
    total_queries = dataset.warmup_query_count + dataset.query_count
    queries = _normalized_random_rows(query_rng, total_queries, dataset.dimension)
    np.save(queries_path, queries, allow_pickle=False)
    return GeneratedEmbeddings(
        corpus_path=corpus_path,
        corpus_norms_path=corpus_norms_path,
        projected_path=projected_path,
        projected_norms_path=projected_norms_path,
        queries_path=queries_path,
        corpus_hash=corpus_digest.hexdigest(),
        projected_hash=projected_digest.hexdigest(),
        query_hash=array_fingerprint(queries),
        corpus_generation_seconds=corpus_seconds,
        projection_generation_seconds=projection_seconds,
    )


def _load_generated(
    config: RetrievalBenchmarkConfig, generated: GeneratedEmbeddings
) -> Tuple[np.memmap, np.ndarray, np.memmap, np.ndarray, np.ndarray]:
    corpus = np.memmap(
        generated.corpus_path,
        mode="r",
        dtype=np.float32,
        shape=(config.dataset.corpus_size, config.dataset.dimension),
    )
    projected = np.memmap(
        generated.projected_path,
        mode="r",
        dtype=np.float32,
        shape=(config.dataset.corpus_size, config.projection.m_prime),
    )
    corpus_norms = np.load(generated.corpus_norms_path, allow_pickle=False)
    projected_norms = np.load(generated.projected_norms_path, allow_pickle=False)
    queries = np.load(generated.queries_path, allow_pickle=False)
    return corpus, corpus_norms, projected, projected_norms, queries


def _candidate_squared_distances(
    corpus: np.ndarray,
    corpus_norms: np.ndarray,
    query: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    candidates = np.asarray(corpus[rows])
    distances = (
        float(np.dot(query, query))
        + corpus_norms[rows]
        - 2.0 * (candidates @ query)
    )
    distances = np.asarray(distances)
    np.maximum(distances, 0.0, out=distances)
    return np.asarray(distances, dtype=np.float64)


def _retention(gt_rows: np.ndarray, candidate_rows: np.ndarray) -> float:
    return len(set(gt_rows.tolist()).intersection(candidate_rows.tolist())) / len(
        gt_rows
    )


def _empty_latency() -> Dict[str, float]:
    return {field: 0.0 for field in LATENCY_FIELDS}


def _base_record(
    method: str, query_index: int, config: RetrievalBenchmarkConfig
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "query_index": query_index,
        "query_id": f"benchmark-query-{query_index:06d}",
        "method": method,
        "chosen_m": None,
        "embedding_retention": None,
        "lid_raw": None,
        "lid_clipped": None,
        "lid_valid": None,
        "lid_failure_reason": None,
        "policy_saturated": False,
        "projected_scan_count": 0,
        "original_scan_count": 0,
        "projected_distance_evaluations": 0,
        "original_distance_evaluations": 0,
        "query_projection_coordinate_ops": 0,
        "projected_distance_coordinate_ops": 0,
        "original_distance_coordinate_ops": 0,
        "projected_vector_bytes_scanned": 0,
        "original_vector_bytes_scanned": 0,
        "rss_before_bytes": _current_rss_bytes(),
        "rss_after_bytes": None,
        "peak_rss_bytes": None,
        "corpus_size": config.dataset.corpus_size,
        "dimension": config.dataset.dimension,
        "m_prime": config.projection.m_prime,
    }
    record.update(_empty_latency())
    return record


def _finish_record(
    record: Dict[str, Any], started: float, config: RetrievalBenchmarkConfig
) -> Dict[str, Any]:
    record["total_ms"] = (perf_counter() - started) * 1000.0
    record["rss_after_bytes"] = _current_rss_bytes()
    record["peak_rss_bytes"] = _peak_rss_bytes()
    record["query_projection_coordinate_ops"] = int(
        record["query_projection_coordinate_ops"]
    )
    record["projected_distance_coordinate_ops"] = int(
        record["projected_distance_evaluations"] * config.projection.m_prime
    )
    record["original_distance_coordinate_ops"] = int(
        record["original_distance_evaluations"] * config.dataset.dimension
    )
    return record


def _run_original_fixed(
    *,
    query_index: int,
    query: np.ndarray,
    original_index: StreamingExactSquaredL2Index,
    config: RetrievalBenchmarkConfig,
) -> Tuple[Dict[str, Any], np.ndarray]:
    method = METHOD_ORIGINAL_FIXED
    record = _base_record(method, query_index, config)
    started = perf_counter()
    result = original_index.search_one(query, config.search.fixed_budget)
    record["original_search_ms"] = result.search_ms
    record["chosen_m"] = config.search.fixed_budget
    record["embedding_retention"] = 1.0
    record["original_scan_count"] = 1
    record["original_distance_evaluations"] = result.distance_evaluations
    record["original_vector_bytes_scanned"] = result.scanned_vector_bytes
    return (
        _finish_record(record, started, config),
        result.rows[: config.search.k_gt].copy(),
    )


def _project_query(query: np.ndarray, matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    started = perf_counter()
    projected = np.asarray(query @ matrix.T, dtype=np.float32)
    return projected, (perf_counter() - started) * 1000.0


def _rerank_candidates(
    *,
    corpus: np.ndarray,
    corpus_norms: np.ndarray,
    query: np.ndarray,
    rows: np.ndarray,
    known_prefix_distances: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, int]:
    started = perf_counter()
    prefix_count = 0 if known_prefix_distances is None else len(known_prefix_distances)
    additional_rows = rows[prefix_count:]
    additional_distances = _candidate_squared_distances(
        corpus, corpus_norms, query, additional_rows
    )
    if known_prefix_distances is None:
        distances = additional_distances
    else:
        distances = np.concatenate([known_prefix_distances, additional_distances])
    order = np.lexsort((rows, distances))
    return rows[order], (perf_counter() - started) * 1000.0, len(additional_rows)


def _run_projected_fixed(
    *,
    query_index: int,
    query: np.ndarray,
    gt_rows: np.ndarray,
    matrix: np.ndarray,
    projected_index: StreamingExactSquaredL2Index,
    corpus: np.ndarray,
    corpus_norms: np.ndarray,
    config: RetrievalBenchmarkConfig,
) -> Dict[str, Any]:
    record = _base_record(METHOD_PROJECTED_FIXED, query_index, config)
    started = perf_counter()
    projected_query, projection_ms = _project_query(query, matrix)
    record["query_projection_ms"] = projection_ms
    record["query_projection_coordinate_ops"] = (
        config.dataset.dimension * config.projection.m_prime
    )
    search = projected_index.search_one(projected_query, config.search.fixed_budget)
    record["expansion_ms"] = search.search_ms
    reranked, rerank_ms, distance_count = _rerank_candidates(
        corpus=corpus,
        corpus_norms=corpus_norms,
        query=query,
        rows=search.rows,
    )
    record["original_rerank_ms"] = rerank_ms
    record["chosen_m"] = config.search.fixed_budget
    record["embedding_retention"] = _retention(gt_rows, reranked[: config.search.k_gt])
    record["projected_scan_count"] = 1
    record["projected_distance_evaluations"] = search.distance_evaluations
    record["original_distance_evaluations"] = distance_count
    record["projected_vector_bytes_scanned"] = search.scanned_vector_bytes
    record["original_vector_bytes_scanned"] = (
        distance_count * config.dataset.dimension * np.dtype(np.float32).itemsize
    )
    return _finish_record(record, started, config)


def _estimate_pilot_lid(
    *,
    corpus: np.ndarray,
    corpus_norms: np.ndarray,
    query: np.ndarray,
    pilot_rows: np.ndarray,
    config: RetrievalBenchmarkConfig,
) -> Tuple[LIDEstimate, np.ndarray, float, float]:
    started = perf_counter()
    pilot_distances = _candidate_squared_distances(
        corpus, corpus_norms, query, pilot_rows
    )
    original_distance_ms = (perf_counter() - started) * 1000.0
    started = perf_counter()
    estimate = estimate_lid_from_squared_distances(
        pilot_distances,
        s_lid=config.search.s_lid,
        min_neighbors=config.search.min_lid_neighbors,
        clip_min=config.lid.clip_min,
        clip_max=config.lid.clip_max,
        duplicate_tolerance=config.lid.duplicate_tolerance,
        fallback=config.lid.fallback,
    )
    estimation_ms = (perf_counter() - started) * 1000.0
    return estimate, pilot_distances, original_distance_ms, estimation_ms


def _run_tri_predict(
    *,
    method: str,
    reuse_projected_distances: bool,
    query_index: int,
    query: np.ndarray,
    gt_rows: np.ndarray,
    matrix: np.ndarray,
    projected_index: StreamingExactSquaredL2Index,
    corpus: np.ndarray,
    corpus_norms: np.ndarray,
    policy: TriPredictPolicy,
    config: RetrievalBenchmarkConfig,
) -> Dict[str, Any]:
    record = _base_record(method, query_index, config)
    started = perf_counter()
    projected_query, projection_ms = _project_query(query, matrix)
    record["query_projection_ms"] = projection_ms
    record["query_projection_coordinate_ops"] = (
        config.dataset.dimension * config.projection.m_prime
    )
    pilot_k = config.search.m_grid[-1] if reuse_projected_distances else config.search.m_pilot
    pilot = projected_index.search_one(projected_query, pilot_k)
    record["pilot_search_ms"] = pilot.search_ms
    pilot_rows = pilot.rows[: config.search.m_pilot]
    estimate, pilot_original_distances, distance_ms, estimation_ms = _estimate_pilot_lid(
        corpus=corpus,
        corpus_norms=corpus_norms,
        query=query,
        pilot_rows=pilot_rows,
        config=config,
    )
    record["pilot_original_distance_ms"] = distance_ms
    record["lid_estimation_ms"] = estimation_ms
    record["lid_total_ms"] = distance_ms + estimation_ms
    record["lid_raw"] = estimate.raw
    record["lid_clipped"] = estimate.clipped
    record["lid_valid"] = estimate.valid
    record["lid_failure_reason"] = estimate.reason
    policy_started = perf_counter()
    decision = policy.choose(estimate.clipped, estimate.valid)
    record["tri_predict_ms"] = (perf_counter() - policy_started) * 1000.0
    expansion_started = perf_counter()
    if reuse_projected_distances:
        candidate_rows = pilot.rows[: decision.budget].copy()
        expansion = None
    else:
        expansion = projected_index.search_one(projected_query, decision.budget)
        candidate_rows = expansion.rows
        if not np.array_equal(candidate_rows[: config.search.m_pilot], pilot_rows):
            raise AssertionError("double-scan expansion changed the exact pilot prefix")
    record["expansion_ms"] = (perf_counter() - expansion_started) * 1000.0
    if expansion is not None:
        record["expansion_ms"] = expansion.search_ms
    reranked, rerank_ms, additional_count = _rerank_candidates(
        corpus=corpus,
        corpus_norms=corpus_norms,
        query=query,
        rows=candidate_rows,
        known_prefix_distances=pilot_original_distances,
    )
    record["original_rerank_ms"] = rerank_ms
    record["chosen_m"] = decision.budget
    record["embedding_retention"] = _retention(
        gt_rows, reranked[: config.search.k_gt]
    )
    record["policy_saturated"] = decision.saturated
    record["projected_scan_count"] = 1 if reuse_projected_distances else 2
    record["projected_distance_evaluations"] = pilot.distance_evaluations + (
        0 if expansion is None else expansion.distance_evaluations
    )
    record["original_distance_evaluations"] = len(pilot_rows) + additional_count
    record["projected_vector_bytes_scanned"] = pilot.scanned_vector_bytes + (
        0 if expansion is None else expansion.scanned_vector_bytes
    )
    record["original_vector_bytes_scanned"] = (
        record["original_distance_evaluations"]
        * config.dataset.dimension
        * np.dtype(np.float32).itemsize
    )
    return _finish_record(record, started, config)


def _percentiles(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _summarize_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}
    for method in METHODS:
        rows = [record for record in records if record["method"] == method]
        if not rows:
            continue
        chosen = [int(record["chosen_m"]) for record in rows]
        summaries[method] = {
            "n": len(rows),
            "latency_ms": {
                field: _percentiles([float(record[field]) for record in rows])
                for field in LATENCY_FIELDS
            },
            "quality": {
                "mean_embedding_retention": float(
                    np.mean([float(record["embedding_retention"]) for record in rows])
                ),
                "minimum_embedding_retention": float(
                    np.min([float(record["embedding_retention"]) for record in rows])
                ),
            },
            "budget": {
                "mean": float(np.mean(chosen)),
                "p95": float(np.quantile(chosen, 0.95)),
                "distribution": {
                    str(value): chosen.count(value) for value in sorted(set(chosen))
                },
                "saturated_n": sum(bool(record["policy_saturated"]) for record in rows),
            },
            "work_per_query": {
                key: float(np.mean([float(record[key]) for record in rows]))
                for key in (
                    "projected_scan_count",
                    "original_scan_count",
                    "projected_distance_evaluations",
                    "original_distance_evaluations",
                    "query_projection_coordinate_ops",
                    "projected_distance_coordinate_ops",
                    "original_distance_coordinate_ops",
                    "projected_vector_bytes_scanned",
                    "original_vector_bytes_scanned",
                )
            },
            "memory": {
                "max_rss_after_bytes": max(int(record["rss_after_bytes"]) for record in rows),
                "max_peak_rss_bytes": max(int(record["peak_rss_bytes"]) for record in rows),
            },
        }
    reuse = summaries.get(METHOD_TRI_REUSE)
    double = summaries.get(METHOD_TRI_DOUBLE)
    if reuse is not None and double is not None:
        summaries["reuse_comparison"] = {
            "projected_distance_evaluation_reduction_fraction": 1.0
            - reuse["work_per_query"]["projected_distance_evaluations"]
            / double["work_per_query"]["projected_distance_evaluations"],
            "projected_bytes_scanned_reduction_fraction": 1.0
            - reuse["work_per_query"]["projected_vector_bytes_scanned"]
            / double["work_per_query"]["projected_vector_bytes_scanned"],
            "mean_total_latency_reduction_fraction": 1.0
            - reuse["latency_ms"]["total_ms"]["mean"]
            / double["latency_ms"]["total_ms"]["mean"],
            "p95_total_latency_reduction_fraction": 1.0
            - reuse["latency_ms"]["total_ms"]["p95"]
            / double["latency_ms"]["total_ms"]["p95"],
        }
    return summaries


def _report(
    manifest: Mapping[str, Any], summary: Mapping[str, Any]
) -> str:
    config = manifest["config"]
    memory = manifest["memory_artifacts"]
    gib = float(1024**3)
    lines = [
        "# Retrieval-only latency benchmark",
        "",
        "## Setup",
        "",
        f"- corpus: {config['dataset']['corpus_size']:,}",
        f"- measured/warmup queries: {config['dataset']['query_count']} / {config['dataset']['warmup_query_count']}",
        f"- original/projected dimensions: {config['dataset']['dimension']} / {config['projection']['m_prime']}",
        f"- fixed M: {config['search']['fixed_budget']}",
        f"- M grid: {config['search']['m_grid']}",
        f"- corpus block rows: {config['search']['corpus_block_rows']}",
        f"- dtype: {config['dataset']['dtype']}",
        "- projected vectors renormalized: false",
        "",
        "## End-to-end latency",
        "",
        "| method | mean ms | p50 | p95 | p99 | mean M | retention | projected scans |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        value = summary[method]
        total = value["latency_ms"]["total_ms"]
        lines.append(
            f"| {method} | {total['mean']:.4f} | {total['p50']:.4f} | "
            f"{total['p95']:.4f} | {total['p99']:.4f} | "
            f"{value['budget']['mean']:.2f} | "
            f"{value['quality']['mean_embedding_retention']:.4f} | "
            f"{value['work_per_query']['projected_scan_count']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Mean stage latency (ms/query)",
            "",
            "| method | query projection | original search | pilot search | LID total | Tri-Predict | expansion | rerank |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in METHODS:
        latency = summary[method]["latency_ms"]
        lines.append(
            f"| {method} | {latency['query_projection_ms']['mean']:.4f} | "
            f"{latency['original_search_ms']['mean']:.4f} | "
            f"{latency['pilot_search_ms']['mean']:.4f} | "
            f"{latency['lid_total_ms']['mean']:.4f} | "
            f"{latency['tri_predict_ms']['mean']:.4f} | "
            f"{latency['expansion_ms']['mean']:.4f} | "
            f"{latency['original_rerank_ms']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Mean work per query",
            "",
            "| method | projected distances | original distances | projected bytes scanned | original bytes read |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in METHODS:
        work = summary[method]["work_per_query"]
        lines.append(
            f"| {method} | {work['projected_distance_evaluations']:.0f} | "
            f"{work['original_distance_evaluations']:.2f} | "
            f"{work['projected_vector_bytes_scanned']:.0f} | "
            f"{work['original_vector_bytes_scanned']:.0f} |"
        )
    comparison = summary["reuse_comparison"]
    lines.extend(
        [
            "",
            "## Pilot/expansion reuse",
            "",
            f"- projected distance reduction: {comparison['projected_distance_evaluation_reduction_fraction']:.2%}",
            f"- projected bytes reduction: {comparison['projected_bytes_scanned_reduction_fraction']:.2%}",
            f"- mean total latency reduction: {comparison['mean_total_latency_reduction_fraction']:.2%}",
            f"- p95 total latency reduction: {comparison['p95_total_latency_reduction_fraction']:.2%}",
            "",
            "The reuse path performs one exact projected scan and retains top-M_max. "
            "The legacy path performs one pilot scan and one expansion scan. Original "
            "pilot distances are reused by reranking in both paths.",
            "",
            "## Memory",
            "",
            f"- original corpus memmap: {memory['original_corpus_bytes'] / gib:.3f} GiB",
            f"- projected corpus memmap: {memory['projected_corpus_bytes'] / gib:.3f} GiB",
            f"- norm arrays: {(memory['corpus_norms_bytes'] + memory['projected_norms_bytes']) / gib:.3f} GiB",
            f"- query array: {memory['queries_bytes'] / gib:.3f} GiB",
            f"- projection matrix: {memory['projection_matrix_bytes'] / gib:.6f} GiB",
            f"- reusable top-M cache per query: {memory['reuse_top_m_cache_bytes_per_query']} bytes",
            f"- measured process peak RSS: {memory['process_peak_rss_bytes'] / gib:.3f} GiB",
            "",
            "This benchmark uses deterministic normalized Gaussian embeddings with "
            "realistic dimensions. It is a systems benchmark, not a semantic retrieval "
            "or answer-quality claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_retrieval_benchmark(
    config: RetrievalBenchmarkConfig, output_dir: Path
) -> Dict[str, Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty benchmark directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    generated = _generate_embeddings(config, output_dir / "data")
    corpus, corpus_norms, projected, projected_norms, queries = _load_generated(
        config, generated
    )
    matrix = dense_gaussian_projection(
        config.projection.m_prime,
        config.dataset.dimension,
        config.seeds.projection,
    ).astype(np.float32)
    original_index = StreamingExactSquaredL2Index(
        corpus,
        squared_norms=corpus_norms,
        block_rows=config.search.corpus_block_rows,
    )
    projected_index = StreamingExactSquaredL2Index(
        projected,
        squared_norms=projected_norms,
        block_rows=config.search.corpus_block_rows,
    )
    policy = TriPredictPolicy(
        corpus_size=config.dataset.corpus_size,
        m_prime=config.projection.m_prime,
        k_gt=config.search.k_gt,
        grid=config.search.m_grid,
        target=config.tri_predict.target,
        max_rank_samples=config.tri_predict.max_rank_samples,
    )
    policy_artifact = policy.serialize()
    records: List[Dict[str, Any]] = []
    warmup_count = config.dataset.warmup_query_count
    total_count = warmup_count + config.dataset.query_count
    for global_index in range(total_count):
        query = np.asarray(queries[global_index], dtype=np.float32)
        original_record, gt_rows = _run_original_fixed(
            query_index=global_index - warmup_count,
            query=query,
            original_index=original_index,
            config=config,
        )
        projected_methods = [
            METHOD_PROJECTED_FIXED,
            METHOD_TRI_REUSE,
            METHOD_TRI_DOUBLE,
        ]
        rotation = global_index % len(projected_methods)
        projected_methods = projected_methods[rotation:] + projected_methods[:rotation]
        query_records: Dict[str, Dict[str, Any]] = {}
        for method in projected_methods:
            if method == METHOD_PROJECTED_FIXED:
                value = _run_projected_fixed(
                    query_index=global_index - warmup_count,
                    query=query,
                    gt_rows=gt_rows,
                    matrix=matrix,
                    projected_index=projected_index,
                    corpus=corpus,
                    corpus_norms=corpus_norms,
                    config=config,
                )
            else:
                value = _run_tri_predict(
                    method=method,
                    reuse_projected_distances=method == METHOD_TRI_REUSE,
                    query_index=global_index - warmup_count,
                    query=query,
                    gt_rows=gt_rows,
                    matrix=matrix,
                    projected_index=projected_index,
                    corpus=corpus,
                    corpus_norms=corpus_norms,
                    policy=policy,
                    config=config,
                )
            query_records[method] = value
        if global_index >= warmup_count:
            records.append(original_record)
            records.extend(query_records[method] for method in METHODS[1:])
    summary = _summarize_records(records)
    dtype_bytes = np.dtype(np.float32).itemsize
    memory_artifacts = {
        "original_corpus_bytes": int(config.dataset.corpus_size * config.dataset.dimension * dtype_bytes),
        "projected_corpus_bytes": int(config.dataset.corpus_size * config.projection.m_prime * dtype_bytes),
        "corpus_norms_bytes": int(corpus_norms.nbytes),
        "projected_norms_bytes": int(projected_norms.nbytes),
        "queries_bytes": int(queries.nbytes),
        "projection_matrix_bytes": int(matrix.nbytes),
        "reuse_top_m_cache_bytes_per_query": int(
            config.search.m_grid[-1]
            * (np.dtype(np.int64).itemsize + np.dtype(np.float64).itemsize)
        ),
        "process_peak_rss_bytes": _peak_rss_bytes(),
    }
    manifest = {
        "schema_version": 1,
        "run_name": config.run_name,
        "created_at_utc": created_at,
        "config_fingerprint": config.config_fingerprint,
        "config": config.raw,
        "data": {
            "kind": "normalized_gaussian_systems_benchmark",
            "semantic_retrieval_claim": False,
            "corpus_hash": generated.corpus_hash,
            "projected_corpus_hash": generated.projected_hash,
            "query_hash": generated.query_hash,
            "corpus_generation_seconds": generated.corpus_generation_seconds,
            "projection_generation_seconds": generated.projection_generation_seconds,
        },
        "projection": projection_metadata(
            dimension=config.dataset.dimension,
            m_prime=config.projection.m_prime,
            seed=config.seeds.projection,
            normalization=True,
            embedding_model="normalized_gaussian_latency_fixture@v1",
            corpus_hash=generated.corpus_hash,
        ),
        "search": {
            "backend": "numpy_streaming_exact_squared_l2",
            "corpus_block_rows": config.search.corpus_block_rows,
            "pilot_expansion_reuse": "top_m_max_from_single_projected_scan",
            "legacy_control": "independent_pilot_and_expansion_projected_scans",
        },
        "memory_artifacts": memory_artifacts,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "thread_environment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
        "policy": policy_artifact,
    }
    manifest["reproducibility_fingerprint"] = fingerprint(
        {
            "config_fingerprint": config.config_fingerprint,
            "corpus_hash": generated.corpus_hash,
            "projected_corpus_hash": generated.projected_hash,
            "query_hash": generated.query_hash,
            "projection_fingerprint": manifest["projection"]["fingerprint"],
            "search": manifest["search"],
            "policy_fingerprint": policy_artifact["fingerprint"],
        }
    )
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "memory.json", memory_artifacts)
    write_json(output_dir / "tri_predict_policy.json", policy_artifact)
    with (output_dir / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    report_path = output_dir / "report.md"
    report_path.write_text(_report(manifest, summary), encoding="utf-8")
    return {
        "manifest.json": output_dir / "manifest.json",
        "summary.json": output_dir / "summary.json",
        "memory.json": output_dir / "memory.json",
        "tri_predict_policy.json": output_dir / "tri_predict_policy.json",
        "per_query.jsonl": output_dir / "per_query.jsonl",
        "report.md": report_path,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    config = load_retrieval_benchmark_config(args.config)
    artifacts = run_retrieval_benchmark(config, args.output)
    print(f"completed retrieval benchmark: {artifacts['report.md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
