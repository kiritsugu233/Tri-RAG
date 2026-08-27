"""Select one fixed projection dimension using SciFact tune queries only."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from .certification import empirical_bernstein
from .embeddings import load_embedding_array
from .projection import (
    dense_gaussian_projection,
    project_rows,
    projection_metadata,
)
from .text_embeddings import (
    load_text_embedding_config,
    validate_text_embedding_cache,
)
from .utils import fingerprint, stable_id_hash, write_json


class RealDimensionSweepError(ValueError):
    pass


@dataclass(frozen=True)
class DimensionProjectionConfig:
    family: str
    seed: int
    candidates: list[int]
    candidate_coupling: str
    post_projection_normalize: bool


@dataclass(frozen=True)
class DimensionSearchConfig:
    normalized_inputs: bool
    distance: str
    arithmetic: str
    stable_tie_break: str
    query_batch_size: int
    k_ctx: int
    k_gt: int
    m_pilot: int
    m_grid: list[int]


@dataclass(frozen=True)
class DimensionSelectionConfig:
    metric: str
    alpha: float
    target: float
    statistic_role: str
    objective: str
    cost_formula: str
    include_query_projection: bool
    tie_break: list[str]


@dataclass(frozen=True)
class RealDimensionSweepConfig:
    schema_version: int
    benchmark: str
    dataset_manifest_fingerprint: str
    embedding_config_fingerprint: str
    embedding_request_fingerprint: str
    embedding_manifest_fingerprint: str
    original_baseline_result_fingerprint: str
    evaluation_split: str
    projection: DimensionProjectionConfig
    search: DimensionSearchConfig
    selection: DimensionSelectionConfig
    raw: Dict[str, Any]
    config_fingerprint: str


_SHA256_LENGTH = 64
_SELECTION_RULE = {
    "eligibility": (
        "for each m_prime, choose the smallest value in the common M_grid whose "
        "query_tune empirical-Bernstein retention statistic reaches target"
    ),
    "cross_dimension": (
        "among eligible (m_prime, fixed_M) pairs, minimize absolute coordinate "
        "multiply-add work (N+d)*m_prime+d*fixed_M"
    ),
    "tie_break": [
        "higher tune empirical-Bernstein lower bound",
        "smaller m_prime",
        "smaller fixed_M",
    ],
    "protected_data": (
        "query_cert and query_test are not loaded or evaluated; the tune statistic "
        "is a selection score, not a certificate"
    ),
    "labels": "evidence qrels and evidence metrics do not enter selection",
}


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RealDimensionSweepError(f"{name} must be a nonempty string")
    return value.strip()


def _fingerprint_string(value: Any, name: str) -> str:
    result = _nonempty_string(value, name).lower()
    if len(result) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise RealDimensionSweepError(f"{name} must be a SHA-256 fingerprint")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RealDimensionSweepError(f"{name} must be a positive integer")
    return value


def _strict_integer_grid(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise RealDimensionSweepError(f"{name} must be a nonempty list")
    result = [_positive_integer(item, f"{name} item") for item in value]
    if result != sorted(set(result)):
        raise RealDimensionSweepError(f"{name} must be strictly increasing")
    return result


def _exact_keys(
    value: Any, expected: set[str], name: str
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise RealDimensionSweepError(f"{name} must be an object")
    if set(value) != expected:
        raise RealDimensionSweepError(
            f"invalid {name} keys; missing={sorted(expected-set(value))}, "
            f"unknown={sorted(set(value)-expected)}"
        )
    return dict(value)


def load_real_dimension_sweep_config(
    path: Union[str, Path],
) -> RealDimensionSweepConfig:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealDimensionSweepError(
            f"cannot load dimension-sweep config {config_path}: {exc}"
        ) from exc
    root = _exact_keys(
        raw,
        {
            "schema_version",
            "benchmark",
            "dataset_manifest_fingerprint",
            "embedding_config_fingerprint",
            "embedding_request_fingerprint",
            "embedding_manifest_fingerprint",
            "original_baseline_result_fingerprint",
            "evaluation_split",
            "projection",
            "search",
            "selection",
        },
        "root",
    )
    if root["schema_version"] != 1:
        raise RealDimensionSweepError("schema_version must be 1")
    if root["benchmark"] != "real_fixed_dimension_tune_sweep_v1":
        raise RealDimensionSweepError(
            "benchmark must be real_fixed_dimension_tune_sweep_v1"
        )
    if root["evaluation_split"] != "query_tune":
        raise RealDimensionSweepError("dimension selection accepts query_tune only")

    projection = _exact_keys(
        root["projection"],
        {
            "family",
            "seed",
            "candidates",
            "candidate_coupling",
            "post_projection_normalize",
        },
        "projection",
    )
    if projection["family"] != "dense_gaussian_n0_variance_1_over_m_prime":
        raise RealDimensionSweepError("projection family is not the frozen Gaussian law")
    if projection["candidate_coupling"] != "same_seed_nested_prefix_rescaled":
        raise RealDimensionSweepError(
            "projection.candidate_coupling must be same_seed_nested_prefix_rescaled"
        )
    if projection["post_projection_normalize"] is not False:
        raise RealDimensionSweepError(
            "projected vectors must not be renormalized"
        )

    search = _exact_keys(
        root["search"],
        {
            "normalized_inputs",
            "distance",
            "arithmetic",
            "stable_tie_break",
            "query_batch_size",
            "k_ctx",
            "k_gt",
            "m_pilot",
            "m_grid",
        },
        "search",
    )
    if search["normalized_inputs"] is not True:
        raise RealDimensionSweepError("search.normalized_inputs must be true")
    if search["distance"] != "squared_l2":
        raise RealDimensionSweepError("search.distance must be squared_l2")
    if search["arithmetic"] != "numpy_float64":
        raise RealDimensionSweepError("search.arithmetic must be numpy_float64")
    if search["stable_tie_break"] != "lexicographic_doc_id":
        raise RealDimensionSweepError(
            "search.stable_tie_break must be lexicographic_doc_id"
        )
    k_ctx = _positive_integer(search["k_ctx"], "search.k_ctx")
    k_gt = _positive_integer(search["k_gt"], "search.k_gt")
    m_pilot = _positive_integer(search["m_pilot"], "search.m_pilot")
    m_grid = _strict_integer_grid(search["m_grid"], "search.m_grid")
    if k_ctx > k_gt or k_gt > m_pilot or m_grid[0] != m_pilot:
        raise RealDimensionSweepError(
            "require k_ctx <= k_gt <= m_pilot == first M_grid value"
        )

    selection = _exact_keys(
        root["selection"],
        {
            "metric",
            "alpha",
            "target",
            "statistic_role",
            "objective",
            "cost_formula",
            "include_query_projection",
            "tie_break",
        },
        "selection",
    )
    if selection["metric"] != "embedding_neighbor_retention_at_k_gt":
        raise RealDimensionSweepError("selection metric is not frozen")
    if selection["statistic_role"] != "tune_selection_score_not_certificate":
        raise RealDimensionSweepError("selection statistic role is not frozen")
    if selection["objective"] != "coordinate_multiply_adds_per_query":
        raise RealDimensionSweepError("selection objective is not frozen")
    expected_formula = (
        "(corpus_size + embedding_dimension) * m_prime + "
        "embedding_dimension * fixed_budget"
    )
    if selection["cost_formula"] != expected_formula:
        raise RealDimensionSweepError("selection cost formula is not frozen")
    if selection["include_query_projection"] is not True:
        raise RealDimensionSweepError("query projection must be included in cost")
    expected_tie_break = [
        "higher_lower_bound",
        "smaller_m_prime",
        "smaller_fixed_budget",
    ]
    if selection["tie_break"] != expected_tie_break:
        raise RealDimensionSweepError("selection tie break is not frozen")
    alpha = selection["alpha"]
    target = selection["target"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise RealDimensionSweepError("selection.alpha must lie in (0,1)")
    if isinstance(target, bool) or not isinstance(target, (int, float)) or not 0 < target <= 1:
        raise RealDimensionSweepError("selection.target must lie in (0,1]")

    return RealDimensionSweepConfig(
        schema_version=1,
        benchmark="real_fixed_dimension_tune_sweep_v1",
        dataset_manifest_fingerprint=_fingerprint_string(
            root["dataset_manifest_fingerprint"],
            "dataset_manifest_fingerprint",
        ),
        embedding_config_fingerprint=_fingerprint_string(
            root["embedding_config_fingerprint"],
            "embedding_config_fingerprint",
        ),
        embedding_request_fingerprint=_fingerprint_string(
            root["embedding_request_fingerprint"],
            "embedding_request_fingerprint",
        ),
        embedding_manifest_fingerprint=_fingerprint_string(
            root["embedding_manifest_fingerprint"],
            "embedding_manifest_fingerprint",
        ),
        original_baseline_result_fingerprint=_fingerprint_string(
            root["original_baseline_result_fingerprint"],
            "original_baseline_result_fingerprint",
        ),
        evaluation_split="query_tune",
        projection=DimensionProjectionConfig(
            family=projection["family"],
            seed=_positive_integer(projection["seed"], "projection.seed"),
            candidates=_strict_integer_grid(
                projection["candidates"], "projection.candidates"
            ),
            candidate_coupling=projection["candidate_coupling"],
            post_projection_normalize=False,
        ),
        search=DimensionSearchConfig(
            normalized_inputs=True,
            distance="squared_l2",
            arithmetic="numpy_float64",
            stable_tie_break="lexicographic_doc_id",
            query_batch_size=_positive_integer(
                search["query_batch_size"], "search.query_batch_size"
            ),
            k_ctx=k_ctx,
            k_gt=k_gt,
            m_pilot=m_pilot,
            m_grid=m_grid,
        ),
        selection=DimensionSelectionConfig(
            metric=selection["metric"],
            alpha=float(alpha),
            target=float(target),
            statistic_role=selection["statistic_role"],
            objective=selection["objective"],
            cost_formula=selection["cost_formula"],
            include_query_projection=True,
            tie_break=list(selection["tie_break"]),
        ),
        raw=root,
        config_fingerprint=fingerprint(root),
    )


def _load_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealDimensionSweepError(
            f"cannot load {description} {path}: {exc}"
        ) from exc


def _load_jsonl(path: Path, description: str) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise RealDimensionSweepError(
                        f"non-object {description} row at {path}:{line_number}"
                    )
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise RealDimensionSweepError(
            f"cannot load {description} {path}: {exc}"
        ) from exc
    if not rows:
        raise RealDimensionSweepError(f"{description} cannot be empty")
    return rows


def _file_identity(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def _validate_original_baseline(
    baseline_dir: Path,
    config: RealDimensionSweepConfig,
    dataset_manifest: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    manifest = _load_json(baseline_dir / "manifest.json", "baseline manifest")
    if manifest.get("kind") != "real_original_exact_tune_manifest_v1":
        raise RealDimensionSweepError("original baseline kind mismatch")
    if manifest.get("data_scope") != "query_tune_only":
        raise RealDimensionSweepError("original baseline is not tune-only")
    if manifest.get("result_fingerprint") != config.original_baseline_result_fingerprint:
        raise RealDimensionSweepError("original baseline result fingerprint mismatch")
    dataset = manifest.get("dataset", {})
    if (
        dataset.get("manifest_fingerprint") != config.dataset_manifest_fingerprint
        or dataset.get("query_split") != "query_tune"
        or dataset.get("query_split_id_hash")
        != dataset_manifest["splits"]["query_tune"]["id_hash"]
    ):
        raise RealDimensionSweepError("original baseline dataset/split identity mismatch")
    embedding = manifest.get("embedding", {})
    if embedding.get("manifest_fingerprint") != config.embedding_manifest_fingerprint:
        raise RealDimensionSweepError("original baseline embedding identity mismatch")
    artifacts = manifest.get("result_artifacts")
    if not isinstance(artifacts, dict):
        raise RealDimensionSweepError("original baseline artifact identities are missing")
    observed_artifacts: Dict[str, Any] = {}
    for name in ("per_query.jsonl", "summary.json", "report.md"):
        observed_artifacts[name] = _file_identity(baseline_dir / name)
        if artifacts.get(name) != observed_artifacts[name]:
            raise RealDimensionSweepError(
                f"original baseline artifact identity mismatch: {name}"
            )
    baseline_result_identity = {
        "config_fingerprint": manifest.get("config_fingerprint"),
        "dataset_manifest_fingerprint": dataset.get("manifest_fingerprint"),
        "embedding_manifest_fingerprint": embedding.get("manifest_fingerprint"),
        "query_split_id_hash": dataset.get("query_split_id_hash"),
        "artifacts": observed_artifacts,
    }
    if fingerprint(baseline_result_identity) != manifest.get("result_fingerprint"):
        raise RealDimensionSweepError("original baseline result identity is invalid")
    records = _load_jsonl(
        baseline_dir / "per_query.jsonl", "original baseline records"
    )
    if (
        len(records) != dataset_manifest["splits"]["query_tune"]["n"]
        or {record.get("split") for record in records} != {"query_tune"}
        or stable_id_hash([str(record.get("query_id")) for record in records])
        != dataset_manifest["splits"]["query_tune"]["id_hash"]
    ):
        raise RealDimensionSweepError("original baseline records are not frozen tune IDs")
    return records


def _exact_projected_rankings(
    corpus: np.ndarray,
    queries: np.ndarray,
    corpus_ids: Sequence[str],
    *,
    k: int,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    """Return exact row rankings with squared-L2 and stable document-ID ties."""
    corpus_values = np.asarray(corpus, dtype=np.float64)
    query_values = np.asarray(queries, dtype=np.float64)
    if corpus_values.ndim != 2 or query_values.ndim != 2:
        raise RealDimensionSweepError("projected arrays must be two-dimensional")
    if corpus_values.shape[1] != query_values.shape[1]:
        raise RealDimensionSweepError("projected corpus/query dimensions differ")
    if not 1 <= k <= len(corpus_values):
        raise RealDimensionSweepError("projected ranking k exceeds corpus size")
    id_values = np.asarray(corpus_ids, dtype=str)
    if len(id_values) != len(corpus_values) or len(set(id_values.tolist())) != len(id_values):
        raise RealDimensionSweepError("corpus IDs are not unique and row-aligned")
    tie_rank = np.argsort(np.argsort(id_values, kind="stable"), kind="stable")
    corpus_norms = np.einsum("ij,ij->i", corpus_values, corpus_values)
    result = np.empty((len(query_values), k), dtype=np.int64)
    started = perf_counter()
    for start in range(0, len(query_values), batch_size):
        batch = query_values[start : start + batch_size]
        query_norms = np.einsum("ij,ij->i", batch, batch)
        distances = (
            query_norms[:, None]
            + corpus_norms[None, :]
            - 2.0 * (batch @ corpus_values.T)
        )
        np.maximum(distances, 0.0, out=distances)
        for offset, row_distances in enumerate(distances):
            result[start + offset] = np.lexsort((tie_rank, row_distances))[:k]
    return result, (perf_counter() - started) * 1000.0


def _ranking_hash(rows: np.ndarray) -> str:
    values = np.asarray(rows, dtype="<i8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _coordinate_work(
    *, corpus_size: int, dimension: int, m_prime: int, budget: int
) -> Dict[str, Any]:
    query_projection = dimension * m_prime
    projected_scan = corpus_size * m_prime
    original_rerank = dimension * budget
    total = query_projection + projected_scan + original_rerank
    original_full_scan = corpus_size * dimension
    return {
        "query_projection": query_projection,
        "projected_full_scan": projected_scan,
        "original_rerank": original_rerank,
        "total": total,
        "original_full_scan_reference": original_full_scan,
        "ratio_to_original_full_scan": total / original_full_scan,
        "reduction_fraction_vs_original_full_scan": 1.0 - total / original_full_scan,
    }


def _report(summary: Mapping[str, Any]) -> str:
    selected = summary["selected"]
    lines = [
        "# SciFact tune-only fixed projection-dimension sweep",
        "",
        "All rankings, retention values, and decisions use `query_tune` only. "
        "The empirical-Bernstein value is a tune selection statistic, not a "
        "certificate. Evidence labels do not enter selection.",
        "",
        "The common objective includes one query projection, one exact projected "
        "full-corpus scan, and exact original-space reranking:",
        "",
        "`work = (N + d) * m_prime + d * M`",
        "",
        "| m_prime | eligible | fixed M | mean retention | tune lower bound | work | vs original |",
        "| ---: | :---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in summary["candidates"]:
        budget = candidate["selected_fixed_budget"]
        bound = candidate["selected_tune_retention_bound"]
        work = candidate["selected_coordinate_work"]
        mean_text = "—" if bound is None else f"{bound['mean']:.6f}"
        lower_text = "—" if bound is None else f"{bound['lower_bound']:.6f}"
        work_text = "—" if work is None else str(work["total"])
        reduction_text = (
            "—"
            if work is None
            else f"{work['reduction_fraction_vs_original_full_scan']:.2%}"
        )
        lines.append(
            f"| {candidate['m_prime']} | "
            f"{'yes' if candidate['eligible'] else 'no'} | "
            f"{budget if budget is not None else '—'} | "
            f"{mean_text} | {lower_text} | {work_text} | {reduction_text} |"
        )
    lines.extend(
        [
            "",
            "## Frozen tune choice",
            "",
            f"- `m_prime`: {selected['m_prime']}",
            f"- fixed reference budget: {selected['fixed_budget']}",
            f"- tune mean retention: {selected['tune_retention_bound']['mean']:.6f}",
            f"- tune lower bound: {selected['tune_retention_bound']['lower_bound']:.6f}",
            f"- target: {summary['selection_target']:.6f}",
            f"- coordinate work: {selected['coordinate_work']['total']}",
            "- reduction versus original full scan: "
            f"{selected['coordinate_work']['reduction_fraction_vs_original_full_scan']:.2%}",
            f"- selection fingerprint: `{summary['selection_fingerprint']}`",
            "",
            "This freezes a projection dimension for later policy tuning. It does "
            "not certify retention, evaluate protected splits, or claim latency or "
            "answer-quality improvement.",
            "",
        ]
    )
    return "\n".join(lines)


def run_real_dimension_sweep(
    config: RealDimensionSweepConfig,
    prepared_dir: Union[str, Path],
    embedding_config_path: Union[str, Path],
    embedding_cache_dir: Union[str, Path],
    original_baseline_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    if config.evaluation_split != "query_tune":
        raise RealDimensionSweepError("dimension selection accepts query_tune only")
    prepared = Path(prepared_dir)
    embedding_cache = Path(embedding_cache_dir)
    baseline = Path(original_baseline_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dimension-sweep output: {output}")

    embedding_config = load_text_embedding_config(embedding_config_path)
    if embedding_config.config_fingerprint != config.embedding_config_fingerprint:
        raise RealDimensionSweepError("embedding config fingerprint mismatch")
    validation = validate_text_embedding_cache(
        embedding_config, prepared, embedding_cache
    )
    dataset_manifest = validation["dataset_manifest"]
    embedding_manifest = validation["embedding_manifest"]
    if dataset_manifest["fingerprint"] != config.dataset_manifest_fingerprint:
        raise RealDimensionSweepError("dataset fingerprint mismatch")
    if validation["request_fingerprint"] != config.embedding_request_fingerprint:
        raise RealDimensionSweepError("embedding request fingerprint mismatch")
    if embedding_manifest["fingerprint"] != config.embedding_manifest_fingerprint:
        raise RealDimensionSweepError("embedding cache fingerprint mismatch")
    baseline_records = _validate_original_baseline(
        baseline, config, dataset_manifest
    )

    corpus_table = load_embedding_array(
        embedding_cache / "corpus_embeddings.f32.npy",
        embedding_cache / "corpus_ids.json",
    )
    query_table = load_embedding_array(
        embedding_cache / "query_embeddings.f32.npy",
        embedding_cache / "query_ids.json",
    )
    query_rows = _load_jsonl(prepared / "queries.jsonl", "prepared queries")
    expected_query_ids = [str(row.get("query_id")) for row in query_rows]
    if query_table.ids.tolist() != expected_query_ids:
        raise RealDimensionSweepError("query embedding rows do not match prepared IDs")
    selected_rows = [
        index
        for index, row in enumerate(query_rows)
        if row.get("split") == "query_tune"
    ]
    selected_query_ids = [expected_query_ids[index] for index in selected_rows]
    if selected_query_ids != [str(record.get("query_id")) for record in baseline_records]:
        raise RealDimensionSweepError(
            "baseline records do not align with prepared tune query order"
        )
    corpus_ids = corpus_table.ids.tolist()
    corpus_id_to_row = {doc_id: row for row, doc_id in enumerate(corpus_ids)}
    exact_top_rows: list[set[int]] = []
    exact_top_ids: list[list[str]] = []
    for record in baseline_records:
        ids = record.get("exact_top_k_ids")
        if not isinstance(ids, list) or len(ids) != config.search.k_gt:
            raise RealDimensionSweepError("baseline exact top-k width mismatch")
        if not all(isinstance(value, str) and value in corpus_id_to_row for value in ids):
            raise RealDimensionSweepError("baseline exact top-k has unknown document ID")
        exact_top_ids.append(list(ids))
        exact_top_rows.append({corpus_id_to_row[value] for value in ids})

    corpus_size, dimension = corpus_table.vectors.shape
    if embedding_config.model.embedding_dimension != dimension:
        raise RealDimensionSweepError("embedding dimension mismatch")
    if config.projection.candidates[-1] > dimension:
        raise RealDimensionSweepError("projection candidate exceeds original dimension")
    if config.search.m_grid[-1] > corpus_size:
        raise RealDimensionSweepError("M_grid exceeds corpus size")
    if config.search.m_grid[-1] != corpus_size:
        raise RealDimensionSweepError(
            "the terminal M_grid value must equal corpus size so failure is explicit"
        )

    tune_queries = query_table.vectors[np.asarray(selected_rows, dtype=np.int64)]
    all_records: list[Dict[str, Any]] = []
    candidates: list[Dict[str, Any]] = []
    timing_candidates: list[Dict[str, Any]] = []
    for m_prime in config.projection.candidates:
        projection_started = perf_counter()
        matrix = dense_gaussian_projection(
            m_prime, dimension, config.projection.seed
        )
        projected_corpus = project_rows(corpus_table.vectors, matrix)
        projected_queries = project_rows(tune_queries, matrix)
        projection_ms = (perf_counter() - projection_started) * 1000.0
        rankings, search_ms = _exact_projected_rankings(
            projected_corpus,
            projected_queries,
            corpus_ids,
            k=config.search.m_grid[-1],
            batch_size=config.search.query_batch_size,
        )
        metadata = projection_metadata(
            dimension=dimension,
            m_prime=m_prime,
            seed=config.projection.seed,
            normalization=True,
            embedding_model=(
                f"{embedding_config.model.name}@{embedding_config.model.revision}"
            ),
            corpus_hash=embedding_manifest["arrays"]["corpus"][
                "array_fingerprint"
            ],
        )
        values_by_budget: Dict[str, list[float]] = {
            str(budget): [] for budget in config.search.m_grid
        }
        for query_index, ranking in enumerate(rankings):
            overlap_counts: Dict[str, int] = {}
            retentions: Dict[str, float] = {}
            truth = exact_top_rows[query_index]
            for budget in config.search.m_grid:
                overlap = len(truth.intersection(ranking[:budget].tolist()))
                overlap_counts[str(budget)] = overlap
                retention = overlap / config.search.k_gt
                retentions[str(budget)] = retention
                values_by_budget[str(budget)].append(retention)
            all_records.append(
                {
                    "query_index": query_index,
                    "query_id": selected_query_ids[query_index],
                    "split": "query_tune",
                    "m_prime": m_prime,
                    "projection_fingerprint": metadata["fingerprint"],
                    "exact_top_k_ids": exact_top_ids[query_index],
                    "projected_ranking_rows_sha256": _ranking_hash(ranking),
                    "ranking_row_identity": {
                        "dtype": "little_endian_int64",
                        "length": len(ranking),
                        "corpus_array_fingerprint": embedding_manifest[
                            "arrays"
                        ]["corpus"]["array_fingerprint"],
                        "corpus_id_hash": dataset_manifest["ids"][
                            "corpus_id_hash"
                        ],
                    },
                    "overlap_count_by_budget": overlap_counts,
                    "embedding_retention_by_budget": retentions,
                }
            )
        by_budget: Dict[str, Any] = {}
        eligible_budgets: list[int] = []
        for budget in config.search.m_grid:
            bound = empirical_bernstein(
                values_by_budget[str(budget)], config.selection.alpha
            ).serialize()
            bound["eligible"] = bool(
                bound["lower_bound"] >= config.selection.target
            )
            by_budget[str(budget)] = bound
            if bound["eligible"]:
                eligible_budgets.append(budget)
        selected_budget = min(eligible_budgets) if eligible_budgets else None
        selected_bound = (
            by_budget[str(selected_budget)] if selected_budget is not None else None
        )
        selected_work = (
            _coordinate_work(
                corpus_size=corpus_size,
                dimension=dimension,
                m_prime=m_prime,
                budget=selected_budget,
            )
            if selected_budget is not None
            else None
        )
        candidates.append(
            {
                "m_prime": m_prime,
                "projection": metadata,
                "eligible": selected_budget is not None,
                "by_budget": by_budget,
                "selected_fixed_budget": selected_budget,
                "selected_tune_retention_bound": selected_bound,
                "selected_coordinate_work": selected_work,
            }
        )
        timing_candidates.append(
            {
                "m_prime": m_prime,
                "projection_and_materialization_ms": projection_ms,
                "exact_projected_search_ms": search_ms,
                "projected_corpus_bytes": int(projected_corpus.nbytes),
                "projected_query_bytes": int(projected_queries.nbytes),
                "distance_evaluations": len(tune_queries) * corpus_size,
            }
        )

    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise RuntimeError(
            "no projection dimension reached the predeclared tune target"
        )
    winner = min(
        eligible,
        key=lambda candidate: (
            candidate["selected_coordinate_work"]["total"],
            -candidate["selected_tune_retention_bound"]["lower_bound"],
            candidate["m_prime"],
            candidate["selected_fixed_budget"],
        ),
    )
    selected = {
        "m_prime": winner["m_prime"],
        "fixed_budget": winner["selected_fixed_budget"],
        "projection_fingerprint": winner["projection"]["fingerprint"],
        "tune_retention_bound": winner["selected_tune_retention_bound"],
        "coordinate_work": winner["selected_coordinate_work"],
    }
    selection = {
        "schema_version": 1,
        "kind": "real_fixed_dimension_tune_selection_v1",
        "data_scope": "query_tune_only",
        "config_fingerprint": config.config_fingerprint,
        "dataset_manifest_fingerprint": dataset_manifest["fingerprint"],
        "embedding_manifest_fingerprint": embedding_manifest["fingerprint"],
        "original_baseline_result_fingerprint": config.original_baseline_result_fingerprint,
        "query_tune_n": len(selected_query_ids),
        "query_tune_id_hash": stable_id_hash(selected_query_ids),
        "selection_metric": config.selection.metric,
        "selection_alpha": config.selection.alpha,
        "selection_target": config.selection.target,
        "selection_rule": _SELECTION_RULE,
        "candidates": candidates,
        "selected": selected,
    }
    selection["selection_fingerprint"] = fingerprint(selection)
    selected_projection = dict(winner["projection"])
    selected_projection.update(
        {
            "schema_version": 1,
            "kind": "frozen_real_projection_v1",
            "selection_fingerprint": selection["selection_fingerprint"],
            "m_pilot": config.search.m_pilot,
            "m_grid": config.search.m_grid,
            "k_gt": config.search.k_gt,
            "k_ctx": config.search.k_ctx,
        }
    )
    selected_projection["frozen_fingerprint"] = fingerprint(selected_projection)
    summary = {
        "schema_version": 1,
        "kind": "real_fixed_dimension_tune_summary_v1",
        "data_scope": "query_tune_only",
        "corpus_size": corpus_size,
        "embedding_dimension": dimension,
        "n_queries": len(selected_query_ids),
        "m_pilot": config.search.m_pilot,
        "m_grid": config.search.m_grid,
        "selection_target": config.selection.target,
        "selection_fingerprint": selection["selection_fingerprint"],
        "candidates": candidates,
        "selected": selected,
    }
    summary["fingerprint"] = fingerprint(summary)
    timings = {
        "role": "systems_diagnostic_excluded_from_result_identity",
        "candidates": timing_candidates,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        paths = {
            "per_query.jsonl": temporary / "per_query.jsonl",
            "selection.json": temporary / "selection.json",
            "selected_projection.json": temporary / "selected_projection.json",
            "summary.json": temporary / "summary.json",
            "timings.json": temporary / "timings.json",
            "report.md": temporary / "report.md",
            "manifest.json": temporary / "manifest.json",
        }
        _write_jsonl(paths["per_query.jsonl"], all_records)
        write_json(paths["selection.json"], selection)
        write_json(paths["selected_projection.json"], selected_projection)
        write_json(paths["summary.json"], summary)
        write_json(paths["timings.json"], timings)
        paths["report.md"].write_text(_report(summary), encoding="utf-8")
        result_artifacts = {
            name: _file_identity(paths[name])
            for name in (
                "per_query.jsonl",
                "selection.json",
                "selected_projection.json",
                "summary.json",
                "report.md",
            )
        }
        result_identity = {
            "config_fingerprint": config.config_fingerprint,
            "dataset_manifest_fingerprint": dataset_manifest["fingerprint"],
            "embedding_manifest_fingerprint": embedding_manifest["fingerprint"],
            "original_baseline_result_fingerprint": config.original_baseline_result_fingerprint,
            "query_tune_id_hash": stable_id_hash(selected_query_ids),
            "artifacts": result_artifacts,
        }
        manifest = {
            "schema_version": 1,
            "kind": "real_fixed_dimension_tune_manifest_v1",
            "data_scope": "query_tune_only",
            "config_fingerprint": config.config_fingerprint,
            "dataset_manifest_fingerprint": dataset_manifest["fingerprint"],
            "embedding_manifest_fingerprint": embedding_manifest["fingerprint"],
            "original_baseline_result_fingerprint": config.original_baseline_result_fingerprint,
            "query_tune_id_hash": stable_id_hash(selected_query_ids),
            "selection_fingerprint": selection["selection_fingerprint"],
            "frozen_projection_fingerprint": selected_projection[
                "frozen_fingerprint"
            ],
            "result_artifacts": result_artifacts,
            "result_fingerprint": fingerprint(result_identity),
            "timings_artifact": "timings.json",
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
        }
        manifest["fingerprint"] = fingerprint(manifest)
        write_json(paths["manifest.json"], manifest)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {name: output / name for name in paths}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--embedding-config", required=True, type=Path)
    parser.add_argument("--embedding-cache", required=True, type=Path)
    parser.add_argument("--original-baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    config = load_real_dimension_sweep_config(args.config)
    paths = run_real_dimension_sweep(
        config,
        args.dataset,
        args.embedding_config,
        args.embedding_cache,
        args.original_baseline,
        args.output,
    )
    print(f"completed tune-only dimension sweep: {paths['report.md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
