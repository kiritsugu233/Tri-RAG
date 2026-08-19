from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .certification import (
    make_certificate,
    per_bin_certificates,
    plan_sample_size,
    validate_certificate_identity,
)
from .config import HarnessConfig, load_config
from .embeddings import normalize_rows
from .indexes import ExactSquaredL2Index
from .lid import estimate_lid_from_squared_distances
from .manifest import build_manifest
from .policies import FixedBudgetPolicy, MonotoneBinnedPolicy
from .projection import dense_gaussian_projection, project_rows, projection_metadata
from .reporting import generate_report
from .synthetic import generate_synthetic_dataset
from .utils import array_fingerprint, fingerprint, stable_id_hash, write_json


def _percentile(values: Sequence[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _retention(gt_rows: np.ndarray, candidate_rows: np.ndarray) -> float:
    return len(set(gt_rows.tolist()).intersection(candidate_rows.tolist())) / len(gt_rows)


def _build_base_records(
    config: HarnessConfig,
    *,
    query_ids: np.ndarray,
    splits: np.ndarray,
    corpus_ids: np.ndarray,
    corpus: np.ndarray,
    queries: np.ndarray,
    projected_corpus: np.ndarray,
    projected_queries: np.ndarray,
) -> List[Dict[str, Any]]:
    retrieval = config.retrieval
    original_index = ExactSquaredL2Index(
        corpus_ids, corpus, batch_size=retrieval.batch_size
    )
    projected_index = ExactSquaredL2Index(
        corpus_ids, projected_corpus, batch_size=retrieval.batch_size
    )
    oracle_k = max(retrieval.k_gt, retrieval.s_lid)
    original = original_index.search(queries, oracle_k)
    pilot = projected_index.search(projected_queries, retrieval.m_pilot)
    expanded = projected_index.search(projected_queries, retrieval.m_grid[-1])
    query_count = len(queries)
    base_records: List[Dict[str, Any]] = []
    for row in range(query_count):
        pilot_rows = pilot.rows[row]
        full_rows = expanded.rows[row]
        pilot_sq = np.einsum(
            "ij,ij->i", corpus[pilot_rows] - queries[row], corpus[pilot_rows] - queries[row]
        )
        if not np.array_equal(pilot_rows, full_rows[: retrieval.m_pilot]):
            raise AssertionError("exact expansion must preserve the pilot prefix")
        lid_kwargs = {
            "s_lid": retrieval.s_lid,
            "min_neighbors": retrieval.min_lid_neighbors,
            "clip_min": config.lid.clip_min,
            "clip_max": config.lid.clip_max,
            "duplicate_tolerance": config.lid.duplicate_tolerance,
            "fallback": config.lid.fallback,
        }
        pilot_lid = estimate_lid_from_squared_distances(pilot_sq, **lid_kwargs)
        oracle_lid = estimate_lid_from_squared_distances(
            original.squared_distances[row, : retrieval.s_lid], **lid_kwargs
        )
        gt_rows = original.rows[row, : retrieval.k_gt]
        retention_by_budget = {
            str(budget): _retention(gt_rows, full_rows[:budget])
            for budget in retrieval.m_grid
        }
        base_records.append(
            {
                "query_id": str(query_ids[row]),
                "split": str(splits[row]),
                "pilot_lid": pilot_lid.serialize(),
                "oracle_lid": oracle_lid.serialize(),
                "lid": pilot_lid.clipped,
                "lid_valid": pilot_lid.valid,
                "retention_by_budget": retention_by_budget,
                "exact_top_k_rows": gt_rows.tolist(),
                "exact_top_k_ids": original.ids[row, : retrieval.k_gt].tolist(),
                "pilot_rows": pilot_rows.tolist(),
                "pilot_squared_distances": pilot_sq.tolist(),
                "expanded_rows": full_rows.tolist(),
                "pilot_search_ms": pilot.search_ms / query_count,
            }
        )
    return base_records


def _materialize_policy_records(
    base_records: List[Mapping[str, Any]],
    policy: MonotoneBinnedPolicy,
    corpus_ids: np.ndarray,
    corpus: np.ndarray,
    queries: np.ndarray,
    projected_corpus: np.ndarray,
    projected_queries: np.ndarray,
    config: HarnessConfig,
) -> List[Dict[str, Any]]:
    decisions = []
    for base in base_records:
        policy_started = perf_counter()
        decision = policy.choose(float(base["lid"]), bool(base["lid_valid"]))
        decisions.append((decision, (perf_counter() - policy_started) * 1000.0))
    projected_index = ExactSquaredL2Index(
        corpus_ids, projected_corpus, batch_size=config.retrieval.batch_size
    )
    expansion_by_row: Dict[int, Any] = {}
    expansion_ms_by_row: Dict[int, float] = {}
    for budget in config.retrieval.m_grid:
        rows = [row for row, (decision, _) in enumerate(decisions) if decision.budget == budget]
        if not rows:
            continue
        search = projected_index.search(projected_queries[rows], budget)
        for local_row, global_row in enumerate(rows):
            expansion_by_row[global_row] = search.rows[local_row]
            expansion_ms_by_row[global_row] = search.search_ms / len(rows)
    results: List[Dict[str, Any]] = []
    for row, base in enumerate(base_records):
        decision, policy_ms = decisions[row]
        candidate_rows = np.asarray(expansion_by_row[row], dtype=np.int64)
        expected_rows = np.asarray(base["expanded_rows"][: decision.budget], dtype=np.int64)
        if not np.array_equal(candidate_rows, expected_rows):
            raise AssertionError("chosen-budget expansion disagrees with evaluation search")
        pilot_rows = np.asarray(base["pilot_rows"], dtype=np.int64)
        if not np.array_equal(candidate_rows[: config.retrieval.m_pilot], pilot_rows):
            raise AssertionError("chosen-budget expansion must reuse the pilot prefix")
        rerank_started = perf_counter()
        pilot_sq = np.asarray(base["pilot_squared_distances"], dtype=np.float64)
        additional_rows = candidate_rows[config.retrieval.m_pilot :]
        additional_sq = np.einsum(
            "ij,ij->i",
            corpus[additional_rows] - queries[row],
            corpus[additional_rows] - queries[row],
        )
        candidate_sq = np.concatenate([pilot_sq, additional_sq])
        candidate_ids = corpus_ids[candidate_rows]
        rerank_order = np.lexsort((candidate_ids, candidate_sq))
        reranked_rows = candidate_rows[rerank_order]
        rerank_ms = (perf_counter() - rerank_started) * 1000.0
        gt_rows = np.asarray(base["exact_top_k_rows"], dtype=np.int64)
        overlap = _retention(gt_rows, candidate_rows)
        reranked_overlap = _retention(gt_rows, reranked_rows[: config.retrieval.k_gt])
        if overlap != reranked_overlap:
            raise AssertionError("exact reranking overlap must equal candidate retention")
        additional = decision.budget - config.retrieval.m_pilot
        total_ms = (
            float(base["pilot_search_ms"])
            + expansion_ms_by_row[row]
            + rerank_ms
            + policy_ms
        )
        results.append(
            {
                "query_id": base["query_id"],
                "split": base["split"],
                "policy_name": "monotone_binned_empirical",
                "policy_version": 1,
                "lid_mode": "pilot_rerank",
                "lid_raw": base["pilot_lid"]["raw"],
                "lid_clipped": base["pilot_lid"]["clipped"],
                "lid_valid": base["pilot_lid"]["valid"],
                "lid_failure_reason": base["pilot_lid"]["reason"],
                "lid_valid_distance_count": base["pilot_lid"]["valid_distance_count"],
                "oracle_lid_raw": base["oracle_lid"]["raw"],
                "oracle_lid_clipped": base["oracle_lid"]["clipped"],
                "oracle_lid_valid": base["oracle_lid"]["valid"],
                "lid_bin": decision.bin_index,
                "chosen_m": decision.budget,
                "policy_saturated": decision.budget == config.retrieval.m_grid[-1],
                "used_lid_fallback": decision.used_fallback,
                "exact_top_k_ids": base["exact_top_k_ids"],
                "projected_candidate_ids": candidate_ids.tolist(),
                "reranked_top_k_ids": corpus_ids[
                    reranked_rows[: config.retrieval.k_gt]
                ].tolist(),
                "embedding_retention": overlap,
                "fixed_retentions": base["retention_by_budget"],
                "pilot_search_ms": float(base["pilot_search_ms"]),
                "expansion_search_ms": expansion_ms_by_row[row],
                "policy_compute_ms": policy_ms,
                "rerank_ms": rerank_ms,
                "total_retrieval_ms": total_ms,
                "pilot_original_distance_count": config.retrieval.m_pilot,
                "additional_original_distance_count": additional,
            }
        )
    return results


def _split_aggregate(records: List[Mapping[str, Any]], grid: Sequence[int]) -> Dict[str, Any]:
    retentions = [float(record["embedding_retention"]) for record in records]
    budgets = [int(record["chosen_m"]) for record in records]
    gaps = [
        abs(float(record["lid_clipped"]) - float(record["oracle_lid_clipped"]))
        for record in records
        if bool(record["lid_valid"]) and bool(record["oracle_lid_valid"])
    ]
    return {
        "n": len(records),
        "adaptive_retention": {
            "mean": float(np.mean(retentions)),
            "median": float(np.median(retentions)),
            "p05": _percentile(retentions, 0.05),
            "minimum": float(np.min(retentions)),
        },
        "budget": {
            "mean": float(np.mean(budgets)),
            "median": float(np.median(budgets)),
            "p95": _percentile(budgets, 0.95),
            "p99": _percentile(budgets, 0.99),
            "distribution": {str(value): budgets.count(value) for value in grid},
        },
        "fixed_retention_mean": {
            str(budget): float(
                np.mean([float(record["fixed_retentions"][str(budget)]) for record in records])
            )
            for budget in grid
        },
        "lid_diagnostic": {
            "paired_valid_n": len(gaps),
            "mean_absolute_gap": float(np.mean(gaps)) if gaps else 0.0,
            "pilot_invalid_n": sum(not bool(record["lid_valid"]) for record in records),
        },
    }


def run_harness(config: HarnessConfig, output_dir: Path) -> Dict[str, Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_dataset(config.synthetic, config.seeds.data)
    corpus = normalize_rows(dataset.corpus.vectors)
    queries = normalize_rows(dataset.queries.vectors)
    matrix = dense_gaussian_projection(
        config.retrieval.m_prime, config.synthetic.dimension, config.seeds.projection
    )
    projected_corpus = project_rows(corpus, matrix)
    projected_queries = project_rows(queries, matrix)
    base_records = _build_base_records(
        config,
        query_ids=dataset.queries.ids,
        splits=dataset.splits,
        corpus_ids=dataset.corpus.ids,
        corpus=corpus,
        queries=queries,
        projected_corpus=projected_corpus,
        projected_queries=projected_queries,
    )
    tune_records = [record for record in base_records if record["split"] == "query_tune"]
    policy = MonotoneBinnedPolicy.fit(
        tune_records,
        grid=config.retrieval.m_grid,
        n_bins=config.policy.n_bins,
        target=config.policy.tune_target,
        safety_margin=config.policy.safety_margin,
        fallback_budget=config.policy.fallback_budget,
    )
    policy_artifact = policy.serialize()
    records = _materialize_policy_records(
        base_records,
        policy,
        dataset.corpus.ids,
        corpus,
        queries,
        projected_corpus,
        projected_queries,
        config,
    )
    records_by_split = {
        split: [record for record in records if record["split"] == split]
        for split in ("query_tune", "query_cert", "query_test")
    }
    query_ids_by_split = {
        split: [str(record["query_id"]) for record in split_records]
        for split, split_records in records_by_split.items()
    }
    manifest = build_manifest(
        config,
        corpus_ids=dataset.corpus.ids.tolist(),
        query_ids_by_split=query_ids_by_split,
        corpus_embeddings=corpus,
        query_embeddings=queries,
        projection_metadata=projection_metadata(
            dimension=config.synthetic.dimension,
            m_prime=config.retrieval.m_prime,
            seed=config.seeds.projection,
            normalization=True,
            embedding_model="synthetic_array_generator@v1",
            corpus_hash=array_fingerprint(corpus),
        ),
        policy_fingerprint=policy_artifact["fingerprint"],
    )
    cert_records = records_by_split["query_cert"]
    cert_split_hash = stable_id_hash(query_ids_by_split["query_cert"])
    planned_n = plan_sample_size(
        config.certification.alpha, config.certification.desired_radius
    )
    certificate = make_certificate(
        [float(record["embedding_retention"]) for record in cert_records],
        alpha=config.certification.alpha,
        target=config.certification.target,
        policy_fingerprint=policy_artifact["fingerprint"],
        split_hash=cert_split_hash,
        planned_n=planned_n,
    )
    if config.certification.per_bin:
        certificate["per_bin"] = per_bin_certificates(
            cert_records,
            bin_count=len(policy.budgets),
            alpha_total=config.certification.alpha,
            target=config.certification.target,
            policy_fingerprint=policy_artifact["fingerprint"],
            split_hash=cert_split_hash,
            min_bin_size=config.certification.min_bin_size,
        )
    validate_certificate_identity(
        certificate,
        policy_fingerprint=policy_artifact["fingerprint"],
        split_hash=cert_split_hash,
    )
    aggregates = {
        split: _split_aggregate(split_records, config.retrieval.m_grid)
        for split, split_records in records_by_split.items()
    }
    fixed_certificates: Dict[str, Any] = {}
    passing_fixed = []
    for budget in config.retrieval.m_grid:
        fixed_policy = FixedBudgetPolicy(
            budget,
            config.retrieval.m_grid,
            max(config.retrieval.k_gt, config.retrieval.k_ctx, config.retrieval.m_pilot),
        )
        fixed_policy_fingerprint = fingerprint(fixed_policy.serialize())
        fixed_certificate = make_certificate(
            [float(record["fixed_retentions"][str(budget)]) for record in cert_records],
            alpha=config.certification.alpha,
            target=config.certification.target,
            policy_fingerprint=fixed_policy_fingerprint,
            split_hash=cert_split_hash,
            planned_n=planned_n,
        )
        fixed_certificates[str(budget)] = fixed_certificate
        if fixed_certificate["passed"]:
            passing_fixed.append(budget)
    smallest_fixed: Optional[int] = min(passing_fixed) if passing_fixed else None
    adaptive_mean_m = aggregates["query_cert"]["budget"]["mean"]
    candidate_saving = (
        None if smallest_fixed is None else 1.0 - adaptive_mean_m / smallest_fixed
    )
    aggregates["certification_comparison"] = {
        "fixed_certificates": fixed_certificates,
        "smallest_certified_fixed_budget": smallest_fixed,
        "adaptive_candidate_saving": candidate_saving,
    }
    timings = {
        "mean_pilot_search_ms": float(np.mean([r["pilot_search_ms"] for r in records])),
        "mean_expansion_search_ms": float(
            np.mean([r["expansion_search_ms"] for r in records])
        ),
        "mean_policy_compute_ms": float(np.mean([r["policy_compute_ms"] for r in records])),
        "mean_rerank_ms": float(np.mean([r["rerank_ms"] for r in records])),
        "mean_total_retrieval_ms": float(
            np.mean([r["total_retrieval_ms"] for r in records])
        ),
        "mean_pilot_original_distance_count": float(
            np.mean([r["pilot_original_distance_count"] for r in records])
        ),
        "mean_additional_original_distance_count": float(
            np.mean([r["additional_original_distance_count"] for r in records])
        ),
    }
    report = generate_report(manifest, policy_artifact, certificate, aggregates, timings)

    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "policy.json", policy_artifact)
    with (output_dir / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    write_json(output_dir / "certification.json", certificate)
    write_json(output_dir / "aggregates.json", aggregates)
    write_json(output_dir / "timings.json", timings)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return {
        name: output_dir / name
        for name in (
            "manifest.json",
            "policy.json",
            "per_query.jsonl",
            "certification.json",
            "aggregates.json",
            "timings.json",
            "report.md",
        )
    }


def _write_validation_manifest(config: HarnessConfig, output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "status": "config_validated",
            "config_fingerprint": config.config_fingerprint,
            "config": config.raw,
        },
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.validate_only:
        _write_validation_manifest(config, args.output)
        print(f"validated config and wrote {args.output / 'manifest.json'}")
        return 0
    artifacts = run_harness(config, args.output)
    print(f"completed {config.run_name}: {artifacts['report.md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
