"""Tune-only global projected-dimension selection followed by fresh certification."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .certification import empirical_bernstein
from .config import HarnessConfig, load_config
from .embeddings import normalize_rows
from .projection import dense_gaussian_projection, project_rows
from .run import _build_base_records, run_harness
from .synthetic import generate_synthetic_dataset
from .tri_predict import tri_predict_retention_grid
from .utils import fingerprint, stable_id_hash, write_json


SELECTION_RULE = {
    "stage_1_within_dimension": (
        "among thresholds whose tune empirical-Bernstein lower bound reaches the "
        "predeclared tune target and whose saturation fraction is within its cap, "
        "minimize mean candidate budget; break ties by lower saturation, higher "
        "lower bound, then higher threshold"
    ),
    "stage_2_across_dimensions": (
        "maximize tune candidate saving against the smallest tune-qualified fixed "
        "budget at the same m_prime; break ties by lower saturation, higher lower "
        "bound, then larger m_prime"
    ),
    "certification_role": (
        "query_cert is evaluated exactly once after selection.json and "
        "selected_config.json have been written; certification never changes the "
        "selected m_prime or threshold"
    ),
}


def _validate_tune_only(records: Sequence[Mapping[str, Any]]) -> None:
    if len(records) < 2:
        raise ValueError("m_prime sweep requires at least two tune records")
    unexpected = sorted({str(record["split"]) for record in records} - {"query_tune"})
    if unexpected:
        raise ValueError(
            "m_prime selection accepts query_tune records only; "
            f"found forbidden splits {unexpected}"
        )


def _tune_fixed_baseline(
    records: Sequence[Mapping[str, Any]], config: HarnessConfig
) -> Dict[str, Any]:
    _validate_tune_only(records)
    results: Dict[str, Any] = {}
    qualified = []
    assert config.m_prime_sweep is not None
    for budget in config.retrieval.m_grid:
        bound = empirical_bernstein(
            [float(record["retention_by_budget"][str(budget)]) for record in records],
            config.certification.alpha,
        ).serialize()
        bound["qualified"] = bool(
            bound["lower_bound"] >= config.m_prime_sweep.tune_lower_bound_target
        )
        results[str(budget)] = bound
        if bound["qualified"]:
            qualified.append(int(budget))
    return {
        "by_budget": results,
        "smallest_tune_qualified_fixed_budget": min(qualified) if qualified else None,
    }


def _raw_prediction_cache(
    records: Sequence[Mapping[str, Any]], config: HarnessConfig
) -> List[Optional[Dict[int, float]]]:
    _validate_tune_only(records)
    predictions: List[Optional[Dict[int, float]]] = []
    corpus_size = config.synthetic.n_clusters * config.synthetic.docs_per_cluster
    for record in records:
        if not bool(record["lid_valid"]):
            predictions.append(None)
            continue
        predictions.append(
            tri_predict_retention_grid(
                lid=float(record["lid"]),
                m_prime=config.retrieval.m_prime,
                k_gt=config.retrieval.k_gt,
                budgets=config.retrieval.m_grid,
                corpus_size=corpus_size,
                max_rank_samples=config.tri_predict.max_rank_samples,
            )
        )
    return predictions


def _evaluate_threshold(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Optional[Mapping[int, float]]],
    *,
    threshold: float,
    config: HarnessConfig,
) -> Dict[str, Any]:
    _validate_tune_only(records)
    if len(records) != len(predictions):
        raise ValueError("prediction cache must align with tune records")
    budgets = []
    retentions = []
    saturated = 0
    for record, prediction in zip(records, predictions):
        chosen = config.retrieval.m_grid[-1]
        is_saturated = True
        if prediction is not None:
            for budget in config.retrieval.m_grid:
                if float(prediction[budget]) >= threshold:
                    chosen = int(budget)
                    is_saturated = False
                    break
        budgets.append(chosen)
        retentions.append(float(record["retention_by_budget"][str(chosen)]))
        saturated += int(is_saturated)
    bound = empirical_bernstein(retentions, config.certification.alpha).serialize()
    saturation_fraction = saturated / len(records)
    assert config.m_prime_sweep is not None
    eligible = bool(
        bound["lower_bound"] >= config.m_prime_sweep.tune_lower_bound_target
        and saturation_fraction
        <= config.m_prime_sweep.max_saturation_fraction
    )
    return {
        "threshold": float(threshold),
        "mean_candidate_budget": float(np.mean(budgets)),
        "budget_distribution": {
            str(budget): budgets.count(budget) for budget in config.retrieval.m_grid
        },
        "saturated_n": saturated,
        "saturation_fraction": saturation_fraction,
        "tune_retention_bound": bound,
        "eligible": eligible,
    }


def _evaluate_dimension(
    config: HarnessConfig,
    *,
    m_prime: int,
    tune_query_ids: np.ndarray,
    tune_splits: np.ndarray,
    corpus_ids: np.ndarray,
    corpus: np.ndarray,
    tune_queries: np.ndarray,
) -> Dict[str, Any]:
    candidate_config = replace(
        config, retrieval=replace(config.retrieval, m_prime=int(m_prime))
    )
    matrix = dense_gaussian_projection(
        m_prime, config.synthetic.dimension, config.seeds.projection
    )
    projected_corpus = project_rows(corpus, matrix)
    projected_tune_queries = project_rows(tune_queries, matrix)
    records = _build_base_records(
        candidate_config,
        query_ids=tune_query_ids,
        splits=tune_splits,
        corpus_ids=corpus_ids,
        corpus=corpus,
        queries=tune_queries,
        projected_corpus=projected_corpus,
        projected_queries=projected_tune_queries,
    )
    _validate_tune_only(records)
    fixed = _tune_fixed_baseline(records, candidate_config)
    predictions = _raw_prediction_cache(records, candidate_config)
    assert config.m_prime_sweep is not None
    thresholds = [
        _evaluate_threshold(
            records,
            predictions,
            threshold=threshold,
            config=candidate_config,
        )
        for threshold in config.m_prime_sweep.threshold_grid
    ]
    eligible = [value for value in thresholds if value["eligible"]]
    selected_threshold = (
        min(
            eligible,
            key=lambda value: (
                value["mean_candidate_budget"],
                value["saturation_fraction"],
                -value["tune_retention_bound"]["lower_bound"],
                -value["threshold"],
            ),
        )
        if eligible
        else None
    )
    fixed_budget = fixed["smallest_tune_qualified_fixed_budget"]
    candidate_saving = (
        None
        if selected_threshold is None or fixed_budget is None
        else 1.0
        - selected_threshold["mean_candidate_budget"] / float(fixed_budget)
    )
    return {
        "m_prime": int(m_prime),
        "projection_seed": config.seeds.projection,
        "tune_query_count": len(records),
        "tune_query_id_hash": stable_id_hash(
            [str(record["query_id"]) for record in records]
        ),
        "fixed_baseline": fixed,
        "thresholds": thresholds,
        "selected_threshold": selected_threshold,
        "tune_candidate_saving": candidate_saving,
        "eligible": selected_threshold is not None and fixed_budget is not None,
    }


def _selection_report(result: Mapping[str, Any]) -> str:
    selected = result["selected"]
    cert = result["fresh_certification"]
    lines = [
        "# Tune-only global m_prime sweep",
        "",
        "All dimension and threshold decisions used `query_tune` only. The frozen "
        "selection artifacts were written before the fresh `query_cert` evaluation.",
        "",
        "| m_prime | eligible | threshold | tune lower bound | mean M | fixed M | saving | saturation |",
        "| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in result["candidates"]:
        threshold = candidate["selected_threshold"]
        if threshold is None:
            threshold_text = lower_text = mean_text = saturation_text = "—"
        else:
            threshold_text = f"{threshold['threshold']:.4f}"
            lower_text = f"{threshold['tune_retention_bound']['lower_bound']:.4f}"
            mean_text = f"{threshold['mean_candidate_budget']:.4f}"
            saturation_text = f"{threshold['saturation_fraction']:.4f}"
        fixed_budget = candidate["fixed_baseline"][
            "smallest_tune_qualified_fixed_budget"
        ]
        saving = candidate["tune_candidate_saving"]
        lines.append(
            f"| {candidate['m_prime']} | {'yes' if candidate['eligible'] else 'no'} | "
            f"{threshold_text} | {lower_text} | {mean_text} | "
            f"{fixed_budget if fixed_budget is not None else '—'} | "
            f"{f'{saving:.4%}' if saving is not None else '—'} | {saturation_text} |"
        )
    lines.extend(
        [
            "",
            "## Frozen choice",
            "",
            f"- `m_prime`: {selected['m_prime']}",
            f"- Tri-Predict threshold: {selected['threshold']}",
            f"- tune candidate saving: {selected['tune_candidate_saving']:.4%}",
            f"- selection fingerprint: `{result['selection_fingerprint']}`",
            "",
            "## Fresh certification",
            "",
            f"- decision: {'PASS' if cert['passed'] else 'FAIL'}",
            f"- mean retention: {cert['mean']:.6f}",
            f"- lower bound: {cert['lower_bound']:.6f}",
            f"- target: {cert['target']:.6f}",
            f"- n: {cert['n']}",
            f"- mean candidate budget: {result['fresh_cert_metrics']['mean_candidate_budget']:.6f}",
            f"- smallest certified fixed budget: {result['fresh_cert_metrics']['smallest_certified_fixed_budget']}",
            f"- candidate saving: {result['fresh_cert_metrics']['candidate_saving']}",
            f"- saturation: {result['fresh_cert_metrics']['saturated_n']}/{cert['n']}",
            "",
            "The certification result is terminal for this frozen selection: a fail is "
            "reported as a fail and does not trigger retuning.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_mprime_sweep(config: HarnessConfig, output_dir: Path) -> Dict[str, Path]:
    if config.m_prime_sweep is None:
        raise ValueError("config must contain m_prime_sweep")
    if config.synthetic.query_tune < 2:
        raise ValueError("m_prime sweep requires at least two tune queries")
    if config.tri_predict.fit_safety_correction:
        raise ValueError(
            "m_prime sweep currently requires tri_predict.fit_safety_correction=false"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty sweep directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_synthetic_dataset(config.synthetic, config.seeds.data)
    corpus = normalize_rows(dataset.corpus.vectors)
    tune_indices = dataset.split_indices()["query_tune"]
    tune_splits = dataset.splits[tune_indices]
    tune_queries = normalize_rows(dataset.queries.vectors[tune_indices])
    if not np.all(tune_splits == "query_tune"):
        raise AssertionError("tune slice contains a forbidden split")

    candidates = [
        _evaluate_dimension(
            config,
            m_prime=m_prime,
            tune_query_ids=dataset.queries.ids[tune_indices],
            tune_splits=tune_splits,
            corpus_ids=dataset.corpus.ids,
            corpus=corpus,
            tune_queries=tune_queries,
        )
        for m_prime in config.m_prime_sweep.candidates
    ]
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        write_json(
            output_dir / "selection_failed.json",
            {
                "schema_version": 1,
                "status": "no_eligible_tune_candidate",
                "selection_rule": SELECTION_RULE,
                "source_config_fingerprint": config.config_fingerprint,
                "candidates": candidates,
            },
        )
        raise RuntimeError("no m_prime candidate met the predeclared tune criteria")
    winner = max(
        eligible,
        key=lambda candidate: (
            candidate["tune_candidate_saving"],
            -candidate["selected_threshold"]["saturation_fraction"],
            candidate["selected_threshold"]["tune_retention_bound"]["lower_bound"],
            candidate["m_prime"],
        ),
    )
    selected = {
        "m_prime": winner["m_prime"],
        "threshold": winner["selected_threshold"]["threshold"],
        "tune_candidate_saving": winner["tune_candidate_saving"],
        "mean_candidate_budget": winner["selected_threshold"][
            "mean_candidate_budget"
        ],
        "saturation_fraction": winner["selected_threshold"]["saturation_fraction"],
        "tune_retention_bound": winner["selected_threshold"][
            "tune_retention_bound"
        ],
        "smallest_tune_qualified_fixed_budget": winner["fixed_baseline"][
            "smallest_tune_qualified_fixed_budget"
        ],
    }
    selection: Dict[str, Any] = {
        "schema_version": 1,
        "status": "frozen_before_certification",
        "data_scope": "query_tune_only",
        "source_config_fingerprint": config.config_fingerprint,
        "data_seed": config.seeds.data,
        "projection_seed": config.seeds.projection,
        "tune_query_count": len(tune_indices),
        "tune_query_id_hash": candidates[0]["tune_query_id_hash"],
        "selection_alpha": config.certification.alpha,
        "selection_rule": SELECTION_RULE,
        "selected": selected,
        "candidates": candidates,
    }
    selection["selection_fingerprint"] = fingerprint(selection)

    selected_raw = copy.deepcopy(config.raw)
    selected_raw["run_name"] = f"{config.run_name}_selected"
    selected_raw["retrieval"]["m_prime"] = selected["m_prime"]
    selected_raw.setdefault(
        "tri_predict",
        {
            "target": config.tri_predict.target,
            "max_rank_samples": config.tri_predict.max_rank_samples,
            "fit_safety_correction": config.tri_predict.fit_safety_correction,
            "safety_quantile": config.tri_predict.safety_quantile,
        },
    )
    selected_raw["tri_predict"]["target"] = selected["threshold"]
    selection_path = output_dir / "selection.json"
    selected_config_path = output_dir / "selected_config.json"
    write_json(selection_path, selection)
    write_json(selected_config_path, selected_raw)

    # This is the first call in the sweep that evaluates query_cert. Both frozen
    # artifacts above already exist and are never modified after this point.
    selected_config = load_config(selected_config_path)
    run_dir = output_dir / "selected_run"
    run_harness(selected_config, run_dir)
    certificate = json.loads(
        (run_dir / "tri_predict_certification.json").read_text(encoding="utf-8")
    )
    aggregates = json.loads(
        (run_dir / "aggregates.json").read_text(encoding="utf-8")
    )
    tri_cert_aggregate = aggregates["tri_predict"]["query_cert"]
    comparison = aggregates["tri_predict_certification_comparison"]
    final_result: Dict[str, Any] = {
        "schema_version": 1,
        "status": "fresh_certification_complete",
        "selection_written_before_certification": True,
        "selection_fingerprint": selection["selection_fingerprint"],
        "selected": selected,
        "candidates": candidates,
        "fresh_certification": certificate,
        "fresh_cert_metrics": {
            "mean_candidate_budget": tri_cert_aggregate["budget"]["mean"],
            "budget_distribution": tri_cert_aggregate["budget"]["distribution"],
            "saturated_n": tri_cert_aggregate["policy_status"]["saturated_n"],
            "smallest_certified_fixed_budget": comparison[
                "smallest_certified_fixed_budget"
            ],
            "candidate_saving": comparison["adaptive_candidate_saving"],
        },
        "selected_run_config_fingerprint": selected_config.config_fingerprint,
        "selected_policy_fingerprint": certificate["policy_fingerprint"],
    }
    result_path = output_dir / "sweep_result.json"
    report_path = output_dir / "sweep_report.md"
    write_json(result_path, final_result)
    report_path.write_text(_selection_report(final_result), encoding="utf-8")
    return {
        "selection.json": selection_path,
        "selected_config.json": selected_config_path,
        "sweep_result.json": result_path,
        "sweep_report.md": report_path,
        "selected_run": run_dir,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    paths = run_mprime_sweep(config, args.output)
    print(f"completed tune-only m_prime sweep: {paths['sweep_report.md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
