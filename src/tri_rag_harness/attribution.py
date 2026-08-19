"""Synthetic attribution of pilot-LID, rank-model, and remaining mean-field error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .certification import empirical_bernstein
from .config import HarnessConfig, load_config
from .embeddings import normalize_rows
from .indexes import ExactSquaredL2Index
from .projection import dense_gaussian_projection, project_rows
from .run import _build_base_records
from .synthetic import generate_synthetic_dataset
from .tri_predict import actual_distance_retention_grid, tri_predict_retention_grid
from .utils import write_json


MODES = ("pilot_lid_rank_model", "oracle_lid_rank_model", "actual_distance_beta")


def _choose_budget(
    predictions: Mapping[int, float], budgets: Sequence[int], threshold: float
) -> tuple[int, bool]:
    passing = [budget for budget in budgets if predictions[budget] >= threshold]
    if passing:
        return int(passing[0]), False
    return int(budgets[-1]), True


def _mode_aggregate(
    records: List[Mapping[str, Any]], *, mode: str, budgets: Sequence[int], alpha: float
) -> Dict[str, Any]:
    errors = []
    chosen_budgets = []
    realized = []
    saturated = 0
    for record in records:
        mode_record = record["modes"][mode]
        predictions = mode_record["predictions"]
        for budget in budgets:
            errors.append(
                float(predictions[str(budget)])
                - float(record["realized_retention_by_budget"][str(budget)])
            )
        chosen_budgets.append(int(mode_record["chosen_m"]))
        realized.append(float(mode_record["realized_retention"]))
        saturated += int(bool(mode_record["saturated"]))
    error_values = np.asarray(errors, dtype=np.float64)
    bound = empirical_bernstein(realized, alpha)
    return {
        "n_queries": len(records),
        "n_query_budget_cells": len(errors),
        "calibration": {
            "signed_bias_predicted_minus_realized": float(np.mean(error_values)),
            "mean_absolute_error": float(np.mean(np.abs(error_values))),
            "root_mean_squared_error": float(
                np.sqrt(np.mean(error_values * error_values))
            ),
        },
        "selection": {
            "mean_m": float(np.mean(chosen_budgets)),
            "mean_realized_retention": float(np.mean(realized)),
            "empirical_bernstein_lower_bound": bound.lower_bound,
            "saturated_n": saturated,
            "budget_distribution": {
                str(budget): chosen_budgets.count(budget) for budget in budgets
            },
        },
    }


def run_attribution(config: HarnessConfig, output_dir: Path) -> Dict[str, Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty attribution directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_dataset(config.synthetic, config.seeds.data)
    corpus = normalize_rows(dataset.corpus.vectors)
    queries = normalize_rows(dataset.queries.vectors)
    projection = dense_gaussian_projection(
        config.retrieval.m_prime,
        config.synthetic.dimension,
        config.seeds.projection,
    )
    projected_corpus = project_rows(corpus, projection)
    projected_queries = project_rows(queries, projection)
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
    full_original = ExactSquaredL2Index(
        dataset.corpus.ids, corpus, batch_size=config.retrieval.batch_size
    ).search(queries, len(corpus))
    budgets = config.retrieval.m_grid
    threshold = config.tri_predict.target
    records: List[Dict[str, Any]] = []
    for row, base in enumerate(base_records):
        pilot_predictions = tri_predict_retention_grid(
            lid=float(base["pilot_lid"]["clipped"]),
            m_prime=config.retrieval.m_prime,
            k_gt=config.retrieval.k_gt,
            budgets=budgets,
            corpus_size=len(corpus),
            max_rank_samples=config.tri_predict.max_rank_samples,
        )
        oracle_predictions = tri_predict_retention_grid(
            lid=float(base["oracle_lid"]["clipped"]),
            m_prime=config.retrieval.m_prime,
            k_gt=config.retrieval.k_gt,
            budgets=budgets,
            corpus_size=len(corpus),
            max_rank_samples=config.tri_predict.max_rank_samples,
        )
        actual_predictions = actual_distance_retention_grid(
            sorted_squared_distances=full_original.squared_distances[row],
            m_prime=config.retrieval.m_prime,
            k_gt=config.retrieval.k_gt,
            budgets=budgets,
        )
        mode_predictions = {
            "pilot_lid_rank_model": pilot_predictions,
            "oracle_lid_rank_model": oracle_predictions,
            "actual_distance_beta": actual_predictions,
        }
        mode_records: Dict[str, Any] = {}
        for mode, predictions in mode_predictions.items():
            chosen, saturated = _choose_budget(predictions, budgets, threshold)
            mode_records[mode] = {
                "predictions": {
                    str(budget): float(predictions[budget]) for budget in budgets
                },
                "chosen_m": chosen,
                "saturated": saturated,
                "realized_retention": float(
                    base["retention_by_budget"][str(chosen)]
                ),
            }
        records.append(
            {
                "query_id": base["query_id"],
                "split": base["split"],
                "pilot_lid": base["pilot_lid"],
                "oracle_lid": base["oracle_lid"],
                "realized_retention_by_budget": base["retention_by_budget"],
                "modes": mode_records,
            }
        )
    by_split = {
        split: [record for record in records if record["split"] == split]
        for split in ("query_tune", "query_cert", "query_test")
    }
    aggregates = {
        split: {
            mode: _mode_aggregate(
                split_records,
                mode=mode,
                budgets=budgets,
                alpha=config.certification.alpha,
            )
            for mode in MODES
        }
        for split, split_records in by_split.items()
    }
    attribution = {}
    for split, split_aggregates in aggregates.items():
        pilot_mae = split_aggregates["pilot_lid_rank_model"]["calibration"][
            "mean_absolute_error"
        ]
        oracle_mae = split_aggregates["oracle_lid_rank_model"]["calibration"][
            "mean_absolute_error"
        ]
        actual_mae = split_aggregates["actual_distance_beta"]["calibration"][
            "mean_absolute_error"
        ]
        attribution[split] = {
            "oracle_minus_pilot_mae": oracle_mae - pilot_mae,
            "pilot_minus_actual_beta_mae": pilot_mae - actual_mae,
            "oracle_minus_actual_beta_mae": oracle_mae - actual_mae,
            "remaining_mae_with_actual_beta": actual_mae,
            "pilot_lid_is_primary_failure": bool(
                oracle_mae + 1e-12 < pilot_mae
                and actual_mae >= 0.5 * pilot_mae
            ),
        }
    artifact = {
        "schema_version": 1,
        "status": "diagnostic_only_not_deployable",
        "config_fingerprint": config.config_fingerprint,
        "prediction_threshold": threshold,
        "modes": {
            "pilot_lid_rank_model": "deployable pilot LID plus LID rank-distance model",
            "oracle_lid_rank_model": "diagnostic exact-neighbor LID plus the same rank model",
            "actual_distance_beta": "full original-distance ratios plus the same orthogonal mean-field aggregation",
        },
        "aggregates": aggregates,
        "attribution": attribution,
        "interpretation_rule": (
            "pilot-vs-oracle isolates LID-estimator substitution; oracle-vs-actual-beta "
            "tests the scalar LID rank model; actual-beta residual retains orthogonality, "
            "structural, conditional-independence, and mean-field error"
        ),
    }
    write_json(output_dir / "attribution.json", artifact)
    with (output_dir / "attribution_per_query.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    return {
        "attribution.json": output_dir / "attribution.json",
        "attribution_per_query.jsonl": output_dir / "attribution_per_query.jsonl",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = run_attribution(load_config(args.config), args.output)
    print(f"completed attribution: {paths['attribution.json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
