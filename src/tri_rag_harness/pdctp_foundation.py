from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .certification import empirical_bernstein
from .embeddings import normalize_rows
from .indexes import ExactSquaredL2Index
from .lid import estimate_lid_from_squared_distances
from .pdctp_calibration import (
    BudgetResidualRecord,
    LIDCalibrationRecord,
    PilotLIDCalibrator,
    TriBudgetResidualCalibrator,
)
from .pdctp_config import PDCTPFoundationConfig, load_pdctp_foundation_config
from .pdctp_features import (
    PilotDistanceFeatureExtractor,
    PilotDistanceObservation,
    PilotFeatureVector,
    stable_sort_pilot_distances,
)
from .pdctp_policies import (
    CalibratedTriPredictPolicy,
    FixedPDCTPPolicy,
    MonotonePDCTPPolicy,
    PDCTPDecisionInput,
    PDCTPDecisionPolicy,
    RawTriPredictPDCTPPolicy,
    validate_policy_suite,
)
from .pdctp_protocol import (
    FIVE_ROLES,
    FiveRoleAssignments,
    FiveRoleProtocolGuard,
)
from .pdctp_statistics import (
    bonferroni_allocation,
    make_paired_bound,
    make_power_plan,
    validate_paired_bound,
)
from .policies import MonotoneBinnedPolicy, TriPredictPolicy
from .projection import dense_gaussian_projection, project_rows
from .utils import array_fingerprint, fingerprint, write_json


class PDCTPFoundationError(RuntimeError):
    pass


def _generate_synthetic(config: PDCTPFoundationConfig) -> Dict[str, Any]:
    synthetic = config.synthetic
    rng = np.random.default_rng(config.data_seed)
    centers = normalize_rows(rng.normal(size=(synthetic.cluster_count, synthetic.dimension)))
    docs_per_cluster = synthetic.corpus_size // synthetic.cluster_count
    corpus_vectors = []
    corpus_ids = []
    corpus_clusters = []
    for cluster in range(synthetic.cluster_count):
        values = centers[cluster] + rng.normal(
            scale=synthetic.corpus_noise,
            size=(docs_per_cluster, synthetic.dimension),
        )
        corpus_vectors.append(values)
        for local_row in range(docs_per_cluster):
            corpus_ids.append(f"pdctp-doc-{cluster:02d}-{local_row:04d}")
            corpus_clusters.append(cluster)

    query_vectors = []
    query_ids = []
    query_roles = []
    query_clusters = []
    global_row = 0
    for role in FIVE_ROLES:
        count = synthetic.role_counts[role]
        for local_row in range(count):
            cluster = (global_row * 3 + local_row) % synthetic.cluster_count
            difficulty = (local_row + 0.5) / count
            scale = synthetic.query_noise_min + difficulty * (
                synthetic.query_noise_max - synthetic.query_noise_min
            )
            query_vectors.append(
                centers[cluster]
                + rng.normal(scale=scale, size=synthetic.dimension)
            )
            query_ids.append(f"pdctp-{role}-{local_row:05d}")
            query_roles.append(role)
            query_clusters.append(cluster)
            global_row += 1
    if set(corpus_ids).intersection(query_ids):
        raise AssertionError("synthetic PDCTP queries must remain external")
    return {
        "corpus_ids": np.asarray(corpus_ids, dtype=str),
        "corpus": normalize_rows(np.vstack(corpus_vectors)),
        "corpus_clusters": np.asarray(corpus_clusters, dtype=np.int64),
        "query_ids": np.asarray(query_ids, dtype=str),
        "queries": normalize_rows(np.vstack(query_vectors)),
        "query_roles": np.asarray(query_roles, dtype=str),
        "query_clusters": np.asarray(query_clusters, dtype=np.int64),
    }


def _jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in records
    )
    path.write_text(payload, encoding="utf-8")


def _retention(gt_rows: np.ndarray, candidate_rows: np.ndarray) -> float:
    return len(set(gt_rows.tolist()).intersection(candidate_rows.tolist())) / len(gt_rows)


def _reranked_rows(
    candidate_rows: np.ndarray,
    *,
    corpus: np.ndarray,
    corpus_ids: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    differences = corpus[candidate_rows] - query
    squared = np.einsum("ij,ij->i", differences, differences)
    order = np.lexsort((corpus_ids[candidate_rows], squared))
    return candidate_rows[order]


def _public_base_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _scan_role(
    config: PDCTPFoundationConfig,
    *,
    role: str,
    label_access: bool,
    dataset: Mapping[str, Any],
    projected_queries: np.ndarray,
    original_index: ExactSquaredL2Index,
    projected_index: ExactSquaredL2Index,
    extractor: PilotDistanceFeatureExtractor,
) -> List[Dict[str, Any]]:
    retrieval = config.retrieval
    corpus = dataset["corpus"]
    corpus_ids = dataset["corpus_ids"]
    query_indices = np.flatnonzero(dataset["query_roles"] == role)
    results: List[Dict[str, Any]] = []
    for query_row in query_indices:
        query = dataset["queries"][query_row]
        projected = projected_index.search(
            projected_queries[query_row : query_row + 1], retrieval.m_grid[-1]
        )
        expanded_rows = projected.rows[0]
        pilot_rows = expanded_rows[: retrieval.m_pilot]
        pilot_differences = corpus[pilot_rows] - query
        pilot_original_sq = np.einsum(
            "ij,ij->i", pilot_differences, pilot_differences
        )
        pilot_projected_sq = projected.squared_distances[0, : retrieval.m_pilot]
        _, sorted_original_sq, sorted_projected_sq = stable_sort_pilot_distances(
            corpus_ids[pilot_rows], pilot_original_sq, pilot_projected_sq
        )
        lid = estimate_lid_from_squared_distances(
            sorted_original_sq,
            s_lid=retrieval.s_lid,
            min_neighbors=retrieval.min_lid_neighbors,
            clip_min=config.calibration.lid_output_domain[0],
            clip_max=config.calibration.lid_output_domain[1],
            duplicate_tolerance=config.feature_spec.duplicate_tolerance,
            fallback=config.calibration.lid_fallback,
        )
        observation = PilotDistanceObservation.from_arrays(
            sorted_original_sq,
            sorted_projected_sq,
            pilot_lid=lid.clipped,
            pilot_lid_valid=lid.valid,
            pilot_lid_failure_reason=lid.reason,
            valid_distance_count=lid.valid_distance_count,
        )
        features = extractor.extract(observation)
        base: Dict[str, Any] = {
            "query_id": str(dataset["query_ids"][query_row]),
            "role": role,
            "labels_accessed": bool(label_access),
            "pilot_lid": lid.clipped,
            "pilot_lid_valid": lid.valid,
            "pilot_lid_failure_reason": lid.reason,
            "features": features.serialize(),
            "projected_scan_count": 1,
            "projected_distance_count": len(corpus),
            "projected_cache_budget": retrieval.m_grid[-1],
            "pilot_is_cached_prefix": bool(
                np.array_equal(pilot_rows, expanded_rows[: retrieval.m_pilot])
            ),
            "pilot_row_hash": array_fingerprint(pilot_rows),
            "expanded_row_hash": array_fingerprint(expanded_rows),
            "_features_obj": features,
            "_expanded_rows": expanded_rows,
            "_query_row": int(query_row),
        }
        if label_access:
            oracle = original_index.search(
                query[None, :], max(retrieval.k_gt, retrieval.s_lid)
            )
            oracle_lid = estimate_lid_from_squared_distances(
                oracle.squared_distances[0, : retrieval.s_lid],
                s_lid=retrieval.s_lid,
                min_neighbors=retrieval.min_lid_neighbors,
                clip_min=config.calibration.lid_output_domain[0],
                clip_max=config.calibration.lid_output_domain[1],
                duplicate_tolerance=config.feature_spec.duplicate_tolerance,
                fallback=config.calibration.lid_fallback,
            )
            gt_rows = oracle.rows[0, : retrieval.k_gt]
            relevant_rows = np.flatnonzero(
                dataset["corpus_clusters"] == dataset["query_clusters"][query_row]
            )
            retention_by_budget: Dict[str, float] = {}
            candidate_evidence_by_budget: Dict[str, float] = {}
            final_evidence_by_budget: Dict[str, float] = {}
            for budget in retrieval.m_grid:
                candidate_rows = expanded_rows[:budget]
                reranked = _reranked_rows(
                    candidate_rows,
                    corpus=corpus,
                    corpus_ids=corpus_ids,
                    query=query,
                )
                retention_by_budget[str(budget)] = _retention(gt_rows, candidate_rows)
                candidate_evidence_by_budget[str(budget)] = len(
                    set(relevant_rows.tolist()).intersection(candidate_rows.tolist())
                ) / len(relevant_rows)
                final_evidence_by_budget[str(budget)] = len(
                    set(relevant_rows.tolist()).intersection(
                        reranked[: retrieval.k_ctx].tolist()
                    )
                ) / len(relevant_rows)
            base.update(
                {
                    "oracle_lid": oracle_lid.clipped,
                    "oracle_lid_valid": oracle_lid.valid,
                    "retention_by_budget": retention_by_budget,
                    "candidate_evidence_by_budget": candidate_evidence_by_budget,
                    "final_evidence_by_budget": final_evidence_by_budget,
                    "_gt_rows": gt_rows,
                }
            )
        results.append(base)
    return results


def _required_budget(
    retention_by_budget: Mapping[str, float],
    grid: Sequence[int],
    level: float,
) -> int:
    for budget in grid:
        if float(retention_by_budget[str(budget)]) >= level:
            return int(budget)
    if float(retention_by_budget[str(grid[-1])]) != 1.0:
        raise AssertionError("complete-corpus candidate set must have unit retention")
    return int(grid[-1])


def _evaluate_policy(
    records: Sequence[Mapping[str, Any]], policy: PDCTPDecisionPolicy
) -> List[Dict[str, Any]]:
    outputs = []
    for record in records:
        features = record["_features_obj"]
        decision = policy.choose(
            PDCTPDecisionInput(
                features=features,
                pilot_lid=float(record["pilot_lid"]),
                pilot_lid_valid=bool(record["pilot_lid_valid"]),
            )
        )
        output: Dict[str, Any] = {
            "query_id": record["query_id"],
            "role": record["role"],
            "policy_name": decision.policy_name,
            "policy_version": decision.policy_version,
            "chosen_m": decision.budget,
            "raw_m": decision.raw_budget,
            "input_lid": decision.input_lid,
            "calibrated_lid": decision.calibrated_lid,
            "residual_correction": decision.residual_correction,
            "used_fallback": decision.used_fallback,
            "saturated": decision.saturated,
            "failure_reason": decision.failure_reason,
            "projected_scan_count": record["projected_scan_count"],
            "pilot_is_cached_prefix": record["pilot_is_cached_prefix"],
            "labels_accessed": record["labels_accessed"],
        }
        if record["labels_accessed"]:
            budget_key = str(decision.budget)
            output.update(
                {
                    "embedding_retention": record["retention_by_budget"][budget_key],
                    "candidate_evidence_recall": record[
                        "candidate_evidence_by_budget"
                    ][budget_key],
                    "final_evidence_recall": record["final_evidence_by_budget"][
                        budget_key
                    ],
                }
            )
        outputs.append(output)
    return outputs


def _candidate_bundle(name: str, artifacts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "name": name,
        "schema_version": 1,
        "candidate_fingerprints": [artifact["fingerprint"] for artifact in artifacts],
        "candidate_count": len(artifacts),
        "fit_role": "query_cal",
    }
    body["fingerprint"] = fingerprint(body)
    return body


def _fit_residual_candidates(
    config: PDCTPFoundationConfig,
    cal_records: Sequence[Mapping[str, Any]],
    raw_policy: TriPredictPolicy,
    *,
    lid_calibrators: Sequence[PilotLIDCalibrator],
    lid_source: str,
) -> List[Tuple[TriBudgetResidualCalibrator, Dict[str, Any]]]:
    candidates: List[Tuple[TriBudgetResidualCalibrator, Dict[str, Any]]] = []
    raw_artifact = raw_policy.serialize()
    calibrator_options: Sequence[Tuple[str, Any]]
    if lid_source == "raw_pilot_lid":
        calibrator_options = (("raw", None),)
    else:
        calibrator_options = tuple(
            (calibrator.fingerprint, calibrator) for calibrator in lid_calibrators
        )
    for lid_key, lid_calibrator in calibrator_options:
        raw_budgets: Dict[str, int] = {}
        for record in cal_records:
            if lid_calibrator is None:
                lid_value = float(record["pilot_lid"])
                lid_valid = bool(record["pilot_lid_valid"] and record["_features_obj"].valid)
            else:
                predicted = lid_calibrator.predict(record["_features_obj"])
                lid_value = predicted.value
                lid_valid = predicted.valid
            raw_budgets[str(record["query_id"])] = raw_policy.choose(
                lid_value, lid_valid
            ).budget
        for level in config.calibration.residual_training_levels:
            fit_rows = [
                BudgetResidualRecord(
                    query_id=str(record["query_id"]),
                    role="query_cal",
                    features=record["_features_obj"],
                    raw_budget=raw_budgets[str(record["query_id"])],
                    required_budget=_required_budget(
                        record["retention_by_budget"], config.retrieval.m_grid, level
                    ),
                    training_level=level,
                )
                for record in cal_records
            ]
            for quantile in config.calibration.residual_quantiles:
                for regularization in config.calibration.residual_regularization_grid:
                    for safety_offset in config.calibration.safety_offsets:
                        fitted = TriBudgetResidualCalibrator.fit(
                            fit_rows,
                            quantile=quantile,
                            regularization=regularization,
                            safety_offset=safety_offset,
                            grid=config.retrieval.m_grid,
                            minimum_budget=max(
                                config.retrieval.k_gt, config.retrieval.m_pilot
                            ),
                            fallback_budget=config.retrieval.m_grid[-1],
                            raw_policy_fingerprint=raw_artifact["fingerprint"],
                            anchor_lid_source=lid_source,
                        )
                        candidates.append(
                            (
                                fitted,
                                {
                                    "raw_policy_fingerprint": raw_artifact["fingerprint"],
                                    "raw_tri_target": raw_policy.target,
                                    "lid_calibrator_fingerprint": lid_key,
                                    "training_level": level,
                                    "quantile": quantile,
                                    "regularization": regularization,
                                    "safety_offset": safety_offset,
                                    "fingerprint": fitted.fingerprint,
                                },
                            )
                        )
    return candidates


def _select_fixed_reference(
    config: PDCTPFoundationConfig, tune_records: Sequence[Mapping[str, Any]]
) -> int:
    for budget in config.retrieval.m_grid:
        bound = empirical_bernstein(
            [float(row["retention_by_budget"][str(budget)]) for row in tune_records],
            alpha=0.05,
        )
        if bound.lower_bound >= config.selection.retention_lower_bound_target:
            return int(budget)
    raise PDCTPFoundationError("no fixed reference meets the frozen tune constraint")


def _select_full_candidate(
    config: PDCTPFoundationConfig,
    tune_records: Sequence[Mapping[str, Any]],
    fixed_budget: int,
    candidates: Sequence[
        Tuple[CalibratedTriPredictPolicy, Mapping[str, Any]]
    ],
) -> Tuple[CalibratedTriPredictPolicy, Dict[str, Any], Dict[str, Any]]:
    fixed_candidate_evidence = float(
        np.mean(
            [
                row["candidate_evidence_by_budget"][str(fixed_budget)]
                for row in tune_records
            ]
        )
    )
    fixed_final_evidence = float(
        np.mean(
            [row["final_evidence_by_budget"][str(fixed_budget)] for row in tune_records]
        )
    )
    outcomes = []
    by_fingerprint: Dict[str, CalibratedTriPredictPolicy] = {}
    metadata_by_fingerprint: Dict[str, Dict[str, Any]] = {}
    for policy, metadata in candidates:
        policy_artifact = policy.serialize()
        policy_fingerprint = str(policy_artifact["fingerprint"])
        by_fingerprint[policy_fingerprint] = policy
        metadata_by_fingerprint[policy_fingerprint] = dict(metadata)
        evaluated = _evaluate_policy(tune_records, policy)
        retentions = [float(row["embedding_retention"]) for row in evaluated]
        retention_bound = empirical_bernstein(retentions, alpha=0.05)
        mean_candidate_evidence = float(
            np.mean([row["candidate_evidence_recall"] for row in evaluated])
        )
        mean_final_evidence = float(
            np.mean([row["final_evidence_recall"] for row in evaluated])
        )
        mean_budget = float(np.mean([row["chosen_m"] for row in evaluated]))
        eligible = bool(
            retention_bound.lower_bound
            >= config.selection.retention_lower_bound_target
            and mean_candidate_evidence - fixed_candidate_evidence
            >= -config.selection.candidate_evidence_noninferiority
            and mean_final_evidence - fixed_final_evidence
            >= -config.selection.final_evidence_noninferiority
        )
        coordinate_work = (
            (config.synthetic.corpus_size + config.synthetic.dimension)
            * config.retrieval.m_prime
            + config.synthetic.dimension * mean_budget
        )
        outcomes.append(
            {
                "policy_fingerprint": policy_fingerprint,
                "metadata": dict(metadata),
                "eligible": eligible,
                "retention_mean": retention_bound.mean,
                "retention_lower_bound": retention_bound.lower_bound,
                "candidate_evidence_mean": mean_candidate_evidence,
                "candidate_evidence_difference_vs_fixed": mean_candidate_evidence
                - fixed_candidate_evidence,
                "final_evidence_mean": mean_final_evidence,
                "final_evidence_difference_vs_fixed": mean_final_evidence
                - fixed_final_evidence,
                "mean_budget": mean_budget,
                "common_coordinate_work": coordinate_work,
            }
        )
    eligible_rows = [row for row in outcomes if row["eligible"]]
    if not eligible_rows:
        best_retention = max(row["retention_lower_bound"] for row in outcomes)
        best_candidate_gap = max(
            row["candidate_evidence_difference_vs_fixed"] for row in outcomes
        )
        best_final_gap = max(
            row["final_evidence_difference_vs_fixed"] for row in outcomes
        )
        raise PDCTPFoundationError(
            "no full PDCTP candidate meets the frozen synthetic tune constraints: "
            f"best_retention_lcb={best_retention:.6f}, "
            f"best_candidate_gap={best_candidate_gap:.6f}, "
            f"best_final_gap={best_final_gap:.6f}"
        )
    chosen = min(
        eligible_rows,
        key=lambda row: (
            row["common_coordinate_work"],
            row["mean_budget"],
            row["policy_fingerprint"],
        ),
    )
    selected_fingerprint = str(chosen["policy_fingerprint"])
    selection: Dict[str, Any] = {
        "name": "pdctp_synthetic_tune_selection",
        "schema_version": 1,
        "role": "query_tune",
        "fixed_reference_budget": fixed_budget,
        "selection_rule": [
            "retention_lower_bound",
            "candidate_evidence_noninferiority",
            "final_evidence_noninferiority",
            "minimum_common_coordinate_work",
            "lower_mean_budget",
            "canonical_fingerprint",
        ],
        "candidate_outcomes": outcomes,
        "selected_policy_fingerprint": selected_fingerprint,
        "selected_metadata": metadata_by_fingerprint[selected_fingerprint],
    }
    selection["fingerprint"] = fingerprint(selection)
    return (
        by_fingerprint[selected_fingerprint],
        metadata_by_fingerprint[selected_fingerprint],
        selection,
    )


def _hypotheses_artifact(config: PDCTPFoundationConfig) -> Dict[str, Any]:
    names = [str(row["name"]) for row in config.certification.hypotheses]
    body: Dict[str, Any] = {
        "name": "pdctp_frozen_certification_hypotheses",
        "schema_version": 1,
        "family_wise_method": "bonferroni",
        "family_wise_alpha": config.certification.family_wise_alpha,
        "alpha_allocation": bonferroni_allocation(
            names, config.certification.family_wise_alpha
        ),
        "hypotheses": list(config.certification.hypotheses),
    }
    body["fingerprint"] = fingerprint(body)
    return body


def _shuffled_tune_diagnostic(
    records: Sequence[Mapping[str, Any]],
    policy: PDCTPDecisionPolicy,
    *,
    seed: int,
    policy_fingerprint: str,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    donor_indices = rng.permutation(len(records))
    observed = _evaluate_policy(records, policy)
    shuffled_rows = []
    for target, donor_index in zip(records, donor_indices):
        donor = records[int(donor_index)]
        decision = policy.choose(
            PDCTPDecisionInput(
                features=donor["_features_obj"],
                pilot_lid=float(donor["pilot_lid"]),
                pilot_lid_valid=bool(donor["pilot_lid_valid"]),
            )
        )
        budget_key = str(decision.budget)
        shuffled_rows.append(
            {
                "query_id": target["query_id"],
                "donor_query_id": donor["query_id"],
                "role": "query_tune",
                "chosen_m": decision.budget,
                "embedding_retention": target["retention_by_budget"][budget_key],
                "candidate_evidence_recall": target[
                    "candidate_evidence_by_budget"
                ][budget_key],
                "final_evidence_recall": target["final_evidence_by_budget"][budget_key],
                "projected_scan_count": target["projected_scan_count"],
                "pilot_is_cached_prefix": target["pilot_is_cached_prefix"],
            }
        )

    def mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
        return float(np.mean([float(row[field]) for row in rows]))

    metrics = (
        "embedding_retention",
        "candidate_evidence_recall",
        "final_evidence_recall",
    )
    body: Dict[str, Any] = {
        "name": "pdctp_shuffled_pilot_profile_tune_diagnostic",
        "schema_version": 1,
        "role": "query_tune",
        "seed": int(seed),
        "policy_fingerprint": policy_fingerprint,
        "used_for_fit": False,
        "used_for_selection": False,
        "used_for_certification": False,
        "n": len(records),
        "observed": {
            "mean_budget": mean(observed, "chosen_m"),
            **{f"mean_{metric}": mean(observed, metric) for metric in metrics},
        },
        "shuffled": {
            "mean_budget": mean(shuffled_rows, "chosen_m"),
            **{f"mean_{metric}": mean(shuffled_rows, metric) for metric in metrics},
        },
        "records": shuffled_rows,
    }
    body["observed_minus_shuffled"] = {
        metric: body["observed"][f"mean_{metric}"]
        - body["shuffled"][f"mean_{metric}"]
        for metric in metrics
    }
    body["fingerprint"] = fingerprint(body)
    return body


def _make_certification_bounds(
    config: PDCTPFoundationConfig,
    evaluated: Mapping[str, Sequence[Mapping[str, Any]]],
    policy_fingerprints: Mapping[str, str],
    hypotheses: Mapping[str, Any],
) -> Dict[str, Any]:
    full = evaluated["full_pdctp"]
    ids = [str(row["query_id"]) for row in full]
    allocation = hypotheses["alpha_allocation"]
    by_name = {str(row["name"]): row for row in config.certification.hypotheses}
    bounds: Dict[str, Any] = {}
    for name, specification in by_name.items():
        comparison = str(specification["comparison"])
        metric = str(specification["metric"])
        if comparison == "zero_anchor":
            left = [float(row["embedding_retention"]) for row in full]
            right = [0.0] * len(left)
            right_fingerprint = "zero_anchor"
        else:
            comparator_key = {
                "fixed_reference": "fixed",
                "monotone_binned": "monotone",
                "raw_tri_predict": "raw_tri",
            }[comparison]
            comparator = evaluated[comparator_key]
            if metric == "candidate_evidence_recall":
                left = [float(row[metric]) for row in full]
                right = [float(row[metric]) for row in comparator]
            elif metric == "final_evidence_recall":
                left = [float(row[metric]) for row in full]
                right = [float(row[metric]) for row in comparator]
            elif metric == "normalized_candidate_budget":
                left = [
                    float(row["chosen_m"]) / config.synthetic.corpus_size
                    for row in full
                ]
                right = [
                    float(row["chosen_m"]) / config.synthetic.corpus_size
                    for row in comparator
                ]
            else:
                raise PDCTPFoundationError(f"unsupported paired metric {metric}")
            right_fingerprint = policy_fingerprints[comparator_key]
        artifact = make_paired_bound(
            ids,
            left,
            right,
            hypothesis=name,
            metric=metric,
            alpha=float(allocation[name]),
            difference_bounds=tuple(specification["difference_bounds"]),
            side=str(specification["side"]),
            margin=float(specification["margin"]),
            left_policy_fingerprint=policy_fingerprints["full_pdctp"],
            right_policy_fingerprint=right_fingerprint,
        )
        validate_paired_bound(artifact)
        bounds[name] = artifact
    body: Dict[str, Any] = {
        "name": "pdctp_synthetic_certification_family",
        "schema_version": 1,
        "role": "query_cert",
        "hypotheses_fingerprint": hypotheses["fingerprint"],
        "bounds": bounds,
        "all_passed": all(value["passed"] for value in bounds.values()),
        "synthetic_only": True,
    }
    body["fingerprint"] = fingerprint(body)
    return body


def run_pdctp_foundation(
    config: PDCTPFoundationConfig, output_dir: Path
) -> Dict[str, Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _generate_synthetic(config)
    matrix = dense_gaussian_projection(
        config.retrieval.m_prime,
        config.synthetic.dimension,
        config.projection_seed,
    )
    projected_corpus = project_rows(dataset["corpus"], matrix)
    projected_queries = project_rows(dataset["queries"], matrix)
    original_index = ExactSquaredL2Index(dataset["corpus_ids"], dataset["corpus"])
    projected_index = ExactSquaredL2Index(dataset["corpus_ids"], projected_corpus)
    extractor = PilotDistanceFeatureExtractor(config.feature_spec)
    ids_by_role = {
        role: tuple(
            str(value)
            for value in dataset["query_ids"][dataset["query_roles"] == role]
        )
        for role in FIVE_ROLES
    }
    text_groups = {
        query_id: f"synthetic-normalized::{query_id}"
        for ids in ids_by_role.values()
        for query_id in ids
    }
    assignments = FiveRoleAssignments(ids_by_role, text_groups)
    guard = FiveRoleProtocolGuard(assignments, config.config_fingerprint)

    guard.open_calibration(ids_by_role["query_cal"])
    cal_records = _scan_role(
        config,
        role="query_cal",
        label_access=True,
        dataset=dataset,
        projected_queries=projected_queries,
        original_index=original_index,
        projected_index=projected_index,
        extractor=extractor,
    )
    if any(not row["_features_obj"].valid for row in cal_records):
        raise PDCTPFoundationError("synthetic calibration unexpectedly produced invalid features")
    lid_candidates = [
        PilotLIDCalibrator.fit(
            [
                LIDCalibrationRecord(
                    query_id=str(row["query_id"]),
                    role="query_cal",
                    features=row["_features_obj"],
                    oracle_lid=float(row["oracle_lid"]),
                )
                for row in cal_records
            ],
            regularization=regularization,
            output_min=config.calibration.lid_output_domain[0],
            output_max=config.calibration.lid_output_domain[1],
            fallback=config.calibration.lid_fallback,
        )
        for regularization in config.calibration.lid_regularization_grid
    ]
    raw_policies = [
        TriPredictPolicy(
            corpus_size=config.synthetic.corpus_size,
            m_prime=config.retrieval.m_prime,
            k_gt=config.retrieval.k_gt,
            grid=config.retrieval.m_grid,
            target=target,
            max_rank_samples=config.retrieval.max_rank_samples,
        )
        for target in config.calibration.raw_tri_threshold_grid
    ]
    raw_policy_by_fingerprint = {
        policy.serialize()["fingerprint"]: policy for policy in raw_policies
    }
    full_residual_candidates = []
    raw_residual_candidates = []
    for candidate_raw_policy in raw_policies:
        full_residual_candidates.extend(
            _fit_residual_candidates(
                config,
                cal_records,
                candidate_raw_policy,
                lid_calibrators=lid_candidates,
                lid_source="calibrated_pilot_lid",
            )
        )
        raw_residual_candidates.extend(
            _fit_residual_candidates(
                config,
                cal_records,
                candidate_raw_policy,
                lid_calibrators=lid_candidates,
                lid_source="raw_pilot_lid",
            )
        )
    lid_bundle = _candidate_bundle(
        "pdctp_lid_calibrator_candidate_bundle",
        [candidate.serialize() for candidate in lid_candidates],
    )
    residual_bundle = _candidate_bundle(
        "pdctp_residual_calibrator_candidate_bundle",
        [candidate.serialize() for candidate, _ in full_residual_candidates]
        + [candidate.serialize() for candidate, _ in raw_residual_candidates],
    )
    guard.register_fit(
        "lid_calibrator",
        role="query_cal",
        ids=ids_by_role["query_cal"],
        artifact_fingerprint=lid_bundle["fingerprint"],
    )
    guard.register_fit(
        "residual_calibrator",
        role="query_cal",
        ids=ids_by_role["query_cal"],
        artifact_fingerprint=residual_bundle["fingerprint"],
    )

    tune_token = guard.open_tune_selection(ids_by_role["query_tune"])
    tune_records = _scan_role(
        config,
        role="query_tune",
        label_access=True,
        dataset=dataset,
        projected_queries=projected_queries,
        original_index=original_index,
        projected_index=projected_index,
        extractor=extractor,
    )
    fixed_budget = _select_fixed_reference(config, tune_records)
    lid_by_fingerprint = {candidate.fingerprint: candidate for candidate in lid_candidates}
    full_candidates = []
    for residual, metadata in full_residual_candidates:
        candidate_raw_policy = raw_policy_by_fingerprint[
            str(metadata["raw_policy_fingerprint"])
        ]
        lid_calibrator = lid_by_fingerprint[
            str(metadata["lid_calibrator_fingerprint"])
        ]
        full_candidates.append(
            (
                CalibratedTriPredictPolicy(
                    mode="full",
                    raw_reference=candidate_raw_policy,
                    minimum_budget=max(config.retrieval.k_gt, config.retrieval.m_pilot),
                    lid_calibrator=lid_calibrator,
                    residual_calibrator=residual,
                ),
                metadata,
            )
        )
    selected_full, selected_metadata, selection = _select_full_candidate(
        config, tune_records, fixed_budget, full_candidates
    )
    selected_lid = lid_by_fingerprint[
        str(selected_metadata["lid_calibrator_fingerprint"])
    ]
    selected_residual = selected_full.residual_calibrator
    assert selected_residual is not None
    residual_only_matches = [
        candidate
        for candidate, metadata in raw_residual_candidates
        if all(
            metadata[key] == selected_metadata[key]
            for key in (
                "raw_policy_fingerprint",
                "training_level",
                "quantile",
                "regularization",
                "safety_offset",
            )
        )
    ]
    if len(residual_only_matches) != 1:
        raise AssertionError("residual-only ablation must have one matched calibrator")
    selected_residual_only = residual_only_matches[0]
    selected_raw_policy = selected_full.raw_reference

    monotone = MonotoneBinnedPolicy.fit(
        [
            {
                "lid": row["pilot_lid"],
                "lid_valid": row["pilot_lid_valid"],
                "retention_by_budget": row["retention_by_budget"],
            }
            for row in tune_records
        ],
        grid=config.retrieval.m_grid,
        n_bins=4,
        target=config.selection.retention_lower_bound_target,
        safety_margin=0.0,
        fallback_budget=config.retrieval.m_grid[-1],
        feature_version="pilot_distance_features_v1",
    )
    minimum_budget = max(config.retrieval.k_gt, config.retrieval.m_pilot)
    policies: Dict[str, PDCTPDecisionPolicy] = {
        "fixed": FixedPDCTPPolicy(fixed_budget, config.retrieval.m_grid, minimum_budget),
        "monotone": MonotonePDCTPPolicy(monotone, minimum_budget=minimum_budget),
        "raw_tri": RawTriPredictPDCTPPolicy(
            selected_raw_policy, minimum_budget=minimum_budget
        ),
        "lid_only": CalibratedTriPredictPolicy(
            mode="lid_only",
            raw_reference=selected_raw_policy,
            minimum_budget=minimum_budget,
            lid_calibrator=selected_lid,
        ),
        "residual_only": CalibratedTriPredictPolicy(
            mode="residual_only",
            raw_reference=selected_raw_policy,
            minimum_budget=minimum_budget,
            residual_calibrator=selected_residual_only,
        ),
        "full_pdctp": selected_full,
    }
    validate_policy_suite(policies)
    policy_artifacts = {name: policy.serialize() for name, policy in policies.items()}
    policy_fingerprints = {
        name: str(artifact["fingerprint"]) for name, artifact in policy_artifacts.items()
    }
    suite_fingerprint = fingerprint(policy_artifacts)
    selection.pop("fingerprint", None)
    selection["frozen_policy_suite_fingerprint"] = suite_fingerprint
    selection["frozen_policy_fingerprints"] = policy_fingerprints
    selection["fingerprint"] = fingerprint(selection)
    guard.freeze_selection(tune_token, selection["fingerprint"])
    shuffled_diagnostic = _shuffled_tune_diagnostic(
        tune_records,
        selected_full,
        seed=config.selection.shuffled_profile_seed,
        policy_fingerprint=policy_fingerprints["full_pdctp"],
    )

    hypotheses = _hypotheses_artifact(config)
    power_plan = make_power_plan(
        config.certification.hypotheses,
        total_alpha=config.certification.family_wise_alpha,
    )
    guard.freeze_hypotheses(hypotheses["fingerprint"])
    cert_token = guard.open_certification(ids_by_role["query_cert"])
    cert_records = _scan_role(
        config,
        role="query_cert",
        label_access=True,
        dataset=dataset,
        projected_queries=projected_queries,
        original_index=original_index,
        projected_index=projected_index,
        extractor=extractor,
    )
    cert_evaluated = {
        name: _evaluate_policy(cert_records, policy) for name, policy in policies.items()
    }
    certification = _make_certification_bounds(
        config, cert_evaluated, policy_fingerprints, hypotheses
    )
    guard.close_certification(cert_token, certification["fingerprint"])

    latency_token = guard.open_latency(
        ids_by_role["query_latency"], labels_requested=False
    )
    latency_records = _scan_role(
        config,
        role="query_latency",
        label_access=False,
        dataset=dataset,
        projected_queries=projected_queries,
        original_index=original_index,
        projected_index=projected_index,
        extractor=extractor,
    )
    latency_evaluated = {
        name: _evaluate_policy(latency_records, policy) for name, policy in policies.items()
    }
    latency_result: Dict[str, Any] = {
        "name": "pdctp_label_free_latency_structural_dry_run",
        "schema_version": 1,
        "role": "query_latency",
        "labels_accessed": False,
        "measured_latency_claim": False,
        "reason": "network_free_foundation_only",
        "records": latency_evaluated,
    }
    latency_result["fingerprint"] = fingerprint(latency_result)
    guard.close_latency(latency_token, latency_result["fingerprint"])

    guard.open_test(ids_by_role["query_test"])
    test_records = _scan_role(
        config,
        role="query_test",
        label_access=True,
        dataset=dataset,
        projected_queries=projected_queries,
        original_index=original_index,
        projected_index=projected_index,
        extractor=extractor,
    )
    test_evaluated = {
        name: _evaluate_policy(test_records, policy) for name, policy in policies.items()
    }

    split_artifact = assignments.serialize()
    manifest: Dict[str, Any] = {
        "name": "pdctp_network_free_foundation_manifest",
        "schema_version": 1,
        "config_fingerprint": config.config_fingerprint,
        "dataset": {
            "synthetic_only": True,
            "real_data_accessed": False,
            "network_accessed": False,
            "corpus_size": len(dataset["corpus"]),
            "query_size": len(dataset["queries"]),
            "queries_external": True,
            "corpus_embedding_fingerprint": array_fingerprint(dataset["corpus"]),
            "query_embedding_fingerprint": array_fingerprint(dataset["queries"]),
        },
        "projection": {
            "family": "dense_gaussian",
            "variance": f"1/{config.retrieval.m_prime}",
            "scale": f"1/sqrt({config.retrieval.m_prime})",
            "seed": config.projection_seed,
            "m_prime": config.retrieval.m_prime,
            "post_projection_normalized": False,
            "matrix_fingerprint": array_fingerprint(matrix),
        },
        "search": {
            "distance": "squared_l2",
            "projected_scan_per_query": 1,
            "pilot_and_expansion_share_scan": True,
            "exact_original_reranking": True,
        },
        "split_fingerprint": split_artifact["fingerprint"],
        "feature_spec_fingerprint": config.feature_spec.fingerprint,
        "lid_candidate_bundle_fingerprint": lid_bundle["fingerprint"],
        "residual_candidate_bundle_fingerprint": residual_bundle["fingerprint"],
        "selection_fingerprint": selection["fingerprint"],
        "shuffled_tune_diagnostic_fingerprint": shuffled_diagnostic["fingerprint"],
        "hypotheses_fingerprint": hypotheses["fingerprint"],
        "power_plan_fingerprint": power_plan["fingerprint"],
        "certification_fingerprint": certification["fingerprint"],
        "latency_dry_run_fingerprint": latency_result["fingerprint"],
        "policy_fingerprints": policy_fingerprints,
        "seeds": {
            "data": config.data_seed,
            "projection": config.projection_seed,
            "latency_method_order": config.latency.method_order_seed,
        },
    }
    manifest["fingerprint"] = fingerprint(manifest)
    report = (
        "# PDCTP network-free foundation\n\n"
        "Synthetic-only gate: PASS. No real data, protected real split, network, "
        "FAISS timing, or LLM was used.\n\n"
        f"Selected synthetic full-policy fingerprint: `{policy_fingerprints['full_pdctp']}`.\n\n"
        f"Synthetic certification family all passed: `{certification['all_passed']}`. "
        "This is a code-path fixture, not a scientific or latency claim.\n\n"
        f"Worst-case planned fresh role size: `{power_plan['required_role_size']}`.\n"
    )

    public_base = {
        "query_cal": [_public_base_record(row) for row in cal_records],
        "query_tune": [_public_base_record(row) for row in tune_records],
        "query_cert": [_public_base_record(row) for row in cert_records],
        "query_latency": [_public_base_record(row) for row in latency_records],
        "query_test": [_public_base_record(row) for row in test_records],
    }
    per_query = []
    for role in FIVE_ROLES:
        per_query.extend(public_base[role])
    for evaluation in cert_evaluated.values():
        per_query.extend(evaluation)
    for evaluation in latency_evaluated.values():
        per_query.extend(evaluation)
    for evaluation in test_evaluated.values():
        per_query.extend(evaluation)

    artifacts = {
        "feature_spec.json": config.feature_spec.serialize(),
        "splits.json": split_artifact,
        "lid_calibrator_candidates.json": {
            **lid_bundle,
            "candidates": [candidate.serialize() for candidate in lid_candidates],
        },
        "residual_calibrator_candidates.json": {
            **residual_bundle,
            "full_candidates": [
                candidate.serialize() for candidate, _ in full_residual_candidates
            ],
            "residual_only_candidates": [
                candidate.serialize() for candidate, _ in raw_residual_candidates
            ],
        },
        "selected_lid_calibrator.json": selected_lid.serialize(),
        "selected_residual_calibrator.json": selected_residual.serialize(),
        "selected_residual_only_calibrator.json": selected_residual_only.serialize(),
        "raw_tri_reference.json": selected_raw_policy.serialize(),
        "monotone_reference.json": monotone.serialize(),
        "policies.json": policy_artifacts,
        "selection.json": selection,
        "shuffled_tune_diagnostic.json": shuffled_diagnostic,
        "hypotheses.json": hypotheses,
        "power_plan.json": power_plan,
        "certification_bounds.json": certification,
        "latency_dry_run.json": latency_result,
        "protocol_state.json": guard.serialize(),
        "manifest.json": manifest,
    }
    paths: Dict[str, Path] = {}
    for name, artifact in artifacts.items():
        path = output_dir / name
        write_json(path, artifact)
        paths[name] = path
    per_query_path = output_dir / "per_query.jsonl"
    _jsonl(per_query_path, per_query)
    paths["per_query.jsonl"] = per_query_path
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    paths["report.md"] = report_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the network-free PDCTP foundation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_pdctp_foundation_config(args.config)
    paths = run_pdctp_foundation(config, args.output)
    print(f"PDCTP network-free foundation wrote {len(paths)} artifacts to {args.output}")


if __name__ == "__main__":
    main()
