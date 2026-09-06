"""TLS-RAG Step 3: synthetic observed-pair profiling and calibration.

This module is deliberately CPU-only, network-free, exact-search-only, and
synthetic-only.  Its calibrated values are reachable-bin event-rate confidence
limits for the frozen fixture, never per-query posteriors or evidence guarantees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import beta as beta_distribution

from .embeddings import normalize_rows
from .indexes import ExactSquaredL2Index
from .projection import project_rows, projection_metadata
from .tls_rag_step2 import (
    Action,
    DecisionInput,
    EvidenceLabelStore,
    EvidencePlan,
    FacetSlot,
    FixedScheduleController,
    PassageEvidence,
    StateValidity,
    Step2Environment,
    SyntheticQuery,
    _evidence_view,
    assert_deployable_only,
    build_decision_input,
    build_evidence_label_store,
    build_step2_environment,
    join_phase_b,
    load_step2_config,
    run_phase_a as run_step2_phase_a,
)
from .tri_law import tri_law_probability
from .utils import array_fingerprint, fingerprint, write_json


CONFIG_SCHEMA = "tls_rag_step3_config_v1"
STATE_SCHEMA = "tls_rag_step3_deployable_state_v1"
PHASE_A_SCHEMA = "tls_rag_step3_phase_a_decision_v1"
PHASE_B_SCHEMA = "tls_rag_step3_phase_b_supervision_v1"

SYNTHETIC_PARTITIONS = (
    "synthetic_model_fit",
    "synthetic_bound_fit",
    "synthetic_evaluation",
)

BASE_FEATURE_NAMES = (
    "current_budget",
    "stage",
    "remaining_grid_steps",
    "original_distance_minimum",
    "original_distance_maximum",
    "original_distance_mean",
    "original_frontier_gap",
    "original_gap_mean",
    "distortion_minimum",
    "distortion_maximum",
    "distortion_mean",
    "distortion_invalid_count",
    "candidate_mean_pairwise_squared_l2",
    "candidate_mean_pairwise_cosine",
    "candidate_duplicate_vector_pairs",
    "zero_original_distance_count",
    "duplicate_original_distance_pairs",
    "duplicate_projected_distance_pairs",
    "base_mandatory_valid",
    "previous_original_mean_delta",
    "previous_original_mean_delta_valid",
)

_PAIR_SUMMARY_FEATURES = (
    "attempted_pair_count",
    "valid_pair_count",
    "invalid_pair_count",
    "tied_pair_count",
    "zero_pair_count",
    "nonfinite_pair_count",
    "collinear_pair_count",
    "probability_mean",
    "probability_maximum",
    "probability_q50",
    "probability_q90",
    "probability_q99",
    "summary_valid",
)

TRI_LAW_FEATURE_NAMES = tuple(
    f"{family}_{name}"
    for family in ("context_boundary", "core_frontier")
    for name in _PAIR_SUMMARY_FEATURES
) + (
    "core_log_distortion_valid_count",
    "core_log_distortion_invalid_count",
    "core_log_distortion_q50",
    "core_log_distortion_q90",
    "core_log_distortion_q99",
    "core_log_distortion_valid",
    "shell_log_distortion_valid_count",
    "shell_log_distortion_invalid_count",
    "shell_log_distortion_q50",
    "shell_log_distortion_q90",
    "shell_log_distortion_q99",
    "shell_log_distortion_valid",
    "risk_duplicate_projected_distance_pairs",
    "risk_required_profile_valid",
    "risk_previous_delta_is_first_state",
) + tuple(
    item
    for metric in (
        "context_boundary_probability_mean",
        "context_boundary_probability_maximum",
        "context_boundary_probability_q50",
        "context_boundary_probability_q90",
        "context_boundary_probability_q99",
        "core_frontier_probability_mean",
        "core_frontier_probability_maximum",
        "core_frontier_probability_q50",
        "core_frontier_probability_q90",
        "core_frontier_probability_q99",
        "core_log_distortion_q50",
        "core_log_distortion_q90",
        "core_log_distortion_q99",
        "shell_log_distortion_q50",
        "shell_log_distortion_q90",
        "shell_log_distortion_q99",
    )
    for item in (f"risk_previous_delta_{metric}", f"risk_previous_delta_{metric}_valid")
)

PLAN_FACET_FEATURE_NAMES = (
    "plan_required_slot_count",
    "plan_required_support_count",
    "plan_independence_rule_count",
    "plan_blocking_contradiction_rule",
    "plan_valid",
    "facet_prediction_count",
    "facet_predicted_covered_count",
    "facet_predicted_coverage_fraction",
    "facet_context_match_count_mean",
    "facet_context_match_count_maximum",
)

WORK_FIELDS = (
    "query_projection_count",
    "projected_full_scan_count",
    "projected_distance_evaluations",
    "pilot_prefix_exposure_count",
    "expansion_prefix_reuse_count",
    "new_original_distance_evaluations",
    "exact_rerank_count",
    "accumulated_rerank_candidates",
    "risk_pair_evaluations",
    "risk_feature_evaluation_count",
    "deterministic_plan_facet_evaluation_count",
    "remaining_score_inference_count",
    "sufficiency_score_inference_count",
    "calibration_lookup_count",
    "controller_evaluation_count",
    "fixed_reference_evaluation_count",
    "final_context_construction_count",
    "final_context_candidates",
)

TIMING_FIELDS = (
    "query_projection_ms",
    "projected_full_scan_and_pilot_ms",
    "expansion_prefix_reuse_ms",
    "new_original_distance_evaluation_ms",
    "exact_reranking_ms",
    "risk_feature_ms",
    "deterministic_plan_facet_ms",
    "remaining_score_inference_ms",
    "sufficiency_score_inference_ms",
    "calibration_lookup_ms",
    "controller_ms",
    "final_context_ms",
)

PORTABLE_ARTIFACTS = (
    "manifest.json",
    "config_identity.json",
    "fixture_identity.json",
    "projection.json",
    "id_maps.json",
    "risk_feature_spec.json",
    "feature_registry.json",
    "synthetic_partitions.json",
    "candidate_registry.json",
    "score_models.json",
    "calibration_tables.json",
    "synthetic_fit_records.jsonl",
    "synthetic_bound_records.jsonl",
    "phase_a_decisions.jsonl",
    "phase_b_supervision.jsonl",
    "ablation_records.jsonl",
    "work_counters.json",
    "aggregates.json",
    "report.md",
)


class Step3ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    row: int

    @property
    def uses_tri_law(self) -> bool:
        return self.row >= 3

    @property
    def uses_plan_facet(self) -> bool:
        return self.row >= 4


@dataclass(frozen=True)
class Step3Config:
    path: Path
    raw: Mapping[str, Any]
    candidates: tuple[CandidateSpec, ...]

    @property
    def config_fingerprint(self) -> str:
        return fingerprint(self.raw)

    def section(self, name: str) -> Mapping[str, Any]:
        return self.raw[name]


def _exact_keys(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise Step3ConfigError(f"{name} must be an object")
    if set(value) != expected:
        raise Step3ConfigError(
            f"invalid {name} keys; missing={sorted(expected-set(value))}, "
            f"unknown={sorted(set(value)-expected)}"
        )
    return value


def load_step3_config(path: Path | str) -> Step3Config:
    config_path = Path(path).resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Step3ConfigError(f"cannot load Step 3 config: {exc}") from exc
    root_keys = {
        "schema_version", "run_name", "upstream_step2", "seeds",
        "synthetic_fixture", "risk_profile", "score_model", "calibration",
        "controller", "scope",
    }
    root = _exact_keys(raw, root_keys, "config")
    if root["schema_version"] != CONFIG_SCHEMA:
        raise Step3ConfigError(f"schema_version must be {CONFIG_SCHEMA}")
    if root["run_name"] != "tls_rag_step3_synthetic_observed_pair_v1":
        raise Step3ConfigError("unexpected Step 3 run identity")
    fixture = root["synthetic_fixture"]
    if tuple(fixture["partition_buckets"]) != SYNTHETIC_PARTITIONS:
        raise Step3ConfigError("synthetic partition names or order changed")
    if set(fixture["partition_buckets"].values()) != set(
        range(fixture["partition_modulus"])
    ):
        raise Step3ConfigError("synthetic partition buckets must cover the modulus")
    if fixture["queries_per_partition"] < 8:
        raise Step3ConfigError("the Step 3 fixture needs at least eight queries per partition")
    risk = root["risk_profile"]
    if risk["quantiles"] != [0.5, 0.9, 0.99] or risk["quantile_method"] != "linear":
        raise Step3ConfigError("risk quantiles are frozen at 0.50/0.90/0.99 linear")
    if risk["equal_distance_tolerance"] != 0.0 or risk["collinear_tolerance"] != 0.0:
        raise Step3ConfigError("Step 3 uses exact equality and exact collinearity")
    calibration = root["calibration"]
    if calibration["independent_unit"] != "query_id":
        raise Step3ConfigError("calibration independent unit must be query_id")
    if calibration["underpowered_interval"] != [0.0, 1.0]:
        raise Step3ConfigError("underpowered cells must use [0,1]")
    if calibration["score_bin_count"] != len(calibration["score_bin_quantiles"]) + 1:
        raise Step3ConfigError("score-bin quantiles do not match bin count")
    controller = root["controller"]
    candidates = tuple(
        CandidateSpec(item["candidate_id"], int(item["row"]))
        for item in controller["candidates"]
    )
    if tuple(item.row for item in candidates) != (2, 3, 4):
        raise Step3ConfigError("Step 3 implements exactly Rows 2--4")
    if len({item.candidate_id for item in candidates}) != len(candidates):
        raise Step3ConfigError("candidate IDs must be unique")
    scope = root["scope"]
    if not all(scope.values()):
        raise Step3ConfigError("all Step 3 scope prohibitions must remain enabled")
    return Step3Config(config_path, raw, candidates)


@dataclass(frozen=True)
class SyntheticBlueprint:
    query_id: str
    partition: str
    label_mode: str


@dataclass(frozen=True)
class Step3Environment:
    upstream: Step2Environment
    queries: tuple[SyntheticQuery, ...]
    partition_ids: tuple[tuple[str, tuple[str, ...]], ...]
    fixture_fingerprint: str

    def partition(self, name: str) -> tuple[str, ...]:
        return dict(self.partition_ids)[name]


@dataclass(frozen=True)
class Step3FixtureBundle:
    environment: Step3Environment
    blueprints: tuple[SyntheticBlueprint, ...]


def stable_synthetic_partition(query_id: str, config: Step3Config) -> str:
    salt = config.section("seeds")["partition_hash_salt"]
    digest = hashlib.sha256(f"{salt}\0{query_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % int(
        config.section("synthetic_fixture")["partition_modulus"]
    )
    reverse = {
        int(value): key
        for key, value in config.section("synthetic_fixture")["partition_buckets"].items()
    }
    return reverse[bucket]


def _read_only(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    result.setflags(write=False)
    return result


def build_step3_environment(config: Step3Config) -> Step3FixtureBundle:
    repo_root = config.path.parents[1]
    upstream_config = load_step2_config(
        repo_root / config.section("upstream_step2")["config_path"]
    )
    upstream = build_step2_environment(upstream_config)
    frozen = config.section("upstream_step2")
    if upstream_config.config_fingerprint != frozen["config_fingerprint"]:
        raise ValueError("frozen Step 2 config fingerprint mismatch")
    if upstream.fixture_fingerprint != frozen["fixture_fingerprint"]:
        raise ValueError("frozen Step 2 fixture fingerprint mismatch")

    count = int(config.section("synthetic_fixture")["queries_per_partition"])
    ids_by_partition: dict[str, list[str]] = {name: [] for name in SYNTHETIC_PARTITIONS}
    serial = 0
    while any(len(values) < count for values in ids_by_partition.values()):
        query_id = f"tls-step3-synthetic-query-{serial:04d}"
        partition = stable_synthetic_partition(query_id, config)
        if len(ids_by_partition[partition]) < count:
            ids_by_partition[partition].append(query_id)
        serial += 1
    ordered_ids = tuple(
        query_id for partition in SYNTHETIC_PARTITIONS for query_id in ids_by_partition[partition]
    )
    rng = np.random.default_rng(int(config.section("seeds")["synthetic_query"]))
    random_vectors = normalize_rows(
        rng.normal(size=(len(ordered_ids), upstream.config.dimension))
    )
    invalid_count = int(
        config.section("synthetic_fixture")["invalid_variants_per_partition"]
    )
    label_modes = tuple(config.section("synthetic_fixture")["label_modes"])
    queries = []
    blueprints = []
    cursor = 0
    for partition in SYNTHETIC_PARTITIONS:
        for local_index, query_id in enumerate(ids_by_partition[partition]):
            variant = local_index if local_index < invalid_count else -1
            vector = random_vectors[cursor]
            if variant == 0:
                row = int.from_bytes(hashlib.sha256(query_id.encode()).digest()[:2], "big") % len(
                    upstream.corpus_ids
                )
                vector = upstream.corpus_embeddings[row].copy()
            facet = f"tls-step3-facet-{query_id}"
            plan = (
                EvidencePlan(upstream.config.evidence_plan_generator, (), True)
                if variant == 1
                else EvidencePlan(
                    upstream.config.evidence_plan_generator,
                    (FacetSlot(facet, 1, False, "synthetic"),),
                    True,
                )
            )
            query = SyntheticQuery(
                query_id=query_id,
                text=f"Synthetic Step 3 external query {query_id}.",
                embedding=_read_only(vector),
                plan=plan,
                deterministic_features_valid=variant != 2,
            )
            mode_digest = hashlib.sha256(f"mode\0{query_id}".encode()).digest()
            mode = label_modes[int.from_bytes(mode_digest[:2], "big") % len(label_modes)]
            queries.append(query)
            blueprints.append(SyntheticBlueprint(query_id, partition, mode))
            cursor += 1
    if set(upstream.corpus_ids).intersection(query.query_id for query in queries):
        raise AssertionError("Step 3 query IDs must remain external")
    query_matrix = np.vstack([query.embedding for query in queries])
    payload = {
        "schema_version": "tls_rag_step3_synthetic_fixture_v1",
        "upstream_fixture_fingerprint": upstream.fixture_fingerprint,
        "query_ids": list(ordered_ids),
        "query_embedding_hash": array_fingerprint(query_matrix),
        "partition_ids": ids_by_partition,
        "plan_identities": {query.query_id: query.plan.to_dict() for query in queries},
        "synthetic_generation_seed": config.section("seeds")["synthetic_query"],
    }
    environment = Step3Environment(
        upstream=upstream,
        queries=tuple(queries),
        partition_ids=tuple((name, tuple(ids_by_partition[name])) for name in SYNTHETIC_PARTITIONS),
        fixture_fingerprint=fingerprint(payload),
    )
    return Step3FixtureBundle(environment, tuple(blueprints))


@dataclass(frozen=True)
class PairRecord:
    near_id: str
    far_id: str
    status: str
    beta: Optional[float]
    rho: Optional[float]
    probability: Optional[float]
    collinear: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "near_id": self.near_id,
            "far_id": self.far_id,
            "status": self.status,
            "beta": self.beta,
            "rho": self.rho,
            "probability": self.probability,
            "collinear": self.collinear,
        }


@dataclass(frozen=True)
class PairSummary:
    attempted_pair_count: int
    valid_pair_count: int
    invalid_pair_count: int
    tied_pair_count: int
    zero_pair_count: int
    nonfinite_pair_count: int
    collinear_pair_count: int
    probability_mean: float
    probability_maximum: float
    probability_q50: float
    probability_q90: float
    probability_q99: float
    summary_valid: bool
    pairs: tuple[PairRecord, ...]

    def feature_dict(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name)) for name in _PAIR_SUMMARY_FEATURES
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **{name: getattr(self, name) for name in _PAIR_SUMMARY_FEATURES},
            "pairs": [pair.to_dict() for pair in self.pairs],
        }


@dataclass(frozen=True)
class DistortionSummary:
    valid_count: int
    invalid_count: int
    q50: float
    q90: float
    q99: float
    valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class PreviousDeltas:
    is_first_state: bool
    values: tuple[tuple[str, float], ...]
    validity: tuple[tuple[str, bool], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_first_state": self.is_first_state,
            "values": dict(self.values),
            "validity": dict(self.validity),
        }


@dataclass(frozen=True)
class ObservedPairRiskProfile:
    schema_version: str
    exposed_candidate_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    shell_candidate_ids: tuple[str, ...]
    context_boundary: PairSummary
    core_frontier: PairSummary
    core_log_distortion: DistortionSummary
    shell_log_distortion: DistortionSummary
    duplicate_projected_distance_pairs: int
    required_profile_valid: bool
    previous_deltas: PreviousDeltas

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "exposed_candidate_ids": list(self.exposed_candidate_ids),
            "context_ids": list(self.context_ids),
            "shell_candidate_ids": list(self.shell_candidate_ids),
            "context_boundary_risk": self.context_boundary.to_dict(),
            "core_frontier_risk": self.core_frontier.to_dict(),
            "core_log_distortion": self.core_log_distortion.to_dict(),
            "shell_log_distortion": self.shell_log_distortion.to_dict(),
            "duplicate_projected_distance_pairs": self.duplicate_projected_distance_pairs,
            "required_profile_valid": self.required_profile_valid,
            "previous_deltas": self.previous_deltas.to_dict(),
        }


def _equal_pair_count(values: Iterable[float]) -> int:
    numeric = tuple(float(value) for value in values)
    return sum(
        numeric[left] == numeric[right]
        for left in range(len(numeric))
        for right in range(left + 1, len(numeric))
    )


def _observed_pair(
    left_id: str,
    right_id: str,
    query: np.ndarray,
    vectors: Mapping[str, np.ndarray],
    m_prime: int,
) -> PairRecord:
    left = np.asarray(vectors[left_id], dtype=np.float64) - query
    right = np.asarray(vectors[right_id], dtype=np.float64) - query
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return PairRecord(left_id, right_id, "nonfinite", None, None, None, False)
    left_squared = float(np.dot(left, left))
    right_squared = float(np.dot(right, right))
    if not np.isfinite(left_squared) or not np.isfinite(right_squared):
        return PairRecord(left_id, right_id, "nonfinite", None, None, None, False)
    if left_squared <= 0.0 or right_squared <= 0.0:
        return PairRecord(left_id, right_id, "zero", None, None, None, False)
    if left_squared == right_squared:
        near_id, far_id = sorted((left_id, right_id))
        return PairRecord(near_id, far_id, "tied", None, None, None, False)
    if left_squared < right_squared:
        near_id, far_id, near, far = left_id, right_id, left, right
        near_squared, far_squared = left_squared, right_squared
    else:
        near_id, far_id, near, far = right_id, left_id, right, left
        near_squared, far_squared = right_squared, left_squared
    beta_value = far_squared / near_squared
    rho_value = float(np.dot(near, far) / np.sqrt(near_squared * far_squared))
    rho_value = float(np.clip(rho_value, -1.0, 1.0))
    collinear = abs(rho_value) == 1.0
    probability = float(tri_law_probability(beta_value, rho_value, m_prime))
    return PairRecord(
        near_id, far_id, "valid", float(beta_value), rho_value, probability, collinear
    )


def _pair_summary(pairs: Sequence[PairRecord]) -> PairSummary:
    probabilities = np.asarray(
        [pair.probability for pair in pairs if pair.status == "valid"],
        dtype=np.float64,
    )
    if probabilities.size:
        quantiles = np.quantile(probabilities, [0.5, 0.9, 0.99], method="linear")
        mean = float(np.mean(probabilities))
        maximum = float(np.max(probabilities))
        q50, q90, q99 = (float(value) for value in quantiles)
    else:
        mean = maximum = q50 = q90 = q99 = 0.0
    invalid = sum(pair.status != "valid" for pair in pairs)
    return PairSummary(
        attempted_pair_count=len(pairs),
        valid_pair_count=int(probabilities.size),
        invalid_pair_count=invalid,
        tied_pair_count=sum(pair.status == "tied" for pair in pairs),
        zero_pair_count=sum(pair.status == "zero" for pair in pairs),
        nonfinite_pair_count=sum(pair.status == "nonfinite" for pair in pairs),
        collinear_pair_count=sum(pair.collinear for pair in pairs),
        probability_mean=mean,
        probability_maximum=maximum,
        probability_q50=q50,
        probability_q90=q90,
        probability_q99=q99,
        summary_valid=bool(probabilities.size),
        pairs=tuple(pairs),
    )


def _distortion_summary(
    ids: Sequence[str],
    projected: Mapping[str, float],
    original: Mapping[str, float],
) -> DistortionSummary:
    values = []
    invalid = 0
    for item in ids:
        projected_value = float(projected[item])
        original_value = float(original[item])
        if (
            not np.isfinite(projected_value)
            or not np.isfinite(original_value)
            or projected_value <= 0.0
            or original_value <= 0.0
        ):
            invalid += 1
        else:
            values.append(float(np.log(projected_value / original_value)))
    if values:
        q50, q90, q99 = (
            float(value)
            for value in np.quantile(values, [0.5, 0.9, 0.99], method="linear")
        )
    else:
        q50 = q90 = q99 = 0.0
    return DistortionSummary(len(values), invalid, q50, q90, q99, bool(values))


def _risk_delta_sources(profile: ObservedPairRiskProfile) -> dict[str, tuple[float, bool]]:
    result: dict[str, tuple[float, bool]] = {}
    for family, summary in (
        ("context_boundary", profile.context_boundary),
        ("core_frontier", profile.core_frontier),
    ):
        for metric in (
            "probability_mean", "probability_maximum", "probability_q50",
            "probability_q90", "probability_q99",
        ):
            result[f"{family}_{metric}"] = (float(getattr(summary, metric)), summary.summary_valid)
    for family, summary in (
        ("core_log_distortion", profile.core_log_distortion),
        ("shell_log_distortion", profile.shell_log_distortion),
    ):
        for metric in ("q50", "q90", "q99"):
            result[f"{family}_{metric}"] = (float(getattr(summary, metric)), summary.valid)
    return result


def build_observed_pair_risk_profile(
    *,
    query_embedding: np.ndarray,
    exposed_candidate_ids: Sequence[str],
    context_ids: Sequence[str],
    shell_candidate_ids: Sequence[str],
    candidate_embeddings_by_id: Mapping[str, np.ndarray],
    projected_squared_distances_by_id: Mapping[str, float],
    original_squared_distances_by_id: Mapping[str, float],
    m_prime: int,
    minimum_valid_pairs_per_family: int = 1,
    collinear_invalidates_required_profile: bool = True,
    previous_profile: Optional[ObservedPairRiskProfile] = None,
    first_state_delta_value: float = 0.0,
    first_state_delta_valid: bool = False,
) -> ObservedPairRiskProfile:
    """Build the exact observed-prefix profile without accepting unseen geometry."""
    exposed = tuple(exposed_candidate_ids)
    exposed_set = set(exposed)
    for name, values in (
        ("candidate_embeddings_by_id", candidate_embeddings_by_id),
        ("projected_squared_distances_by_id", projected_squared_distances_by_id),
        ("original_squared_distances_by_id", original_squared_distances_by_id),
    ):
        if set(values) != exposed_set:
            raise ValueError(f"{name} must contain every and only exposed candidates")
    if not set(context_ids).issubset(exposed_set):
        raise ValueError("context contains an unseen candidate")
    if not set(shell_candidate_ids).issubset(exposed_set):
        raise ValueError("shell contains an unseen candidate")
    if not context_ids:
        boundary_pairs: list[PairRecord] = []
    else:
        boundary_id = max(
            context_ids,
            key=lambda item: (float(original_squared_distances_by_id[item]), item),
        )
        boundary_pairs = [
            _observed_pair(
                boundary_id, item, np.asarray(query_embedding, dtype=np.float64),
                candidate_embeddings_by_id, m_prime,
            )
            for item in exposed
            if item not in set(context_ids)
        ]
    frontier_pairs = [
        _observed_pair(
            core_id, shell_id, np.asarray(query_embedding, dtype=np.float64),
            candidate_embeddings_by_id, m_prime,
        )
        for core_id in context_ids
        for shell_id in shell_candidate_ids
        if core_id != shell_id
    ]
    boundary = _pair_summary(boundary_pairs)
    frontier = _pair_summary(frontier_pairs)
    core_distortion = _distortion_summary(
        context_ids, projected_squared_distances_by_id, original_squared_distances_by_id
    )
    shell_distortion = _distortion_summary(
        shell_candidate_ids,
        projected_squared_distances_by_id,
        original_squared_distances_by_id,
    )
    duplicate_projected = _equal_pair_count(projected_squared_distances_by_id.values())
    required_valid = (
        boundary.valid_pair_count >= minimum_valid_pairs_per_family
        and frontier.valid_pair_count >= minimum_valid_pairs_per_family
        and boundary.zero_pair_count + frontier.zero_pair_count == 0
        and boundary.nonfinite_pair_count + frontier.nonfinite_pair_count == 0
        and core_distortion.valid
        and shell_distortion.valid
        and (
            not collinear_invalidates_required_profile
            or boundary.collinear_pair_count + frontier.collinear_pair_count == 0
        )
    )
    provisional = ObservedPairRiskProfile(
        schema_version="tls_rag_observed_pair_profile_v1",
        exposed_candidate_ids=exposed,
        context_ids=tuple(context_ids),
        shell_candidate_ids=tuple(shell_candidate_ids),
        context_boundary=boundary,
        core_frontier=frontier,
        core_log_distortion=core_distortion,
        shell_log_distortion=shell_distortion,
        duplicate_projected_distance_pairs=duplicate_projected,
        required_profile_valid=required_valid,
        previous_deltas=PreviousDeltas(True, (), ()),
    )
    current_sources = _risk_delta_sources(provisional)
    if previous_profile is None:
        delta_values = tuple((name, float(first_state_delta_value)) for name in current_sources)
        delta_validity = tuple((name, bool(first_state_delta_valid)) for name in current_sources)
        is_first = True
    else:
        previous_sources = _risk_delta_sources(previous_profile)
        value_rows = []
        valid_rows = []
        for name, (current_value, current_valid) in current_sources.items():
            previous_value, previous_valid = previous_sources[name]
            valid = bool(current_valid and previous_valid)
            value_rows.append((name, current_value - previous_value if valid else 0.0))
            valid_rows.append((name, valid))
        delta_values = tuple(value_rows)
        delta_validity = tuple(valid_rows)
        is_first = False
    return ObservedPairRiskProfile(
        **{
            field.name: getattr(provisional, field.name)
            for field in fields(provisional)
            if field.name != "previous_deltas"
        },
        previous_deltas=PreviousDeltas(is_first, delta_values, delta_validity),
    )


@dataclass(frozen=True)
class PreparedStage:
    query_id: str
    stage: int
    decision_input: DecisionInput
    risk_profile: ObservedPairRiskProfile
    base_features: tuple[tuple[str, float], ...]
    tri_law_features: tuple[tuple[str, float], ...]
    plan_facet_features: tuple[tuple[str, float], ...]
    base_work: tuple[tuple[str, int], ...]
    base_timing: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class PreparedQuery:
    query_id: str
    stages: tuple[PreparedStage, ...]
    full_projected_ranking_ids: tuple[str, ...]
    full_projected_squared_distances: tuple[float, ...]


def _base_features(state: DecisionInput, previous: Optional[DecisionInput]) -> tuple[tuple[str, float], ...]:
    gaps = np.asarray(state.original_distance_gaps, dtype=np.float64)
    previous_valid = previous is not None
    values = {
        "current_budget": float(state.current_budget),
        "stage": float(state.step),
        "remaining_grid_steps": float(state.remaining_grid_steps),
        "original_distance_minimum": state.original_distance_summary.minimum,
        "original_distance_maximum": state.original_distance_summary.maximum,
        "original_distance_mean": state.original_distance_summary.mean,
        "original_frontier_gap": float(gaps[-1]) if gaps.size else 0.0,
        "original_gap_mean": float(np.mean(gaps)) if gaps.size else 0.0,
        "distortion_minimum": state.distortion_summary.minimum,
        "distortion_maximum": state.distortion_summary.maximum,
        "distortion_mean": state.distortion_summary.mean,
        "distortion_invalid_count": float(state.distortion_invalid_count),
        "candidate_mean_pairwise_squared_l2": state.candidate_mean_pairwise_squared_l2,
        "candidate_mean_pairwise_cosine": state.candidate_mean_pairwise_cosine,
        "candidate_duplicate_vector_pairs": float(state.candidate_duplicate_vector_pairs),
        "zero_original_distance_count": float(state.validity.zero_original_distance_count),
        "duplicate_original_distance_pairs": float(state.validity.duplicate_original_distance_pairs),
        "duplicate_projected_distance_pairs": float(state.validity.duplicate_projected_distance_pairs),
        "base_mandatory_valid": float(state.validity.mandatory_valid),
        "previous_original_mean_delta": (
            state.original_distance_summary.mean - previous.original_distance_summary.mean
            if previous_valid else 0.0
        ),
        "previous_original_mean_delta_valid": float(previous_valid),
    }
    if tuple(values) != BASE_FEATURE_NAMES:
        raise AssertionError("base feature registry drift")
    return tuple(values.items())


def _tri_law_features(profile: ObservedPairRiskProfile) -> tuple[tuple[str, float], ...]:
    values: dict[str, float] = {}
    for family, summary in (
        ("context_boundary", profile.context_boundary),
        ("core_frontier", profile.core_frontier),
    ):
        for name, value in summary.feature_dict().items():
            values[f"{family}_{name}"] = value
    for family, summary in (
        ("core_log_distortion", profile.core_log_distortion),
        ("shell_log_distortion", profile.shell_log_distortion),
    ):
        values[f"{family}_valid_count"] = float(summary.valid_count)
        values[f"{family}_invalid_count"] = float(summary.invalid_count)
        values[f"{family}_q50"] = summary.q50
        values[f"{family}_q90"] = summary.q90
        values[f"{family}_q99"] = summary.q99
        values[f"{family}_valid"] = float(summary.valid)
    values["risk_duplicate_projected_distance_pairs"] = float(
        profile.duplicate_projected_distance_pairs
    )
    values["risk_required_profile_valid"] = float(profile.required_profile_valid)
    values["risk_previous_delta_is_first_state"] = float(
        profile.previous_deltas.is_first_state
    )
    delta_values = dict(profile.previous_deltas.values)
    delta_validity = dict(profile.previous_deltas.validity)
    for name in delta_values:
        values[f"risk_previous_delta_{name}"] = float(delta_values[name])
        values[f"risk_previous_delta_{name}_valid"] = float(delta_validity[name])
    if tuple(values) != TRI_LAW_FEATURE_NAMES:
        raise AssertionError("Tri-Law feature registry drift")
    return tuple(values.items())


def _plan_features(state: DecisionInput) -> tuple[tuple[str, float], ...]:
    raw_plan = dict(state.evidence_plan_features)
    predictions = state.facet_match_predictions
    counts = [prediction.context_match_count for prediction in predictions]
    covered = sum(prediction.predicted_covered for prediction in predictions)
    values = {
        "plan_required_slot_count": raw_plan["required_slot_count"],
        "plan_required_support_count": raw_plan["required_support_count"],
        "plan_independence_rule_count": raw_plan["independence_rule_count"],
        "plan_blocking_contradiction_rule": raw_plan["blocking_contradiction_rule"],
        "plan_valid": float(state.validity.plan_valid),
        "facet_prediction_count": float(len(predictions)),
        "facet_predicted_covered_count": float(covered),
        "facet_predicted_coverage_fraction": (
            float(covered / len(predictions)) if predictions else 0.0
        ),
        "facet_context_match_count_mean": float(np.mean(counts)) if counts else 0.0,
        "facet_context_match_count_maximum": float(np.max(counts)) if counts else 0.0,
    }
    if tuple(values) != PLAN_FACET_FEATURE_NAMES:
        raise AssertionError("plan/facet feature registry drift")
    return tuple(values.items())


def prepare_queries(
    environment: Step3Environment,
    config: Step3Config,
    query_ids: Optional[Sequence[str]] = None,
) -> tuple[PreparedQuery, ...]:
    """Compute one exact projected scan and all frozen prefixes label-free."""
    requested = set(query_ids) if query_ids is not None else {
        query.query_id for query in environment.queries
    }
    queries = [query for query in environment.queries if query.query_id in requested]
    if {query.query_id for query in queries} != requested:
        raise ValueError("requested unknown synthetic query ID")
    upstream = environment.upstream
    index = ExactSquaredL2Index(
        upstream.corpus_ids, upstream.projected_corpus, batch_size=1
    )
    row_by_id = {item: row for row, item in enumerate(upstream.corpus_ids)}
    risk_config = config.section("risk_profile")
    results = []
    for query in queries:
        projection_started = perf_counter()
        projected_query = project_rows(
            query.embedding[None, :], upstream.projection_matrix
        )[0]
        projection_ms = (perf_counter() - projection_started) * 1000.0
        search = index.search(projected_query, upstream.config.corpus_size)
        full_ids = tuple(str(value) for value in search.ids[0].tolist())
        full_projected = tuple(float(value) for value in search.squared_distances[0])
        cache: dict[str, float] = {}
        stages = []
        previous_budget = 0
        previous_state: Optional[DecisionInput] = None
        previous_risk: Optional[ObservedPairRiskProfile] = None
        for stage, budget in enumerate(upstream.config.budget_grid):
            prefix_started = perf_counter()
            exposed = full_ids[:budget]
            new_ids = full_ids[previous_budget:budget]
            prefix_ms = (perf_counter() - prefix_started) * 1000.0
            original_started = perf_counter()
            for item in new_ids:
                if item in cache:
                    raise AssertionError("original distance evaluated more than once")
                difference = upstream.corpus_embeddings[row_by_id[item]] - query.embedding
                cache[item] = float(np.dot(difference, difference))
            original_ms = (perf_counter() - original_started) * 1000.0
            rerank_started = perf_counter()
            distance_values = np.asarray([cache[item] for item in exposed])
            order = np.lexsort((np.asarray(exposed, dtype=str), distance_values))
            reranked = tuple(exposed[int(index_value)] for index_value in order)
            rerank_ms = (perf_counter() - rerank_started) * 1000.0
            context_started = perf_counter()
            context = reranked[: upstream.config.k_ctx]
            context_ms = (perf_counter() - context_started) * 1000.0
            decision_input = build_decision_input(
                environment=upstream,
                query=query,
                step=stage,
                projected_ids=exposed,
                projected_distances=full_projected[:budget],
                original_distances_by_id=cache,
                reranked_ids=reranked,
                context_ids=context,
            )
            shell = (
                exposed[-min(int(risk_config["shell_size"]), budget):]
                if stage == 0
                else new_ids[-int(risk_config["shell_size"]):]
            )
            prefix_vectors = {
                item: upstream.corpus_embeddings[row_by_id[item]] for item in exposed
            }
            projected_map = {
                item: full_projected[position] for position, item in enumerate(exposed)
            }
            original_map = {item: cache[item] for item in exposed}
            risk_started = perf_counter()
            risk = build_observed_pair_risk_profile(
                query_embedding=query.embedding,
                exposed_candidate_ids=exposed,
                context_ids=context,
                shell_candidate_ids=shell,
                candidate_embeddings_by_id=prefix_vectors,
                projected_squared_distances_by_id=projected_map,
                original_squared_distances_by_id=original_map,
                m_prime=upstream.config.m_prime,
                minimum_valid_pairs_per_family=int(
                    risk_config["minimum_valid_pairs_per_family"]
                ),
                collinear_invalidates_required_profile=bool(
                    risk_config["collinear_invalidates_required_profile"]
                ),
                previous_profile=previous_risk,
                first_state_delta_value=float(risk_config["first_state_delta_value"]),
                first_state_delta_valid=bool(risk_config["first_state_delta_valid"]),
            )
            risk_ms = (perf_counter() - risk_started) * 1000.0
            plan_started = perf_counter()
            plan_features = _plan_features(decision_input)
            plan_ms = (perf_counter() - plan_started) * 1000.0
            work = {
                "query_projection_count": int(stage == 0),
                "projected_full_scan_count": int(stage == 0),
                "projected_distance_evaluations": upstream.config.corpus_size if stage == 0 else 0,
                "pilot_prefix_exposure_count": int(stage == 0),
                "expansion_prefix_reuse_count": int(stage > 0),
                "new_original_distance_evaluations": len(new_ids),
                "exact_rerank_count": 1,
                "accumulated_rerank_candidates": budget,
                "final_context_construction_count": 1,
                "final_context_candidates": len(context),
            }
            timing = {
                "query_projection_ms": projection_ms if stage == 0 else 0.0,
                "projected_full_scan_and_pilot_ms": search.search_ms if stage == 0 else 0.0,
                "expansion_prefix_reuse_ms": prefix_ms if stage > 0 else 0.0,
                "new_original_distance_evaluation_ms": original_ms,
                "exact_reranking_ms": rerank_ms,
                "risk_feature_ms": risk_ms,
                "deterministic_plan_facet_ms": plan_ms,
                "final_context_ms": context_ms,
            }
            stages.append(
                PreparedStage(
                    query_id=query.query_id,
                    stage=stage,
                    decision_input=decision_input,
                    risk_profile=risk,
                    base_features=_base_features(decision_input, previous_state),
                    tri_law_features=_tri_law_features(risk),
                    plan_facet_features=plan_features,
                    base_work=tuple(work.items()),
                    base_timing=tuple(timing.items()),
                )
            )
            previous_budget = budget
            previous_state = decision_input
            previous_risk = risk
        results.append(PreparedQuery(query.query_id, tuple(stages), full_ids, full_projected))
    return tuple(results)


def feature_names_for_row(row: int) -> tuple[str, ...]:
    if row == 2:
        return BASE_FEATURE_NAMES
    if row == 3:
        return BASE_FEATURE_NAMES + TRI_LAW_FEATURE_NAMES
    if row == 4:
        return BASE_FEATURE_NAMES + TRI_LAW_FEATURE_NAMES + PLAN_FACET_FEATURE_NAMES
    raise ValueError("Step 3 has feature registries only for Rows 2--4")


def _feature_values_for_candidate(stage: PreparedStage, candidate: CandidateSpec) -> tuple[tuple[str, float], ...]:
    values = stage.base_features
    if candidate.uses_tri_law:
        values += stage.tri_law_features
    if candidate.uses_plan_facet:
        values += stage.plan_facet_features
    if tuple(name for name, _ in values) != feature_names_for_row(candidate.row):
        raise AssertionError("candidate feature schema drift")
    return values


@dataclass(frozen=True)
class Step3DeployableState:
    schema_version: str
    candidate_id: str
    row: int
    query_id: str
    stage: int
    current_budget: int
    remaining_grid_steps: int
    context_ids: tuple[str, ...]
    feature_values: tuple[tuple[str, float], ...]
    feature_schema_fingerprint: str
    base_state_valid: bool
    required_risk_profile_valid: bool
    evidence_plan_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "row": self.row,
            "query_id": self.query_id,
            "stage": self.stage,
            "current_budget": self.current_budget,
            "remaining_grid_steps": self.remaining_grid_steps,
            "context_ids": list(self.context_ids),
            "feature_values": dict(self.feature_values),
            "feature_schema_fingerprint": self.feature_schema_fingerprint,
            "base_state_valid": self.base_state_valid,
            "required_risk_profile_valid": self.required_risk_profile_valid,
            "evidence_plan_valid": self.evidence_plan_valid,
        }


def make_deployable_state(stage: PreparedStage, candidate: CandidateSpec) -> Step3DeployableState:
    feature_values = _feature_values_for_candidate(stage, candidate)
    names = tuple(name for name, _ in feature_values)
    state = Step3DeployableState(
        schema_version=STATE_SCHEMA,
        candidate_id=candidate.candidate_id,
        row=candidate.row,
        query_id=stage.query_id,
        stage=stage.stage,
        current_budget=stage.decision_input.current_budget,
        remaining_grid_steps=stage.decision_input.remaining_grid_steps,
        context_ids=stage.decision_input.context_ids,
        feature_values=feature_values,
        feature_schema_fingerprint=fingerprint(
            {"schema_version": "tls_rag_step3_feature_names_v1", "names": list(names)}
        ),
        base_state_valid=stage.decision_input.validity.mandatory_valid,
        required_risk_profile_valid=(
            stage.risk_profile.required_profile_valid if candidate.uses_tri_law else True
        ),
        evidence_plan_valid=stage.decision_input.validity.plan_valid,
    )
    assert_deployable_only(state)
    return state


@dataclass(frozen=True)
class ScorePrediction:
    score: float
    valid: bool
    reason: str


@dataclass(frozen=True)
class LinearScoreModel:
    schema_version: str
    outcome: str
    candidate_id: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    regularization: float
    fit_query_ids: tuple[str, ...]
    fit_row_count: int
    excluded_invalid_row_count: int
    model_fingerprint: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "outcome": self.outcome,
            "candidate_id": self.candidate_id,
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "regularization": self.regularization,
            "fit_query_ids": list(self.fit_query_ids),
            "fit_row_count": self.fit_row_count,
            "excluded_invalid_row_count": self.excluded_invalid_row_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "fingerprint": self.model_fingerprint}

    def predict(self, state: Step3DeployableState) -> ScorePrediction:
        if state.candidate_id != self.candidate_id:
            return ScorePrediction(0.0, False, "candidate_mismatch")
        values = dict(state.feature_values)
        if tuple(values) != self.feature_names:
            return ScorePrediction(0.0, False, "feature_schema_mismatch")
        numeric = np.asarray([values[name] for name in self.feature_names], dtype=np.float64)
        if not np.all(np.isfinite(numeric)):
            return ScorePrediction(0.0, False, "nonfinite_feature")
        standardized = (numeric - np.asarray(self.means)) / np.asarray(self.scales)
        score = self.intercept + float(np.dot(standardized, self.coefficients))
        if not np.isfinite(score):
            return ScorePrediction(0.0, False, "prediction_failure")
        return ScorePrediction(float(np.clip(score, 0.0, 1.0)), True, "valid")


def _fit_score_model(
    *,
    outcome: str,
    candidate: CandidateSpec,
    rows: Sequence[tuple[str, Step3DeployableState, bool]],
    config: Step3Config,
) -> LinearScoreModel:
    feature_names = feature_names_for_row(candidate.row)
    valid_rows = []
    excluded = 0
    for query_id, state, target in rows:
        numeric = np.asarray([value for _, value in state.feature_values], dtype=np.float64)
        mandatory = (
            state.base_state_valid
            and state.evidence_plan_valid
            and state.required_risk_profile_valid
        )
        if mandatory and np.all(np.isfinite(numeric)):
            valid_rows.append((query_id, numeric, float(target)))
        else:
            excluded += 1
    if not valid_rows:
        raise ValueError("no valid synthetic model-fit rows")
    matrix = np.vstack([row[1] for row in valid_rows])
    targets = np.asarray([row[2] for row in valid_rows], dtype=np.float64)
    means = np.mean(matrix, axis=0)
    scales = np.std(matrix, axis=0)
    floor = float(config.section("score_model")["scale_floor"])
    scales = np.where(scales > floor, scales, 1.0)
    standardized = (matrix - means) / scales
    design = np.column_stack((np.ones(len(standardized)), standardized))
    regularization = float(config.section("score_model")["regularization"])
    penalty = np.eye(design.shape[1]) * regularization
    penalty[0, 0] = 0.0
    parameters = np.linalg.pinv(design.T @ design + penalty) @ design.T @ targets
    payload = {
        "schema_version": config.section("score_model")["version"],
        "outcome": outcome,
        "candidate_id": candidate.candidate_id,
        "feature_names": list(feature_names),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "coefficients": parameters[1:].tolist(),
        "intercept": float(parameters[0]),
        "regularization": regularization,
        "fit_query_ids": sorted({row[0] for row in valid_rows}),
        "fit_row_count": len(valid_rows),
        "excluded_invalid_row_count": excluded,
    }
    return LinearScoreModel(
        schema_version=payload["schema_version"],
        outcome=outcome,
        candidate_id=candidate.candidate_id,
        feature_names=feature_names,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in parameters[1:]),
        intercept=float(parameters[0]),
        regularization=regularization,
        fit_query_ids=tuple(payload["fit_query_ids"]),
        fit_row_count=len(valid_rows),
        excluded_invalid_row_count=excluded,
        model_fingerprint=fingerprint(payload),
    )


def _query_map(environment: Step3Environment) -> dict[str, SyntheticQuery]:
    return {query.query_id: query for query in environment.queries}


def build_synthetic_label_store(
    bundle: Step3FixtureBundle,
    prepared: Mapping[str, PreparedQuery],
    allowed_query_ids: Sequence[str],
) -> EvidenceLabelStore:
    """Construct labels only for the explicitly opened synthetic partition."""
    allowed = set(allowed_query_ids)
    blueprints = {item.query_id: item for item in bundle.blueprints}
    queries = _query_map(bundle.environment)
    if not allowed.issubset(blueprints) or not allowed.issubset(prepared):
        raise ValueError("label-store request contains an unknown query")
    facets: dict[str, set[str]] = {
        item: set() for item in bundle.environment.upstream.corpus_ids
    }
    for query_id in sorted(allowed):
        query = queries[query_id]
        if not query.plan.valid:
            continue
        mode = blueprints[query_id].label_mode
        stages = prepared[query_id].stages
        target: Optional[str] = None
        if mode == "pilot":
            target = stages[0].decision_input.context_ids[0]
        elif mode == "middle":
            earlier = set(stages[0].decision_input.context_ids)
            target = next(
                (item for item in stages[1].decision_input.context_ids if item not in earlier),
                stages[1].decision_input.context_ids[-1],
            )
        elif mode == "terminal":
            earlier = set(stages[1].decision_input.context_ids)
            target = next(
                (item for item in stages[2].decision_input.context_ids if item not in earlier),
                stages[2].decision_input.context_ids[-1],
            )
        elif mode != "never":
            raise ValueError("unknown synthetic label mode")
        if target is not None:
            facets[target].add(query.plan.slots[0].facet)
    annotations = tuple(
        PassageEvidence(
            passage_id=item,
            facets=tuple(sorted(facets[item])),
            source_group=f"tls-step3-source-{row:02d}",
            contradiction_facets=(),
            invalid=False,
        )
        for row, item in enumerate(bundle.environment.upstream.corpus_ids)
    )
    partition_identity = fingerprint(sorted(allowed))[:16]
    return EvidenceLabelStore(
        f"tls_rag_step3_synthetic_label_store_v1_{partition_identity}", annotations
    )


def reconstruct_synthetic_supervision(
    prepared: PreparedQuery,
    query: SyntheticQuery,
    environment: Step3Environment,
    store: EvidenceLabelStore,
) -> tuple[dict[str, Any], ...]:
    upstream = environment.upstream
    candidate_views = {
        stage.stage: _evidence_view(
            stage.decision_input.exposed_candidate_ids, query.plan, store
        )
        for stage in prepared.stages
    }
    context_views = {
        stage.stage: _evidence_view(stage.decision_input.context_ids, query.plan, store)
        for stage in prepared.stages
    }
    full_distances = np.einsum(
        "ij,ij->i",
        upstream.corpus_embeddings - query.embedding,
        upstream.corpus_embeddings - query.embedding,
    )
    exact_order = np.lexsort((np.asarray(upstream.corpus_ids, dtype=str), full_distances))
    exact_top_k_ids = tuple(
        upstream.corpus_ids[int(row)] for row in exact_order[: upstream.config.k_gt]
    )
    records = []
    for stage in prepared.stages:
        index = stage.stage
        current_candidate = candidate_views[index]
        current_context = context_views[index]
        later_indices = range(index + 1, len(prepared.stages))
        next_index = index + 1 if index + 1 < len(prepared.stages) else None
        candidate_gain = bool(
            next_index is not None
            and set(candidate_views[next_index]["covered_slots"])
            - set(current_candidate["covered_slots"])
        )
        context_gain = bool(
            next_index is not None
            and context_views[next_index]["coverage"] > current_context["coverage"]
        )
        remaining = any(
            context_views[later]["coverage"] > current_context["coverage"]
            or (
                not current_context["sufficient"]
                and context_views[later]["sufficient"]
            )
            for later in later_indices
        )
        candidate_ids = set(stage.decision_input.exposed_candidate_ids)
        retention = len(candidate_ids.intersection(exact_top_k_ids)) / len(exact_top_k_ids)
        records.append(
            {
                "query_id": query.query_id,
                "stage": index,
                "budget": stage.decision_input.current_budget,
                "candidate_coverage": current_candidate["coverage"],
                "context_coverage": current_context["coverage"],
                "current_context_is_sufficient": current_context["sufficient"],
                "next_candidate_has_new_evidence": candidate_gain,
                "next_context_has_new_evidence": context_gain,
                "later_context_has_useful_evidence": bool(remaining),
                "candidate_evidence_ids": current_candidate["supporting_passage_ids"],
                "context_evidence_ids": current_context["supporting_passage_ids"],
                "exact_top_k_ids": list(exact_top_k_ids),
                "exact_top_k_retention": float(retention),
            }
        )
    return tuple(records)


def _supervision_by_query(
    query_ids: Sequence[str],
    prepared: Mapping[str, PreparedQuery],
    environment: Step3Environment,
    store: EvidenceLabelStore,
) -> dict[str, tuple[dict[str, Any], ...]]:
    queries = _query_map(environment)
    return {
        query_id: reconstruct_synthetic_supervision(
            prepared[query_id], queries[query_id], environment, store
        )
        for query_id in query_ids
    }


def fit_score_models(
    candidates: Sequence[CandidateSpec],
    prepared: Mapping[str, PreparedQuery],
    supervision: Mapping[str, Sequence[Mapping[str, Any]]],
    fit_query_ids: Sequence[str],
    config: Step3Config,
) -> tuple[tuple[LinearScoreModel, LinearScoreModel], ...]:
    models = []
    for candidate in candidates:
        gain_rows = []
        sufficiency_rows = []
        for query_id in fit_query_ids:
            labels = {int(row["stage"]): row for row in supervision[query_id]}
            for stage in prepared[query_id].stages:
                state = make_deployable_state(stage, candidate)
                gain_rows.append(
                    (query_id, state, bool(labels[stage.stage]["later_context_has_useful_evidence"]))
                )
                sufficiency_rows.append(
                    (query_id, state, bool(labels[stage.stage]["current_context_is_sufficient"]))
                )
        models.append(
            (
                _fit_score_model(
                    outcome="remaining_useful_evidence_event",
                    candidate=candidate,
                    rows=gain_rows,
                    config=config,
                ),
                _fit_score_model(
                    outcome="current_context_sufficiency_event",
                    candidate=candidate,
                    rows=sufficiency_rows,
                    config=config,
                ),
            )
        )
    return tuple(models)


def clopper_pearson_one_sided(
    successes: int, trials: int, alpha: float
) -> tuple[float, float]:
    if (
        isinstance(successes, bool)
        or isinstance(trials, bool)
        or not isinstance(successes, (int, np.integer))
        or not isinstance(trials, (int, np.integer))
        or trials < 0
        or successes < 0
        or successes > trials
    ):
        raise ValueError("successes/trials must satisfy 0 <= successes <= trials")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if trials == 0:
        return 0.0, 1.0
    lower = 0.0 if successes == 0 else float(
        beta_distribution.ppf(alpha, successes, trials - successes + 1)
    )
    upper = 1.0 if successes == trials else float(
        beta_distribution.ppf(1.0 - alpha, successes + 1, trials - successes)
    )
    return lower, upper


@dataclass(frozen=True)
class CalibrationCell:
    candidate_id: str
    stage: int
    outcome: str
    bin_index: int
    lower_edge: float
    upper_edge: float
    query_ids: tuple[str, ...]
    successes: int
    trials: int
    lower_limit: float
    upper_limit: float
    valid: bool
    alpha_share: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "stage": self.stage,
            "outcome": self.outcome,
            "bin_index": self.bin_index,
            "lower_edge": self.lower_edge,
            "upper_edge": self.upper_edge,
            "query_ids": list(self.query_ids),
            "successes": self.successes,
            "trials": self.trials,
            "lower_limit": self.lower_limit,
            "upper_limit": self.upper_limit,
            "valid": self.valid,
            "alpha_share": self.alpha_share,
        }


@dataclass(frozen=True)
class CalibrationTable:
    schema_version: str
    candidate_id: str
    bin_edges: tuple[tuple[int, str, tuple[float, ...]], ...]
    cells: tuple[CalibrationCell, ...]
    reachable_query_ids_by_stage: tuple[tuple[int, tuple[str, ...]], ...]
    table_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "bin_edges": [
                {"stage": stage, "outcome": outcome, "edges": list(edges)}
                for stage, outcome, edges in self.bin_edges
            ],
            "cells": [cell.to_dict() for cell in self.cells],
            "reachable_query_ids_by_stage": [
                {"stage": stage, "query_ids": list(ids)}
                for stage, ids in self.reachable_query_ids_by_stage
            ],
        }
        return {**payload, "fingerprint": self.table_fingerprint}

    def edges(self, stage: int, outcome: str) -> tuple[float, ...]:
        matches = [edges for item_stage, item_outcome, edges in self.bin_edges if item_stage == stage and item_outcome == outcome]
        if len(matches) != 1:
            raise KeyError("calibration bin edges unavailable")
        return matches[0]

    def lookup(self, stage: int, outcome: str, score: float) -> CalibrationCell:
        edges = self.edges(stage, outcome)
        bin_index = int(np.searchsorted(edges[1:-1], score, side="right"))
        matches = [
            cell for cell in self.cells
            if cell.stage == stage and cell.outcome == outcome and cell.bin_index == bin_index
        ]
        if len(matches) != 1:
            raise KeyError("calibration cell unavailable")
        return matches[0]


@dataclass(frozen=True)
class ControllerEvaluation:
    action: Action
    reason: str
    next_budget: Optional[int]
    remaining_score: float
    sufficiency_score: float
    remaining_score_valid: bool
    sufficiency_score_valid: bool
    remaining_bin: Optional[int]
    sufficiency_bin: Optional[int]
    remaining_upper_limit: float
    sufficiency_lower_limit: float
    remaining_cell_valid: bool
    sufficiency_cell_valid: bool


class CalibratedStep3Controller:
    """Immutable-artifact controller with no supervision-store handle."""

    def __init__(
        self,
        *,
        candidate: CandidateSpec,
        gain_model: LinearScoreModel,
        sufficiency_model: LinearScoreModel,
        calibration_table: CalibrationTable,
        budget_grid: Sequence[int],
        maximum_expansions: int,
        delta_gain: float,
        tau_sufficient: float,
    ):
        self._candidate = candidate
        self._gain_model = gain_model
        self._sufficiency_model = sufficiency_model
        self._calibration_table = calibration_table
        self._budget_grid = tuple(int(value) for value in budget_grid)
        self._maximum_expansions = int(maximum_expansions)
        self._delta_gain = float(delta_gain)
        self._tau_sufficient = float(tau_sufficient)

    def choose(self, state: Step3DeployableState) -> ControllerEvaluation:
        if not isinstance(state, Step3DeployableState):
            raise TypeError("controller accepts only Step3DeployableState")
        assert_deployable_only(state)
        if state.schema_version != STATE_SCHEMA or state.candidate_id != self._candidate.candidate_id:
            return self._fallback(state, "schema_or_candidate_mismatch")
        if state.stage >= len(self._budget_grid) or state.current_budget != self._budget_grid[state.stage]:
            return self._fallback(state, "grid_state_mismatch")
        gain_prediction = self._gain_model.predict(state)
        suff_prediction = self._sufficiency_model.predict(state)
        gain_cell: Optional[CalibrationCell] = None
        suff_cell: Optional[CalibrationCell] = None
        if gain_prediction.valid:
            try:
                gain_cell = self._calibration_table.lookup(
                    state.stage, "remaining_useful_evidence_event", gain_prediction.score
                )
            except KeyError:
                gain_cell = None
        if suff_prediction.valid:
            try:
                suff_cell = self._calibration_table.lookup(
                    state.stage, "current_context_sufficiency_event", suff_prediction.score
                )
            except KeyError:
                suff_cell = None
        mandatory_valid = (
            state.base_state_valid
            and state.evidence_plan_valid
            and state.required_risk_profile_valid
            and gain_prediction.valid
            and suff_prediction.valid
            and gain_cell is not None
            and suff_cell is not None
            and gain_cell.valid
            and suff_cell.valid
        )
        upper_gain = gain_cell.upper_limit if gain_cell is not None else 1.0
        lower_sufficiency = suff_cell.lower_limit if suff_cell is not None else 0.0
        stop_bounds = (
            mandatory_valid
            and upper_gain <= self._delta_gain
            and lower_sufficiency >= self._tau_sufficient
        )
        if stop_bounds:
            action = Action.STOP
            reason = "dual_bound_stop"
            next_budget = None
        elif state.remaining_grid_steps > 0 and state.stage < self._maximum_expansions:
            action = Action.EXPAND_TO_NEXT_GRID_VALUE
            next_budget = self._budget_grid[state.stage + 1]
            if not mandatory_valid:
                reason = "conservative_invalid_or_uncalibrated_expand"
            elif upper_gain > self._delta_gain and lower_sufficiency < self._tau_sufficient:
                reason = "both_bounds_require_expand"
            elif upper_gain > self._delta_gain:
                reason = "remaining_gain_bound_requires_expand"
            else:
                reason = "sufficiency_bound_requires_expand"
        else:
            action = Action.STOP
            next_budget = None
            if not state.evidence_plan_valid:
                reason = "invalid_evidence_plan"
            elif not mandatory_valid:
                reason = "invalid_state_at_corpus_exhaustion"
            elif not stop_bounds:
                reason = "evidence_nonattainment"
            else:
                reason = "maximum_expansions_reached"
        return ControllerEvaluation(
            action=action,
            reason=reason,
            next_budget=next_budget,
            remaining_score=gain_prediction.score,
            sufficiency_score=suff_prediction.score,
            remaining_score_valid=gain_prediction.valid,
            sufficiency_score_valid=suff_prediction.valid,
            remaining_bin=gain_cell.bin_index if gain_cell is not None else None,
            sufficiency_bin=suff_cell.bin_index if suff_cell is not None else None,
            remaining_upper_limit=upper_gain,
            sufficiency_lower_limit=lower_sufficiency,
            remaining_cell_valid=bool(gain_cell is not None and gain_cell.valid),
            sufficiency_cell_valid=bool(suff_cell is not None and suff_cell.valid),
        )

    def _fallback(self, state: Step3DeployableState, reason: str) -> ControllerEvaluation:
        can_expand = (
            0 <= state.stage < len(self._budget_grid) - 1
            and state.stage < self._maximum_expansions
        )
        return ControllerEvaluation(
            action=Action.EXPAND_TO_NEXT_GRID_VALUE if can_expand else Action.STOP,
            reason=reason if can_expand else f"{reason}_at_terminal_budget",
            next_budget=self._budget_grid[state.stage + 1] if can_expand else None,
            remaining_score=0.0,
            sufficiency_score=0.0,
            remaining_score_valid=False,
            sufficiency_score_valid=False,
            remaining_bin=None,
            sufficiency_bin=None,
            remaining_upper_limit=1.0,
            sufficiency_lower_limit=0.0,
            remaining_cell_valid=False,
            sufficiency_cell_valid=False,
        )


def _score_bin_edges(scores: Sequence[float], config: Step3Config) -> tuple[float, ...]:
    calibration = config.section("calibration")
    if scores:
        interior = tuple(
            float(value)
            for value in np.quantile(
                np.asarray(scores, dtype=np.float64),
                calibration["score_bin_quantiles"],
                method=calibration["score_bin_method"],
            )
        )
    else:
        interior = tuple(float(value) for value in calibration["score_bin_quantiles"])
    return (0.0, *interior, 1.0)


def _table_with_fingerprint(
    candidate_id: str,
    bin_edges: Sequence[tuple[int, str, tuple[float, ...]]],
    cells: Sequence[CalibrationCell],
    reachable: Sequence[tuple[int, tuple[str, ...]]],
    schema_version: str,
) -> CalibrationTable:
    provisional = CalibrationTable(
        schema_version=schema_version,
        candidate_id=candidate_id,
        bin_edges=tuple(bin_edges),
        cells=tuple(cells),
        reachable_query_ids_by_stage=tuple(reachable),
        table_fingerprint="",
    )
    payload = provisional.to_dict()
    payload.pop("fingerprint")
    return CalibrationTable(
        schema_version=provisional.schema_version,
        candidate_id=provisional.candidate_id,
        bin_edges=provisional.bin_edges,
        cells=provisional.cells,
        reachable_query_ids_by_stage=provisional.reachable_query_ids_by_stage,
        table_fingerprint=fingerprint(payload),
    )


def build_candidate_calibration_table(
    *,
    candidate: CandidateSpec,
    gain_model: LinearScoreModel,
    sufficiency_model: LinearScoreModel,
    prepared: Mapping[str, PreparedQuery],
    supervision: Mapping[str, Sequence[Mapping[str, Any]]],
    bound_query_ids: Sequence[str],
    config: Step3Config,
) -> tuple[CalibrationTable, tuple[dict[str, Any], ...]]:
    """Build candidate-specific stage tables from its own reachable queries."""
    calibration = config.section("calibration")
    upstream = next(iter(prepared.values()))
    stage_count = len(upstream.stages)
    bin_count = int(calibration["score_bin_count"])
    outcome_count = 2
    alpha_share = float(calibration["family_wise_alpha"]) / (
        len(config.candidates) * stage_count * bin_count * outcome_count
    )
    minimum_count = int(calibration["minimum_cell_query_count"])
    controller_config = config.section("controller")
    budget_grid = tuple(stage.decision_input.current_budget for stage in upstream.stages)
    maximum_expansions = len(budget_grid) - 1
    reached = tuple(sorted(bound_query_ids))
    cells: list[CalibrationCell] = []
    edge_rows: list[tuple[int, str, tuple[float, ...]]] = []
    reachability: list[tuple[int, tuple[str, ...]]] = []
    bound_records: list[dict[str, Any]] = []
    for stage_index in range(stage_count):
        reachability.append((stage_index, reached))
        states = {
            query_id: make_deployable_state(prepared[query_id].stages[stage_index], candidate)
            for query_id in reached
        }
        predictions = {
            "remaining_useful_evidence_event": {
                query_id: gain_model.predict(state) for query_id, state in states.items()
            },
            "current_context_sufficiency_event": {
                query_id: sufficiency_model.predict(state) for query_id, state in states.items()
            },
        }
        outcome_fields = {
            "remaining_useful_evidence_event": "later_context_has_useful_evidence",
            "current_context_sufficiency_event": "current_context_is_sufficient",
        }
        for outcome, model_predictions in predictions.items():
            eligible_scores = [
                prediction.score
                for query_id, prediction in model_predictions.items()
                if prediction.valid
                and states[query_id].base_state_valid
                and states[query_id].evidence_plan_valid
                and states[query_id].required_risk_profile_valid
            ]
            edges = _score_bin_edges(eligible_scores, config)
            edge_rows.append((stage_index, outcome, edges))
            for bin_index in range(bin_count):
                member_ids = []
                events = []
                for query_id, prediction in model_predictions.items():
                    state = states[query_id]
                    if not (
                        prediction.valid
                        and state.base_state_valid
                        and state.evidence_plan_valid
                        and state.required_risk_profile_valid
                    ):
                        continue
                    assigned = int(
                        np.searchsorted(edges[1:-1], prediction.score, side="right")
                    )
                    if assigned == bin_index:
                        member_ids.append(query_id)
                        labels = {int(row["stage"]): row for row in supervision[query_id]}
                        events.append(bool(labels[stage_index][outcome_fields[outcome]]))
                successes = int(sum(events))
                trials = len(member_ids)
                if trials < minimum_count:
                    lower, upper, valid = 0.0, 1.0, False
                else:
                    lower, upper = clopper_pearson_one_sided(successes, trials, alpha_share)
                    valid = True
                cells.append(
                    CalibrationCell(
                        candidate_id=candidate.candidate_id,
                        stage=stage_index,
                        outcome=outcome,
                        bin_index=bin_index,
                        lower_edge=edges[bin_index],
                        upper_edge=edges[bin_index + 1],
                        query_ids=tuple(sorted(member_ids)),
                        successes=successes,
                        trials=trials,
                        lower_limit=lower,
                        upper_limit=upper,
                        valid=valid,
                        alpha_share=alpha_share,
                    )
                )
        partial_table = _table_with_fingerprint(
            candidate.candidate_id,
            edge_rows,
            cells,
            reachability,
            calibration["version"],
        )
        controller = CalibratedStep3Controller(
            candidate=candidate,
            gain_model=gain_model,
            sufficiency_model=sufficiency_model,
            calibration_table=partial_table,
            budget_grid=budget_grid,
            maximum_expansions=maximum_expansions,
            delta_gain=float(controller_config["delta_gain"]),
            tau_sufficient=float(controller_config["tau_sufficient"]),
        )
        next_reached = []
        for query_id in reached:
            state = states[query_id]
            decision = controller.choose(state)
            labels = {int(row["stage"]): row for row in supervision[query_id]}
            bound_records.append(
                {
                    "schema_version": "tls_rag_step3_synthetic_bound_record_v1",
                    "candidate_id": candidate.candidate_id,
                    "row": candidate.row,
                    "query_id": query_id,
                    "stage": stage_index,
                    "remaining_score": decision.remaining_score,
                    "sufficiency_score": decision.sufficiency_score,
                    "remaining_bin": decision.remaining_bin,
                    "sufficiency_bin": decision.sufficiency_bin,
                    "remaining_cell_valid": decision.remaining_cell_valid,
                    "sufficiency_cell_valid": decision.sufficiency_cell_valid,
                    "observed_remaining_event": bool(
                        labels[stage_index]["later_context_has_useful_evidence"]
                    ),
                    "observed_sufficiency_event": bool(
                        labels[stage_index]["current_context_is_sufficient"]
                    ),
                    "action": decision.action.value,
                    "action_reason": decision.reason,
                }
            )
            if decision.action is Action.EXPAND_TO_NEXT_GRID_VALUE:
                next_reached.append(query_id)
        reached = tuple(sorted(next_reached))
    table = _table_with_fingerprint(
        candidate.candidate_id,
        edge_rows,
        cells,
        reachability,
        calibration["version"],
    )
    return table, tuple(bound_records)


def build_calibration_tables(
    *,
    candidates: Sequence[CandidateSpec],
    models: Sequence[tuple[LinearScoreModel, LinearScoreModel]],
    prepared: Mapping[str, PreparedQuery],
    supervision: Mapping[str, Sequence[Mapping[str, Any]]],
    bound_query_ids: Sequence[str],
    config: Step3Config,
) -> tuple[tuple[CalibrationTable, ...], tuple[dict[str, Any], ...]]:
    tables = []
    records = []
    for candidate, (gain_model, sufficiency_model) in zip(candidates, models):
        table, candidate_records = build_candidate_calibration_table(
            candidate=candidate,
            gain_model=gain_model,
            sufficiency_model=sufficiency_model,
            prepared=prepared,
            supervision=supervision,
            bound_query_ids=bound_query_ids,
            config=config,
        )
        tables.append(table)
        records.extend(candidate_records)
    return tuple(tables), tuple(records)


@dataclass(frozen=True)
class PhaseAResult:
    records: tuple[dict[str, Any], ...]
    timing_records: tuple[dict[str, Any], ...]
    decision_fingerprint: str

    def recompute_fingerprint(self) -> str:
        return fingerprint(
            {
                "schema_version": "tls_rag_step3_closed_phase_a_trajectory_v1",
                "records": list(self.records),
            }
        )


@dataclass(frozen=True)
class PhaseBResult:
    records: tuple[dict[str, Any], ...]
    before_fingerprint: str
    after_fingerprint: str
    supervision_fingerprint: str


def _retrieval_state_record(stage: PreparedStage) -> dict[str, Any]:
    state = stage.decision_input
    return {
        "query_id": state.query_id,
        "stage": state.step,
        "current_budget": state.current_budget,
        "remaining_grid_steps": state.remaining_grid_steps,
        "exposed_candidate_ids": list(state.exposed_candidate_ids),
        "projected_ranks": list(state.projected_ranks),
        "projected_squared_distances": list(state.projected_squared_distances),
        "cached_original_squared_distances": list(state.cached_original_squared_distances),
        "exact_reranked_ids": list(state.exact_reranked_ids),
        "exact_reranked_squared_distances": list(state.exact_reranked_squared_distances),
        "context_ids": list(state.context_ids),
        "validity": state.validity.to_dict(),
    }


def _incremental_work(stage: PreparedStage, candidate: Optional[CandidateSpec]) -> dict[str, int]:
    base = dict(stage.base_work)
    uses_tri = candidate is not None and candidate.uses_tri_law
    uses_plan = candidate is not None and candidate.uses_plan_facet
    work = {
        "query_projection_count": base["query_projection_count"],
        "projected_full_scan_count": base["projected_full_scan_count"],
        "projected_distance_evaluations": base["projected_distance_evaluations"],
        "pilot_prefix_exposure_count": base["pilot_prefix_exposure_count"],
        "expansion_prefix_reuse_count": base["expansion_prefix_reuse_count"],
        "new_original_distance_evaluations": base["new_original_distance_evaluations"],
        "exact_rerank_count": base["exact_rerank_count"],
        "accumulated_rerank_candidates": base["accumulated_rerank_candidates"],
        "risk_pair_evaluations": (
            stage.risk_profile.context_boundary.attempted_pair_count
            + stage.risk_profile.core_frontier.attempted_pair_count
            if uses_tri else 0
        ),
        "risk_feature_evaluation_count": int(uses_tri),
        "deterministic_plan_facet_evaluation_count": int(uses_plan),
        "remaining_score_inference_count": int(candidate is not None),
        "sufficiency_score_inference_count": int(candidate is not None),
        "calibration_lookup_count": 2 if candidate is not None else 0,
        "controller_evaluation_count": int(candidate is not None),
        "fixed_reference_evaluation_count": int(candidate is None),
        "final_context_construction_count": base["final_context_construction_count"],
        "final_context_candidates": base["final_context_candidates"],
    }
    if tuple(work) != WORK_FIELDS:
        raise AssertionError("Step 3 work schema drift")
    return work


def _cumulative_fixed_work(stages: Sequence[PreparedStage]) -> dict[str, int]:
    if not stages:
        raise ValueError("fixed reference requires one frozen budget state")
    first = dict(stages[0].base_work)
    final = dict(stages[-1].base_work)
    total = {name: 0 for name in WORK_FIELDS}
    total.update(
        {
            "query_projection_count": first["query_projection_count"],
            "projected_full_scan_count": first["projected_full_scan_count"],
            "projected_distance_evaluations": first["projected_distance_evaluations"],
            "pilot_prefix_exposure_count": 1,
            "new_original_distance_evaluations": sum(
                dict(stage.base_work)["new_original_distance_evaluations"]
                for stage in stages
            ),
            "exact_rerank_count": 1,
            "accumulated_rerank_candidates": final["accumulated_rerank_candidates"],
            "fixed_reference_evaluation_count": 1,
            "final_context_construction_count": 1,
            "final_context_candidates": final["final_context_candidates"],
        }
    )
    return total


def _fixed_timing_record(stages: Sequence[PreparedStage]) -> dict[str, float]:
    if not stages:
        raise ValueError("fixed reference requires one frozen budget state")
    first = dict(stages[0].base_timing)
    final = dict(stages[-1].base_timing)
    timing = {name: 0.0 for name in TIMING_FIELDS}
    timing.update(
        {
            "query_projection_ms": first["query_projection_ms"],
            "projected_full_scan_and_pilot_ms": first["projected_full_scan_and_pilot_ms"],
            "new_original_distance_evaluation_ms": sum(
                dict(stage.base_timing)["new_original_distance_evaluation_ms"]
                for stage in stages
            ),
            "exact_reranking_ms": final["exact_reranking_ms"],
            "final_context_ms": final["final_context_ms"],
        }
    )
    return timing


def _base_timing_record(stage: PreparedStage, candidate: Optional[CandidateSpec]) -> dict[str, float]:
    base = dict(stage.base_timing)
    return {
        "query_projection_ms": base["query_projection_ms"],
        "projected_full_scan_and_pilot_ms": base["projected_full_scan_and_pilot_ms"],
        "expansion_prefix_reuse_ms": base["expansion_prefix_reuse_ms"],
        "new_original_distance_evaluation_ms": base["new_original_distance_evaluation_ms"],
        "exact_reranking_ms": base["exact_reranking_ms"],
        "risk_feature_ms": base["risk_feature_ms"] if candidate is not None and candidate.uses_tri_law else 0.0,
        "deterministic_plan_facet_ms": base["deterministic_plan_facet_ms"] if candidate is not None and candidate.uses_plan_facet else 0.0,
        "remaining_score_inference_ms": 0.0,
        "sufficiency_score_inference_ms": 0.0,
        "calibration_lookup_ms": 0.0,
        "controller_ms": 0.0,
        "final_context_ms": base["final_context_ms"],
    }


def _terminal_flags(
    state: Step3DeployableState, decision: ControllerEvaluation, maximum_expansions: int
) -> list[str]:
    if decision.action is not Action.STOP:
        return []
    flags = []
    if state.remaining_grid_steps == 0:
        flags.append("corpus_exhausted")
    if state.stage >= maximum_expansions:
        flags.append("maximum_expansions_reached")
    if not state.evidence_plan_valid:
        flags.append("invalid_evidence_plan")
    if not (
        state.base_state_valid
        and state.required_risk_profile_valid
        and decision.remaining_cell_valid
        and decision.sufficiency_cell_valid
    ):
        flags.append("invalid_state")
    if decision.reason == "evidence_nonattainment":
        flags.append("evidence_nonattainment")
    if not flags:
        flags.append("calibrated_dual_bound_stop")
    return flags


def run_evaluation_phase_a(
    *,
    config: Step3Config,
    environment: Step3Environment,
    prepared: Mapping[str, PreparedQuery],
    models: Sequence[tuple[LinearScoreModel, LinearScoreModel]],
    tables: Sequence[CalibrationTable],
) -> PhaseAResult:
    """Close all evaluation actions without receiving an evaluation label store."""
    evaluation_ids = environment.partition("synthetic_evaluation")
    model_map = {
        candidate.candidate_id: pair for candidate, pair in zip(config.candidates, models)
    }
    table_map = {
        candidate.candidate_id: table for candidate, table in zip(config.candidates, tables)
    }
    controller_config = config.section("controller")
    grid = environment.upstream.config.budget_grid
    maximum_expansions = environment.upstream.config.maximum_expansions
    records: list[dict[str, Any]] = []
    timing_records: list[dict[str, Any]] = []
    for query_id in evaluation_ids:
        prepared_query = prepared[query_id]
        for stage_index, fixed_budget in enumerate(grid):
            stage = prepared_query.stages[stage_index]
            cumulative_work = _cumulative_fixed_work(prepared_query.stages[: stage_index + 1])
            record = {
                "schema_version": PHASE_A_SCHEMA,
                "method_id": f"tls-rag-row1-fixed-{fixed_budget}-v1",
                "row": 1,
                "query_id": query_id,
                "stage": stage_index,
                "budget": fixed_budget,
                "retrieval_state": _retrieval_state_record(stage),
                "controller_feature_names": [],
                "controller_feature_values": {},
                "action": Action.STOP.value,
                "action_reason": "fixed_grid_reference",
                "next_budget": None,
                "terminal_flags": ["fixed_grid_reference"],
                "work": cumulative_work,
            }
            assert_deployable_only(record)
            records.append(record)
            timing = _fixed_timing_record(prepared_query.stages[: stage_index + 1])
            timing_records.append(
                {"method_id": record["method_id"], "query_id": query_id, "stage": stage_index, **timing}
            )
        for candidate in config.candidates:
            gain_model, sufficiency_model = model_map[candidate.candidate_id]
            table = table_map[candidate.candidate_id]
            controller = CalibratedStep3Controller(
                candidate=candidate,
                gain_model=gain_model,
                sufficiency_model=sufficiency_model,
                calibration_table=table,
                budget_grid=grid,
                maximum_expansions=maximum_expansions,
                delta_gain=float(controller_config["delta_gain"]),
                tau_sufficient=float(controller_config["tau_sufficient"]),
            )
            for stage in prepared_query.stages:
                state = make_deployable_state(stage, candidate)
                gain_started = perf_counter()
                gain_model.predict(state)
                gain_ms = (perf_counter() - gain_started) * 1000.0
                suff_started = perf_counter()
                sufficiency_model.predict(state)
                suff_ms = (perf_counter() - suff_started) * 1000.0
                lookup_started = perf_counter()
                gain_probe = gain_model.predict(state)
                suff_probe = sufficiency_model.predict(state)
                if gain_probe.valid:
                    try:
                        table.lookup(stage.stage, "remaining_useful_evidence_event", gain_probe.score)
                    except KeyError:
                        pass
                if suff_probe.valid:
                    try:
                        table.lookup(stage.stage, "current_context_sufficiency_event", suff_probe.score)
                    except KeyError:
                        pass
                lookup_ms = (perf_counter() - lookup_started) * 1000.0
                controller_started = perf_counter()
                decision = controller.choose(state)
                controller_ms = (perf_counter() - controller_started) * 1000.0
                record = {
                    "schema_version": PHASE_A_SCHEMA,
                    "method_id": candidate.candidate_id,
                    "row": candidate.row,
                    "query_id": query_id,
                    "stage": stage.stage,
                    "budget": state.current_budget,
                    "retrieval_state": _retrieval_state_record(stage),
                    "controller_state": state.to_dict(),
                    "controller_feature_names": list(feature_names_for_row(candidate.row)),
                    "controller_feature_values": dict(state.feature_values),
                    "observed_pair_profile": stage.risk_profile.to_dict() if candidate.uses_tri_law else None,
                    "deterministic_plan_facet_features": dict(stage.plan_facet_features) if candidate.uses_plan_facet else None,
                    "remaining_score": decision.remaining_score,
                    "sufficiency_score": decision.sufficiency_score,
                    "remaining_score_valid": decision.remaining_score_valid,
                    "sufficiency_score_valid": decision.sufficiency_score_valid,
                    "remaining_bin": decision.remaining_bin,
                    "sufficiency_bin": decision.sufficiency_bin,
                    "remaining_upper_limit": decision.remaining_upper_limit,
                    "sufficiency_lower_limit": decision.sufficiency_lower_limit,
                    "remaining_cell_valid": decision.remaining_cell_valid,
                    "sufficiency_cell_valid": decision.sufficiency_cell_valid,
                    "action": decision.action.value,
                    "action_reason": decision.reason,
                    "next_budget": decision.next_budget,
                    "terminal_flags": _terminal_flags(state, decision, maximum_expansions),
                    "work": _incremental_work(stage, candidate),
                }
                assert_deployable_only(record)
                records.append(record)
                timing = _base_timing_record(stage, candidate)
                timing["remaining_score_inference_ms"] = gain_ms
                timing["sufficiency_score_inference_ms"] = suff_ms
                timing["calibration_lookup_ms"] = lookup_ms
                timing["controller_ms"] = controller_ms
                timing_records.append(
                    {"method_id": candidate.candidate_id, "query_id": query_id, "stage": stage.stage, **timing}
                )
                if decision.action is Action.STOP:
                    break
                if decision.next_budget != grid[stage.stage + 1]:
                    raise AssertionError("Step 3 controller skipped a grid value")
    provisional = PhaseAResult(tuple(records), tuple(timing_records), "")
    return PhaseAResult(
        provisional.records,
        provisional.timing_records,
        provisional.recompute_fingerprint(),
    )


def join_evaluation_phase_b(
    phase_a: PhaseAResult,
    supervision: Mapping[str, Sequence[Mapping[str, Any]]],
) -> PhaseBResult:
    before = phase_a.recompute_fingerprint()
    if before != phase_a.decision_fingerprint:
        raise ValueError("Phase A changed before evaluation label join")
    rows = []
    for decision in phase_a.records:
        labels = {
            int(item["stage"]): item for item in supervision[decision["query_id"]]
        }[int(decision["stage"])]
        rows.append(
            {
                "schema_version": PHASE_B_SCHEMA,
                "method_id": decision["method_id"],
                "row": decision["row"],
                "query_id": decision["query_id"],
                "stage": decision["stage"],
                "budget": decision["budget"],
                "phase_a_decision_fingerprint": phase_a.decision_fingerprint,
                **{key: value for key, value in labels.items() if key not in {"query_id", "stage", "budget"}},
            }
        )
    after = phase_a.recompute_fingerprint()
    if before != after:
        raise AssertionError("evaluation supervision mutated Phase A")
    payload = {
        "schema_version": "tls_rag_step3_phase_b_records_v1",
        "phase_a_decision_fingerprint": phase_a.decision_fingerprint,
        "records": rows,
    }
    return PhaseBResult(tuple(rows), before, after, fingerprint(payload))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                    allow_nan=False,
                ) + "\n"
            )


def _fit_records(
    config: Step3Config,
    prepared: Mapping[str, PreparedQuery],
    supervision: Mapping[str, Sequence[Mapping[str, Any]]],
    fit_ids: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    rows = []
    for candidate in config.candidates:
        for query_id in fit_ids:
            labels = {int(item["stage"]): item for item in supervision[query_id]}
            for stage in prepared[query_id].stages:
                state = make_deployable_state(stage, candidate)
                rows.append(
                    {
                        "schema_version": "tls_rag_step3_synthetic_fit_record_v1",
                        "candidate_id": candidate.candidate_id,
                        "row": candidate.row,
                        "query_id": query_id,
                        "stage": stage.stage,
                        "feature_values": dict(state.feature_values),
                        "input_valid": bool(
                            state.base_state_valid
                            and state.evidence_plan_valid
                            and state.required_risk_profile_valid
                        ),
                        "target_remaining_event": bool(
                            labels[stage.stage]["later_context_has_useful_evidence"]
                        ),
                        "target_sufficiency_event": bool(
                            labels[stage.stage]["current_context_is_sufficient"]
                        ),
                    }
                )
    return tuple(rows)


def _artifact_with_fingerprint(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["fingerprint"] = fingerprint(payload)
    return result


def _canonicalize_floats(value: Any, decimals: int) -> Any:
    """Canonicalize only finite floating leaves for cross-platform audit hashes."""
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_floats(child, decimals)
            for key, child in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_canonicalize_floats(child, decimals) for child in value]
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("cannot canonicalize a nonfinite Step 2 trajectory value")
        rounded = round(numeric, decimals)
        return 0.0 if rounded == 0.0 else rounded
    return value


def step2_semantic_trajectory_fingerprint(
    phase_a: Any, *, float_canonical_decimals: int
) -> str:
    """Hash all Step 2 decision fields after a frozen decimal float lattice."""
    if (
        isinstance(float_canonical_decimals, bool)
        or not isinstance(float_canonical_decimals, int)
        or not 0 <= float_canonical_decimals <= 15
    ):
        raise ValueError("float_canonical_decimals must be an integer in [0,15]")
    payload = {
        "schema_version": "tls_rag_step2_semantic_trajectory_v1",
        "config_fingerprint": phase_a.config_fingerprint,
        "fixture_fingerprint": phase_a.fixture_fingerprint,
        "float_canonical_decimals": float_canonical_decimals,
        "records": _canonicalize_floats(
            phase_a.portable_records(), float_canonical_decimals
        ),
    }
    return fingerprint(payload)


def step2_semantic_supervision_fingerprint(
    phase_b: Any,
    *,
    phase_a_semantic_fingerprint: str,
    float_canonical_decimals: int,
) -> str:
    """Hash Step 2 supervision against the semantic, not raw, Phase A identity."""
    if (
        not isinstance(phase_a_semantic_fingerprint, str)
        or len(phase_a_semantic_fingerprint) != 64
    ):
        raise ValueError("phase_a_semantic_fingerprint must be a SHA-256 hex string")
    normalized_records = []
    for raw_record in phase_b.supervision_records:
        record = dict(raw_record)
        if "phase_a_decision_fingerprint" not in record:
            raise ValueError("Step 2 supervision record lacks its Phase A identity")
        record["phase_a_decision_fingerprint"] = phase_a_semantic_fingerprint
        normalized_records.append(record)
    payload = {
        "schema_version": "tls_rag_step2_semantic_supervision_v1",
        "phase_a_semantic_trajectory_fingerprint": phase_a_semantic_fingerprint,
        "float_canonical_decimals": float_canonical_decimals,
        "records": _canonicalize_floats(
            normalized_records, float_canonical_decimals
        ),
    }
    return fingerprint(payload)


def _work_artifact(phase_a: PhaseAResult) -> dict[str, Any]:
    records = [
        {
            "method_id": row["method_id"],
            "query_id": row["query_id"],
            "stage": row["stage"],
            **row["work"],
        }
        for row in phase_a.records
    ]
    totals = {name: sum(row[name] for row in records) for name in WORK_FIELDS}
    by_method = {}
    for method_id in sorted({row["method_id"] for row in records}):
        subset = [row for row in records if row["method_id"] == method_id]
        by_method[method_id] = {
            name: sum(row[name] for row in subset) for name in WORK_FIELDS
        }
    return {
        "schema_version": "tls_rag_step3_work_counters_v1",
        "records": records,
        "totals": totals,
        "totals_by_method": by_method,
    }


def _aggregates(phase_a: PhaseAResult, phase_b: PhaseBResult) -> dict[str, Any]:
    methods = sorted({row["method_id"] for row in phase_a.records})
    by_method = {}
    for method_id in methods:
        decisions = [row for row in phase_a.records if row["method_id"] == method_id]
        joined = [row for row in phase_b.records if row["method_id"] == method_id]
        final_decisions = {}
        final_joined = {}
        for row in decisions:
            final_decisions[row["query_id"]] = row
        for row in joined:
            final_joined[row["query_id"]] = row
        by_method[method_id] = {
            "row": decisions[0]["row"],
            "query_count": len(final_decisions),
            "stage_record_count": len(decisions),
            "stop_count": sum(row["action"] == Action.STOP.value for row in decisions),
            "expand_count": sum(
                row["action"] == Action.EXPAND_TO_NEXT_GRID_VALUE.value for row in decisions
            ),
            "mean_terminal_budget": float(
                np.mean([row["budget"] for row in final_decisions.values()])
            ),
            "mean_terminal_context_coverage": float(
                np.mean([row["context_coverage"] for row in final_joined.values()])
            ),
            "mean_terminal_exact_top_k_retention": float(
                np.mean([row["exact_top_k_retention"] for row in final_joined.values()])
            ),
            "terminal_context_sufficient_count": sum(
                row["current_context_is_sufficient"] for row in final_joined.values()
            ),
            "terminal_reason_counts": {
                reason: sum(row["action_reason"] == reason for row in final_decisions.values())
                for reason in sorted({row["action_reason"] for row in final_decisions.values()})
            },
        }
    return {
        "schema_version": "tls_rag_step3_diagnostic_aggregates_v1",
        "phase_a_decision_fingerprint": phase_a.decision_fingerprint,
        "phase_b_supervision_fingerprint": phase_b.supervision_fingerprint,
        "methods": by_method,
        "interpretation": "synthetic code-path diagnostics only; no method was selected or certified",
    }


def _ablation_records(
    phase_a: PhaseAResult, phase_b: PhaseBResult
) -> tuple[dict[str, Any], ...]:
    labels = {
        (row["method_id"], row["query_id"], row["stage"]): row
        for row in phase_b.records
    }
    return tuple(
        {
            "schema_version": "tls_rag_step3_ablation_query_stage_v1",
            "method_id": row["method_id"],
            "row": row["row"],
            "query_id": row["query_id"],
            "stage": row["stage"],
            "budget": row["budget"],
            "feature_names": row["controller_feature_names"],
            "action": row["action"],
            "action_reason": row["action_reason"],
            "context_coverage": labels[(row["method_id"], row["query_id"], row["stage"])]["context_coverage"],
            "current_context_is_sufficient": labels[(row["method_id"], row["query_id"], row["stage"])]["current_context_is_sufficient"],
            "later_context_has_useful_evidence": labels[(row["method_id"], row["query_id"], row["stage"])]["later_context_has_useful_evidence"],
            "exact_top_k_retention": labels[(row["method_id"], row["query_id"], row["stage"])]["exact_top_k_retention"],
            "work": row["work"],
        }
        for row in phase_a.records
    )


def run_step3(config: Step3Config, output_dir: Path) -> dict[str, Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    offline_started = perf_counter()
    bundle = build_step3_environment(config)
    environment = bundle.environment

    # Prove the frozen Step 2 path still resolves to its exact identities.
    step2_controller = FixedScheduleController(environment.upstream.config)
    step2_phase_a = run_step2_phase_a(environment.upstream, step2_controller)
    step2_phase_b = join_phase_b(
        step2_phase_a,
        environment.upstream,
        build_evidence_label_store(environment.upstream),
    )
    upstream_frozen = config.section("upstream_step2")
    step2_phase_a_exact_match = (
        step2_phase_a.decision_fingerprint == upstream_frozen["phase_a_fingerprint"]
    )
    step2_phase_a_semantic_fingerprint = step2_semantic_trajectory_fingerprint(
        step2_phase_a,
        float_canonical_decimals=int(
            upstream_frozen["phase_a_float_canonical_decimals"]
        ),
    )
    step2_phase_a_semantic_match = (
        step2_phase_a_semantic_fingerprint
        == upstream_frozen["phase_a_semantic_fingerprint"]
    )
    if not step2_phase_a_exact_match and not step2_phase_a_semantic_match:
        raise ValueError(
            "frozen Step 2 Phase A trajectory changed beyond the allowed "
            "cross-platform float lattice; "
            f"observed_exact={step2_phase_a.decision_fingerprint}, "
            f"expected_exact={upstream_frozen['phase_a_fingerprint']}, "
            f"observed_semantic={step2_phase_a_semantic_fingerprint}, "
            f"expected_semantic={upstream_frozen['phase_a_semantic_fingerprint']}"
        )
    step2_phase_b_exact_match = (
        step2_phase_b.supervision_fingerprint == upstream_frozen["phase_b_fingerprint"]
    )
    step2_phase_b_semantic_fingerprint = step2_semantic_supervision_fingerprint(
        step2_phase_b,
        phase_a_semantic_fingerprint=step2_phase_a_semantic_fingerprint,
        float_canonical_decimals=int(
            upstream_frozen["phase_a_float_canonical_decimals"]
        ),
    )
    step2_phase_b_semantic_match = (
        step2_phase_b_semantic_fingerprint
        == upstream_frozen["phase_b_semantic_fingerprint"]
    )
    if not step2_phase_b_exact_match and not step2_phase_b_semantic_match:
        raise ValueError(
            "frozen Step 2 Phase B supervision changed beyond the allowed "
            "cross-platform float lattice; "
            f"observed_exact={step2_phase_b.supervision_fingerprint}, "
            f"expected_exact={upstream_frozen['phase_b_fingerprint']}, "
            f"observed_semantic={step2_phase_b_semantic_fingerprint}, "
            f"expected_semantic={upstream_frozen['phase_b_semantic_fingerprint']}"
        )

    prepared_tuple = prepare_queries(environment, config)
    prepared = {item.query_id: item for item in prepared_tuple}
    fixture_feature_setup_ms = (perf_counter() - offline_started) * 1000.0
    fit_ids = environment.partition("synthetic_model_fit")
    bound_ids = environment.partition("synthetic_bound_fit")
    evaluation_ids = environment.partition("synthetic_evaluation")

    model_started = perf_counter()
    fit_store = build_synthetic_label_store(bundle, prepared, fit_ids)
    fit_supervision = _supervision_by_query(
        fit_ids, prepared, environment, fit_store
    )
    models = fit_score_models(
        config.candidates, prepared, fit_supervision, fit_ids, config
    )
    model_ms = (perf_counter() - model_started) * 1000.0
    fit_records = _fit_records(config, prepared, fit_supervision, fit_ids)

    calibration_started = perf_counter()
    bound_store = build_synthetic_label_store(bundle, prepared, bound_ids)
    bound_supervision = _supervision_by_query(
        bound_ids, prepared, environment, bound_store
    )
    tables, bound_records = build_calibration_tables(
        candidates=config.candidates,
        models=models,
        prepared=prepared,
        supervision=bound_supervision,
        bound_query_ids=bound_ids,
        config=config,
    )
    calibration_ms = (perf_counter() - calibration_started) * 1000.0

    # Evaluation supervision is still unopened here.
    phase_a = run_evaluation_phase_a(
        config=config,
        environment=environment,
        prepared=prepared,
        models=models,
        tables=tables,
    )
    _write_jsonl(output_dir / "phase_a_decisions.jsonl", phase_a.records)
    if phase_a.recompute_fingerprint() != phase_a.decision_fingerprint:
        raise AssertionError("serialized Step 3 Phase A fingerprint mismatch")

    # Only after serialization do we construct and join evaluation labels.
    evaluation_join_started = perf_counter()
    evaluation_store = build_synthetic_label_store(bundle, prepared, evaluation_ids)
    evaluation_supervision = _supervision_by_query(
        evaluation_ids, prepared, environment, evaluation_store
    )
    phase_b = join_evaluation_phase_b(phase_a, evaluation_supervision)
    evaluation_join_ms = (perf_counter() - evaluation_join_started) * 1000.0
    _write_jsonl(output_dir / "phase_b_supervision.jsonl", phase_b.records)
    ablation_records = _ablation_records(phase_a, phase_b)
    _write_jsonl(output_dir / "ablation_records.jsonl", ablation_records)
    _write_jsonl(output_dir / "synthetic_fit_records.jsonl", fit_records)
    _write_jsonl(output_dir / "synthetic_bound_records.jsonl", bound_records)

    config_identity = _artifact_with_fingerprint(
        {
            "schema_version": "tls_rag_step3_config_identity_v1",
            "config": config.raw,
            "config_fingerprint": config.config_fingerprint,
        }
    )
    query_matrix = np.vstack([query.embedding for query in environment.queries])
    fixture_identity = _artifact_with_fingerprint(
        {
            "schema_version": "tls_rag_step3_fixture_identity_v1",
            "fixture_fingerprint": environment.fixture_fingerprint,
            "upstream_step2_fixture_fingerprint": environment.upstream.fixture_fingerprint,
            "corpus_embedding_hash": array_fingerprint(environment.upstream.corpus_embeddings),
            "synthetic_query_embedding_hash": array_fingerprint(query_matrix),
            "synthetic_blueprint_fingerprint": fingerprint(
                [
                    {
                        "query_id": item.query_id,
                        "partition": item.partition,
                        "label_mode": item.label_mode,
                    }
                    for item in bundle.blueprints
                ]
            ),
            "fit_label_store_fingerprint": fit_store.to_dict()["fingerprint"],
            "bound_label_store_fingerprint": bound_store.to_dict()["fingerprint"],
            "evaluation_label_store_fingerprint": evaluation_store.to_dict()["fingerprint"],
        }
    )
    projection = projection_metadata(
        dimension=environment.upstream.config.dimension,
        m_prime=environment.upstream.config.m_prime,
        seed=environment.upstream.config.projection_seed,
        normalization=True,
        embedding_model="tls_rag_synthetic_array_fixture@step3_reuses_step2",
        corpus_hash=array_fingerprint(environment.upstream.corpus_embeddings),
    )
    projection["matrix_hash"] = array_fingerprint(environment.upstream.projection_matrix)
    projection["schema_version"] = "tls_rag_step3_projection_identity_v1"
    projection["fingerprint"] = fingerprint(
        {key: value for key, value in projection.items() if key != "fingerprint"}
    )
    id_maps = _artifact_with_fingerprint(
        {
            "schema_version": "tls_rag_step3_ordered_id_maps_v1",
            "corpus_ids": list(environment.upstream.corpus_ids),
            "synthetic_query_ids": [query.query_id for query in environment.queries],
        }
    )
    risk_spec = _artifact_with_fingerprint(
        {
            "schema_version": "tls_rag_step3_risk_feature_spec_v1",
            "config": config.section("risk_profile"),
            "pair_orientation": "strictly_smaller_original_displacement_then_farther",
            "pair_input_scope": "current_exposed_prefix_only",
            "context_boundary": "farthest_context_vs_each_exposed_noncontext",
            "core_frontier": "each_context_vs_frozen_newest_shell",
            "interpretation": "dependent ex-ante single-pair geometric features; not posterior probabilities or guarantees",
        }
    )
    feature_registry = _artifact_with_fingerprint(
        {
            "schema_version": "tls_rag_step3_feature_registry_v1",
            "rows": {
                "2": list(feature_names_for_row(2)),
                "3": list(feature_names_for_row(3)),
                "4": list(feature_names_for_row(4)),
            },
            "row3_minus_row2": list(TRI_LAW_FEATURE_NAMES),
            "row4_minus_row3": list(PLAN_FACET_FEATURE_NAMES),
        }
    )
    partitions = _artifact_with_fingerprint(
        {
            "schema_version": "tls_rag_step3_synthetic_partition_proof_v1",
            "assignment": "salted_sha256_query_id_modulus_label_free",
            "salt": config.section("seeds")["partition_hash_salt"],
            "partitions": {name: list(environment.partition(name)) for name in SYNTHETIC_PARTITIONS},
            "pairwise_disjoint": all(
                not set(environment.partition(left)).intersection(environment.partition(right))
                for left_index, left in enumerate(SYNTHETIC_PARTITIONS)
                for right in SYNTHETIC_PARTITIONS[left_index + 1 :]
            ),
            "real_role_names_used": False,
        }
    )
    candidate_registry = _artifact_with_fingerprint(
        {
            "schema_version": "tls_rag_step3_candidate_registry_v1",
            "fixed_row1_budgets": list(config.section("controller")["fixed_reference_budgets"]),
            "adaptive_candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "row": item.row,
                    "feature_schema_fingerprint": fingerprint(
                        {
                            "schema_version": "tls_rag_step3_feature_names_v1",
                            "names": list(feature_names_for_row(item.row)),
                        }
                    ),
                }
                for item in config.candidates
            ],
            "selection_performed": False,
        }
    )
    score_models = _artifact_with_fingerprint(
        {
            "schema_version": "tls_rag_step3_score_models_v1",
            "models": [model.to_dict() for pair in models for model in pair],
            "fit_partition": "synthetic_model_fit",
            "missing_behavior": config.section("score_model")["missing_behavior"],
        }
    )
    calibration_tables = _artifact_with_fingerprint(
        {
            "schema_version": "tls_rag_step3_calibration_tables_v1",
            "tables": [table.to_dict() for table in tables],
            "family_wise_alpha": config.section("calibration")["family_wise_alpha"],
            "allocation": config.section("calibration")["allocation"],
            "independent_unit": "query_id",
            "interpretation": "candidate-stage-bin-reachability event-rate limits, not individual posteriors",
        }
    )
    work = _work_artifact(phase_a)
    aggregates = _aggregates(phase_a, phase_b)
    report = "\n".join(
        (
            "# TLS-RAG Step 3 synthetic diagnostics",
            "",
            "This CPU/network-free run evaluates all frozen Rows 1--4 on deterministic synthetic external queries. It reports observed-pair Tri-Law geometric features, transparent scores, candidate-specific reachable-bin Clopper--Pearson limits, and conservative dual-bound actions.",
            "",
            f"- Synthetic evaluation queries: {len(evaluation_ids)}",
            f"- Phase A fingerprint: `{phase_a.decision_fingerprint}`",
            f"- Phase B fingerprint: `{phase_b.supervision_fingerprint}`",
            f"- Candidate table fingerprints: {', '.join(f'`{table.table_fingerprint}`' for table in tables)}",
            "",
            "No real or protected data was accessed. No method was selected or scientifically certified. Timings are nonportable code-path measurements, not a latency claim. Tri-Law aggregates are dependent ex-ante geometric features, not posterior missing-evidence or relevance probabilities. The bounds do not guarantee evidence sufficiency or answer correctness. No approximate index, GPU, network, LLM, answer generation, answer evaluation, or Step 4 work was used.",
            "",
        )
    )
    timings = {
        "schema_version": "tls_rag_step3_nonportable_timings_v1",
        "portable": False,
        "offline_fixture_feature_setup_ms": fixture_feature_setup_ms,
        "offline_model_fit_ms": model_ms,
        "offline_calibration_setup_ms": calibration_ms,
        "offline_evaluation_label_join_ms": evaluation_join_ms,
        "query_stage_records": list(phase_a.timing_records),
        "totals_ms": {
            name: float(sum(row[name] for row in phase_a.timing_records))
            for name in TIMING_FIELDS
        },
    }
    manifest_payload = {
        "schema_version": "tls_rag_step3_manifest_v1",
        "run_name": config.raw["run_name"],
        "scope": "cpu_network_free_pure_synthetic_step3_diagnostics",
        "config_fingerprint": config.config_fingerprint,
        "fixture_fingerprint": environment.fixture_fingerprint,
        "projection_fingerprint": projection["fingerprint"],
        "id_map_fingerprint": id_maps["fingerprint"],
        "risk_feature_spec_fingerprint": risk_spec["fingerprint"],
        "feature_registry_fingerprint": feature_registry["fingerprint"],
        "partition_fingerprint": partitions["fingerprint"],
        "candidate_registry_fingerprint": candidate_registry["fingerprint"],
        "score_models_fingerprint": score_models["fingerprint"],
        "calibration_tables_fingerprint": calibration_tables["fingerprint"],
        "phase_a_decision_fingerprint": phase_a.decision_fingerprint,
        "phase_b_supervision_fingerprint": phase_b.supervision_fingerprint,
        "phase_a_serialized_before_evaluation_label_store_opened": True,
        "phase_a_fingerprint_unchanged_by_join": phase_b.before_fingerprint == phase_b.after_fingerprint,
        "step2_compatibility": {
            "phase_a_observed_fingerprint": step2_phase_a.decision_fingerprint,
            "phase_a_reference_fingerprint": upstream_frozen["phase_a_fingerprint"],
            "phase_a_exact_reference_match": step2_phase_a_exact_match,
            "phase_a_semantic_fingerprint": step2_phase_a_semantic_fingerprint,
            "phase_a_semantic_reference_fingerprint": upstream_frozen[
                "phase_a_semantic_fingerprint"
            ],
            "phase_a_semantic_reference_match": step2_phase_a_semantic_match,
            "phase_a_float_canonical_decimals": upstream_frozen[
                "phase_a_float_canonical_decimals"
            ],
            "phase_b_observed_fingerprint": step2_phase_b.supervision_fingerprint,
            "phase_b_reference_fingerprint": upstream_frozen["phase_b_fingerprint"],
            "phase_b_exact_reference_match": step2_phase_b_exact_match,
            "phase_b_semantic_fingerprint": step2_phase_b_semantic_fingerprint,
            "phase_b_semantic_reference_fingerprint": upstream_frozen[
                "phase_b_semantic_fingerprint"
            ],
            "phase_b_semantic_reference_match": step2_phase_b_semantic_match,
        },
        "portable_artifacts": list(PORTABLE_ARTIFACTS),
        "timings_portable": False,
        "prohibitions_observed": config.section("scope"),
        "claims_denied": [
            "real_data", "selection", "certification", "latency",
            "posterior_probability", "evidence_guarantee", "answer_quality",
        ],
    }
    manifest = _artifact_with_fingerprint(manifest_payload)
    artifacts = {
        "manifest.json": manifest,
        "config_identity.json": config_identity,
        "fixture_identity.json": fixture_identity,
        "projection.json": projection,
        "id_maps.json": id_maps,
        "risk_feature_spec.json": risk_spec,
        "feature_registry.json": feature_registry,
        "synthetic_partitions.json": partitions,
        "candidate_registry.json": candidate_registry,
        "score_models.json": score_models,
        "calibration_tables.json": calibration_tables,
        "work_counters.json": work,
        "aggregates.json": aggregates,
        "timings.json": timings,
    }
    for name, value in artifacts.items():
        write_json(output_dir / name, value)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return {name: output_dir / name for name in (*PORTABLE_ARTIFACTS, "timings.json")}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    config = load_step3_config(args.config)
    artifacts = run_step3(config, args.output)
    print(f"completed {config.raw['run_name']}: {artifacts['report.md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
