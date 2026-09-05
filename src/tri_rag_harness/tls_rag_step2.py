"""TLS-RAG Step 2: deterministic, label-isolated synthetic skeleton.

This module intentionally contains no learned controller, calibration, risk
aggregation, real-data adapter, approximate index, or answer-generation path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .embeddings import normalize_rows
from .indexes import ExactSquaredL2Index
from .projection import dense_gaussian_projection, project_rows, projection_metadata
from .utils import array_fingerprint, fingerprint, write_json


CONFIG_SCHEMA = "tls_rag_step2_config_v1"
DECISION_SCHEMA = "tls_rag_decision_input_v1"
PHASE_A_SCHEMA = "tls_rag_phase_a_decision_v1"
PHASE_B_SCHEMA = "tls_rag_phase_b_supervision_v1"

QUERY_IDS = (
    "tls-query-candidate-without-context",
    "tls-query-empty-plan",
    "tls-query-invalid-feature",
    "tls-query-later-useful-evidence",
    "tls-query-nonattainment",
    "tls-query-pilot-stop",
    "tls-query-zero-distance",
)

PORTABLE_ARTIFACTS = (
    "manifest.json",
    "projection.json",
    "id_maps.json",
    "evidence_plan_schema.json",
    "evidence_label_store.json",
    "phase_a_decisions.jsonl",
    "phase_b_supervision.jsonl",
    "work_counters.json",
    "aggregates.json",
    "report.md",
)

FORBIDDEN_DECISION_FIELDS = frozenset(
    {
        "split",
        "split_role",
        "role",
        "qrel",
        "qrels",
        "evidence_label",
        "evidence_labels",
        "facet_label",
        "facet_labels",
        "support_label",
        "support_labels",
        "evidence_id",
        "evidence_ids",
        "candidate_gain",
        "candidate_evidence_gain",
        "context_gain",
        "final_context_evidence_gain",
        "remaining_gain",
        "remaining_useful_evidence",
        "sufficiency",
        "current_sufficiency",
        "answer_label",
        "answer_labels",
        "generated_answer",
        "answer_correctness",
        "oracle_lid",
        "effective_lid",
        "exact_top_k_ids",
        "exact_full_corpus_top_k_ids",
        "realized_retention",
        "future_expansion_outcome",
        "future_expansion_outcomes",
        "protected_role_outcome",
        "protected_role_outcomes",
    }
)


class Step2ConfigError(ValueError):
    pass


class ForbiddenDecisionFieldError(ValueError):
    pass


class Action(str, Enum):
    STOP = "STOP"
    EXPAND_TO_NEXT_GRID_VALUE = "EXPAND_TO_NEXT_GRID_VALUE"


@dataclass(frozen=True)
class ScheduleEntry:
    query_id: str
    stop_after_expansions: int


@dataclass(frozen=True)
class Step2Config:
    run_name: str
    data_seed: int
    projection_seed: int
    dimension: int
    corpus_size: int
    evidence_plan_generator: str
    m_prime: int
    k_gt: int
    k_ctx: int
    m_pilot: int
    budget_grid: tuple[int, ...]
    maximum_expansions: int
    schedules: tuple[ScheduleEntry, ...]
    controller_name: str
    controller_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONFIG_SCHEMA,
            "run_name": self.run_name,
            "seeds": {
                "data": self.data_seed,
                "projection": self.projection_seed,
            },
            "fixture": {
                "dimension": self.dimension,
                "corpus_size": self.corpus_size,
                "evidence_plan_generator": self.evidence_plan_generator,
            },
            "retrieval": {
                "backend": "numpy_exact_squared_l2",
                "m_prime": self.m_prime,
                "k_gt": self.k_gt,
                "k_ctx": self.k_ctx,
                "m_pilot": self.m_pilot,
                "budget_grid": list(self.budget_grid),
                "maximum_expansions": self.maximum_expansions,
                "tie_break": "stable_string_id",
                "post_projection_normalized": False,
            },
            "controller": {
                "name": self.controller_name,
                "version": self.controller_version,
                "stop_after_expansions": {
                    entry.query_id: entry.stop_after_expansions
                    for entry in self.schedules
                },
            },
        }

    @property
    def config_fingerprint(self) -> str:
        return fingerprint(self.to_dict())


def _require_exact_keys(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise Step2ConfigError(f"{name} must be an object")
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise Step2ConfigError(
            f"invalid {name} keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise Step2ConfigError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Step2ConfigError(f"{name} must be a nonnegative integer")
    return value


def load_step2_config(path: Path | str) -> Step2Config:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Step2ConfigError(f"cannot load Step 2 config: {exc}") from exc
    root = _require_exact_keys(
        raw,
        {"schema_version", "run_name", "seeds", "fixture", "retrieval", "controller"},
        "config",
    )
    if root["schema_version"] != CONFIG_SCHEMA:
        raise Step2ConfigError(f"schema_version must be {CONFIG_SCHEMA}")
    if not isinstance(root["run_name"], str) or not root["run_name"].startswith(
        "tls_rag"
    ):
        raise Step2ConfigError("run_name must be a nonempty tls_rag identity")
    seeds = _require_exact_keys(root["seeds"], {"data", "projection"}, "seeds")
    fixture = _require_exact_keys(
        root["fixture"],
        {"dimension", "corpus_size", "evidence_plan_generator"},
        "fixture",
    )
    retrieval = _require_exact_keys(
        root["retrieval"],
        {
            "backend",
            "m_prime",
            "k_gt",
            "k_ctx",
            "m_pilot",
            "budget_grid",
            "maximum_expansions",
            "tie_break",
            "post_projection_normalized",
        },
        "retrieval",
    )
    controller = _require_exact_keys(
        root["controller"],
        {"name", "version", "stop_after_expansions"},
        "controller",
    )
    if retrieval["backend"] != "numpy_exact_squared_l2":
        raise Step2ConfigError("Step 2 permits only the NumPy exact squared-L2 backend")
    if retrieval["tie_break"] != "stable_string_id":
        raise Step2ConfigError("Step 2 requires stable string-ID ties")
    if retrieval["post_projection_normalized"] is not False:
        raise Step2ConfigError("projected vectors must not be renormalized")
    if fixture["corpus_size"] != 12:
        raise Step2ConfigError("the checked Step 2 fixture requires exactly 12 passages")
    if not isinstance(fixture["evidence_plan_generator"], str) or not fixture[
        "evidence_plan_generator"
    ].startswith("tls_rag"):
        raise Step2ConfigError("evidence-plan generator must use a tls_rag identity")
    grid_raw = retrieval["budget_grid"]
    if not isinstance(grid_raw, list) or not grid_raw:
        raise Step2ConfigError("budget_grid must be a nonempty list")
    grid = tuple(_positive_int(value, "budget_grid item") for value in grid_raw)
    if list(grid) != sorted(set(grid)):
        raise Step2ConfigError("budget_grid must be strictly increasing")
    m_pilot = _positive_int(retrieval["m_pilot"], "m_pilot")
    if grid[0] != m_pilot:
        raise Step2ConfigError("budget_grid must begin exactly at m_pilot")
    if grid[-1] != fixture["corpus_size"]:
        raise Step2ConfigError("budget_grid must end at the complete tiny corpus")
    k_gt = _positive_int(retrieval["k_gt"], "k_gt")
    k_ctx = _positive_int(retrieval["k_ctx"], "k_ctx")
    if m_pilot < max(k_gt, k_ctx):
        raise Step2ConfigError("m_pilot must be at least max(k_gt, k_ctx)")
    maximum_expansions = _nonnegative_int(
        retrieval["maximum_expansions"], "maximum_expansions"
    )
    if maximum_expansions != len(grid) - 1:
        raise Step2ConfigError(
            "the exhaustion fixture freezes maximum_expansions to all grid transitions"
        )
    schedule_raw = controller["stop_after_expansions"]
    if not isinstance(schedule_raw, dict) or set(schedule_raw) != set(QUERY_IDS):
        raise Step2ConfigError("fixed schedule must contain every and only Step 2 query ID")
    schedules = tuple(
        ScheduleEntry(query_id, _nonnegative_int(schedule_raw[query_id], query_id))
        for query_id in QUERY_IDS
    )
    if any(entry.stop_after_expansions > maximum_expansions for entry in schedules):
        raise Step2ConfigError("scheduled stop cannot exceed maximum_expansions")
    if controller["name"] != "tls_rag_fixed_schedule_controller":
        raise Step2ConfigError("Step 2 permits only the frozen schedule controller")
    if controller["version"] != 1:
        raise Step2ConfigError("controller version must be 1")
    return Step2Config(
        run_name=root["run_name"],
        data_seed=_nonnegative_int(seeds["data"], "seeds.data"),
        projection_seed=_nonnegative_int(seeds["projection"], "seeds.projection"),
        dimension=_positive_int(fixture["dimension"], "fixture.dimension"),
        corpus_size=_positive_int(fixture["corpus_size"], "fixture.corpus_size"),
        evidence_plan_generator=fixture["evidence_plan_generator"],
        m_prime=_positive_int(retrieval["m_prime"], "m_prime"),
        k_gt=k_gt,
        k_ctx=k_ctx,
        m_pilot=m_pilot,
        budget_grid=grid,
        maximum_expansions=maximum_expansions,
        schedules=schedules,
        controller_name=controller["name"],
        controller_version=controller["version"],
    )


@dataclass(frozen=True)
class FacetSlot:
    facet: str
    required_supports: int
    require_distinct_source_groups: bool
    match_term: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "facet": self.facet,
            "required_supports": self.required_supports,
            "require_distinct_source_groups": self.require_distinct_source_groups,
            "match_term": self.match_term,
        }


@dataclass(frozen=True)
class EvidencePlan:
    generator: str
    slots: tuple[FacetSlot, ...]
    block_contradictions: bool

    @property
    def valid(self) -> bool:
        return bool(self.slots) and all(
            slot.facet
            and slot.match_term
            and slot.required_supports > 0
            and isinstance(slot.require_distinct_source_groups, bool)
            for slot in self.slots
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator": self.generator,
            "slots": [slot.to_dict() for slot in self.slots],
            "block_contradictions": self.block_contradictions,
            "valid": self.valid,
        }


@dataclass(frozen=True)
class SyntheticQuery:
    query_id: str
    text: str
    embedding: np.ndarray
    plan: EvidencePlan
    deterministic_features_valid: bool


@dataclass(frozen=True)
class Step2Environment:
    config: Step2Config
    corpus_ids: tuple[str, ...]
    corpus_texts: tuple[str, ...]
    corpus_embeddings: np.ndarray
    queries: tuple[SyntheticQuery, ...]
    projection_matrix: np.ndarray
    projected_corpus: np.ndarray
    fixture_fingerprint: str


@dataclass(frozen=True)
class PassageEvidence:
    passage_id: str
    facets: tuple[str, ...]
    source_group: str
    contradiction_facets: tuple[str, ...]
    invalid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "passage_id": self.passage_id,
            "facets": list(self.facets),
            "source_group": self.source_group,
            "contradiction_facets": list(self.contradiction_facets),
            "invalid": self.invalid,
        }


@dataclass(frozen=True)
class EvidenceLabelStore:
    schema_version: str
    annotations: tuple[PassageEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "annotations": [annotation.to_dict() for annotation in self.annotations],
        }
        return {**payload, "fingerprint": fingerprint(payload)}


def _read_only(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    result.setflags(write=False)
    return result


def _unit_vector(rng: np.random.Generator, dimension: int) -> np.ndarray:
    return normalize_rows(rng.normal(size=(1, dimension)))[0]


def _stable_rank(ids: tuple[str, ...], corpus: np.ndarray, query: np.ndarray) -> np.ndarray:
    distances = np.einsum("ij,ij->i", corpus - query, corpus - query)
    return np.lexsort((np.asarray(ids, dtype=str), distances))


def _projected_rank(
    ids: tuple[str, ...], projected_corpus: np.ndarray, projected_query: np.ndarray
) -> np.ndarray:
    distances = np.einsum(
        "ij,ij->i",
        projected_corpus - projected_query,
        projected_corpus - projected_query,
    )
    return np.lexsort((np.asarray(ids, dtype=str), distances))


def _plan(generator: str, facet: str, *, supports: int = 1) -> EvidencePlan:
    return EvidencePlan(
        generator=generator,
        slots=(
            FacetSlot(
                facet=facet,
                required_supports=supports,
                require_distinct_source_groups=supports > 1,
                match_term="synthetic",
            ),
        ),
        block_contradictions=True,
    )


def build_step2_environment(config: Step2Config) -> Step2Environment:
    rng = np.random.default_rng(config.data_seed)
    corpus = normalize_rows(rng.normal(size=(config.corpus_size, config.dimension)))
    corpus[1] = corpus[0]
    corpus_ids = tuple(f"tls-passage-{row:02d}" for row in range(config.corpus_size))
    corpus_texts = tuple(
        f"Synthetic passage {row:02d} with deterministic surface tokens."
        for row in range(config.corpus_size)
    )
    matrix = dense_gaussian_projection(
        config.m_prime, config.dimension, config.projection_seed
    )
    projected_corpus = project_rows(corpus, matrix)

    ordinary = [_unit_vector(rng, config.dimension) for _ in range(5)]
    later_query: Optional[np.ndarray] = None
    for _ in range(10_000):
        candidate = _unit_vector(rng, config.dimension)
        original_top = _stable_rank(corpus_ids, corpus, candidate)[: config.k_ctx]
        projected_query = project_rows(candidate[None, :], matrix)[0]
        projected_order = _projected_rank(
            corpus_ids, projected_corpus, projected_query
        )
        projected_positions = {
            int(row): position for position, row in enumerate(projected_order.tolist())
        }
        if any(
            projected_positions[int(row)] >= config.budget_grid[-2]
            for row in original_top
        ):
            later_query = candidate
            break
    if later_query is None:
        raise RuntimeError("data seed could not realize the later-evidence fixture")

    generator = config.evidence_plan_generator
    query_rows = (
        (
            QUERY_IDS[0],
            "Find the candidate-only synthetic facet.",
            ordinary[0],
            _plan(generator, "candidate_only"),
            True,
        ),
        (
            QUERY_IDS[1],
            "This deterministic fixture intentionally has no evidence slots.",
            ordinary[1],
            EvidencePlan(generator, (), True),
            True,
        ),
        (
            QUERY_IDS[2],
            "Exercise a deterministic invalid feature flag.",
            ordinary[2],
            _plan(generator, "invalid_feature_fact"),
            False,
        ),
        (
            QUERY_IDS[3],
            "Require evidence absent from the immediate shell but present later.",
            later_query,
            _plan(generator, "later_fact"),
            True,
        ),
        (
            QUERY_IDS[4],
            "Require a valid facet that no synthetic passage supports.",
            ordinary[3],
            _plan(generator, "never_attained"),
            True,
        ),
        (
            QUERY_IDS[5],
            "Stop at pilot after two independent synthetic supports.",
            ordinary[4],
            _plan(generator, "pilot_fact", supports=2),
            True,
        ),
        (
            QUERY_IDS[6],
            "Exercise an external ID with a zero embedding displacement.",
            corpus[2].copy(),
            _plan(generator, "zero_distance_fact"),
            True,
        ),
    )
    queries = tuple(
        SyntheticQuery(query_id, text, _read_only(vector), plan, feature_valid)
        for query_id, text, vector, plan, feature_valid in query_rows
    )
    if set(corpus_ids).intersection(query.query_id for query in queries):
        raise AssertionError("synthetic query IDs must be external to corpus IDs")
    if not np.allclose(np.linalg.norm(corpus, axis=1), 1.0, atol=1e-12):
        raise AssertionError("corpus embeddings must be normalized")
    query_matrix = np.vstack([query.embedding for query in queries])
    if not np.allclose(np.linalg.norm(query_matrix, axis=1), 1.0, atol=1e-12):
        raise AssertionError("query embeddings must be normalized")
    fixture_payload = {
        "schema_version": "tls_rag_step2_fixture_v1",
        "config_fingerprint": config.config_fingerprint,
        "corpus_ids": list(corpus_ids),
        "query_ids": [query.query_id for query in queries],
        "corpus_embedding_hash": array_fingerprint(corpus),
        "query_embedding_hash": array_fingerprint(query_matrix),
        "projection_matrix_hash": array_fingerprint(matrix),
        "plans": {query.query_id: query.plan.to_dict() for query in queries},
    }
    return Step2Environment(
        config=config,
        corpus_ids=corpus_ids,
        corpus_texts=corpus_texts,
        corpus_embeddings=_read_only(corpus),
        queries=queries,
        projection_matrix=_read_only(matrix),
        projected_corpus=_read_only(projected_corpus),
        fixture_fingerprint=fingerprint(fixture_payload),
    )


def _query_rankings(environment: Step2Environment, query: SyntheticQuery) -> tuple[np.ndarray, np.ndarray]:
    projected_query = project_rows(
        query.embedding[None, :], environment.projection_matrix
    )[0]
    projected_order = _projected_rank(
        environment.corpus_ids, environment.projected_corpus, projected_query
    )
    original_order = _stable_rank(
        environment.corpus_ids, environment.corpus_embeddings, query.embedding
    )
    return projected_order, original_order


def build_evidence_label_store(environment: Step2Environment) -> EvidenceLabelStore:
    """Open the synthetic labels after Phase A has been closed by the caller."""
    facets: dict[str, set[str]] = {passage_id: set() for passage_id in environment.corpus_ids}
    query_by_id = {query.query_id: query for query in environment.queries}

    candidate_query = query_by_id["tls-query-candidate-without-context"]
    projected, _ = _query_rankings(environment, candidate_query)
    shell_rows = projected[
        environment.config.budget_grid[0] : environment.config.budget_grid[1]
    ]
    exposed_rows = projected[: environment.config.budget_grid[1]]
    exposed_distances = np.einsum(
        "ij,ij->i",
        environment.corpus_embeddings[exposed_rows] - candidate_query.embedding,
        environment.corpus_embeddings[exposed_rows] - candidate_query.embedding,
    )
    reranked = exposed_rows[
        np.lexsort(
            (
                np.asarray(environment.corpus_ids, dtype=str)[exposed_rows],
                exposed_distances,
            )
        )
    ]
    context_rows = set(reranked[: environment.config.k_ctx].tolist())
    candidate_only_row = next(int(row) for row in shell_rows if int(row) not in context_rows)
    facets[environment.corpus_ids[candidate_only_row]].add("candidate_only")

    later_query = query_by_id["tls-query-later-useful-evidence"]
    projected, original = _query_rankings(environment, later_query)
    prefix_rows = set(projected[: environment.config.budget_grid[-2]].tolist())
    later_row = next(
        int(row)
        for row in original[: environment.config.k_ctx]
        if int(row) not in prefix_rows
    )
    facets[environment.corpus_ids[later_row]].add("later_fact")

    pilot_query = query_by_id["tls-query-pilot-stop"]
    projected, _ = _query_rankings(environment, pilot_query)
    pilot_rows = projected[: environment.config.m_pilot]
    pilot_distances = np.einsum(
        "ij,ij->i",
        environment.corpus_embeddings[pilot_rows] - pilot_query.embedding,
        environment.corpus_embeddings[pilot_rows] - pilot_query.embedding,
    )
    pilot_context = pilot_rows[
        np.lexsort(
            (
                np.asarray(environment.corpus_ids, dtype=str)[pilot_rows],
                pilot_distances,
            )
        )
    ][: environment.config.k_ctx]
    for row in pilot_context:
        facets[environment.corpus_ids[int(row)]].add("pilot_fact")

    invalid_query = query_by_id["tls-query-invalid-feature"]
    projected, _ = _query_rankings(environment, invalid_query)
    facets[environment.corpus_ids[int(projected[0])]].add("invalid_feature_fact")
    facets[environment.corpus_ids[2]].add("zero_distance_fact")

    unused_rows = [
        row for row, passage_id in enumerate(environment.corpus_ids) if not facets[passage_id]
    ]
    invalid_row = unused_rows[-1]
    contradiction_row = unused_rows[0]
    annotations = tuple(
        PassageEvidence(
            passage_id=passage_id,
            facets=tuple(sorted(facets[passage_id])),
            source_group=f"tls-source-{row:02d}",
            contradiction_facets=("unused_contradiction",)
            if row == contradiction_row
            else (),
            invalid=row == invalid_row,
        )
        for row, passage_id in enumerate(environment.corpus_ids)
    )
    return EvidenceLabelStore("tls_rag_evidence_label_store_v1", annotations)


@dataclass(frozen=True)
class NumericSummary:
    minimum: float
    maximum: float
    mean: float

    def to_dict(self) -> dict[str, float]:
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
        }


@dataclass(frozen=True)
class StateValidity:
    plan_valid: bool
    deterministic_features_valid: bool
    all_numeric_features_finite: bool
    zero_original_distance_count: int
    duplicate_original_distance_pairs: int
    duplicate_projected_distance_pairs: int
    mandatory_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            field.name: getattr(self, field.name) for field in fields(self)
        }


@dataclass(frozen=True)
class FacetMatchPrediction:
    facet: str
    context_match_count: int
    predicted_covered: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "facet": self.facet,
            "context_match_count": self.context_match_count,
            "predicted_covered": self.predicted_covered,
        }


@dataclass(frozen=True)
class DecisionInput:
    schema_version: str
    query_id: str
    query_features: tuple[tuple[str, float], ...]
    evidence_plan_features: tuple[tuple[str, float], ...]
    current_budget: int
    step: int
    remaining_grid_steps: int
    exposed_candidate_ids: tuple[str, ...]
    projected_ranks: tuple[int, ...]
    projected_squared_distances: tuple[float, ...]
    cached_original_squared_distances: tuple[float, ...]
    exact_reranked_ids: tuple[str, ...]
    exact_reranked_squared_distances: tuple[float, ...]
    original_distance_summary: NumericSummary
    original_distance_gaps: tuple[float, ...]
    distortion_summary: NumericSummary
    distortion_invalid_count: int
    candidate_mean_pairwise_squared_l2: float
    candidate_mean_pairwise_cosine: float
    candidate_duplicate_vector_pairs: int
    context_ids: tuple[str, ...]
    facet_match_predictions: tuple[FacetMatchPrediction, ...]
    validity: StateValidity

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "query_id": self.query_id,
            "query_features": dict(self.query_features),
            "evidence_plan_features": dict(self.evidence_plan_features),
            "current_budget": self.current_budget,
            "step": self.step,
            "remaining_grid_steps": self.remaining_grid_steps,
            "exposed_candidate_ids": list(self.exposed_candidate_ids),
            "projected_ranks": list(self.projected_ranks),
            "projected_squared_distances": list(self.projected_squared_distances),
            "cached_original_squared_distances": list(
                self.cached_original_squared_distances
            ),
            "exact_reranked_ids": list(self.exact_reranked_ids),
            "exact_reranked_squared_distances": list(
                self.exact_reranked_squared_distances
            ),
            "original_distance_summary": self.original_distance_summary.to_dict(),
            "original_distance_gaps": list(self.original_distance_gaps),
            "distortion_summary": self.distortion_summary.to_dict(),
            "distortion_invalid_count": self.distortion_invalid_count,
            "candidate_mean_pairwise_squared_l2": self.candidate_mean_pairwise_squared_l2,
            "candidate_mean_pairwise_cosine": self.candidate_mean_pairwise_cosine,
            "candidate_duplicate_vector_pairs": self.candidate_duplicate_vector_pairs,
            "context_ids": list(self.context_ids),
            "facet_match_predictions": [
                prediction.to_dict() for prediction in self.facet_match_predictions
            ],
            "validity": self.validity.to_dict(),
        }


def assert_deployable_only(value: Any, path: str = "decision_input") -> None:
    if is_dataclass(value):
        assert_deployable_only(
            {field.name: getattr(value, field.name) for field in fields(value)}, path
        )
        return
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower().replace("-", "_")
            if key in FORBIDDEN_DECISION_FIELDS:
                raise ForbiddenDecisionFieldError(f"forbidden field at {path}.{key}")
            assert_deployable_only(child, f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            assert_deployable_only(child, f"{path}[{index}]")
        return
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(f"nonfinite deployable value at {path}")


@dataclass(frozen=True)
class ControllerDecision:
    action: Action
    reason: str
    next_budget: Optional[int]


class FixedScheduleController:
    """Label-free fixture controller; it has no fit, score, threshold, or bound."""

    def __init__(self, config: Step2Config):
        self._grid = config.budget_grid
        self._maximum_expansions = config.maximum_expansions
        self._schedule = tuple(
            (entry.query_id, entry.stop_after_expansions) for entry in config.schedules
        )

    def choose(self, state: DecisionInput) -> ControllerDecision:
        if not isinstance(state, DecisionInput):
            raise TypeError("controller accepts only DecisionInput")
        assert_deployable_only(state)
        if state.schema_version != DECISION_SCHEMA:
            raise ValueError("decision-input schema mismatch")
        if state.current_budget != self._grid[state.step]:
            raise ValueError("state budget is not the current grid value")
        scheduled = dict(self._schedule).get(state.query_id)
        if scheduled is None:
            raise ValueError("query is absent from the frozen action schedule")
        can_expand = (
            state.remaining_grid_steps > 0 and state.step < self._maximum_expansions
        )
        if not state.validity.mandatory_valid and can_expand:
            return ControllerDecision(
                Action.EXPAND_TO_NEXT_GRID_VALUE,
                "conservative_invalid_state",
                self._grid[state.step + 1],
            )
        if state.remaining_grid_steps == 0:
            if not state.validity.plan_valid:
                reason = "invalid_evidence_plan"
            elif not state.validity.mandatory_valid:
                reason = "invalid_state_at_corpus_exhaustion"
            else:
                reason = "corpus_exhausted"
            return ControllerDecision(Action.STOP, reason, None)
        if state.step >= self._maximum_expansions:
            return ControllerDecision(Action.STOP, "maximum_expansions_reached", None)
        if state.step >= scheduled:
            return ControllerDecision(Action.STOP, "fixed_schedule_stop", None)
        return ControllerDecision(
            Action.EXPAND_TO_NEXT_GRID_VALUE,
            "fixed_schedule_expand",
            self._grid[state.step + 1],
        )


def _numeric_summary(values: np.ndarray) -> NumericSummary:
    numeric = np.asarray(values, dtype=np.float64)
    if numeric.size == 0:
        return NumericSummary(0.0, 0.0, 0.0)
    return NumericSummary(
        float(np.min(numeric)), float(np.max(numeric)), float(np.mean(numeric))
    )


def _equal_pair_count(values: np.ndarray) -> int:
    numeric = np.asarray(values, dtype=np.float64)
    return sum(
        bool(numeric[left] == numeric[right])
        for left in range(len(numeric))
        for right in range(left + 1, len(numeric))
    )


def _candidate_diversity(vectors: np.ndarray) -> tuple[float, float, int]:
    if len(vectors) < 2:
        return 0.0, 0.0, 0
    squared_distances = []
    cosines = []
    duplicate_pairs = 0
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            difference = vectors[left] - vectors[right]
            distance = float(np.dot(difference, difference))
            cosine = float(np.dot(vectors[left], vectors[right]))
            squared_distances.append(distance)
            cosines.append(cosine)
            duplicate_pairs += int(distance == 0.0)
    return (
        float(np.mean(squared_distances)),
        float(np.mean(cosines)),
        duplicate_pairs,
    )


def build_decision_input(
    *,
    environment: Step2Environment,
    query: SyntheticQuery,
    step: int,
    projected_ids: tuple[str, ...],
    projected_distances: tuple[float, ...],
    original_distances_by_id: Mapping[str, float],
    reranked_ids: tuple[str, ...],
    context_ids: tuple[str, ...],
) -> DecisionInput:
    config = environment.config
    original_in_projected_order = np.asarray(
        [original_distances_by_id[passage_id] for passage_id in projected_ids],
        dtype=np.float64,
    )
    reranked_distances = np.asarray(
        [original_distances_by_id[passage_id] for passage_id in reranked_ids],
        dtype=np.float64,
    )
    positive = original_in_projected_order > 0.0
    distortion = np.asarray(projected_distances, dtype=np.float64)[positive] / (
        original_in_projected_order[positive]
    )
    corpus_rows = {passage_id: row for row, passage_id in enumerate(environment.corpus_ids)}
    candidate_vectors = environment.corpus_embeddings[
        [corpus_rows[passage_id] for passage_id in projected_ids]
    ]
    pairwise_l2, pairwise_cosine, duplicate_vectors = _candidate_diversity(
        candidate_vectors
    )
    facet_predictions = []
    text_by_id = dict(zip(environment.corpus_ids, environment.corpus_texts))
    for slot in query.plan.slots:
        count = sum(slot.match_term.casefold() in text_by_id[item].casefold() for item in context_ids)
        facet_predictions.append(
            FacetMatchPrediction(slot.facet, int(count), count >= slot.required_supports)
        )
    plan_features = (
        ("required_slot_count", float(len(query.plan.slots))),
        (
            "required_support_count",
            float(sum(slot.required_supports for slot in query.plan.slots)),
        ),
        (
            "independence_rule_count",
            float(sum(slot.require_distinct_source_groups for slot in query.plan.slots)),
        ),
        ("blocking_contradiction_rule", float(query.plan.block_contradictions)),
    )
    query_features = (
        ("text_character_count", float(len(query.text))),
        ("text_token_count", float(len(query.text.split()))),
    )
    finite_values = np.asarray(
        [
            *[value for _, value in query_features],
            *[value for _, value in plan_features],
            *original_in_projected_order.tolist(),
            *projected_distances,
            *reranked_distances.tolist(),
            *distortion.tolist(),
            pairwise_l2,
            pairwise_cosine,
        ],
        dtype=np.float64,
    )
    zero_count = int(np.count_nonzero(original_in_projected_order == 0.0))
    validity = StateValidity(
        plan_valid=query.plan.valid,
        deterministic_features_valid=query.deterministic_features_valid,
        all_numeric_features_finite=bool(np.all(np.isfinite(finite_values))),
        zero_original_distance_count=zero_count,
        duplicate_original_distance_pairs=_equal_pair_count(original_in_projected_order),
        duplicate_projected_distance_pairs=_equal_pair_count(
            np.asarray(projected_distances, dtype=np.float64)
        ),
        mandatory_valid=(
            query.plan.valid
            and query.deterministic_features_valid
            and bool(np.all(np.isfinite(finite_values)))
            and zero_count == 0
        ),
    )
    state = DecisionInput(
        schema_version=DECISION_SCHEMA,
        query_id=query.query_id,
        query_features=query_features,
        evidence_plan_features=plan_features,
        current_budget=config.budget_grid[step],
        step=step,
        remaining_grid_steps=len(config.budget_grid) - step - 1,
        exposed_candidate_ids=projected_ids,
        projected_ranks=tuple(range(1, len(projected_ids) + 1)),
        projected_squared_distances=projected_distances,
        cached_original_squared_distances=tuple(original_in_projected_order.tolist()),
        exact_reranked_ids=reranked_ids,
        exact_reranked_squared_distances=tuple(reranked_distances.tolist()),
        original_distance_summary=_numeric_summary(reranked_distances),
        original_distance_gaps=tuple(np.diff(reranked_distances).tolist()),
        distortion_summary=_numeric_summary(distortion),
        distortion_invalid_count=int(len(original_in_projected_order) - len(distortion)),
        candidate_mean_pairwise_squared_l2=pairwise_l2,
        candidate_mean_pairwise_cosine=pairwise_cosine,
        candidate_duplicate_vector_pairs=duplicate_vectors,
        context_ids=context_ids,
        facet_match_predictions=tuple(facet_predictions),
        validity=validity,
    )
    assert_deployable_only(state)
    return state


WORK_FIELDS = (
    "query_projection_count",
    "projected_full_scan_count",
    "projected_distance_evaluations",
    "pilot_prefix_exposure_count",
    "expansion_prefix_reuse_count",
    "new_original_distance_evaluations",
    "exact_rerank_count",
    "accumulated_rerank_candidates",
    "evidence_plan_computation_count",
    "fixed_controller_evaluation_count",
    "final_context_construction_count",
    "final_context_candidates",
)

TIMING_FIELDS = (
    "query_projection_ms",
    "pilot_projected_scan_and_ranking_ms",
    "expansion_prefix_reuse_ms",
    "new_original_distance_evaluation_ms",
    "exact_reranking_ms",
    "evidence_plan_computation_ms",
    "fixed_controller_evaluation_ms",
    "final_context_construction_ms",
)


@dataclass(frozen=True)
class DecisionRecord:
    query_id: str
    stage: int
    decision_input: DecisionInput
    action: Action
    action_reason: str
    next_budget: Optional[int]
    terminal_flags: tuple[str, ...]
    work: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PHASE_A_SCHEMA,
            "query_id": self.query_id,
            "stage": self.stage,
            "decision_input": self.decision_input.to_dict(),
            "action": self.action.value,
            "action_reason": self.action_reason,
            "next_budget": self.next_budget,
            "terminal_flags": list(self.terminal_flags),
            "work": dict(self.work),
        }


@dataclass(frozen=True)
class PhaseAQueryTrace:
    query_id: str
    records: tuple[DecisionRecord, ...]
    full_projected_ranking_ids: tuple[str, ...]
    full_projected_squared_distances: tuple[float, ...]
    evaluated_original_ids: tuple[str, ...]
    original_evaluation_counts: tuple[tuple[str, int], ...]
    projected_scan_count: int


@dataclass(frozen=True)
class PhaseAResult:
    config_fingerprint: str
    fixture_fingerprint: str
    queries: tuple[PhaseAQueryTrace, ...]
    timing_records: tuple[dict[str, Any], ...]
    decision_fingerprint: str

    def portable_records(self) -> list[dict[str, Any]]:
        return [
            record.to_dict() for query in self.queries for record in query.records
        ]

    def recompute_decision_fingerprint(self) -> str:
        return fingerprint(
            {
                "schema_version": "tls_rag_phase_a_trajectory_v1",
                "config_fingerprint": self.config_fingerprint,
                "fixture_fingerprint": self.fixture_fingerprint,
                "records": self.portable_records(),
            }
        )


def _terminal_flags(state: DecisionInput, decision: ControllerDecision, config: Step2Config) -> tuple[str, ...]:
    if decision.action is not Action.STOP:
        return ()
    flags = []
    if state.remaining_grid_steps == 0:
        flags.append("corpus_exhausted")
    if state.step >= config.maximum_expansions:
        flags.append("maximum_expansions_reached")
    if not state.validity.plan_valid:
        flags.append("invalid_evidence_plan")
    if not state.validity.mandatory_valid:
        flags.append("invalid_state")
    if not flags:
        flags.append("fixed_schedule_stop")
    return tuple(flags)


def run_phase_a(
    environment: Step2Environment, controller: FixedScheduleController
) -> PhaseAResult:
    """Build and close the complete label-free state/action trajectory."""
    config = environment.config
    projected_index = ExactSquaredL2Index(
        environment.corpus_ids, environment.projected_corpus, batch_size=1
    )
    query_traces = []
    timing_records = []
    for query in environment.queries:
        plan_started = perf_counter()
        plan = query.plan
        plan_ms = (perf_counter() - plan_started) * 1000.0
        projection_started = perf_counter()
        projected_query = project_rows(
            query.embedding[None, :], environment.projection_matrix
        )[0]
        projection_ms = (perf_counter() - projection_started) * 1000.0
        search_result = projected_index.search(projected_query, config.corpus_size)
        full_ids = tuple(str(value) for value in search_result.ids[0].tolist())
        full_projected_distances = tuple(
            float(value) for value in search_result.squared_distances[0].tolist()
        )
        cache: dict[str, float] = {}
        evaluation_counts: dict[str, int] = {}
        records = []
        previous_budget = 0
        for step, budget in enumerate(config.budget_grid):
            prefix_started = perf_counter()
            exposed_ids = full_ids[:budget]
            new_ids = full_ids[previous_budget:budget]
            prefix_ms = (perf_counter() - prefix_started) * 1000.0
            original_started = perf_counter()
            corpus_rows = {
                passage_id: row for row, passage_id in enumerate(environment.corpus_ids)
            }
            for passage_id in new_ids:
                if passage_id in cache:
                    raise AssertionError("an original distance would be evaluated twice")
                difference = (
                    environment.corpus_embeddings[corpus_rows[passage_id]]
                    - query.embedding
                )
                cache[passage_id] = float(np.dot(difference, difference))
                evaluation_counts[passage_id] = evaluation_counts.get(passage_id, 0) + 1
            original_ms = (perf_counter() - original_started) * 1000.0
            rerank_started = perf_counter()
            exposed_distances = np.asarray(
                [cache[passage_id] for passage_id in exposed_ids], dtype=np.float64
            )
            rerank_order = np.lexsort(
                (np.asarray(exposed_ids, dtype=str), exposed_distances)
            )
            reranked_ids = tuple(exposed_ids[index] for index in rerank_order)
            rerank_ms = (perf_counter() - rerank_started) * 1000.0
            context_started = perf_counter()
            context_ids = reranked_ids[: config.k_ctx]
            context_ms = (perf_counter() - context_started) * 1000.0
            state = build_decision_input(
                environment=environment,
                query=SyntheticQuery(
                    query.query_id,
                    query.text,
                    query.embedding,
                    plan,
                    query.deterministic_features_valid,
                ),
                step=step,
                projected_ids=exposed_ids,
                projected_distances=full_projected_distances[:budget],
                original_distances_by_id=cache,
                reranked_ids=reranked_ids,
                context_ids=context_ids,
            )
            controller_started = perf_counter()
            decision = controller.choose(state)
            controller_ms = (perf_counter() - controller_started) * 1000.0
            work_values = {
                "query_projection_count": int(step == 0),
                "projected_full_scan_count": int(step == 0),
                "projected_distance_evaluations": config.corpus_size if step == 0 else 0,
                "pilot_prefix_exposure_count": int(step == 0),
                "expansion_prefix_reuse_count": int(step > 0),
                "new_original_distance_evaluations": len(new_ids),
                "exact_rerank_count": 1,
                "accumulated_rerank_candidates": budget,
                "evidence_plan_computation_count": int(step == 0),
                "fixed_controller_evaluation_count": 1,
                "final_context_construction_count": 1,
                "final_context_candidates": len(context_ids),
            }
            if tuple(work_values) != WORK_FIELDS:
                raise AssertionError("work-counter schema drift")
            record = DecisionRecord(
                query_id=query.query_id,
                stage=step,
                decision_input=state,
                action=decision.action,
                action_reason=decision.reason,
                next_budget=decision.next_budget,
                terminal_flags=_terminal_flags(state, decision, config),
                work=tuple(work_values.items()),
            )
            records.append(record)
            timing = {
                "query_id": query.query_id,
                "stage": step,
                "query_projection_ms": projection_ms if step == 0 else 0.0,
                "pilot_projected_scan_and_ranking_ms": search_result.search_ms
                if step == 0
                else 0.0,
                "expansion_prefix_reuse_ms": prefix_ms if step > 0 else 0.0,
                "new_original_distance_evaluation_ms": original_ms,
                "exact_reranking_ms": rerank_ms,
                "evidence_plan_computation_ms": plan_ms if step == 0 else 0.0,
                "fixed_controller_evaluation_ms": controller_ms,
                "final_context_construction_ms": context_ms,
            }
            if tuple(key for key in timing if key not in {"query_id", "stage"}) != TIMING_FIELDS:
                raise AssertionError("timing schema drift")
            timing_records.append(timing)
            if decision.action is Action.STOP:
                break
            if decision.next_budget != config.budget_grid[step + 1]:
                raise AssertionError("expansion skipped a frozen grid value")
            previous_budget = budget
        if not records or records[-1].action is not Action.STOP:
            raise AssertionError("each Phase A trajectory must close with STOP")
        query_traces.append(
            PhaseAQueryTrace(
                query_id=query.query_id,
                records=tuple(records),
                full_projected_ranking_ids=full_ids,
                full_projected_squared_distances=full_projected_distances,
                evaluated_original_ids=tuple(cache),
                original_evaluation_counts=tuple(evaluation_counts.items()),
                projected_scan_count=1,
            )
        )
    provisional = PhaseAResult(
        config_fingerprint=config.config_fingerprint,
        fixture_fingerprint=environment.fixture_fingerprint,
        queries=tuple(query_traces),
        timing_records=tuple(timing_records),
        decision_fingerprint="",
    )
    decision_fingerprint = provisional.recompute_decision_fingerprint()
    return PhaseAResult(
        config_fingerprint=provisional.config_fingerprint,
        fixture_fingerprint=provisional.fixture_fingerprint,
        queries=provisional.queries,
        timing_records=provisional.timing_records,
        decision_fingerprint=decision_fingerprint,
    )


def _annotation_map(store: EvidenceLabelStore) -> dict[str, PassageEvidence]:
    annotations = {annotation.passage_id: annotation for annotation in store.annotations}
    if len(annotations) != len(store.annotations):
        raise ValueError("evidence-label store has duplicate passage IDs")
    return annotations


def _evidence_view(
    passage_ids: Sequence[str], plan: EvidencePlan, store: EvidenceLabelStore
) -> dict[str, Any]:
    if not plan.valid:
        return {
            "covered_slots": [],
            "coverage": 0.0,
            "sufficient": False,
            "supporting_passage_ids": [],
            "blocking_contradiction": False,
        }
    annotations = _annotation_map(store)
    covered_slots = []
    supporting_ids = set()
    required_facets = {slot.facet for slot in plan.slots}
    blocking = False
    for passage_id in passage_ids:
        annotation = annotations[passage_id]
        if annotation.invalid:
            continue
        if required_facets.intersection(annotation.contradiction_facets):
            blocking = True
    for slot in plan.slots:
        supports = []
        for passage_id in passage_ids:
            annotation = annotations[passage_id]
            if not annotation.invalid and slot.facet in annotation.facets:
                supports.append((passage_id, annotation.source_group))
        support_count = (
            len({source_group for _, source_group in supports})
            if slot.require_distinct_source_groups
            else len(supports)
        )
        if support_count >= slot.required_supports:
            covered_slots.append(slot.facet)
            supporting_ids.update(passage_id for passage_id, _ in supports)
    coverage = len(covered_slots) / len(plan.slots)
    return {
        "covered_slots": sorted(covered_slots),
        "coverage": float(coverage),
        "sufficient": bool(
            len(covered_slots) == len(plan.slots)
            and not (plan.block_contradictions and blocking)
        ),
        "supporting_passage_ids": sorted(supporting_ids),
        "blocking_contradiction": bool(blocking),
    }


def _all_budget_views(
    environment: Step2Environment, query: SyntheticQuery, trace: PhaseAQueryTrace
) -> dict[int, dict[str, tuple[str, ...]]]:
    rows = {passage_id: row for row, passage_id in enumerate(environment.corpus_ids)}
    views = {}
    for budget in environment.config.budget_grid:
        candidates = trace.full_projected_ranking_ids[:budget]
        distances = np.asarray(
            [
                float(
                    np.dot(
                        environment.corpus_embeddings[rows[passage_id]] - query.embedding,
                        environment.corpus_embeddings[rows[passage_id]] - query.embedding,
                    )
                )
                for passage_id in candidates
            ],
            dtype=np.float64,
        )
        order = np.lexsort((np.asarray(candidates, dtype=str), distances))
        reranked = tuple(candidates[index] for index in order)
        views[budget] = {
            "candidates": tuple(candidates),
            "context": reranked[: environment.config.k_ctx],
        }
    return views


@dataclass(frozen=True)
class PhaseBResult:
    decision_fingerprint_before_join: str
    decision_fingerprint_after_join: str
    supervision_records: tuple[dict[str, Any], ...]
    supervision_fingerprint: str


def join_phase_b(
    phase_a: PhaseAResult,
    environment: Step2Environment,
    label_store: EvidenceLabelStore,
) -> PhaseBResult:
    """Join labels only after Phase A and reconstruct every Step 2 target."""
    before = phase_a.recompute_decision_fingerprint()
    if before != phase_a.decision_fingerprint:
        raise ValueError("Phase A trajectory changed before supervision join")
    query_by_id = {query.query_id: query for query in environment.queries}
    supervision_records = []
    for trace in phase_a.queries:
        query = query_by_id[trace.query_id]
        views = _all_budget_views(environment, query, trace)
        candidate_evidence = {
            budget: _evidence_view(view["candidates"], query.plan, label_store)
            for budget, view in views.items()
        }
        context_evidence = {
            budget: _evidence_view(view["context"], query.plan, label_store)
            for budget, view in views.items()
        }
        full_distances = np.einsum(
            "ij,ij->i",
            environment.corpus_embeddings - query.embedding,
            environment.corpus_embeddings - query.embedding,
        )
        exact_order = np.lexsort(
            (np.asarray(environment.corpus_ids, dtype=str), full_distances)
        )
        exact_top_k_ids = tuple(
            environment.corpus_ids[int(row)]
            for row in exact_order[: environment.config.k_gt]
        )
        for record in trace.records:
            budget = record.decision_input.current_budget
            grid_index = environment.config.budget_grid.index(budget)
            later_budgets = environment.config.budget_grid[grid_index + 1 :]
            next_budget = later_budgets[0] if later_budgets else None
            candidate_current = candidate_evidence[budget]
            context_current = context_evidence[budget]
            if next_budget is None:
                candidate_gain = False
                context_gain = False
            else:
                candidate_gain = bool(
                    set(candidate_evidence[next_budget]["covered_slots"])
                    - set(candidate_current["covered_slots"])
                )
                context_gain = bool(
                    context_evidence[next_budget]["coverage"]
                    > context_current["coverage"]
                )
            remaining = any(
                context_evidence[later_budget]["coverage"]
                > context_current["coverage"]
                or (
                    not context_current["sufficient"]
                    and context_evidence[later_budget]["sufficient"]
                )
                for later_budget in later_budgets
            )
            candidate_ids = set(views[budget]["candidates"])
            retention = len(candidate_ids.intersection(exact_top_k_ids)) / len(
                exact_top_k_ids
            )
            terminal_nonattainment = bool(
                record.action is Action.STOP
                and query.plan.valid
                and not context_current["sufficient"]
                and not remaining
            )
            supervision_records.append(
                {
                    "schema_version": PHASE_B_SCHEMA,
                    "query_id": trace.query_id,
                    "stage": record.stage,
                    "budget": budget,
                    "phase_a_decision_fingerprint": phase_a.decision_fingerprint,
                    "candidate_covered_facets": candidate_current["covered_slots"],
                    "context_covered_facets": context_current["covered_slots"],
                    "candidate_coverage": candidate_current["coverage"],
                    "context_coverage": context_current["coverage"],
                    "current_final_context_sufficiency": context_current["sufficient"],
                    "marginal_candidate_evidence_gain": candidate_gain,
                    "marginal_final_context_evidence_gain": context_gain,
                    "remaining_useful_evidence": bool(remaining),
                    "candidate_evidence_ids": candidate_current[
                        "supporting_passage_ids"
                    ],
                    "context_evidence_ids": context_current[
                        "supporting_passage_ids"
                    ],
                    "blocking_contradiction": context_current[
                        "blocking_contradiction"
                    ],
                    "exact_top_k_ids": list(exact_top_k_ids),
                    "exact_top_k_retention": float(retention),
                    "evidence_plan_valid": query.plan.valid,
                    "terminal_evidence_nonattainment": terminal_nonattainment,
                }
            )
    after = phase_a.recompute_decision_fingerprint()
    if before != after or after != phase_a.decision_fingerprint:
        raise AssertionError("Phase B changed the closed Phase A trajectory")
    supervision_payload = {
        "schema_version": "tls_rag_phase_b_records_v1",
        "phase_a_decision_fingerprint": phase_a.decision_fingerprint,
        "records": supervision_records,
    }
    return PhaseBResult(
        decision_fingerprint_before_join=before,
        decision_fingerprint_after_join=after,
        supervision_records=tuple(supervision_records),
        supervision_fingerprint=fingerprint(supervision_payload),
    )


def same_distance_different_angle_fixture() -> dict[str, np.ndarray]:
    """Return normalized vectors for the Step 1 angle counterexample."""
    theta_near = 0.45
    theta_far = 1.05
    query = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    near = np.asarray(
        [np.cos(theta_near), np.sin(theta_near), 0.0], dtype=np.float64
    )
    far_same_plane = np.asarray(
        [np.cos(theta_far), np.sin(theta_far), 0.0], dtype=np.float64
    )
    far_orthogonal_plane = np.asarray(
        [np.cos(theta_far), 0.0, np.sin(theta_far)], dtype=np.float64
    )
    return {
        "query": _read_only(query),
        "near": _read_only(near),
        "far_same_plane": _read_only(far_same_plane),
        "far_orthogonal_plane": _read_only(far_orthogonal_plane),
    }


def _aggregate_records(
    phase_a: PhaseAResult, phase_b: PhaseBResult
) -> dict[str, Any]:
    decisions = phase_a.portable_records()
    supervision = list(phase_b.supervision_records)
    final_supervision = {}
    for record in supervision:
        final_supervision[record["query_id"]] = record
    return {
        "schema_version": "tls_rag_step2_aggregates_v1",
        "query_count": len(phase_a.queries),
        "stage_count": len(decisions),
        "actions": {
            action.value: sum(record["action"] == action.value for record in decisions)
            for action in Action
        },
        "terminal_reasons": {
            reason: sum(record["action_reason"] == reason for record in decisions)
            for reason in sorted(
                {record["action_reason"] for record in decisions if record["action"] == "STOP"}
            )
        },
        "candidate_gain_stage_count": sum(
            record["marginal_candidate_evidence_gain"] for record in supervision
        ),
        "context_gain_stage_count": sum(
            record["marginal_final_context_evidence_gain"] for record in supervision
        ),
        "remaining_gain_stage_count": sum(
            record["remaining_useful_evidence"] for record in supervision
        ),
        "sufficient_stage_count": sum(
            record["current_final_context_sufficiency"] for record in supervision
        ),
        "terminal_evidence_nonattainment_query_count": sum(
            record["terminal_evidence_nonattainment"]
            for record in final_supervision.values()
        ),
        "mean_final_context_coverage": float(
            np.mean([record["context_coverage"] for record in final_supervision.values()])
        ),
        "mean_final_exact_top_k_retention": float(
            np.mean(
                [record["exact_top_k_retention"] for record in final_supervision.values()]
            )
        ),
        "phase_a_decision_fingerprint": phase_a.decision_fingerprint,
        "phase_b_supervision_fingerprint": phase_b.supervision_fingerprint,
    }


def _work_artifact(phase_a: PhaseAResult) -> dict[str, Any]:
    records = [
        {
            "query_id": record.query_id,
            "stage": record.stage,
            **dict(record.work),
        }
        for query in phase_a.queries
        for record in query.records
    ]
    totals = {
        field: sum(record[field] for record in records) for field in WORK_FIELDS
    }
    return {
        "schema_version": "tls_rag_step2_work_counters_v1",
        "records": records,
        "totals": totals,
    }


def _plan_schema_artifact(environment: Step2Environment) -> dict[str, Any]:
    plans = {query.query_id: query.plan.to_dict() for query in environment.queries}
    payload = {
        "schema_version": "tls_rag_evidence_plan_and_annotation_schema_v1",
        "generator": environment.config.evidence_plan_generator,
        "plans": plans,
        "annotation_schema": {
            "passage_id": "stable_string",
            "facets": "list[stable_string]",
            "source_group": "stable_string",
            "contradiction_facets": "list[stable_string]",
            "invalid": "boolean",
        },
        "contains_passage_evidence_labels": False,
    }
    return {**payload, "fingerprint": fingerprint(payload)}


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )


def run_step2(config: Step2Config, output_dir: Path) -> dict[str, Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_started = perf_counter()
    environment = build_step2_environment(config)
    setup_ms = (perf_counter() - setup_started) * 1000.0
    controller = FixedScheduleController(config)
    phase_a = run_phase_a(environment, controller)

    # Serialize and close Phase A before the evidence-label store is constructed.
    phase_a_path = output_dir / "phase_a_decisions.jsonl"
    _write_jsonl(phase_a_path, phase_a.portable_records())
    if phase_a.recompute_decision_fingerprint() != phase_a.decision_fingerprint:
        raise AssertionError("serialized Phase A fingerprint mismatch")

    label_store = build_evidence_label_store(environment)
    phase_b = join_phase_b(phase_a, environment, label_store)
    _write_jsonl(
        output_dir / "phase_b_supervision.jsonl", phase_b.supervision_records
    )
    aggregates = _aggregate_records(phase_a, phase_b)
    work = _work_artifact(phase_a)
    projection = projection_metadata(
        dimension=config.dimension,
        m_prime=config.m_prime,
        seed=config.projection_seed,
        normalization=True,
        embedding_model="tls_rag_synthetic_array_fixture@step2",
        corpus_hash=array_fingerprint(environment.corpus_embeddings),
    )
    projection["matrix_hash"] = array_fingerprint(environment.projection_matrix)
    projection["schema_version"] = "tls_rag_projection_identity_v1"
    projection["fingerprint"] = fingerprint(
        {key: value for key, value in projection.items() if key != "fingerprint"}
    )
    id_maps = {
        "schema_version": "tls_rag_ordered_id_maps_v1",
        "corpus_ids": list(environment.corpus_ids),
        "query_ids": [query.query_id for query in environment.queries],
    }
    id_maps["fingerprint"] = fingerprint(id_maps)
    plan_schema = _plan_schema_artifact(environment)
    evidence_labels = label_store.to_dict()
    manifest_payload = {
        "schema_version": "tls_rag_step2_manifest_v1",
        "run_name": config.run_name,
        "scope": "cpu_network_free_step2_code_path_fixture",
        "config": config.to_dict(),
        "config_fingerprint": config.config_fingerprint,
        "fixture_fingerprint": environment.fixture_fingerprint,
        "projection_fingerprint": projection["fingerprint"],
        "id_map_fingerprint": id_maps["fingerprint"],
        "evidence_plan_schema_fingerprint": plan_schema["fingerprint"],
        "evidence_label_store_fingerprint": evidence_labels["fingerprint"],
        "phase_a_decision_fingerprint": phase_a.decision_fingerprint,
        "phase_b_supervision_fingerprint": phase_b.supervision_fingerprint,
        "phase_a_closed_before_label_store_opened": True,
        "phase_a_fingerprint_unchanged_by_join": (
            phase_b.decision_fingerprint_before_join
            == phase_b.decision_fingerprint_after_join
        ),
        "queries_are_external": not bool(
            set(environment.corpus_ids).intersection(
                query.query_id for query in environment.queries
            )
        ),
        "portable_artifacts": list(PORTABLE_ARTIFACTS),
        "timings_portable": False,
        "seeds": {
            "data": config.data_seed,
            "projection": config.projection_seed,
        },
        "prohibitions_observed": {
            "network": True,
            "real_data_or_model": True,
            "approximate_index": True,
            "llm": True,
            "answer_generation": True,
            "learned_or_calibrated_controller": True,
        },
    }
    manifest = {**manifest_payload, "fingerprint": fingerprint(manifest_payload)}
    report = "\n".join(
        (
            "# TLS-RAG Step 2 synthetic fixture",
            "",
            "This run exercises the fixed `STOP` / `EXPAND_TO_NEXT_GRID_VALUE` interface, one exact projected scan with prefix reuse, exact original-space reranking, fixed top-`k_ctx` context construction, and a closed Phase A followed by a separate Phase B label join.",
            "",
            f"- Queries: {aggregates['query_count']}",
            f"- Visited stages: {aggregates['stage_count']}",
            f"- Phase A decision fingerprint: `{phase_a.decision_fingerprint}`",
            f"- Phase B supervision fingerprint: `{phase_b.supervision_fingerprint}`",
            f"- Terminal evidence nonattainment queries: {aggregates['terminal_evidence_nonattainment_query_count']}",
            "",
            "The fixture includes candidate gain without context gain, an empty immediate shell followed by later context evidence, stable duplicate/equal distances, a zero displacement, an empty plan, invalid deterministic features, maximum-expansion/full-corpus exhaustion, and terminal nonattainment.",
            "",
            "This is only a Step 2 code-path fixture. It is not a learned or calibrated controller result, real-data claim, certificate, latency claim, Tri-Law posterior claim, or answer-quality result. No network, real dataset/model, approximate index, LLM, or answer generation was used.",
            "",
        )
    )
    timings = {
        "schema_version": "tls_rag_step2_timings_v1",
        "portable": False,
        "setup_and_fixture_generation_ms": setup_ms,
        "records": list(phase_a.timing_records),
        "totals_ms": {
            field: float(sum(record[field] for record in phase_a.timing_records))
            for field in TIMING_FIELDS
        },
    }

    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "projection.json", projection)
    write_json(output_dir / "id_maps.json", id_maps)
    write_json(output_dir / "evidence_plan_schema.json", plan_schema)
    write_json(output_dir / "evidence_label_store.json", evidence_labels)
    write_json(output_dir / "work_counters.json", work)
    write_json(output_dir / "aggregates.json", aggregates)
    write_json(output_dir / "timings.json", timings)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return {
        name: output_dir / name for name in (*PORTABLE_ARTIFACTS, "timings.json")
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    config = load_step2_config(args.config)
    artifacts = run_step2(config, args.output)
    print(f"completed {config.run_name}: {artifacts['report.md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
