from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence

import numpy as np

from .pdctp_calibration import PilotLIDCalibrator, TriBudgetResidualCalibrator
from .pdctp_features import PilotFeatureVector
from .policies import MonotoneBinnedPolicy, PolicyDecision, TriPredictPolicy
from .utils import fingerprint


class PDCTPPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class PDCTPDecisionInput:
    """Inference input with no labels, exact neighbors, qrels, or split role."""

    features: PilotFeatureVector
    pilot_lid: float
    pilot_lid_valid: bool


@dataclass(frozen=True)
class PDCTPDecision:
    policy_name: str
    policy_version: int
    budget: int
    raw_budget: Optional[int]
    input_lid: Optional[float]
    calibrated_lid: Optional[float]
    residual_correction: Optional[float]
    used_fallback: bool
    saturated: bool
    failure_reason: Optional[str]


class PDCTPDecisionPolicy(Protocol):
    grid: Sequence[int]
    minimum_budget: int

    def choose(self, observation: PDCTPDecisionInput) -> PDCTPDecision:
        ...

    def serialize(self) -> Dict[str, Any]:
        ...


def _validate_grid(grid: Sequence[int], minimum_budget: int) -> tuple[int, ...]:
    values = tuple(int(value) for value in grid)
    if not values or list(values) != sorted(set(values)):
        raise PDCTPPolicyError("PDCTP budget grid must be strictly increasing")
    if minimum_budget not in values or values[0] < minimum_budget:
        raise PDCTPPolicyError("PDCTP grid violates the frozen minimum budget")
    return values


class FixedPDCTPPolicy:
    NAME = "fixed_budget_pdctp_interface"
    VERSION = 1

    def __init__(self, budget: int, grid: Sequence[int], minimum_budget: int):
        self.minimum_budget = int(minimum_budget)
        self.grid = _validate_grid(grid, self.minimum_budget)
        self.budget = int(budget)
        if self.budget not in self.grid:
            raise PDCTPPolicyError("fixed budget must come from the PDCTP grid")

    def choose(self, observation: PDCTPDecisionInput) -> PDCTPDecision:
        return PDCTPDecision(
            self.NAME,
            self.VERSION,
            self.budget,
            None,
            None,
            None,
            None,
            False,
            False,
            None,
        )

    def serialize(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "name": self.NAME,
            "version": self.VERSION,
            "budget": self.budget,
            "grid": list(self.grid),
            "minimum_budget": self.minimum_budget,
        }
        body["fingerprint"] = fingerprint(body)
        return body


class MonotonePDCTPPolicy:
    NAME = "monotone_binned_pdctp_interface"
    VERSION = 1

    def __init__(self, reference: MonotoneBinnedPolicy, *, minimum_budget: int):
        self.reference = reference
        self.minimum_budget = int(minimum_budget)
        self.grid = _validate_grid(reference.grid, self.minimum_budget)
        if any(budget < self.minimum_budget for budget in reference.budgets):
            raise PDCTPPolicyError("monotone policy violates the PDCTP lower bound")

    def choose(self, observation: PDCTPDecisionInput) -> PDCTPDecision:
        decision = self.reference.choose(
            observation.pilot_lid, observation.pilot_lid_valid
        )
        return PDCTPDecision(
            self.NAME,
            self.VERSION,
            decision.budget,
            None,
            observation.pilot_lid if observation.pilot_lid_valid else None,
            None,
            None,
            decision.used_fallback,
            decision.saturated,
            "invalid_pilot_lid" if decision.used_fallback else None,
        )

    def serialize(self) -> Dict[str, Any]:
        reference_artifact = self.reference.serialize()
        body: Dict[str, Any] = {
            "name": self.NAME,
            "version": self.VERSION,
            "reference_name": reference_artifact["name"],
            "reference_fingerprint": reference_artifact["fingerprint"],
            "grid": list(self.grid),
            "minimum_budget": self.minimum_budget,
        }
        body["fingerprint"] = fingerprint(body)
        return body


class RawTriPredictPDCTPPolicy:
    """A no-mutation adapter around the immutable Raw Tri-Predict API."""

    NAME = "raw_tri_predict_v1_algorithm_pdctp_interface"
    VERSION = 1

    def __init__(self, reference: TriPredictPolicy, *, minimum_budget: int):
        self.reference = reference
        self.minimum_budget = int(minimum_budget)
        self.grid = _validate_grid(reference.grid, self.minimum_budget)
        self.reference_fingerprint = str(reference.serialize()["fingerprint"])

    def _raw_decision(self, lid: float, valid: bool) -> PolicyDecision:
        return self.reference.choose(lid, valid)

    def choose(self, observation: PDCTPDecisionInput) -> PDCTPDecision:
        decision = self._raw_decision(
            observation.pilot_lid, observation.pilot_lid_valid
        )
        return PDCTPDecision(
            self.NAME,
            self.VERSION,
            decision.budget,
            decision.budget,
            observation.pilot_lid if observation.pilot_lid_valid else None,
            None,
            None,
            decision.used_fallback,
            decision.saturated,
            "invalid_pilot_lid" if decision.used_fallback else None,
        )

    def serialize(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "name": self.NAME,
            "version": self.VERSION,
            "reference_algorithm": "query_adaptive_tri_predict",
            "reference_policy_fingerprint": self.reference_fingerprint,
            "reference_policy_schema_version": self.reference.VERSION,
            "grid": list(self.grid),
            "minimum_budget": self.minimum_budget,
            "behavior": "delegated_without_mutation",
        }
        body["fingerprint"] = fingerprint(body)
        return body


class CalibratedTriPredictPolicy:
    """Explicit LID-only, residual-only, or full PDCTP policy."""

    NAMES = {
        "lid_only": "pdctp_lid_calibration_only",
        "residual_only": "pdctp_budget_residual_only",
        "full": "pilot_distance_calibrated_tri_predict",
    }
    SCHEMA = "calibrated_tri_predict_policy_v1"
    VERSION = 1

    def __init__(
        self,
        *,
        mode: str,
        raw_reference: TriPredictPolicy,
        minimum_budget: int,
        lid_calibrator: Optional[PilotLIDCalibrator] = None,
        residual_calibrator: Optional[TriBudgetResidualCalibrator] = None,
    ):
        if mode not in self.NAMES:
            raise PDCTPPolicyError("unsupported calibrated policy mode")
        if mode in {"lid_only", "full"} and lid_calibrator is None:
            raise PDCTPPolicyError("selected mode requires a LID calibrator")
        if mode in {"residual_only", "full"} and residual_calibrator is None:
            raise PDCTPPolicyError("selected mode requires a residual calibrator")
        self.mode = mode
        self.name = self.NAMES[mode]
        self.raw_reference = raw_reference
        self.raw_reference_fingerprint = str(
            raw_reference.serialize()["fingerprint"]
        )
        self.minimum_budget = int(minimum_budget)
        self.grid = _validate_grid(raw_reference.grid, self.minimum_budget)
        self.lid_calibrator = lid_calibrator
        self.residual_calibrator = residual_calibrator
        if (
            residual_calibrator is not None
            and residual_calibrator.raw_policy_fingerprint
            != self.raw_reference_fingerprint
        ):
            raise PDCTPPolicyError("residual calibrator references another Raw Tri policy")
        if residual_calibrator is not None and tuple(residual_calibrator.grid) != self.grid:
            raise PDCTPPolicyError("residual calibrator grid does not match Raw Tri")

    def choose(self, observation: PDCTPDecisionInput) -> PDCTPDecision:
        input_lid: Optional[float]
        calibrated_lid: Optional[float] = None
        lid_for_raw = float(observation.pilot_lid)
        lid_valid = bool(observation.pilot_lid_valid and observation.features.valid)
        failure_reason: Optional[str] = None
        used_fallback = False
        if self.mode in {"lid_only", "full"}:
            assert self.lid_calibrator is not None
            calibrated = self.lid_calibrator.predict(observation.features)
            calibrated_lid = calibrated.value
            lid_for_raw = calibrated.value
            lid_valid = calibrated.valid
            used_fallback = calibrated.used_fallback
            failure_reason = calibrated.failure_reason
        input_lid = lid_for_raw if lid_valid else None
        raw_decision = self.raw_reference.choose(lid_for_raw, lid_valid)
        used_fallback = used_fallback or raw_decision.used_fallback
        if raw_decision.used_fallback and failure_reason is None:
            failure_reason = "invalid_lid_for_raw_tri"

        if self.mode in {"residual_only", "full"}:
            assert self.residual_calibrator is not None
            if raw_decision.used_fallback:
                return PDCTPDecision(
                    self.name,
                    self.VERSION,
                    self.grid[-1],
                    raw_decision.budget,
                    input_lid,
                    calibrated_lid,
                    None,
                    True,
                    True,
                    failure_reason,
                )
            calibrated_budget = self.residual_calibrator.choose_budget(
                raw_decision.budget, observation.features
            )
            return PDCTPDecision(
                self.name,
                self.VERSION,
                calibrated_budget.budget,
                raw_decision.budget,
                input_lid,
                calibrated_lid,
                calibrated_budget.residual,
                calibrated_budget.used_fallback,
                calibrated_budget.saturated,
                calibrated_budget.failure_reason,
            )
        return PDCTPDecision(
            self.name,
            self.VERSION,
            raw_decision.budget,
            raw_decision.budget,
            input_lid,
            calibrated_lid,
            None,
            used_fallback,
            raw_decision.saturated,
            failure_reason,
        )

    def serialize(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "name": self.name,
            "schema": self.SCHEMA,
            "version": self.VERSION,
            "mode": self.mode,
            "raw_theory_anchor": {
                "name": "query_adaptive_tri_predict",
                "policy_fingerprint": self.raw_reference_fingerprint,
            },
            "lid_calibrator_fingerprint": (
                None if self.lid_calibrator is None else self.lid_calibrator.fingerprint
            ),
            "residual_calibrator_fingerprint": (
                None
                if self.residual_calibrator is None
                else self.residual_calibrator.fingerprint
            ),
            "grid": list(self.grid),
            "minimum_budget": self.minimum_budget,
            "inference_fields": [
                "pilot_original_squared_distances",
                "pilot_projected_squared_distances",
                "pilot_lid",
                "pilot_lid_validity",
                "frozen_metadata",
                "raw_tri_prediction",
            ],
            "forbidden_inference_fields": [
                "oracle_lid",
                "exact_top_k",
                "qrels",
                "retention",
                "answer_labels",
                "split_role",
            ],
        }
        body["fingerprint"] = fingerprint(body)
        return body


def validate_policy_suite(policies: Mapping[str, PDCTPDecisionPolicy]) -> None:
    required = {
        "fixed",
        "monotone",
        "raw_tri",
        "lid_only",
        "residual_only",
        "full_pdctp",
    }
    if set(policies) != required:
        raise PDCTPPolicyError("policy suite must contain all baselines and ablations")
    grids = {tuple(policy.grid) for policy in policies.values()}
    minimums = {int(policy.minimum_budget) for policy in policies.values()}
    if len(grids) != 1 or len(minimums) != 1:
        raise PDCTPPolicyError("all policies must share one grid and lower bound")
