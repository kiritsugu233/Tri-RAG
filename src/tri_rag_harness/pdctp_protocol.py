from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .utils import fingerprint


class LeakageError(RuntimeError):
    """Raised before any split access that would violate the v2 protocol."""


FIVE_ROLES = (
    "query_cal",
    "query_tune",
    "query_cert",
    "query_latency",
    "query_test",
)


@dataclass(frozen=True)
class FiveRoleAssignments:
    ids_by_role: Mapping[str, Tuple[str, ...]]
    normalized_text_group_by_id: Mapping[str, str]

    def __post_init__(self) -> None:
        if set(self.ids_by_role) != set(FIVE_ROLES):
            raise LeakageError("exactly five PDCTP query roles are required")
        normalized: Dict[str, Tuple[str, ...]] = {}
        seen_ids: set[str] = set()
        seen_groups: Dict[str, str] = {}
        for role in FIVE_ROLES:
            ids = tuple(str(value) for value in self.ids_by_role[role])
            if not ids or len(set(ids)) != len(ids) or any(not value for value in ids):
                raise LeakageError(f"{role} IDs must be nonempty and unique")
            overlap = seen_ids.intersection(ids)
            if overlap:
                raise LeakageError("query IDs cross PDCTP roles")
            seen_ids.update(ids)
            for query_id in ids:
                group = self.normalized_text_group_by_id.get(query_id)
                if not isinstance(group, str) or not group:
                    raise LeakageError("every query ID needs a normalized-text group")
                previous_role = seen_groups.get(group)
                if previous_role is not None and previous_role != role:
                    raise LeakageError("normalized duplicate query text crosses roles")
                seen_groups[group] = role
            normalized[role] = ids
        extra_groups = set(self.normalized_text_group_by_id).difference(seen_ids)
        if extra_groups:
            raise LeakageError("normalized-text map contains unassigned query IDs")
        object.__setattr__(self, "ids_by_role", normalized)
        object.__setattr__(
            self,
            "normalized_text_group_by_id",
            {query_id: str(self.normalized_text_group_by_id[query_id]) for query_id in seen_ids},
        )

    def validate_ids(self, role: str, ids: Sequence[str], *, require_all: bool) -> None:
        if role not in FIVE_ROLES:
            raise LeakageError("unknown PDCTP query role")
        values = tuple(str(value) for value in ids)
        if len(set(values)) != len(values):
            raise LeakageError("requested query IDs are not unique")
        expected = self.ids_by_role[role]
        if not set(values).issubset(expected):
            raise LeakageError(f"query IDs do not belong exclusively to {role}")
        if require_all and values != expected:
            raise LeakageError(f"{role} access requires the complete frozen ID order")

    def serialize(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "name": "pdctp_five_role_assignments",
            "schema_version": 1,
            "roles": {
                role: {
                    "ordered_ids": list(self.ids_by_role[role]),
                    "ordered_id_hash": fingerprint(list(self.ids_by_role[role])),
                    "n": len(self.ids_by_role[role]),
                }
                for role in FIVE_ROLES
            },
            "normalized_text_groups_disjoint": True,
        }
        body["fingerprint"] = fingerprint(body)
        return body

    @property
    def fingerprint(self) -> str:
        return str(self.serialize()["fingerprint"])


@dataclass(frozen=True)
class RoleAccessToken:
    role: str
    purpose: str
    ordered_id_hash: str
    labels_allowed: bool
    upstream_state_fingerprint: str

    def serialize(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "schema": "pdctp_role_access_token_v1",
            "role": self.role,
            "purpose": self.purpose,
            "ordered_id_hash": self.ordered_id_hash,
            "labels_allowed": self.labels_allowed,
            "upstream_state_fingerprint": self.upstream_state_fingerprint,
        }
        body["fingerprint"] = fingerprint(body)
        return body


class FiveRoleProtocolGuard:
    """State machine that refuses protected access before upstream freezing."""

    VERSION = 1
    REQUIRED_FITS = frozenset({"lid_calibrator", "residual_calibrator"})

    def __init__(self, assignments: FiveRoleAssignments, config_fingerprint: str):
        if not config_fingerprint:
            raise LeakageError("five-role guard requires a frozen config fingerprint")
        self.assignments = assignments
        self.config_fingerprint = str(config_fingerprint)
        self._fit_artifacts: Dict[str, str] = {}
        self._cal_opened = False
        self._selection_fingerprint: Optional[str] = None
        self._hypotheses_fingerprint: Optional[str] = None
        self._certification_result_fingerprint: Optional[str] = None
        self._latency_result_fingerprint: Optional[str] = None
        self._cert_opened = False
        self._latency_opened = False
        self._test_opened = False

    def _state_body(self) -> Dict[str, Any]:
        return {
            "name": "pdctp_five_role_protocol_guard",
            "version": self.VERSION,
            "config_fingerprint": self.config_fingerprint,
            "assignments_fingerprint": self.assignments.fingerprint,
            "fit_artifacts": dict(sorted(self._fit_artifacts.items())),
            "calibration_opened": self._cal_opened,
            "selection_fingerprint": self._selection_fingerprint,
            "hypotheses_fingerprint": self._hypotheses_fingerprint,
            "certification_opened": self._cert_opened,
            "certification_result_fingerprint": self._certification_result_fingerprint,
            "latency_opened": self._latency_opened,
            "latency_result_fingerprint": self._latency_result_fingerprint,
            "test_opened": self._test_opened,
            "mutation_after_selection_allowed": False,
            "mutation_after_certification_allowed": False,
        }

    @property
    def state_fingerprint(self) -> str:
        return fingerprint(self._state_body())

    def serialize(self) -> Dict[str, Any]:
        body = self._state_body()
        body["fingerprint"] = fingerprint(body)
        return body

    def register_fit(
        self,
        name: str,
        *,
        role: str,
        ids: Sequence[str],
        artifact_fingerprint: str,
    ) -> None:
        if not self._cal_opened:
            raise LeakageError("query_cal must be opened before calibration fitting")
        if self._selection_fingerprint is not None or self._cert_opened:
            raise LeakageError("calibrators cannot refit after tune selection")
        if role != "query_cal":
            raise LeakageError("calibration fitting is restricted to query_cal")
        if name not in self.REQUIRED_FITS:
            raise LeakageError("unknown required calibration fit")
        if name in self._fit_artifacts:
            raise LeakageError("a calibration fit cannot be replaced")
        if not artifact_fingerprint:
            raise LeakageError("calibration fit requires an artifact fingerprint")
        self.assignments.validate_ids(role, ids, require_all=True)
        self._fit_artifacts[name] = str(artifact_fingerprint)

    def open_calibration(self, ids: Sequence[str]) -> RoleAccessToken:
        if self._cal_opened or self._fit_artifacts or self._selection_fingerprint is not None:
            raise LeakageError("query_cal may be opened only once before fitting")
        self.assignments.validate_ids("query_cal", ids, require_all=True)
        self._cal_opened = True
        return RoleAccessToken(
            "query_cal",
            "calibration_fit",
            fingerprint(list(ids)),
            True,
            self.state_fingerprint,
        )

    def open_tune_selection(self, ids: Sequence[str]) -> RoleAccessToken:
        if set(self._fit_artifacts) != self.REQUIRED_FITS:
            raise LeakageError("all query_cal fits must be frozen before query_tune")
        if self._selection_fingerprint is not None:
            raise LeakageError("query_tune selection is already frozen")
        self.assignments.validate_ids("query_tune", ids, require_all=True)
        return RoleAccessToken(
            "query_tune",
            "policy_selection",
            fingerprint(list(ids)),
            True,
            self.state_fingerprint,
        )

    def freeze_selection(self, token: RoleAccessToken, policy_fingerprint: str) -> None:
        if token.role != "query_tune" or token.purpose != "policy_selection":
            raise LeakageError("selection token is invalid")
        if token.upstream_state_fingerprint != self.state_fingerprint:
            raise LeakageError("selection token is stale")
        if not policy_fingerprint:
            raise LeakageError("selected policy requires a fingerprint")
        self._selection_fingerprint = str(policy_fingerprint)

    def freeze_hypotheses(self, artifact_fingerprint: str) -> None:
        if self._selection_fingerprint is None:
            raise LeakageError("selection must be frozen before hypotheses")
        if self._hypotheses_fingerprint is not None or self._cert_opened:
            raise LeakageError("certification hypotheses are immutable")
        if not artifact_fingerprint:
            raise LeakageError("hypotheses require an artifact fingerprint")
        self._hypotheses_fingerprint = str(artifact_fingerprint)

    def open_certification(self, ids: Sequence[str]) -> RoleAccessToken:
        if self._selection_fingerprint is None or self._hypotheses_fingerprint is None:
            raise LeakageError("selection and hypotheses must freeze before query_cert")
        if self._cert_opened or self._certification_result_fingerprint is not None:
            raise LeakageError("query_cert may be opened only once")
        self.assignments.validate_ids("query_cert", ids, require_all=True)
        self._cert_opened = True
        return RoleAccessToken(
            "query_cert",
            "scientific_certification",
            fingerprint(list(ids)),
            True,
            self.state_fingerprint,
        )

    def close_certification(
        self, token: RoleAccessToken, result_fingerprint: str
    ) -> None:
        if token.role != "query_cert" or token.purpose != "scientific_certification":
            raise LeakageError("certification token is invalid")
        if token.ordered_id_hash != fingerprint(
            list(self.assignments.ids_by_role["query_cert"])
        ) or token.upstream_state_fingerprint != self.state_fingerprint:
            raise LeakageError("certification token identity is stale or mismatched")
        if not self._cert_opened or self._certification_result_fingerprint is not None:
            raise LeakageError("certification is not open or is already terminal")
        if not result_fingerprint:
            raise LeakageError("terminal certification needs a result fingerprint")
        self._certification_result_fingerprint = str(result_fingerprint)

    def open_latency(
        self, ids: Sequence[str], *, labels_requested: bool
    ) -> RoleAccessToken:
        if self._certification_result_fingerprint is None:
            raise LeakageError("latency cannot run before terminal certification")
        if labels_requested:
            raise LeakageError("query_latency is label-free")
        if self._latency_opened or self._latency_result_fingerprint is not None:
            raise LeakageError("query_latency may be opened only once")
        self.assignments.validate_ids("query_latency", ids, require_all=True)
        self._latency_opened = True
        return RoleAccessToken(
            "query_latency",
            "paired_systems_measurement",
            fingerprint(list(ids)),
            False,
            self.state_fingerprint,
        )

    def close_latency(self, token: RoleAccessToken, result_fingerprint: str) -> None:
        if token.role != "query_latency" or token.labels_allowed:
            raise LeakageError("latency token is invalid")
        if token.ordered_id_hash != fingerprint(
            list(self.assignments.ids_by_role["query_latency"])
        ) or token.upstream_state_fingerprint != self.state_fingerprint:
            raise LeakageError("latency token identity is stale or mismatched")
        if not self._latency_opened or self._latency_result_fingerprint is not None:
            raise LeakageError("latency is not open or is already terminal")
        if not result_fingerprint:
            raise LeakageError("terminal latency needs a result fingerprint")
        self._latency_result_fingerprint = str(result_fingerprint)

    def open_test(self, ids: Sequence[str]) -> RoleAccessToken:
        if (
            self._certification_result_fingerprint is None
            or self._latency_result_fingerprint is None
        ):
            raise LeakageError("query_test requires terminal certification and latency")
        if self._test_opened:
            raise LeakageError("query_test may be opened only once")
        self.assignments.validate_ids("query_test", ids, require_all=True)
        self._test_opened = True
        return RoleAccessToken(
            "query_test",
            "descriptive_final_report",
            fingerprint(list(ids)),
            True,
            self.state_fingerprint,
        )
