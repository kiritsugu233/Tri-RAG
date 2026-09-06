"""Frozen Step 4 protocol and CPU/offline synthetic input-guard exercise.

No real-data loader, selector, certifier or systems benchmark is implemented.
The upstream two-action controller is exercised only with explicit mock states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Sequence

from .tls_rag_step3 import (
    Action,
    CalibrationCell,
    CalibrationTable,
    CalibratedStep3Controller,
    CandidateSpec,
    LinearScoreModel,
    STATE_SCHEMA,
    Step3DeployableState,
    assert_deployable_only,
    feature_names_for_row,
    load_step3_config,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/tls_rag_step4_protocol_v1.json"
PROTOCOL_FINGERPRINT = "35a3ae863249d1f1d7aec7f0b2fbaae877871093216523177b67e5c40b0a802e"
ROLES = (
    "query_cal_model_fit", "query_cal_bound_fit", "query_tune",
    "query_cert", "query_latency", "query_test",
)
ARTIFACT_NAMES = (
    "protocol_identity.json", "synthetic_role_proof.json",
    "synthetic_transitions.json", "synthetic_controller_probes.json",
    "denial_report.json", "manifest.json",
)


class ProtocolError(ValueError):
    """A closed gate or an invalid synthetic protocol request."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class Protocol:
    # Immutable serialized storage; callers receive fresh copies, not mutable
    # mappings that could change after fingerprint validation.
    payload: bytes

    @property
    def raw(self) -> dict[str, Any]:
        return json.loads(self.payload)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


def parse_protocol(text: str, expected_fingerprint: str) -> Protocol:
    if expected_fingerprint != PROTOCOL_FINGERPRINT:
        raise ProtocolError("pre-data protocol fingerprint mismatch")
    try:
        raw = json.loads(text, object_pairs_hook=_unique_object)
        payload = canonical(raw)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"invalid protocol JSON: {exc}") from exc
    if hashlib.sha256(payload).hexdigest() != PROTOCOL_FINGERPRINT:
        raise ProtocolError("frozen protocol content changed; new version required")
    return Protocol(payload)


def load_protocol(expected_fingerprint: str) -> Protocol:
    # Check the caller's pin before touching even the fixed protocol path.
    if expected_fingerprint != PROTOCOL_FINGERPRINT:
        raise ProtocolError("pre-data protocol fingerprint mismatch")
    return parse_protocol(PROTOCOL_PATH.read_text(encoding="utf-8"), expected_fingerprint)


def validate_protocol(protocol: Protocol, expected_fingerprint: str) -> None:
    if type(protocol) is not Protocol or expected_fingerprint != PROTOCOL_FINGERPRINT:
        raise ProtocolError("pre-data protocol fingerprint mismatch")
    if protocol.fingerprint != PROTOCOL_FINGERPRINT:
        raise ProtocolError("frozen protocol content changed; new version required")


def guard_request(request: Any) -> None:
    """Reject arbitrary paths, role inputs and unknown fields without I/O."""
    if type(request) is not dict or set(request) != {"schema_version", "input_kind"}:
        raise ProtocolError("only the exact synthetic request schema is accepted")
    if request != {
        "schema_version": "tls_rag_step4_readiness_request_v1",
        "input_kind": "generated_synthetic",
    }:
        raise ProtocolError("real-data gate closed; only generated synthetic input is allowed")


@dataclass(frozen=True)
class SyntheticIdentity:
    query_id: str
    independent_unit_id: str
    role: str

    def to_dict(self) -> dict[str, str]:
        return dict(vars(self))


@dataclass(frozen=True)
class SyntheticFixture:
    schema_version: str
    seed: int
    corpus_ids: tuple[str, ...]
    identities: tuple[SyntheticIdentity, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "corpus_ids": list(self.corpus_ids),
            "identities": [item.to_dict() for item in self.identities],
        }

    @property
    def fingerprint(self) -> str:
        return digest(self.to_dict())


def build_synthetic_fixture(protocol: Protocol) -> SyntheticFixture:
    validate_protocol(protocol, PROTOCOL_FINGERPRINT)
    spec = protocol.raw["synthetic_fixture"]
    count = spec["query_count_per_role"]
    seed = spec["seed"]
    units = [f"tls-step4-synthetic-unit-{i:04d}" for i in range(len(ROLES) * count)]
    split_seed = protocol.raw["roles"]["split_seed"]
    units.sort(key=lambda unit: (digest([split_seed, unit]), unit))
    identities = tuple(
        SyntheticIdentity(
            f"tls-step4-synthetic-query-{unit.rsplit('-', 1)[1]}", unit,
            ROLES[index // count],
        )
        for index, unit in enumerate(units)
    )
    fixture = SyntheticFixture(
        spec["schema_version"], seed,
        tuple(f"tls-step4-synthetic-corpus-{i:04d}" for i in range(spec["corpus_size"])),
        identities,
    )
    validate_fixture(fixture, protocol)
    return fixture


def validate_fixture(fixture: SyntheticFixture, protocol: Protocol) -> None:
    validate_protocol(protocol, PROTOCOL_FINGERPRINT)
    spec = protocol.raw["synthetic_fixture"]
    if type(fixture) is not SyntheticFixture or fixture.schema_version != spec["schema_version"]:
        raise ProtocolError("only a synthetic fixture is accepted")
    if type(fixture.seed) is not int or fixture.seed != spec["seed"]:
        raise ProtocolError("synthetic fixture seed mismatch")
    if type(fixture.identities) is not tuple or type(fixture.corpus_ids) is not tuple:
        raise ProtocolError("fixture containers must be immutable tuples")
    queries, units = [], []
    for item in fixture.identities:
        if type(item) is not SyntheticIdentity or item.role not in ROLES:
            raise ProtocolError("unknown synthetic identity or role")
        for value in (item.query_id, item.independent_unit_id):
            if type(value) is not str or not value:
                raise ProtocolError("IDs must be nonempty stable strings")
        queries.append(item.query_id)
        units.append(item.independent_unit_id)
    if len(queries) != len(set(queries)):
        raise ProtocolError("query IDs overlap within or across roles")
    if len(units) != len(set(units)):
        raise ProtocolError("independent units overlap within or across roles")
    if any(type(item) is not str or not item for item in fixture.corpus_ids):
        raise ProtocolError("corpus IDs must be nonempty stable strings")
    if len(set(fixture.corpus_ids)) != len(fixture.corpus_ids):
        raise ProtocolError("duplicate corpus IDs")
    if set(queries).intersection(fixture.corpus_ids):
        raise ProtocolError("queries must be external to corpus IDs")
    for values, kind in ((queries, "query"), (units, "unit"), (fixture.corpus_ids, "corpus")):
        if any(re.fullmatch(rf"tls-step4-synthetic-{kind}-[0-9]{{4}}", value) is None for value in values):
            raise ProtocolError("only the generated synthetic ID namespace is allowed")
    expected_counts = {role: spec["query_count_per_role"] for role in ROLES}
    if Counter(item.role for item in fixture.identities) != expected_counts:
        raise ProtocolError("synthetic role counts are incomplete")
    if len(fixture.corpus_ids) != spec["corpus_size"]:
        raise ProtocolError("synthetic corpus count mismatch")
    count = spec["query_count_per_role"]
    expected_units = [f"tls-step4-synthetic-unit-{i:04d}" for i in range(len(ROLES) * count)]
    split_seed = protocol.raw["roles"]["split_seed"]
    expected_units.sort(key=lambda unit: (digest([split_seed, unit]), unit))
    expected_assignment = tuple(
        SyntheticIdentity(f"tls-step4-synthetic-query-{unit.rsplit('-', 1)[1]}", unit, ROLES[index // count])
        for index, unit in enumerate(expected_units)
    )
    if fixture.identities != expected_assignment:
        raise ProtocolError("frozen label-free synthetic role assignment changed")
    if fixture.corpus_ids != tuple(f"tls-step4-synthetic-corpus-{i:04d}" for i in range(spec["corpus_size"])):
        raise ProtocolError("frozen synthetic corpus identity changed")


@dataclass(frozen=True)
class SyntheticDecision:
    """Toy receipt record for a role-opening simulation, not a retrieval result."""
    query_id: str
    budget: int
    action: str

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


class SyntheticSession:
    """Sequential role/receipt guards. No loader or arbitrary artifact handle."""

    def __init__(self, protocol: Protocol, fixture: SyntheticFixture):
        validate_fixture(fixture, protocol)
        self._protocol = protocol
        self._fixture = fixture
        self._cursor = 0
        self._phase = "sealed"
        self._terminal = False
        self._decision_payload: Optional[bytes] = None
        self._events: list[dict[str, Any]] = []
        self._previous_receipt = digest([protocol.fingerprint, fixture.fingerprint])

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def events(self) -> list[dict[str, Any]]:
        return json.loads(canonical(self._events))

    @property
    def previous_receipt(self) -> str:
        return self._previous_receipt

    def _record(self, event: str, **details: Any) -> None:
        self._events.append({
            "sequence": len(self._events), "event": event,
            "protocol_fingerprint": self._protocol.fingerprint,
            "fixture_fingerprint": self._fixture.fingerprint, **details,
        })

    def _deny(self, reason: str) -> None:
        self._terminal = True
        self._record("terminal_protocol_failure", reason=reason)
        raise ProtocolError(reason)

    def _require_active_role(self, role: str) -> None:
        if self._terminal:
            self._deny("session terminal; no reopening or retry")
        if self._cursor >= len(ROLES) or role != ROLES[self._cursor]:
            self._deny("role opening order mismatch")

    def open_inputs(self, role: str, protocol_fingerprint: str, previous_receipt: str) -> None:
        self._require_active_role(role)
        if protocol_fingerprint != self._protocol.fingerprint:
            self._deny("pre-data protocol fingerprint mismatch")
        if previous_receipt != self._previous_receipt:
            self._deny("prior frozen artifact receipt mismatch")
        if self._phase != "sealed":
            self._deny("role inputs already opened")
        self._phase = "inputs_open"
        self._record("synthetic_inputs_open", role=role, prior_receipt=previous_receipt)

    def close_phase_a(self, role: str, records: tuple[SyntheticDecision, ...]) -> str:
        self._require_active_role(role)
        if self._phase != "inputs_open" or type(records) is not tuple:
            self._deny("Phase A must close once after inputs and before labels")
        if any(type(row) is not SyntheticDecision for row in records):
            self._deny("Phase A only accepts label-free synthetic decisions")
        if any(type(row.query_id) is not str or not row.query_id for row in records):
            self._deny("Phase A query IDs must be stable strings")
        ids = [row.query_id for row in records]
        expected = [item.query_id for item in self._fixture.identities if item.role == role]
        if len(ids) != len(set(ids)) or set(ids) != set(expected):
            self._deny("Phase A must contain each role query exactly once")
        for row in records:
            if type(row.budget) is not int or row.budget not in (3, 6, 12) or row.action != "STOP":
                self._deny("invalid synthetic terminal decision or grid growth")
        payload = {
            "schema_version": "tls_rag_step4_synthetic_closed_phase_a_v1",
            "role": role, "protocol_fingerprint": self._protocol.fingerprint,
            "fixture_fingerprint": self._fixture.fingerprint,
            "prior_receipt": self._previous_receipt,
            "records": [row.to_dict() for row in sorted(records, key=lambda row: row.query_id)],
        }
        self._decision_payload = canonical(payload)
        result = hashlib.sha256(self._decision_payload).hexdigest()
        self._phase = "decisions_closed"
        self._record("synthetic_phase_a_closed", role=role, fingerprint=result, payload=payload)
        return result

    def open_synthetic_labels(self, role: str, phase_a_fingerprint: str) -> tuple[tuple[str, bool], ...]:
        self._require_active_role(role)
        if role == "query_latency":
            self._deny("latency labels are never allowed")
        if self._phase != "decisions_closed" or self._decision_payload is None:
            self._deny("labels require a closed Phase A")
        if hashlib.sha256(self._decision_payload).hexdigest() != phase_a_fingerprint:
            self._deny("Phase A fingerprint mismatch before label opening")
        # Deliberately generated here, after the complete immutable Phase A.
        labels = tuple(
            (item.query_id, int(digest([self._fixture.seed, item.query_id])[:8], 16) % 2 == 0)
            for item in self._fixture.identities if item.role == role
        )
        self._phase = "labels_open"
        self._record("synthetic_labels_open", role=role, phase_a_fingerprint=phase_a_fingerprint,
                     synthetic_label_fingerprint=digest(labels))
        return labels

    def finish_role(self, role: str, *, simulated_gate_pass: bool) -> str:
        self._require_active_role(role)
        required_phase = "decisions_closed" if role == "query_latency" else "labels_open"
        if self._phase != required_phase or type(simulated_gate_pass) is not bool:
            self._deny("role cannot finish without its required closed phases")
        if not simulated_gate_pass:
            self._deny("simulated gate failure; no selection retry, top-up or later role")
        receipt = digest({
            "schema_version": "tls_rag_step4_synthetic_role_receipt_v1",
            "protocol_fingerprint": self._protocol.fingerprint,
            "fixture_fingerprint": self._fixture.fingerprint,
            "prior_receipt": self._previous_receipt, "role": role,
            "events": self._events,
        })
        self._record("synthetic_role_closed", role=role, receipt=receipt,
                     interpretation="transition_simulation_not_selection_or_certification")
        self._previous_receipt = receipt
        self._cursor += 1
        self._phase = "sealed"
        self._decision_payload = None
        self._terminal = self._cursor == len(ROLES)
        return receipt


def guard_deployable_state(state: Any, candidate: CandidateSpec) -> None:
    """Positive schema/type checks before passing a mock state upstream."""
    if type(state) is not Step3DeployableState:
        raise ProtocolError("only the exact upstream deployable state type is accepted")
    registered = {
        2: "tls-rag-row2-sequential-v1", 3: "tls-rag-row3-observed-pair-v1",
        4: "tls-rag-row4-plan-facet-v1",
    }
    if type(candidate) is not CandidateSpec or type(candidate.row) is not int or registered.get(candidate.row) != candidate.candidate_id:
        raise ProtocolError("unregistered frozen candidate")
    if state.schema_version != STATE_SCHEMA or state.candidate_id != candidate.candidate_id or state.row != candidate.row:
        raise ProtocolError("candidate or state schema mismatch")
    if type(state.stage) is not int or not 0 <= state.stage < 3:
        raise ProtocolError("stage must be a nonnegative frozen-grid index")
    if type(state.current_budget) is not int or state.current_budget != (3, 6, 12)[state.stage]:
        raise ProtocolError("budget grid mismatch")
    if type(state.remaining_grid_steps) is not int or state.remaining_grid_steps != 2 - state.stage:
        raise ProtocolError("remaining grid steps mismatch")
    if type(state.query_id) is not str or not re.fullmatch(r"tls-step4-synthetic-query-[0-9]{4}", state.query_id):
        raise ProtocolError("only synthetic state IDs are accepted")
    if type(state.context_ids) is not tuple or len(state.context_ids) != 2 or len(set(state.context_ids)) != 2:
        raise ProtocolError("context must contain two unique synthetic IDs")
    if any(type(item) is not str or not re.fullmatch(r"tls-step4-synthetic-corpus-[0-9]{4}", item) for item in state.context_ids):
        raise ProtocolError("only synthetic context IDs are accepted")
    if type(state.feature_values) is not tuple or any(type(pair) is not tuple or len(pair) != 2 for pair in state.feature_values):
        raise ProtocolError("features must be immutable name/value pairs")
    names = feature_names_for_row(candidate.row)
    if tuple(name for name, _ in state.feature_values) != names:
        raise ProtocolError("unknown, missing, reordered or duplicate feature")
    if any(type(value) not in (int, float) for _, value in state.feature_values):
        raise ProtocolError("features accept numeric scalars only; no nested payload")
    if any(not math.isfinite(value) for _, value in state.feature_values):
        raise ProtocolError("nonfinite deployable value; no imputation or decision")
    if any(type(value) is not bool for value in (
        state.base_state_valid, state.required_risk_profile_valid, state.evidence_plan_valid,
    )):
        raise ProtocolError("validity flags must be booleans")
    expected_schema = digest({"schema_version": "tls_rag_step3_feature_names_v1", "names": list(names)})
    if state.feature_schema_fingerprint != expected_schema:
        raise ProtocolError("feature schema fingerprint mismatch")
    assert_deployable_only(state)


def mock_state(candidate: CandidateSpec, stage: int) -> Step3DeployableState:
    names = feature_names_for_row(candidate.row)
    return Step3DeployableState(
        STATE_SCHEMA, candidate.candidate_id, candidate.row,
        "tls-step4-synthetic-query-0000", stage, (3, 6, 12)[stage], 2 - stage,
        ("tls-step4-synthetic-corpus-0000", "tls-step4-synthetic-corpus-0001"),
        tuple((name, 0.0) for name in names),
        digest({"schema_version": "tls_rag_step3_feature_names_v1", "names": list(names)}),
        True, True, True,
    )


def mock_controller(candidate: CandidateSpec, *, gain_upper: float, suff_lower: float,
                    cell_valid: bool = True) -> CalibratedStep3Controller:
    """Manual constant models and bounds solely for controller branch probes."""
    names = feature_names_for_row(candidate.row)
    model = LinearScoreModel(
        "tls_rag_transparent_ridge_score_v1", "synthetic_probe", candidate.candidate_id,
        names, tuple(0.0 for _ in names), tuple(1.0 for _ in names),
        tuple(0.0 for _ in names), 0.5, 0.01,
        ("tls-step4-synthetic-query-0001",), 1, 0, "synthetic_probe_not_fitted",
    )
    outcomes = ("remaining_useful_evidence_event", "current_context_sufficiency_event")
    cells = tuple(
        CalibrationCell(
            candidate.candidate_id, stage, outcome, 0, 0.0, 1.0, (), 0, 0,
            suff_lower if outcome == outcomes[1] else 0.0,
            gain_upper if outcome == outcomes[0] else 1.0, cell_valid, 0.2 / 36,
        )
        for stage in range(3) for outcome in outcomes
    )
    table = CalibrationTable(
        "tls_rag_reachable_clopper_pearson_v1", candidate.candidate_id,
        tuple((stage, outcome, (0.0, 1.0)) for stage in range(3) for outcome in outcomes),
        cells, (), "synthetic_manual_table_not_calibrated",
    )
    return CalibratedStep3Controller(
        candidate=candidate, gain_model=model, sufficiency_model=model,
        calibration_table=table, budget_grid=(3, 6, 12), maximum_expansions=2,
        delta_gain=0.65, tau_sufficient=0.15,
    )


def synthetic_controller_probes(protocol: Protocol) -> list[dict[str, Any]]:
    validate_protocol(protocol, PROTOCOL_FINGERPRINT)
    rows = []
    scenarios = (
        ("dual_bound_pass", 0.1, 0.9, True),
        ("remaining_bound_fail", 0.8, 0.9, True),
        ("sufficiency_bound_fail", 0.1, 0.1, True),
        ("underpowered_cell", 1.0, 0.0, False),
        ("invalid_state", 0.1, 0.9, True),
        ("invalid_plan", 0.1, 0.9, True),
        ("invalid_risk", 0.1, 0.9, True),
    )
    for item in protocol.raw["candidates"]["adaptive"]:
        candidate = CandidateSpec(item["upstream_id"], item["row"])
        for scenario, gain, suff, valid in scenarios:
            controller = mock_controller(candidate, gain_upper=gain, suff_lower=suff, cell_valid=valid)
            for stage in range(3):
                state = mock_state(candidate, stage)
                if scenario == "invalid_state":
                    state = replace(state, base_state_valid=False)
                elif scenario == "invalid_plan":
                    state = replace(state, evidence_plan_valid=False)
                elif scenario == "invalid_risk":
                    state = replace(state, required_risk_profile_valid=False)
                guard_deployable_state(state, candidate)
                choice = controller.choose(state)
                expected = Action.STOP if scenario == "dual_bound_pass" or stage == 2 else Action.EXPAND_TO_NEXT_GRID_VALUE
                if choice.action is not expected:
                    raise AssertionError("frozen controller synthetic branch behavior changed")
                if choice.next_budget != ((3, 6, 12)[stage + 1] if expected is Action.EXPAND_TO_NEXT_GRID_VALUE else None):
                    raise AssertionError("frozen controller grew or skipped the grid")
                rows.append({
                    "schema_version": "tls_rag_step4_synthetic_controller_probe_v1",
                    "candidate_id": item["id"], "upstream_candidate_id": candidate.candidate_id,
                    "row": candidate.row, "scenario": scenario, "stage": stage,
                    "budget": state.current_budget, "action": choice.action.value,
                    "next_budget": choice.next_budget, "reason": choice.reason,
                    "terminal_failure": stage == 2 and scenario != "dual_bound_pass",
                    "interpretation": "manual_synthetic_branch_probe_not_evidence_or_quality_result",
                })
    return rows


def simulate_roles(protocol: Protocol, fixture: SyntheticFixture) -> list[dict[str, Any]]:
    session = SyntheticSession(protocol, fixture)
    for role in ROLES:
        session.open_inputs(role, protocol.fingerprint, session.previous_receipt)
        records = tuple(SyntheticDecision(item.query_id, 12, "STOP")
                        for item in fixture.identities if item.role == role)
        closed = session.close_phase_a(role, records)
        if role != "query_latency":
            session.open_synthetic_labels(role, closed)
        session.finish_role(role, simulated_gate_pass=True)
    return session.events


def _artifact(schema: str, **payload: Any) -> dict[str, Any]:
    body = {"schema_version": schema, **payload}
    return {**body, "fingerprint": digest(body)}


def run_readiness(output: Path, expected_fingerprint: str, request: Any) -> dict[str, str]:
    guard_request(request)
    protocol = load_protocol(expected_fingerprint)
    upstream = load_step3_config(ROOT / "configs/tls_rag_step3_synthetic_v1.json")
    if upstream.config_fingerprint != protocol.raw["upstream"]["step3_config_fingerprint"]:
        raise ProtocolError("frozen upstream config fingerprint mismatch before fixture generation")
    # The output must be new. Never enumerate or read an existing run directory.
    # Refuse symlink ancestors before creating/writing any synthetic artifact.
    output = Path(output).absolute()
    if any(parent.is_symlink() for parent in (output, *output.parents)):
        raise ProtocolError("output path must not contain symlinks")
    if output.exists():
        raise FileExistsError("output must be absent; existing directories are never inspected")
    fixture = build_synthetic_fixture(protocol)
    transitions = simulate_roles(protocol, fixture)
    probes = synthetic_controller_probes(protocol)
    artifacts = {
        "protocol_identity.json": _artifact(
            "tls_rag_step4_protocol_identity_v1", protocol=protocol.raw,
            protocol_fingerprint=protocol.fingerprint,
        ),
        "synthetic_role_proof.json": _artifact(
            "tls_rag_step4_synthetic_role_proof_v1", fixture=fixture.to_dict(),
            fixture_fingerprint=fixture.fingerprint, pairwise_disjoint=True,
            independent_sampling_proven=False, real_roles_opened=False,
        ),
        "synthetic_transitions.json": _artifact(
            "tls_rag_step4_synthetic_transition_audit_v1", events=transitions,
            simulation_only=True,
        ),
        "synthetic_controller_probes.json": _artifact(
            "tls_rag_step4_synthetic_probe_collection_v1", records=probes,
            mock_models_and_tables=True,
        ),
        "denial_report.json": _artifact(
            "tls_rag_step4_denial_report_v1", real_data_gate="closed",
            claims_denied=["real_data_run", "selection", "certification", "latency",
                           "posterior_probability", "internal_simultaneous_coverage",
                           "evidence_guarantee", "answer_quality", "production"],
            internal_alpha=0.2, future_cert_alpha=0.05, shared_error_budget=False,
            step5_started=False,
        ),
    }
    artifacts["manifest.json"] = _artifact(
        "tls_rag_step4_readiness_manifest_v1",
        protocol_fingerprint=protocol.fingerprint,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixture_fingerprint=fixture.fingerprint,
        artifact_fingerprints={name: value["fingerprint"] for name, value in artifacts.items()},
        artifact_file_sha256={name: hashlib.sha256(canonical(value) + b"\n").hexdigest()
                              for name, value in artifacts.items()},
        real_data_gate="closed", scope="cpu_offline_generated_synthetic_protocol_guards_only",
        portable_artifacts=list(ARTIFACT_NAMES),
    )
    output.mkdir(parents=True, exist_ok=False)
    for name, value in artifacts.items():
        (output / name).write_bytes(canonical(value) + b"\n")
    return {name: value["fingerprint"] for name, value in artifacts.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-protocol-fingerprint", required=True)
    args = parser.parse_args(argv)
    identities = run_readiness(args.output, args.expected_protocol_fingerprint, {
        "schema_version": "tls_rag_step4_readiness_request_v1",
        "input_kind": "generated_synthetic",
    })
    print(json.dumps({"real_data_gate": "closed", "artifact_fingerprints": identities}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
