import hashlib
import json
import socket
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from tri_rag_harness.tls_rag_step3 import CandidateSpec, feature_names_for_row
from tri_rag_harness.tls_rag_step4 import (
    ARTIFACT_NAMES, PROTOCOL_FINGERPRINT, PROTOCOL_PATH, ROLES,
    Protocol, ProtocolError, SyntheticDecision, SyntheticIdentity,
    SyntheticSession, build_synthetic_fixture, canonical, digest,
    guard_deployable_state, guard_request, load_protocol, main,
    mock_controller, mock_state, parse_protocol, run_readiness,
    simulate_roles, synthetic_controller_probes, validate_fixture,
)


REQUEST = {"schema_version": "tls_rag_step4_readiness_request_v1", "input_kind": "generated_synthetic"}


class TlsRagStep4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(PROTOCOL_FINGERPRINT)
        cls.fixture = build_synthetic_fixture(cls.protocol)
        cls.candidate = CandidateSpec("tls-rag-row3-observed-pair-v1", 3)

    def session(self):
        return SyntheticSession(self.protocol, self.fixture)

    def decisions(self, role):
        return tuple(SyntheticDecision(item.query_id, 12, "STOP")
                     for item in self.fixture.identities if item.role == role)

    def complete_role(self, session, role, passed=True):
        session.open_inputs(role, PROTOCOL_FINGERPRINT, session.previous_receipt)
        closed = session.close_phase_a(role, self.decisions(role))
        if role != "query_latency":
            session.open_synthetic_labels(role, closed)
        return session.finish_role(role, simulated_gate_pass=passed)

    def test_frozen_protocol_rejects_every_top_level_change(self):
        self.assertEqual(digest(self.protocol.raw), PROTOCOL_FINGERPRINT)
        for key in self.protocol.raw:
            with self.subTest(key=key):
                changed = self.protocol.raw
                changed[key] = None
                with self.assertRaisesRegex(ProtocolError, "content changed"):
                    parse_protocol(json.dumps(changed), PROTOCOL_FINGERPRINT)
        changed = self.protocol.raw
        changed["unregistered_path"] = "must-not-open"
        with self.assertRaises(ProtocolError):
            parse_protocol(json.dumps(changed), PROTOCOL_FINGERPRINT)

    def test_duplicate_keys_nonfinite_and_type_substitution_are_rejected(self):
        for value in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', 'null', '[]'):
            with self.subTest(value=value), self.assertRaises(ProtocolError):
                parse_protocol(value, PROTOCOL_FINGERPRINT)
        changed = self.protocol.raw
        changed["scope"]["synthetic_only"] = 1
        with self.assertRaises(ProtocolError):
            parse_protocol(json.dumps(changed), PROTOCOL_FINGERPRINT)

    def test_protocol_storage_cannot_be_changed_through_returned_mapping(self):
        copy = self.protocol.raw
        copy["retrieval"]["budget_grid"].append(24)
        self.assertEqual(self.protocol.raw["retrieval"]["budget_grid"], [3, 6, 12])
        self.assertEqual(self.protocol.fingerprint, PROTOCOL_FINGERPRINT)
        with self.assertRaises(ProtocolError):
            build_synthetic_fixture(Protocol(canonical(copy)))

    def test_wrong_pin_rejects_before_any_path_read_or_fixture_generation(self):
        with mock.patch.object(Path, "read_text", side_effect=AssertionError("unexpected read")) as read:
            with self.assertRaisesRegex(ProtocolError, "pre-data"):
                load_protocol("0" * 64)
            read.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "absent"
            with mock.patch("tri_rag_harness.tls_rag_step4.build_synthetic_fixture") as builder:
                with self.assertRaises(ProtocolError):
                    run_readiness(output, "0" * 64, REQUEST)
                builder.assert_not_called()
            self.assertFalse(output.exists())

    def test_bad_upstream_identity_rejects_before_fixture_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "absent"
            with mock.patch("tri_rag_harness.tls_rag_step4.load_step3_config",
                            return_value=mock.Mock(config_fingerprint="0" * 64)):
                with mock.patch("tri_rag_harness.tls_rag_step4.build_synthetic_fixture") as builder:
                    with self.assertRaisesRegex(ProtocolError, "upstream"):
                        run_readiness(output, PROTOCOL_FINGERPRINT, REQUEST)
                    builder.assert_not_called()
            self.assertFalse(output.exists())

    def test_real_protected_and_arbitrary_path_requests_fail_without_io(self):
        requests = [
            {**REQUEST, "input_kind": kind} for kind in ("real", "protected", "query_cert", "file")
        ] + [
            {**REQUEST, key: {"nested": "unopened-fixture-path"}}
            for key in ("path", "corpus", "labels", "role_manifest", "adapter", "metadata")
        ]
        with mock.patch.object(Path, "open", side_effect=AssertionError("path opened")) as opened:
            with mock.patch.object(Path, "stat", side_effect=AssertionError("path inspected")) as inspected:
                for request in requests:
                    with self.subTest(request=request), self.assertRaises(ProtocolError):
                        run_readiness(Path("unopened-output"), PROTOCOL_FINGERPRINT, request)
                opened.assert_not_called()
                inspected.assert_not_called()

    def test_cli_has_no_real_data_or_config_override_switch(self):
        with mock.patch("sys.stderr"):
            for flag in ("--data", "--enable-real", "--config", "--role-manifest"):
                with self.subTest(flag=flag), self.assertRaises(SystemExit) as error:
                    main(["--output", "unopened-output", "--expected-protocol-fingerprint",
                          PROTOCOL_FINGERPRINT, flag, "unopened-input"])
                self.assertEqual(error.exception.code, 2)

    def test_synthetic_roles_are_complete_disjoint_and_external(self):
        validate_fixture(self.fixture, self.protocol)
        identities = self.fixture.identities
        self.assertEqual(len({item.query_id for item in identities}), 24)
        self.assertEqual(len({item.independent_unit_id for item in identities}), 24)
        self.assertFalse(set(self.fixture.corpus_ids).intersection(item.query_id for item in identities))
        for role in ROLES:
            self.assertEqual(sum(item.role == role for item in identities), 4)
        self.assertEqual(self.fixture.to_dict(), build_synthetic_fixture(self.protocol).to_dict())

    def test_duplicate_query_and_family_rejected_within_and_across_roles(self):
        rows = self.fixture.identities
        for target in (1, 4):
            for field in ("query_id", "independent_unit_id"):
                changed = list(rows)
                changed[target] = replace(rows[target], **{field: getattr(rows[0], field)})
                with self.subTest(target=target, field=field), self.assertRaisesRegex(ProtocolError, "overlap"):
                    validate_fixture(replace(self.fixture, identities=tuple(changed)), self.protocol)

    def test_unknown_role_nonstring_ids_missing_queries_and_corpus_overlap_reject(self):
        first, *rest = self.fixture.identities
        for changed in (replace(first, role="query_cal"), replace(first, query_id=0),
                        replace(first, independent_unit_id=""), replace(first, query_id="external-id")):
            with self.subTest(changed=changed), self.assertRaises(ProtocolError):
                validate_fixture(replace(self.fixture, identities=(changed, *rest)), self.protocol)
        for fixture in (
            replace(self.fixture, identities=tuple(rest)),
            replace(self.fixture, corpus_ids=(first.query_id, *self.fixture.corpus_ids[1:])),
            replace(self.fixture, corpus_ids=(self.fixture.corpus_ids[1], *self.fixture.corpus_ids[1:])),
            replace(self.fixture, seed=0),
        ):
            with self.assertRaises(ProtocolError):
                validate_fixture(fixture, self.protocol)

    def test_role_order_prior_receipt_and_pin_fail_closed(self):
        for role, pin, receipt in ((ROLES[2], PROTOCOL_FINGERPRINT, None),
                                   (ROLES[0], "0" * 64, None),
                                   (ROLES[0], PROTOCOL_FINGERPRINT, "1" * 64)):
            session = self.session()
            with self.assertRaises(ProtocolError):
                session.open_inputs(role, pin, receipt or session.previous_receipt)
            self.assertTrue(session.terminal)
            with self.assertRaisesRegex(ProtocolError, "terminal"):
                session.open_inputs(ROLES[0], PROTOCOL_FINGERPRINT, session.previous_receipt)

    def test_labels_cannot_open_before_all_role_decisions_close(self):
        session = self.session()
        session.open_inputs(ROLES[0], PROTOCOL_FINGERPRINT, session.previous_receipt)
        with self.assertRaisesRegex(ProtocolError, "closed Phase A"):
            session.open_synthetic_labels(ROLES[0], "0" * 64)
        for records in (self.decisions(ROLES[0])[:-1],
                        self.decisions(ROLES[0]) + (self.decisions(ROLES[0])[0],),
                        self.decisions(ROLES[1])):
            session = self.session()
            session.open_inputs(ROLES[0], PROTOCOL_FINGERPRINT, session.previous_receipt)
            with self.assertRaisesRegex(ProtocolError, "each role query"):
                session.close_phase_a(ROLES[0], records)

    def test_phase_a_rejects_labels_extra_payload_and_grid_growth(self):
        for records in (({"query_id": "q", "labels": True},),
                        (replace(self.decisions(ROLES[0])[0], budget=24), *self.decisions(ROLES[0])[1:])):
            session = self.session()
            session.open_inputs(ROLES[0], PROTOCOL_FINGERPRINT, session.previous_receipt)
            with self.assertRaises(ProtocolError):
                session.close_phase_a(ROLES[0], records)
            self.assertTrue(session.terminal)

    def test_phase_a_receipt_binds_records_and_survives_label_opening(self):
        session = self.session()
        role = ROLES[0]
        session.open_inputs(role, PROTOCOL_FINGERPRINT, session.previous_receipt)
        closed = session.close_phase_a(role, self.decisions(role))
        before = session.events[-1]
        self.assertEqual(digest(before["payload"]), closed)
        labels = session.open_synthetic_labels(role, closed)
        self.assertEqual(len(labels), 4)
        self.assertEqual(session.events[-2], before)
        copy = session.events
        copy[1]["payload"]["records"][0]["budget"] = 3
        self.assertEqual(session.events[-2], before)
        bad = self.session()
        bad.open_inputs(role, PROTOCOL_FINGERPRINT, bad.previous_receipt)
        bad.close_phase_a(role, self.decisions(role))
        with self.assertRaisesRegex(ProtocolError, "fingerprint mismatch"):
            bad.open_synthetic_labels(role, "0" * 64)

    def test_latency_never_opens_labels_even_with_valid_receipt(self):
        session = self.session()
        for role in ROLES[:4]:
            self.complete_role(session, role)
        role = "query_latency"
        session.open_inputs(role, PROTOCOL_FINGERPRINT, session.previous_receipt)
        closed = session.close_phase_a(role, self.decisions(role))
        with self.assertRaisesRegex(ProtocolError, "never"):
            session.open_synthetic_labels(role, closed)

    def test_failed_tune_or_cert_is_terminal_without_later_roles_or_retry(self):
        for failed in ("query_tune", "query_cert"):
            with self.subTest(failed=failed):
                session = self.session()
                for role in ROLES[:ROLES.index(failed)]:
                    self.complete_role(session, role)
                with self.assertRaisesRegex(ProtocolError, "simulated gate failure"):
                    self.complete_role(session, failed, passed=False)
                self.assertTrue(session.terminal)
                for role in (failed, ROLES[ROLES.index(failed) + 1]):
                    with self.assertRaisesRegex(ProtocolError, "terminal"):
                        session.open_inputs(role, PROTOCOL_FINGERPRINT, session.previous_receipt)

    def test_simulated_receipt_chain_and_all_phase_orders_reconstruct(self):
        events = simulate_roles(self.protocol, self.fixture)
        previous = digest([PROTOCOL_FINGERPRINT, self.fixture.fingerprint])
        for index, event in enumerate(events):
            self.assertEqual(event["sequence"], index)
            if event["event"] == "synthetic_inputs_open":
                self.assertEqual(event["prior_receipt"], previous)
            elif event["event"] == "synthetic_role_closed":
                reconstructed = digest({
                    "schema_version": "tls_rag_step4_synthetic_role_receipt_v1",
                    "protocol_fingerprint": PROTOCOL_FINGERPRINT,
                    "fixture_fingerprint": self.fixture.fingerprint,
                    "prior_receipt": previous, "role": event["role"], "events": events[:index],
                })
                self.assertEqual(event["receipt"], reconstructed)
                previous = reconstructed
        opened = [event["role"] for event in events if event["event"] == "synthetic_inputs_open"]
        self.assertEqual(opened, list(ROLES))
        self.assertEqual(events, simulate_roles(self.protocol, self.fixture))

    def test_state_guard_rejects_nested_labels_unknown_features_and_subclasses(self):
        state = mock_state(self.candidate, 0)
        for field in ("exact_top_k_ids", "evidence_labels", "realized_recall", "answer", "oracle_exact"):
            for value in ({"nested": [{field: True}]}, ((field, 1.0),)):
                with self.subTest(field=field, value=value), self.assertRaises((ProtocolError, ValueError, TypeError)):
                    guard_deployable_state(replace(state, feature_values=value), self.candidate)
        class ExtendedState(type(state)):
            labels = True
        extended = ExtendedState(**vars(state))
        with self.assertRaises(ProtocolError):
            guard_deployable_state(extended, self.candidate)
        with self.assertRaises(ProtocolError):
            guard_deployable_state({**state.to_dict(), "labels": {}}, self.candidate)

    def test_state_guard_rejects_malformed_grid_ids_flags_and_feature_identity(self):
        state = mock_state(self.candidate, 0)
        for change in (
            {"stage": -1}, {"stage": True}, {"current_budget": 24},
            {"remaining_grid_steps": 0}, {"remaining_grid_steps": True},
            {"query_id": 0}, {"query_id": "real-query"}, {"base_state_valid": 1},
            {"feature_schema_fingerprint": "0" * 64},
            {"context_ids": ("tls-step4-synthetic-corpus-0000",) * 2},
            {"feature_values": state.feature_values[::-1]},
            {"feature_values": state.feature_values + (state.feature_values[0],)},
            {"feature_values": ((state.feature_values[0][0], {"labels": True}), *state.feature_values[1:])},
        ):
            with self.subTest(change=change), self.assertRaises(ProtocolError):
                guard_deployable_state(replace(state, **change), self.candidate)

    def test_input_guard_does_not_change_upstream_decisions(self):
        for stage in range(3):
            state = mock_state(self.candidate, stage)
            controller = mock_controller(self.candidate, gain_upper=0.8, suff_lower=0.1)
            before = controller.choose(state)
            serialized = canonical(state.to_dict())
            guard_deployable_state(state, self.candidate)
            self.assertEqual(before, controller.choose(state))
            self.assertEqual(serialized, canonical(state.to_dict()))

    def test_nonfinite_features_are_rejected_without_imputation_or_controller_call(self):
        controller = mock_controller(self.candidate, gain_upper=0.1, suff_lower=0.9)
        for value in (float("nan"), float("inf"), -float("inf")):
            for stage in range(3):
                state = mock_state(self.candidate, stage)
                state = replace(state, feature_values=((state.feature_values[0][0], value), *state.feature_values[1:]))
                with mock.patch.object(controller, "choose") as choose:
                    with self.assertRaisesRegex(ProtocolError, "nonfinite"):
                        guard_deployable_state(state, self.candidate)
                        controller.choose(state)
                    choose.assert_not_called()

    def test_all_controller_branches_rows_and_terminal_failures(self):
        probes = synthetic_controller_probes(self.protocol)
        self.assertEqual(len(probes), 3 * 7 * 3)
        self.assertEqual({row["row"] for row in probes}, {2, 3, 4})
        for row in probes:
            if row["scenario"] == "dual_bound_pass":
                self.assertEqual(row["reason"], "dual_bound_stop")
                self.assertFalse(row["terminal_failure"])
            elif row["stage"] == 2:
                self.assertTrue(row["terminal_failure"])
                self.assertEqual(row["action"], "STOP")
                self.assertIsNone(row["next_budget"])
                self.assertNotEqual(row["reason"], "dual_bound_stop")
            else:
                self.assertEqual(row["next_budget"], [6, 12][row["stage"]])
        self.assertEqual(self.protocol.raw["candidates"]["fixed_reference_budgets"], [3, 6, 12])
        self.assertTrue(set(feature_names_for_row(2)) < set(feature_names_for_row(3)) < set(feature_names_for_row(4)))

    def test_internal_and_formal_alpha_budgets_and_claims_are_separate(self):
        raw = self.protocol.raw
        internal = raw["calibration"]
        cert = raw["certification"]
        self.assertEqual(internal["internal_family_size"], 36)
        self.assertAlmostEqual(36 * (internal["internal_alpha"] / 36), 0.2)
        self.assertFalse(internal["certification_error_budget_shared"])
        self.assertEqual(cert["family_wise_alpha"], 0.05)
        self.assertEqual(len(cert["endpoints"]), 3)
        self.assertFalse(cert["performed_in_readiness"])
        self.assertIn("joint_coverage_unproved", internal["interpretation"])
        self.assertFalse(raw["reporting"]["stages_independent"])
        self.assertIn("fresh_independent_roles", cert["no_retuning"])

    def test_existing_output_not_enumerated_or_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            marker = output / "sentinel"
            marker.write_text("unchanged")
            with mock.patch.object(Path, "iterdir", side_effect=AssertionError("enumerated old output")):
                with self.assertRaises(FileExistsError):
                    run_readiness(output, PROTOCOL_FINGERPRINT, REQUEST)
            self.assertEqual(marker.read_text(), "unchanged")

    def test_symlink_output_is_rejected_without_writing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            target = base / "target"
            target.mkdir()
            link = base / "link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ProtocolError, "symlink"):
                run_readiness(link / "run", PROTOCOL_FINGERPRINT, REQUEST)
            self.assertFalse((target / "run").exists())

    def test_offline_two_runs_are_byte_identical_and_all_artifacts_rehash(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            with mock.patch.object(socket, "socket", side_effect=AssertionError("network forbidden")):
                with mock.patch.object(socket, "create_connection", side_effect=AssertionError("network forbidden")):
                    first = run_readiness(base / "a", PROTOCOL_FINGERPRINT, REQUEST)
                    second = run_readiness(base / "b", PROTOCOL_FINGERPRINT, REQUEST)
            self.assertEqual(first, second)
            self.assertEqual(set(first), set(ARTIFACT_NAMES))
            for name in ARTIFACT_NAMES:
                one = (base / "a" / name).read_bytes()
                self.assertEqual(one, (base / "b" / name).read_bytes())
                value = json.loads(one)
                fingerprint = value.pop("fingerprint")
                self.assertEqual(fingerprint, digest(value))
            manifest = json.loads((base / "a" / "manifest.json").read_bytes())
            for name, fingerprint in manifest["artifact_file_sha256"].items():
                self.assertEqual(fingerprint, hashlib.sha256((base / "a" / name).read_bytes()).hexdigest())
            self.assertEqual(manifest["real_data_gate"], "closed")
            denial = json.loads((base / "a" / "denial_report.json").read_bytes())
            for claim in ("selection", "certification", "latency", "evidence_guarantee", "production"):
                self.assertIn(claim, denial["claims_denied"])
            self.assertFalse(denial["step5_started"])


if __name__ == "__main__":
    unittest.main()
