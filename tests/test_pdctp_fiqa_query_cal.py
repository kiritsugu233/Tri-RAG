import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tri_rag_harness.pdctp_calibration import (
    BudgetResidualRecord,
    TriBudgetResidualCalibrator,
)
from tri_rag_harness.pdctp_features import (
    PilotDistanceFeatureSpec,
    PilotFeatureVector,
)
from tri_rag_harness.pdctp_fiqa_query_cal import (
    PDCTPQueryCalConfig,
    PDCTPQueryCalError,
    build_query_cal_records,
    fit_query_cal_candidates,
    load_pdctp_query_cal_config,
    reconstruct_residual_candidate,
    validate_query_cal_documents,
)
from tri_rag_harness.pdctp_protocol import (
    FIVE_ROLES,
    FiveRoleAssignments,
    FiveRoleProtocolGuard,
)
from tri_rag_harness.utils import fingerprint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "pdctp_fiqa_query_cal_v1.json"
ACCEPTED_AUDIT = (
    ROOT
    / "artifacts"
    / "pdctp_fiqa_query_cal_v1"
    / "query_cal_audit.json"
)


def _fingerprinted(value):
    result = copy.deepcopy(value)
    result["fingerprint"] = fingerprint(result)
    return result


def _synthetic_protocol():
    return {
        "embedding": {"model": {"name": "fixture-e5"}},
        "retrieval": {
            "corpus_size": 64,
            "embedding_dimension": 12,
            "query_batch_size": 4,
            "k_gt": 4,
            "k_ctx": 2,
            "m_pilot": 16,
            "s_lid": 8,
            "min_lid_neighbors": 6,
            "m_grid": [16, 24, 32, 48, 64],
            "max_rank_samples": 32,
            "projection": {
                "family": "dense_gaussian",
                "variance": "1/m_prime",
                "seed": 83047,
                "m_prime": 6,
                "post_projection_normalize": False,
            },
        },
        "features": {
            "schema": "pilot_distance_features_v1",
            "lid_boundary": 8,
            "minimum_count": 8,
            "gap_quantiles": [0.25, 0.5, 0.75],
            "epsilon": 0.0,
            "duplicate_tolerance": 1e-12,
            "invalid_fill": 0.0,
            "output_decimals": 10,
        },
        "candidate_suite": {
            "raw_tri_threshold_grid": [0.8, 0.9],
            "lid_regularization_grid": [0.0, 0.1],
            "lid_output_domain": [1.0, 100.0],
            "lid_fallback": 100.0,
            "residual_training_levels": [0.75, 1.0],
            "residual_quantiles": [0.5],
            "residual_regularization_grid": [0.0, 0.1],
            "safety_offsets": [0.0, 0.1],
            "expected_full_pdctp_tuples": 32,
        },
    }


def _normalized_random(seed, rows, dimension):
    values = np.random.default_rng(seed).normal(size=(rows, dimension))
    return values / np.linalg.norm(values, axis=1, keepdims=True)


class PDCTPFiQAQueryCalTests(unittest.TestCase):
    def test_checked_in_query_cal_audit_is_self_consistent_and_fit_only(self):
        audit = json.loads(ACCEPTED_AUDIT.read_text())
        body = dict(audit)
        claimed = body.pop("fingerprint")
        self.assertEqual(fingerprint(body), claimed)
        self.assertEqual(
            claimed,
            "e3cd09d125a868b685df02b23f0706926fa5786752d863f17bf13d6293de7884",
        )
        self.assertEqual(
            audit["decision"],
            "ACCEPT_QUERY_CAL_FITS_READY_TO_IMPLEMENT_QUERY_TUNE",
        )
        self.assertEqual(audit["query_cal_records"], 1966)
        self.assertEqual(audit["candidate_audit"]["base_models"], 675)
        self.assertEqual(
            audit["candidate_audit"]["full_pdctp_operating_points"], 1620
        )
        self.assertEqual(
            audit["candidate_audit"]["residual_only_operating_points"], 405
        )
        self.assertTrue(audit["checks"]["candidate_refit_from_records_exact"])
        self.assertTrue(
            audit["portability_audit"][
                "candidate_bundles_portable_value_identical"
            ]
        )
        self.assertFalse(
            audit["portability_audit"]["query_records_byte_identical"]
        )
        for role in ("query_tune", "query_cert", "query_latency", "query_test"):
            self.assertFalse(audit["checks"][f"{role}_accessed"])
        self.assertFalse(audit["checks"]["qrels_or_relevance_accessed"])
        self.assertTrue(audit["checks"]["no_policy_selected"])
        for identity in audit["file_identities"].values():
            self.assertEqual(len(identity), 64)
            self.assertTrue(all(character in "0123456789abcdef" for character in identity))

    def test_checked_config_binds_accepted_embedding_gate(self):
        config = load_pdctp_query_cal_config(CONFIG)
        self.assertEqual(
            config.config_fingerprint,
            "7ff0bdf656ebc22026702622e933975ffe56b3814bb384b1db99effde51df36b",
        )
        self.assertEqual(config.query_batch_size, 32)
        self.assertEqual(
            config.raw["bindings"]["embedding_audit_fingerprint"],
            "54af315d5b94b43a81be71ea29ab860635f0748a97108e0cda120a510947dd71",
        )
        self.assertFalse(config.raw["access"]["qrels_or_relevance_allowed"])
        self.assertFalse(config.raw["access"]["selection_allowed"])

    def test_compact_quantile_solver_round_trip_and_legacy_equivalence(self):
        spec = PilotDistanceFeatureSpec(lid_boundary=5, minimum_count=5)
        feature = PilotFeatureVector(
            "pilot_distance_features_v1",
            spec.fingerprint,
            spec.feature_names,
            tuple(float(index + 1) for index in range(len(spec.feature_names))),
            True,
            None,
        )
        required = [16, 16, 32, 32, 64]
        rows = [
            BudgetResidualRecord(
                query_id=f"q{index}",
                role="query_cal",
                features=feature,
                raw_budget=32,
                required_budget=value,
                training_level=1.0,
            )
            for index, value in enumerate(required)
        ]
        kwargs = {
            "quantile": 0.5,
            "regularization": 0.0,
            "safety_offset": 0.0,
            "grid": [16, 32, 64],
            "minimum_budget": 16,
            "fallback_budget": 64,
            "raw_policy_fingerprint": "raw-fixture",
        }
        legacy = TriBudgetResidualCalibrator.fit(rows, **kwargs)
        compact = TriBudgetResidualCalibrator.fit_compact(rows, **kwargs)
        self.assertEqual(compact.solver, TriBudgetResidualCalibrator.COMPACT_SOLVER)
        self.assertAlmostEqual(compact.objective_value, legacy.objective_value, places=5)
        self.assertEqual(
            TriBudgetResidualCalibrator.from_serialized(compact.serialize()).serialize(),
            compact.serialize(),
        )
        shifted = compact.with_operating_point(safety_offset=math.log(1.1))
        self.assertEqual(shifted.intercept, compact.intercept)
        self.assertNotEqual(shifted.fingerprint, compact.fingerprint)

    def test_exact_query_cal_records_are_deterministic_and_single_scan(self):
        protocol = _synthetic_protocol()
        corpus = _normalized_random(11, 64, 12)
        queries = _normalized_random(12, 12, 12)
        corpus_ids = [f"doc-{index:03d}" for index in range(64)]
        query_ids = [f"query-cal-{index:03d}" for index in range(12)]
        first, projection_one = build_query_cal_records(
            protocol,
            query_ids,
            corpus_ids,
            corpus,
            queries,
            batch_size=4,
            record_distance_decimals=10,
        )
        second, projection_two = build_query_cal_records(
            protocol,
            query_ids,
            corpus_ids,
            corpus,
            queries,
            batch_size=4,
            record_distance_decimals=10,
        )
        public_first = [{k: v for k, v in row.items() if not k.startswith("_")} for row in first]
        public_second = [{k: v for k, v in row.items() if not k.startswith("_")} for row in second]
        self.assertEqual(public_first, public_second)
        self.assertEqual(projection_one, projection_two)
        self.assertFalse(projection_one["post_projection_normalized"])
        for row in first:
            self.assertEqual(row["role"], "query_cal")
            self.assertFalse(row["supervision"]["qrels_or_relevance"])
            self.assertEqual(row["work"]["projected_scan_count"], 1)
            retention = list(row["retention_by_budget"].values())
            self.assertEqual(retention, sorted(retention))
            self.assertEqual(retention[-1], 1.0)

    def test_all_candidates_fit_without_selection_and_reconstruct(self):
        protocol = _synthetic_protocol()
        corpus = _normalized_random(21, 64, 12)
        queries = _normalized_random(22, 12, 12)
        records, _ = build_query_cal_records(
            protocol,
            [f"query-cal-{index:03d}" for index in range(12)],
            [f"doc-{index:03d}" for index in range(64)],
            corpus,
            queries,
            batch_size=4,
            record_distance_decimals=10,
        )
        lid_bundle, residual_bundle = fit_query_cal_candidates(protocol, records)
        self.assertEqual(lid_bundle["candidate_count"], 2)
        self.assertFalse(lid_bundle["selection_performed"])
        self.assertEqual(residual_bundle["counts"]["full_pdctp_operating_points"], 32)
        self.assertEqual(residual_bundle["counts"]["residual_only_operating_points"], 16)
        self.assertFalse(residual_bundle["selection_performed"])
        for section in ("full_operating_points", "residual_only_operating_points"):
            point = residual_bundle[section][-1]
            restored = reconstruct_residual_candidate(
                residual_bundle, point["fingerprint"]
            )
            self.assertEqual(
                restored.fingerprint, point["effective_calibrator_fingerprint"]
            )
        tampered = copy.deepcopy(residual_bundle)
        tampered["shared_fit"]["ordered_ids"][0] = "wrong-role-id"
        with self.assertRaisesRegex(PDCTPQueryCalError, "bundle fingerprint"):
            reconstruct_residual_candidate(
                tampered, tampered["full_operating_points"][0]["fingerprint"]
            )

    def test_upstream_tamper_is_rejected_while_all_roles_are_closed(self):
        ids_by_role = {
            role: (f"{role}-0", f"{role}-1") for role in FIVE_ROLES
        }
        all_ids = [query_id for role in FIVE_ROLES for query_id in ids_by_role[role]]
        assignments = FiveRoleAssignments(
            ids_by_role,
            {query_id: query_id for query_id in all_ids},
        )
        state = FiveRoleProtocolGuard(assignments, "frozen-config").serialize()
        assignment_body = assignments.serialize()
        roles = _fingerprinted(
            {
                "roles": assignment_body["roles"],
                "all_roles_initially_closed": True,
                "authorizes_outcome_access": False,
            }
        )
        protocol = _fingerprinted(
            {
                "decision": "READY_FOR_DATASET_AND_EMBEDDING_AUDIT_ONLY",
                "authorizes_method_evaluation": False,
                "authorizes_protected_outcome_access": False,
                "initial_guard_state_fingerprint": state["fingerprint"],
                "config_fingerprint": "frozen-config",
                "resolved_roles": {"assignment_fingerprint": roles["fingerprint"]},
            }
        )
        dataset = _fingerprinted(
            {
                "protocol_freeze_fingerprint": protocol["fingerprint"],
                "role_assignments_fingerprint": roles["fingerprint"],
            }
        )
        arrays = {"corpus": {"sha256": "c" * 64}, "queries": {"sha256": "q" * 64}}
        embedding = _fingerprinted(
            {"dataset": {"manifest_fingerprint": dataset["fingerprint"]}, "arrays": arrays}
        )
        audit = _fingerprinted(
            {
                "decision": "READY_TO_OPEN_QUERY_CAL",
                "protocol_freeze_fingerprint": protocol["fingerprint"],
                "protocol_state_fingerprint": state["fingerprint"],
                "role_assignments_fingerprint": roles["fingerprint"],
                "dataset_manifest_fingerprint": dataset["fingerprint"],
                "embedding_config_fingerprint": "e" * 64,
                "embedding_manifest_fingerprint": embedding["fingerprint"],
                "arrays": arrays,
                "checks": {
                    "all_roles_remained_closed": True,
                    "qrels_or_relevance_opened": False,
                },
                "scope_guards": {
                    "contains_qrels_or_relevance": False,
                    "contains_retrieval_or_policy_outcomes": False,
                    "fits_or_selects_a_method": False,
                    "runs_an_llm": False,
                    "uses_an_approximate_index": False,
                },
            }
        )
        audit_sha = "a" * 64
        bindings = {
            "protocol_freeze_fingerprint": protocol["fingerprint"],
            "protocol_state_fingerprint": state["fingerprint"],
            "role_assignments_fingerprint": roles["fingerprint"],
            "dataset_manifest_fingerprint": dataset["fingerprint"],
            "embedding_config_fingerprint": "e" * 64,
            "embedding_manifest_fingerprint": embedding["fingerprint"],
            "embedding_audit_fingerprint": audit["fingerprint"],
            "embedding_audit_sha256": audit_sha,
            "query_cal_ordered_id_hash": fingerprint(list(ids_by_role["query_cal"])),
        }
        config = PDCTPQueryCalConfig(
            raw={"bindings": bindings},
            config_fingerprint="gate-config",
            run_name="fixture",
            query_batch_size=2,
            record_distance_decimals=10,
        )
        restored = validate_query_cal_documents(
            config,
            protocol,
            state,
            roles,
            audit,
            dataset,
            embedding,
            embedding_audit_sha256=audit_sha,
        )
        self.assertEqual(restored.fingerprint, assignments.fingerprint)
        self.assertFalse(state["calibration_opened"])
        tampered = copy.deepcopy(audit)
        tampered["decision"] = "OPEN_QUERY_TUNE"
        with self.assertRaisesRegex(PDCTPQueryCalError, "authorize query_cal"):
            validate_query_cal_documents(
                config,
                protocol,
                state,
                roles,
                tampered,
                dataset,
                embedding,
                embedding_audit_sha256=audit_sha,
            )


if __name__ == "__main__":
    unittest.main()
