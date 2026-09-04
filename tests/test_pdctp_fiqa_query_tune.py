import copy
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from tri_rag_harness.pdctp_fiqa_query_cal import (
    build_query_cal_records,
    fit_query_cal_candidates,
)
from tri_rag_harness.pdctp_fiqa_query_tune import (
    PDCTPQueryTuneConfig,
    PDCTPQueryTuneError,
    build_query_tune_records,
    load_pdctp_query_tune_config,
    load_query_tune_qrels,
    reconstruct_frozen_policy_suite,
    select_query_tune_policies,
    validate_query_tune_documents,
)
from tri_rag_harness.pdctp_protocol import (
    FIVE_ROLES,
    FiveRoleAssignments,
    FiveRoleProtocolGuard,
)
from tri_rag_harness.utils import fingerprint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "pdctp_fiqa_query_tune_v1.json"


def _fingerprinted(value):
    result = copy.deepcopy(value)
    result["fingerprint"] = fingerprint(result)
    return result


def _normalized_random(seed, rows, dimension):
    values = np.random.default_rng(seed).normal(size=(rows, dimension))
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _protocol():
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
            "fixed_budgets": [16, 24, 32, 48, 64],
            "methods": [
                "fixed",
                "monotone_binned",
                "raw_tri_predict",
                "lid_calibration_only",
                "budget_residual_only",
                "pdctp",
            ],
            "monotone_binned": {
                "n_bins_grid": [2, 3],
                "bin_target_grid": [0.5, 1.0],
                "fallback_budget": 64,
            },
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
        "selection": {
            "schema": "pdctp_fiqa_selection_v1",
            "role": "query_tune",
            "retention_lower_bound_target": 0.5,
            "candidate_evidence_noninferiority": 1.0,
            "final_evidence_noninferiority": 1.0,
            "tune_bound_alpha": 0.05,
            "objective": "common_coordinate_work",
            "tie_breaks": ["lower_mean_budget", "canonical_fingerprint"],
            "comparator_eligibility": "must_meet_same_retention_and_evidence_constraints",
            "shuffled_profile_scope": "query_tune_diagnostic_only",
        },
    }


def _fit_fixture():
    protocol = _protocol()
    corpus = _normalized_random(31, 64, 12)
    queries = _normalized_random(32, 20, 12)
    records, _ = build_query_cal_records(
        protocol,
        [f"query-cal-{index:03d}" for index in range(len(queries))],
        [f"doc-{index:03d}" for index in range(len(corpus))],
        corpus,
        queries,
        batch_size=4,
        record_distance_decimals=10,
    )
    lid, residual = fit_query_cal_candidates(protocol, records)
    return protocol, records, lid, residual


class PDCTPFiQAQueryTuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.cal_records, cls.lid_bundle, cls.residual_bundle = (
            _fit_fixture()
        )

    def test_checked_config_freezes_query_tune_only_and_all_family_counts(self):
        config = load_pdctp_query_tune_config(CONFIG)
        self.assertEqual(
            config.config_fingerprint,
            "06c647625bb01192b54ae0698e9e4150fe4fec0d2b4407858de74c763573d7d0",
        )
        self.assertEqual(config.raw["access"]["role"], "query_tune")
        self.assertFalse(config.raw["access"]["calibrator_fit_allowed"])
        self.assertEqual(
            config.raw["access"]["blocked_roles"],
            ["query_cert", "query_latency", "query_test"],
        )
        self.assertEqual(
            sum(config.raw["selection_contract"]["expected_candidate_counts"].values()),
            2086,
        )
        tampered = json.loads(CONFIG.read_text())
        tampered["selection_contract"]["family_selection"] = "joint_cherry_pick"
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "config.json"
            path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(PDCTPQueryTuneError, "selection contract"):
                load_pdctp_query_tune_config(path)

    def test_qrel_parser_skips_non_tune_outcomes_before_parsing(self):
        tune_id = "pdctp-beir-fiqa:query:tune-1"
        member = "fiqa/qrels/train.tsv"
        content = (
            "query-id\tcorpus-id\tscore\n"
            "cal-1\tthis-row-is-not-parsed\tnot-an-integer\n"
            "tune-1\tdoc-2\t1\n"
            "cert-1\talso-not-parsed\tnot-an-integer\n"
        ).encode()
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "fiqa.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(member, content)
            metadata = {
                "path": member,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            qrels, audit = load_query_tune_qrels(
                path, metadata, [tune_id], minimum_relevance=1
            )
            self.assertEqual(qrels[tune_id], ("pdctp-beir-fiqa:doc:doc-2",))
            self.assertEqual(audit["non_tune_rows_skipped_before_outcome_parse"], 2)
            self.assertFalse(audit["non_tune_qrel_outcomes_parsed"])
            changed = dict(metadata)
            changed["sha256"] = "0" * 64
            with self.assertRaisesRegex(PDCTPQueryTuneError, "member identity"):
                load_query_tune_qrels(path, changed, [tune_id], minimum_relevance=1)

    def test_exact_tune_records_are_deterministic_reconstructable_and_one_scan(self):
        protocol = self.protocol
        corpus = _normalized_random(41, 64, 12)
        queries = _normalized_random(42, 12, 12)
        corpus_ids = [f"pdctp-beir-fiqa:doc:{index:03d}" for index in range(64)]
        query_ids = [f"pdctp-beir-fiqa:query:{index:03d}" for index in range(12)]
        qrels = {
            query_id: (corpus_ids[(index * 3) % len(corpus_ids)],)
            for index, query_id in enumerate(query_ids)
        }
        first, projection_one = build_query_tune_records(
            protocol,
            query_ids,
            corpus_ids,
            corpus,
            queries,
            qrels,
            batch_size=4,
            record_distance_decimals=10,
        )
        second, projection_two = build_query_tune_records(
            protocol,
            query_ids,
            corpus_ids,
            corpus,
            queries,
            qrels,
            batch_size=4,
            record_distance_decimals=10,
        )
        public_first = [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in first
        ]
        public_second = [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in second
        ]
        self.assertEqual(public_first, public_second)
        self.assertEqual(projection_one, projection_two)
        self.assertFalse(projection_one["post_projection_normalized"])
        for row in first:
            self.assertNotIn("oracle_lid", row)
            self.assertEqual(row["role"], "query_tune")
            self.assertFalse(row["supervision"]["oracle_lid"])
            self.assertFalse(row["supervision"]["non_tune_qrel_outcomes"])
            self.assertEqual(row["work"]["projected_scan_count"], 1)
            self.assertEqual(list(row["retention_by_budget"].values())[-1], 1.0)
            relevant_ranks = row["projected_rank_of_positive_qrels"]
            for budget in protocol["retrieval"]["m_grid"]:
                expected = sum(rank <= budget for rank in relevant_ranks) / len(
                    relevant_ranks
                )
                self.assertEqual(
                    row["candidate_evidence_by_budget"][str(budget)], expected
                )
            self.assertEqual(
                row["final_top_k_doc_ids_by_budget"]["64"],
                row["exact_original_top_k_doc_ids"][:2],
            )

    def _perfect_tune_records(self):
        grid = self.protocol["retrieval"]["m_grid"]
        rows = []
        for index in range(64):
            source = self.cal_records[index % len(self.cal_records)]
            row = {
                "query_id": f"query-tune-{index:03d}",
                "role": "query_tune",
                "pilot": copy.deepcopy(source["pilot"]),
                "retention_by_budget": {str(value): 1.0 for value in grid},
                "candidate_evidence_by_budget": {str(value): 1.0 for value in grid},
                "final_evidence_by_budget": {str(value): 1.0 for value in grid},
                "_features_obj": source["_features_obj"],
            }
            rows.append(row)
        return rows

    def test_all_candidate_families_select_independently_and_suite_reconstructs(self):
        records = self._perfect_tune_records()
        expected = {
            "fixed": 5,
            "monotone_binned": 4,
            "raw_tri_predict": 2,
            "lid_calibration_only": 4,
            "budget_residual_only": 16,
            "pdctp": 32,
        }
        before_raw = copy.deepcopy(self.residual_bundle["raw_policies"])
        first = select_query_tune_policies(
            self.protocol,
            records,
            self.lid_bundle,
            self.residual_bundle,
            selection_float_decimals=12,
            expected_candidate_counts=expected,
        )
        second = select_query_tune_policies(
            self.protocol,
            records,
            self.lid_bundle,
            self.residual_bundle,
            selection_float_decimals=12,
            expected_candidate_counts=expected,
        )
        self.assertTrue(first["success"])
        self.assertEqual(first["candidate_outcomes"], second["candidate_outcomes"])
        np.testing.assert_array_equal(first["budget_matrix"], second["budget_matrix"])
        self.assertEqual(first["candidate_outcomes"]["counts"], expected)
        self.assertEqual(
            set(first["selection"]["selected_by_method"]), set(expected)
        )
        self.assertEqual(
            first["selection"]["family_selection_contract"]["family_selection"],
            "independent_minimum_work_eligible_candidate_per_method",
        )
        self.assertFalse(first["selection"]["calibrator_refit"])
        restored = reconstruct_frozen_policy_suite(
            first["policies"], first["component_registry"], first["policy_suite"]
        )
        self.assertEqual(
            {name: policy.serialize() for name, policy in restored.items()},
            first["policies"]["policies"],
        )
        self.assertEqual(before_raw, self.residual_bundle["raw_policies"])
        tampered = copy.deepcopy(first["component_registry"])
        tampered["components"][0]["artifact"]["fingerprint"] = "0" * 64
        with self.assertRaisesRegex(PDCTPQueryTuneError, "registry fingerprint"):
            reconstruct_frozen_policy_suite(
                first["policies"], tampered, first["policy_suite"]
            )

    def test_missing_eligible_family_is_terminal_without_policy_substitution(self):
        records = self._perfect_tune_records()
        terminal = self.protocol["retrieval"]["m_grid"][-1]
        for row in records:
            for field in (
                "retention_by_budget",
                "candidate_evidence_by_budget",
                "final_evidence_by_budget",
            ):
                row[field] = {
                    key: float(int(key) == terminal) for key in row[field]
                }
        expected = {
            "fixed": 5,
            "monotone_binned": 4,
            "raw_tri_predict": 2,
            "lid_calibration_only": 4,
            "budget_residual_only": 16,
            "pdctp": 32,
        }
        result = select_query_tune_policies(
            self.protocol,
            records,
            self.lid_bundle,
            self.residual_bundle,
            selection_float_decimals=12,
            expected_candidate_counts=expected,
        )
        self.assertFalse(result["success"])
        self.assertEqual(
            result["selection"]["decision"],
            "TERMINAL_QUERY_TUNE_FAILURE_NO_RETUNING",
        )
        self.assertIn(
            "pdctp", result["selection"]["missing_eligible_method_families"]
        )
        self.assertIsNone(result["selection"]["selected_by_method"])
        self.assertNotIn("policy_suite", result)

    def test_upstream_tamper_is_rejected_before_tune_token(self):
        ids_by_role = {
            role: (f"{role}-0", f"{role}-1") for role in FIVE_ROLES
        }
        all_ids = [value for role in FIVE_ROLES for value in ids_by_role[role]]
        assignments = FiveRoleAssignments(
            ids_by_role, {query_id: query_id for query_id in all_ids}
        )
        roles = _fingerprinted(
            {
                "roles": assignments.serialize()["roles"],
                "all_roles_initially_closed": True,
                "authorizes_outcome_access": False,
            }
        )
        source = _fingerprinted(
            {
                "decision": "GO_TO_PROTOCOL_FREEZE",
                "source": {
                    "archive": {"sha256": "s" * 64},
                    "members": {"qrels_train": {"sha256": "q" * 64}},
                }
            }
        )
        protocol = _fingerprinted(
            {
                "decision": "READY_FOR_DATASET_AND_EMBEDDING_AUDIT_ONLY",
                "config_fingerprint": "protocol-config",
                "resolved_roles": {"assignment_fingerprint": roles["fingerprint"]},
                "resolved_inputs": {
                    "source_audit": {"fingerprint": source["fingerprint"]}
                },
            }
        )
        dataset = _fingerprinted(
            {
                "protocol_freeze_fingerprint": protocol["fingerprint"],
                "role_assignments_fingerprint": roles["fingerprint"],
                "scope_guards": {
                    "contains_qrels": False,
                    "contains_relevance_values": False,
                },
            }
        )
        embedding = _fingerprinted(
            {"dataset": {"manifest_fingerprint": dataset["fingerprint"]}}
        )
        embedding_audit = _fingerprinted(
            {
                "decision": "READY_TO_OPEN_QUERY_CAL",
                "embedding_config_fingerprint": "e" * 64,
                "embedding_manifest_fingerprint": embedding["fingerprint"],
                "checks": {
                    "all_roles_remained_closed": True,
                    "qrels_or_relevance_opened": False,
                },
            }
        )
        guard = FiveRoleProtocolGuard(assignments, "protocol-config")
        guard.open_calibration(ids_by_role["query_cal"])
        guard.register_fit(
            "lid_calibrator",
            role="query_cal",
            ids=ids_by_role["query_cal"],
            artifact_fingerprint=self.lid_bundle["fingerprint"],
        )
        guard.register_fit(
            "residual_calibrator",
            role="query_cal",
            ids=ids_by_role["query_cal"],
            artifact_fingerprint=self.residual_bundle["fingerprint"],
        )
        cal_state = guard.serialize()
        cal_access = _fingerprinted(
            {
                "role": "query_cal",
                "roles_opened": ["query_cal"],
                "qrels_or_relevance_accessed": False,
                "selection_performed": False,
            }
        )
        cal_projection = _fingerprinted({"role": "query_cal"})
        file_ids = {
            name: {"bytes": index + 1, "sha256": str(index) * 64}
            for index, name in enumerate(
                (
                    "query_cal_access.json",
                    "projection.json",
                    "query_cal_records.jsonl",
                    "lid_calibrator_candidates.json",
                    "residual_calibrator_candidates.json",
                    "protocol_state_after_query_cal.json",
                    "manifest.json",
                    "report.md",
                ),
                start=1,
            )
        }
        cal_manifest = _fingerprinted(
            {
                "decision": "QUERY_CAL_FITS_FROZEN_READY_FOR_QUERY_TUNE",
                "protocol_state_fingerprint": cal_state["fingerprint"],
                "query_cal_access_fingerprint": cal_access["fingerprint"],
                "projection_fingerprint": cal_projection["fingerprint"],
                "lid_candidate_bundle_fingerprint": self.lid_bundle["fingerprint"],
                "residual_candidate_bundle_fingerprint": self.residual_bundle[
                    "fingerprint"
                ],
                "checks": {
                    "only_query_cal_opened": True,
                    "query_tune_accessed": False,
                    "selection_performed": False,
                },
                "artifacts": {},
            }
        )
        cal_audit = _fingerprinted(
            {
                "decision": "ACCEPT_QUERY_CAL_FITS_READY_TO_IMPLEMENT_QUERY_TUNE",
                "checks": {
                    "candidate_refit_from_records_exact": True,
                    "only_query_cal_opened": True,
                    "query_tune_accessed": False,
                    "query_cert_accessed": False,
                    "query_latency_accessed": False,
                    "query_test_accessed": False,
                    "qrels_or_relevance_accessed": False,
                    "no_policy_selected": True,
                },
                "file_identities": {
                    name: identity["sha256"] for name, identity in file_ids.items()
                },
            }
        )
        bindings = {
            "protocol_freeze_fingerprint": protocol["fingerprint"],
            "role_assignments_fingerprint": roles["fingerprint"],
            "source_audit_fingerprint": source["fingerprint"],
            "source_audit_sha256": "a" * 64,
            "source_archive_sha256": "s" * 64,
            "qrels_train_member_sha256": "q" * 64,
            "dataset_manifest_fingerprint": dataset["fingerprint"],
            "embedding_config_fingerprint": "e" * 64,
            "embedding_manifest_fingerprint": embedding["fingerprint"],
            "embedding_audit_fingerprint": embedding_audit["fingerprint"],
            "query_cal_audit_fingerprint": cal_audit["fingerprint"],
            "query_cal_audit_sha256": "b" * 64,
            "query_cal_manifest_fingerprint": cal_manifest["fingerprint"],
            "query_cal_protocol_state_fingerprint": cal_state["fingerprint"],
            "lid_candidate_bundle_fingerprint": self.lid_bundle["fingerprint"],
            "residual_candidate_bundle_fingerprint": self.residual_bundle[
                "fingerprint"
            ],
            "query_tune_ordered_id_hash": fingerprint(
                list(ids_by_role["query_tune"])
            ),
            "power_plan_fingerprint": "p" * 64,
        }
        config = PDCTPQueryTuneConfig(
            raw={"bindings": bindings},
            config_fingerprint="tune-config",
            run_name="fixture",
            query_batch_size=2,
            record_distance_decimals=10,
            selection_float_decimals=12,
        )
        restored, restored_guard = validate_query_tune_documents(
            config,
            protocol,
            roles,
            source,
            embedding_audit,
            dataset,
            embedding,
            cal_audit,
            cal_manifest,
            cal_state,
            cal_access,
            cal_projection,
            self.lid_bundle,
            self.residual_bundle,
            source_audit_sha256="a" * 64,
            query_cal_audit_sha256="b" * 64,
            query_cal_file_identities=file_ids,
        )
        self.assertEqual(restored.ids_by_role["query_tune"], ids_by_role["query_tune"])
        self.assertIsNone(restored_guard.serialize()["selection_fingerprint"])
        tampered = copy.deepcopy(cal_audit)
        tampered["decision"] = "ACCEPT_AND_OPEN_CERT"
        tampered = _fingerprinted(
            {key: value for key, value in tampered.items() if key != "fingerprint"}
        )
        changed_bindings = dict(bindings)
        changed_bindings["query_cal_audit_fingerprint"] = tampered["fingerprint"]
        changed_config = PDCTPQueryTuneConfig(
            raw={"bindings": changed_bindings},
            config_fingerprint="tune-config",
            run_name="fixture",
            query_batch_size=2,
            record_distance_decimals=10,
            selection_float_decimals=12,
        )
        with self.assertRaisesRegex(PDCTPQueryTuneError, "does not authorize tune"):
            validate_query_tune_documents(
                changed_config,
                protocol,
                roles,
                source,
                embedding_audit,
                dataset,
                embedding,
                tampered,
                cal_manifest,
                cal_state,
                cal_access,
                cal_projection,
                self.lid_bundle,
                self.residual_bundle,
                source_audit_sha256="a" * 64,
                query_cal_audit_sha256="b" * 64,
                query_cal_file_identities=file_ids,
            )


if __name__ == "__main__":
    unittest.main()
