import copy
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from tri_rag_harness.pdctp_fiqa_query_cert import (
    PDCTPQueryCertError,
    _validate_query_tune_file_inventory,
    build_query_cert_records,
    load_pdctp_query_cert_config,
    load_query_cert_qrels,
    make_query_certification,
)
from tri_rag_harness.pdctp_policies import FixedPDCTPPolicy
from tri_rag_harness.pdctp_policies import PDCTPDecisionInput
from tri_rag_harness.pdctp_statistics import validate_paired_bound
from tri_rag_harness.pdctp_protocol import (
    FIVE_ROLES,
    FiveRoleAssignments,
    FiveRoleProtocolGuard,
    LeakageError,
)
from tri_rag_harness.utils import fingerprint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "pdctp_fiqa_query_cert_v1.json"
ACCEPTED_TUNE_AUDIT = (
    ROOT / "artifacts" / "pdctp_fiqa_query_tune_v1" / "query_tune_audit.json"
)


def _normalized_random(seed, rows, dimension):
    values = np.random.default_rng(seed).normal(size=(rows, dimension))
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _hypothesis_rows():
    return [
        {
            "name": "pdctp_absolute_retention",
            "metric": "embedding_retention",
            "comparison": "zero_anchor",
            "side": "lower",
            "margin": 0.0,
            "difference_bounds": [0.0, 1.0],
            "desired_radius": 0.8,
        },
        {
            "name": "candidate_evidence_noninferiority_fixed",
            "metric": "candidate_evidence_recall",
            "comparison": "fixed_reference",
            "side": "lower",
            "margin": -1.0,
            "difference_bounds": [-1.0, 1.0],
            "desired_radius": 0.8,
        },
        {
            "name": "final_evidence_noninferiority_fixed",
            "metric": "final_evidence_recall",
            "comparison": "fixed_reference",
            "side": "lower",
            "margin": -1.0,
            "difference_bounds": [-1.0, 1.0],
            "desired_radius": 0.8,
        },
        {
            "name": "normalized_budget_superiority_fixed",
            "metric": "normalized_candidate_budget",
            "comparison": "fixed_reference",
            "side": "upper",
            "margin": 1.1,
            "difference_bounds": [-1.0, 1.0],
            "desired_radius": 0.8,
        },
        {
            "name": "normalized_budget_superiority_monotone",
            "metric": "normalized_candidate_budget",
            "comparison": "monotone_binned",
            "side": "upper",
            "margin": 1.1,
            "difference_bounds": [-1.0, 1.0],
            "desired_radius": 0.8,
        },
        {
            "name": "normalized_budget_superiority_raw_tri",
            "metric": "normalized_candidate_budget",
            "comparison": "raw_tri_predict",
            "side": "upper",
            "margin": 1.1,
            "difference_bounds": [-1.0, 1.0],
            "desired_radius": 0.8,
        },
    ]


def _protocol(query_count=12):
    hypotheses = _hypothesis_rows()
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
            "lid_output_domain": [1.0, 100.0],
            "lid_fallback": 100.0,
        },
        "certification": {
            "schema": "pdctp_fiqa_certification_v1",
            "role": "query_cert",
            "required_query_count": query_count,
            "family_wise_method": "bonferroni",
            "family_wise_alpha": 0.05,
            "hypotheses": hypotheses,
            "failure_behavior": "terminal_no_retuning_no_budget_expansion",
        },
    }


def _policies():
    grid = [16, 24, 32, 48, 64]
    return {
        "fixed": FixedPDCTPPolicy(32, grid, 16),
        "monotone": FixedPDCTPPolicy(24, grid, 16),
        "raw_tri": FixedPDCTPPolicy(32, grid, 16),
        "lid_only": FixedPDCTPPolicy(24, grid, 16),
        "residual_only": FixedPDCTPPolicy(64, grid, 16),
        "full_pdctp": FixedPDCTPPolicy(16, grid, 16),
    }


def _hypotheses(protocol):
    names = [row["name"] for row in protocol["certification"]["hypotheses"]]
    body = {
        "name": "fixture_hypotheses",
        "family_wise_method": "bonferroni",
        "family_wise_alpha": 0.05,
        "alpha_allocation": {name: 0.05 / len(names) for name in names},
        "hypotheses": protocol["certification"]["hypotheses"],
        "required_query_count": protocol["certification"]["required_query_count"],
        "power_plan_fingerprint": "p" * 64,
        "frozen_before_query_cert": True,
    }
    body["fingerprint"] = fingerprint(body)
    return body


class PDCTPFiQAQueryCertTests(unittest.TestCase):
    def test_checked_config_freezes_one_time_certification_only(self):
        config = load_pdctp_query_cert_config(CONFIG)
        self.assertEqual(
            config.config_fingerprint,
            "c6357a748f7f3262f481c1f597d3f25acb76892a9dbfc621735effe6b0bd8143",
        )
        self.assertEqual(config.raw["access"]["role"], "query_cert")
        self.assertFalse(config.raw["access"]["calibrator_fit_allowed"])
        self.assertFalse(config.raw["access"]["selection_allowed"])
        self.assertEqual(
            config.raw["access"]["blocked_roles"], ["query_latency", "query_test"]
        )
        self.assertEqual(
            config.raw["certification_contract"]["failure_behavior"],
            "terminal_no_retuning_no_budget_expansion",
        )
        tampered = json.loads(CONFIG.read_text())
        tampered["access"]["selection_allowed"] = True
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "config.json"
            path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(PDCTPQueryCertError, "access scope"):
                load_pdctp_query_cert_config(path)

    def test_accepted_tune_audit_is_still_pre_cert_and_terminally_frozen(self):
        audit = json.loads(ACCEPTED_TUNE_AUDIT.read_text())
        body = dict(audit)
        claimed = body.pop("fingerprint")
        self.assertEqual(fingerprint(body), claimed)
        self.assertEqual(
            claimed,
            "f9a375115ed2c7f461bbd16add72735a6b9da44c309dd91f4782ecb01f0e5924",
        )
        self.assertEqual(
            audit["decision"],
            "ACCEPT_QUERY_TUNE_SELECTION_READY_TO_IMPLEMENT_QUERY_CERT",
        )
        self.assertFalse(audit["checks"]["query_cert_accessed"])
        self.assertFalse(audit["checks"]["calibrator_refit"])
        self.assertEqual(len(audit["file_identities"]), 16)
        self.assertEqual(
            audit["candidate_audit"]["hypotheses_fingerprint"],
            "d06e5a04ced82456f12e5f4bcc1a92cc1f599c5e6a1f1e61eab04acdd9056700",
        )
        identities = {
            name: {"bytes": index, "sha256": sha256}
            for index, (name, sha256) in enumerate(
                audit["file_identities"].items(), start=1
            )
        }
        _validate_query_tune_file_inventory(audit, identities)
        tampered = copy.deepcopy(identities)
        tampered["selection.json"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(PDCTPQueryCertError, "selection.json"):
            _validate_query_tune_file_inventory(audit, tampered)

    def test_qrel_parser_skips_every_non_cert_outcome_before_parsing(self):
        cert_id = "pdctp-beir-fiqa:query:cert-1"
        member = "fiqa/qrels/train.tsv"
        content = (
            "query-id\tcorpus-id\tscore\n"
            "cal-1\tthis-is-not-parsed\tnot-an-integer\n"
            "cert-1\tdoc-2\t1\n"
            "test-1\talso-not-parsed\tnot-an-integer\n"
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
            qrels, audit = load_query_cert_qrels(
                path, metadata, [cert_id], minimum_relevance=1
            )
            self.assertEqual(qrels[cert_id], ("pdctp-beir-fiqa:doc:doc-2",))
            self.assertEqual(audit["non_cert_rows_skipped_before_outcome_parse"], 2)
            self.assertFalse(audit["non_cert_qrel_outcomes_parsed"])
            changed = dict(metadata)
            changed["sha256"] = "0" * 64
            with self.assertRaisesRegex(PDCTPQueryCertError, "member identity"):
                load_query_cert_qrels(path, changed, [cert_id], minimum_relevance=1)

    def test_certification_is_one_time_terminal_and_keeps_later_roles_closed(self):
        ids_by_role = {
            role: tuple(f"{role}-{index}" for index in range(2)) for role in FIVE_ROLES
        }
        all_ids = [query_id for role in FIVE_ROLES for query_id in ids_by_role[role]]
        assignments = FiveRoleAssignments(
            ids_by_role, {query_id: query_id for query_id in all_ids}
        )
        guard = FiveRoleProtocolGuard(assignments, "frozen-protocol")
        guard.open_calibration(ids_by_role["query_cal"])
        for name in ("lid_calibrator", "residual_calibrator"):
            guard.register_fit(
                name,
                role="query_cal",
                ids=ids_by_role["query_cal"],
                artifact_fingerprint=f"{name}-fingerprint",
            )
        tune_token = guard.open_tune_selection(ids_by_role["query_tune"])
        guard.freeze_selection(tune_token, "frozen-selection")
        guard.freeze_hypotheses("frozen-hypotheses")
        cert_token = guard.open_certification(ids_by_role["query_cert"])
        self.assertEqual(cert_token.purpose, "scientific_certification")
        self.assertTrue(cert_token.labels_allowed)
        guard.close_certification(cert_token, "terminal-result")
        state = guard.serialize()
        self.assertEqual(state["certification_result_fingerprint"], "terminal-result")
        self.assertFalse(state["latency_opened"])
        self.assertFalse(state["test_opened"])
        with self.assertRaisesRegex(LeakageError, "only once"):
            guard.open_certification(ids_by_role["query_cert"])
        with self.assertRaisesRegex(LeakageError, "cannot refit"):
            guard.register_fit(
                "lid_calibrator",
                role="query_cal",
                ids=ids_by_role["query_cal"],
                artifact_fingerprint="replacement",
            )
        with self.assertRaisesRegex(LeakageError, "query_test requires"):
            guard.open_test(ids_by_role["query_test"])

    def test_exact_cert_records_are_deterministic_label_safe_and_one_scan(self):
        protocol = _protocol()
        corpus = _normalized_random(71, 64, 12)
        queries = _normalized_random(72, 12, 12)
        corpus_ids = [f"pdctp-beir-fiqa:doc:{index:03d}" for index in range(64)]
        query_ids = [f"pdctp-beir-fiqa:query:{index:03d}" for index in range(12)]
        qrels = {
            query_id: (corpus_ids[(index * 5) % len(corpus_ids)],)
            for index, query_id in enumerate(query_ids)
        }
        first, projection_one = build_query_cert_records(
            protocol,
            query_ids,
            corpus_ids,
            corpus,
            queries,
            qrels,
            _policies(),
            batch_size=4,
            record_distance_decimals=10,
        )
        second, projection_two = build_query_cert_records(
            protocol,
            query_ids,
            corpus_ids,
            corpus,
            queries,
            qrels,
            _policies(),
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
        policies = _policies()
        for row in first:
            self.assertEqual(row["role"], "query_cert")
            self.assertTrue(row["policy_decisions_completed_before_supervision"])
            self.assertFalse(row["supervision"]["oracle_lid_for_policy_decision"])
            self.assertFalse(row["supervision"]["exact_top_k_for_policy_decision"])
            self.assertFalse(row["supervision"]["qrels_for_policy_decision"])
            self.assertFalse(row["supervision"]["non_cert_qrel_outcomes"])
            self.assertEqual(row["shared_work"]["projected_scan_count"], 1)
            self.assertEqual(set(row["methods"]), set(_hypothesis_method_names()))
            observation = PDCTPDecisionInput(
                row["_features_obj"], row["pilot"]["lid"], row["pilot"]["lid_valid"]
            )
            for family, policy_key in {
                "fixed": "fixed",
                "monotone_binned": "monotone",
                "raw_tri_predict": "raw_tri",
                "lid_calibration_only": "lid_only",
                "budget_residual_only": "residual_only",
                "pdctp": "full_pdctp",
            }.items():
                self.assertEqual(
                    row["methods"][family]["decision"],
                    vars(policies[policy_key].choose(observation)),
                )
            ranks = row["projected_rank_of_positive_qrels"]
            for method in row["methods"].values():
                expected = sum(rank <= method["chosen_m"] for rank in ranks) / len(
                    ranks
                )
                self.assertEqual(method["candidate_evidence_recall"], expected)
            self.assertEqual(
                row["methods"]["budget_residual_only"]["final_top_k_doc_ids"],
                row["exact_original_top_k_doc_ids"][:2],
            )

    def test_all_six_paired_bounds_reconstruct_and_failure_is_terminal(self):
        protocol = _protocol()
        corpus = _normalized_random(81, 64, 12)
        queries = _normalized_random(82, 12, 12)
        corpus_ids = [f"pdctp-beir-fiqa:doc:{index:03d}" for index in range(64)]
        query_ids = [f"pdctp-beir-fiqa:query:{index:03d}" for index in range(12)]
        qrels = {
            query_id: (corpus_ids[(index * 7) % len(corpus_ids)],)
            for index, query_id in enumerate(query_ids)
        }
        records, _ = build_query_cert_records(
            protocol,
            query_ids,
            corpus_ids,
            corpus,
            queries,
            qrels,
            _policies(),
            batch_size=4,
            record_distance_decimals=10,
        )
        hypotheses = _hypotheses(protocol)
        passed = make_query_certification(
            protocol, records, _policies(), hypotheses, "suite-fingerprint"
        )
        self.assertTrue(passed["all_passed"])
        self.assertTrue(passed["terminal"])
        self.assertFalse(passed["calibrator_refit"])
        self.assertFalse(passed["selection_performed"])
        self.assertEqual(len(passed["bounds"]), 6)
        for bound in passed["bounds"].values():
            validate_paired_bound(bound)
            self.assertEqual(len(bound["per_query"]), len(query_ids))

        failed_protocol = copy.deepcopy(protocol)
        failed_protocol["certification"]["hypotheses"][0]["margin"] = 1.0
        failed_hypotheses = _hypotheses(failed_protocol)
        failed = make_query_certification(
            failed_protocol,
            records,
            _policies(),
            failed_hypotheses,
            "suite-fingerprint",
        )
        self.assertFalse(failed["all_passed"])
        self.assertEqual(
            failed["decision"],
            "TERMINAL_CERTIFICATION_FAIL_NO_RETUNING_READY_FOR_LATENCY_IMPLEMENTATION",
        )
        self.assertEqual(
            failed["failure_behavior"],
            "terminal_no_retuning_no_budget_expansion",
        )


def _hypothesis_method_names():
    return (
        "fixed",
        "monotone_binned",
        "raw_tri_predict",
        "lid_calibration_only",
        "budget_residual_only",
        "pdctp",
    )


if __name__ == "__main__":
    unittest.main()
