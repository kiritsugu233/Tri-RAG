import copy
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from tri_rag_harness.pdctp_config import (
    PDCTPConfigError,
    load_pdctp_foundation_config,
)
from tri_rag_harness.pdctp_protocol import (
    FIVE_ROLES,
    FiveRoleAssignments,
    FiveRoleProtocolGuard,
    LeakageError,
)
from tri_rag_harness.pdctp_statistics import (
    PairedBoundError,
    bonferroni_allocation,
    make_paired_bound,
    make_power_plan,
    validate_paired_bound,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "pdctp_network_free_foundation_v1.json"


class PDCTPProtocolStatisticsTests(unittest.TestCase):
    def _assignments(self):
        ids_by_role = {
            role: (f"{role}-0", f"{role}-1") for role in FIVE_ROLES
        }
        groups = {
            query_id: f"normalized::{query_id}"
            for ids in ids_by_role.values()
            for query_id in ids
        }
        return FiveRoleAssignments(ids_by_role, groups)

    def test_checked_in_versioned_config_has_all_foundation_schemas(self):
        config = load_pdctp_foundation_config(CONFIG)
        self.assertEqual(config.feature_spec.schema, "pilot_distance_features_v1")
        self.assertEqual(set(config.synthetic.role_counts), set(FIVE_ROLES))
        self.assertEqual(config.retrieval.m_grid[-1], config.synthetic.corpus_size)
        self.assertEqual(config.latency.backends[0], "faiss_cpu_exact")
        self.assertEqual(config.latency.gpu_device_count, 1)
        self.assertEqual(config.latency.boundary_tie_overfetch, 64)
        self.assertEqual(config.latency.required_packages["faiss"], "1.10.0")
        self.assertEqual(config.calibration.raw_tri_threshold_grid, (0.8, 0.85))
        self.assertGreater(len(config.calibration.lid_regularization_grid), 1)
        self.assertGreater(len(config.certification.hypotheses), 1)

    def test_config_rejects_role_selection_and_latency_mutations(self):
        original = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = []
        role_mutation = copy.deepcopy(original)
        del role_mutation["roles"]["counts"]["query_cal"]
        mutations.append((role_mutation, "roles.counts"))
        selection_mutation = copy.deepcopy(original)
        selection_mutation["selection"]["objective"] = "mean_budget"
        mutations.append((selection_mutation, "selection"))
        latency_mutation = copy.deepcopy(original)
        latency_mutation["latency"]["backends"] = ["approximate"]
        mutations.append((latency_mutation, "latency"))
        with tempfile.TemporaryDirectory() as directory_name:
            for index, (raw, message) in enumerate(mutations):
                path = Path(directory_name) / f"mutated-{index}.json"
                path.write_text(json.dumps(raw), encoding="utf-8")
                with self.subTest(index=index):
                    with self.assertRaisesRegex(PDCTPConfigError, message):
                        load_pdctp_foundation_config(path)

    def test_duplicate_text_and_identity_crossing_are_rejected(self):
        assignments = self._assignments()
        crossed_ids = dict(assignments.ids_by_role)
        crossed_ids["query_tune"] = (
            crossed_ids["query_cal"][0],
            crossed_ids["query_tune"][1],
        )
        with self.assertRaisesRegex(LeakageError, "IDs cross"):
            FiveRoleAssignments(crossed_ids, assignments.normalized_text_group_by_id)

        crossed_groups = dict(assignments.normalized_text_group_by_id)
        crossed_groups["query_tune-0"] = crossed_groups["query_cal-0"]
        with self.assertRaisesRegex(LeakageError, "duplicate"):
            FiveRoleAssignments(assignments.ids_by_role, crossed_groups)

    def test_five_role_state_machine_refuses_early_or_labeled_access(self):
        assignments = self._assignments()
        guard = FiveRoleProtocolGuard(assignments, "config-fingerprint")
        with self.assertRaisesRegex(LeakageError, "fits"):
            guard.open_tune_selection(assignments.ids_by_role["query_tune"])
        with self.assertRaisesRegex(LeakageError, "selection"):
            guard.open_certification(assignments.ids_by_role["query_cert"])
        guard.open_calibration(assignments.ids_by_role["query_cal"])
        guard.register_fit(
            "lid_calibrator",
            role="query_cal",
            ids=assignments.ids_by_role["query_cal"],
            artifact_fingerprint="lid-fit",
        )
        guard.register_fit(
            "residual_calibrator",
            role="query_cal",
            ids=assignments.ids_by_role["query_cal"],
            artifact_fingerprint="residual-fit",
        )
        tune = guard.open_tune_selection(assignments.ids_by_role["query_tune"])
        guard.freeze_selection(tune, "selected-policy")
        with self.assertRaisesRegex(LeakageError, "refit"):
            guard.register_fit(
                "lid_calibrator",
                role="query_cal",
                ids=assignments.ids_by_role["query_cal"],
                artifact_fingerprint="changed",
            )
        guard.freeze_hypotheses("hypotheses")
        cert = guard.open_certification(assignments.ids_by_role["query_cert"])
        guard.close_certification(cert, "terminal-cert")
        with self.assertRaisesRegex(LeakageError, "label-free"):
            guard.open_latency(
                assignments.ids_by_role["query_latency"], labels_requested=True
            )
        with self.assertRaisesRegex(LeakageError, "terminal certification and latency"):
            guard.open_test(assignments.ids_by_role["query_test"])
        latency = guard.open_latency(
            assignments.ids_by_role["query_latency"], labels_requested=False
        )
        self.assertFalse(latency.labels_allowed)
        guard.close_latency(latency, "terminal-latency")
        test = guard.open_test(assignments.ids_by_role["query_test"])
        self.assertTrue(test.labels_allowed)
        self.assertTrue(guard.serialize()["test_opened"])

    def test_paired_bound_is_hand_reconstructable_and_tamper_evident(self):
        ids = ["q0", "q1", "q2", "q3"]
        artifact = make_paired_bound(
            ids,
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            hypothesis="candidate_evidence_noninferiority",
            metric="candidate_evidence_recall",
            alpha=0.01,
            difference_bounds=(-1.0, 1.0),
            side="lower",
            margin=-0.1,
            left_policy_fingerprint="pdctp",
            right_policy_fingerprint="fixed",
        )
        self.assertEqual(
            [row["difference"] for row in artifact["per_query"]],
            [0.0, 1.0, 0.0, 1.0],
        )
        self.assertEqual(artifact["n"], 4)
        validate_paired_bound(artifact)
        tampered = copy.deepcopy(artifact)
        tampered["per_query"][0]["left"] = 0.0
        with self.assertRaisesRegex(PairedBoundError, "fingerprint"):
            validate_paired_bound(tampered)

    def test_budget_superiority_uses_upper_paired_bound(self):
        artifact = make_paired_bound(
            [f"q{index}" for index in range(64)],
            [0.1] * 64,
            [0.9] * 64,
            hypothesis="budget_superiority_fixed",
            metric="normalized_candidate_budget",
            alpha=0.01,
            difference_bounds=(-1.0, 1.0),
            side="upper",
            margin=0.0,
            left_policy_fingerprint="pdctp",
            right_policy_fingerprint="fixed",
        )
        self.assertLess(artifact["upper_bound"], 0.0)
        self.assertTrue(artifact["passed"])

    def test_familywise_allocation_and_power_plan_are_deterministic(self):
        config = load_pdctp_foundation_config(CONFIG)
        names = [row["name"] for row in config.certification.hypotheses]
        allocation = bonferroni_allocation(
            names, config.certification.family_wise_alpha
        )
        self.assertAlmostEqual(sum(allocation.values()), 0.05)
        one = make_power_plan(
            config.certification.hypotheses,
            total_alpha=config.certification.family_wise_alpha,
        )
        two = make_power_plan(
            config.certification.hypotheses,
            total_alpha=config.certification.family_wise_alpha,
        )
        self.assertEqual(one, two)
        self.assertGreater(one["required_role_size"], 2)
        self.assertEqual(one["family_wise_method"], "bonferroni")
        checked_in = json.loads(
            (ROOT / "artifacts" / "pdctp_network_free" / "power_plan_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(one, checked_in)


if __name__ == "__main__":
    unittest.main()
