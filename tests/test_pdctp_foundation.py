import json
import tempfile
import unittest
from pathlib import Path

from tri_rag_harness.pdctp_config import load_pdctp_foundation_config
from tri_rag_harness.pdctp_foundation import run_pdctp_foundation
from tri_rag_harness.pdctp_protocol import FIVE_ROLES
from tri_rag_harness.pdctp_statistics import validate_paired_bound


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "pdctp_network_free_foundation_v1.json"


class PDCTPFoundationEndToEndTests(unittest.TestCase):
    def test_five_role_walking_skeleton_is_deterministic_and_auditable(self):
        config = load_pdctp_foundation_config(CONFIG)
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            paths_one = run_pdctp_foundation(config, directory / "one")
            paths_two = run_pdctp_foundation(config, directory / "two")
            self.assertEqual(set(paths_one), set(paths_two))
            for name in paths_one:
                with self.subTest(name=name):
                    self.assertEqual(
                        paths_one[name].read_bytes(), paths_two[name].read_bytes()
                    )

            manifest = json.loads(paths_one["manifest.json"].read_text())
            self.assertTrue(manifest["dataset"]["synthetic_only"])
            self.assertFalse(manifest["dataset"]["real_data_accessed"])
            self.assertFalse(manifest["dataset"]["network_accessed"])
            self.assertEqual(manifest["projection"]["m_prime"], 8)
            self.assertFalse(manifest["projection"]["post_projection_normalized"])
            self.assertTrue(manifest["search"]["pilot_and_expansion_share_scan"])

            splits = json.loads(paths_one["splits.json"].read_text())
            self.assertEqual(set(splits["roles"]), set(FIVE_ROLES))
            id_sets = [
                set(splits["roles"][role]["ordered_ids"]) for role in FIVE_ROLES
            ]
            for index, left in enumerate(id_sets):
                for right in id_sets[index + 1 :]:
                    self.assertTrue(left.isdisjoint(right))

            records = [
                json.loads(line)
                for line in paths_one["per_query.jsonl"].read_text().splitlines()
            ]
            self.assertTrue(records)
            self.assertTrue(all(row["projected_scan_count"] == 1 for row in records))
            self.assertTrue(all(row["pilot_is_cached_prefix"] for row in records))
            latency_rows = [row for row in records if row["role"] == "query_latency"]
            self.assertTrue(latency_rows)
            self.assertTrue(all(row["labels_accessed"] is False for row in latency_rows))
            self.assertTrue(
                all(
                    "oracle_lid" not in row
                    and "embedding_retention" not in row
                    and "candidate_evidence_recall" not in row
                    for row in latency_rows
                )
            )

            policies = json.loads(paths_one["policies.json"].read_text())
            self.assertEqual(
                set(policies),
                {
                    "fixed",
                    "monotone",
                    "raw_tri",
                    "lid_only",
                    "residual_only",
                    "full_pdctp",
                },
            )
            self.assertEqual(len({row["name"] for row in policies.values()}), 6)
            raw_reference = json.loads(
                paths_one["raw_tri_reference.json"].read_text()
            )
            self.assertIn(
                raw_reference["target"], config.calibration.raw_tri_threshold_grid
            )
            residual_candidates = json.loads(
                paths_one["residual_calibrator_candidates.json"].read_text()
            )
            fitted_targets = {
                artifact["anchor"]["policy_fingerprint"]
                for artifact in residual_candidates["full_candidates"]
            }
            self.assertEqual(len(fitted_targets), len(config.calibration.raw_tri_threshold_grid))
            shuffled = json.loads(
                paths_one["shuffled_tune_diagnostic.json"].read_text()
            )
            self.assertEqual(shuffled["role"], "query_tune")
            self.assertFalse(shuffled["used_for_selection"])
            self.assertTrue(
                all(row["role"] == "query_tune" for row in shuffled["records"])
            )
            certification = json.loads(
                paths_one["certification_bounds.json"].read_text()
            )
            for bound in certification["bounds"].values():
                validate_paired_bound(bound)
                self.assertEqual(bound["n"], config.synthetic.role_counts["query_cert"])
            power = json.loads(paths_one["power_plan.json"].read_text())
            self.assertGreater(
                power["required_role_size"],
                config.synthetic.role_counts["query_cert"],
            )
            protocol = json.loads(paths_one["protocol_state.json"].read_text())
            self.assertTrue(protocol["certification_opened"])
            self.assertTrue(protocol["latency_opened"])
            self.assertTrue(protocol["test_opened"])
            self.assertFalse(protocol["mutation_after_certification_allowed"])


if __name__ == "__main__":
    unittest.main()
