import json
import tempfile
import unittest
from pathlib import Path

from tri_rag_harness.config import load_config
from tri_rag_harness.mprime_sweep import _validate_tune_only, run_mprime_sweep
from tri_rag_harness.utils import fingerprint


ROOT = Path(__file__).resolve().parents[1]


class MPrimeSweepTests(unittest.TestCase):
    def _small_config(self, directory: Path):
        raw = json.loads(
            (ROOT / "configs" / "synthetic_mprime_sweep_fresh.json").read_text()
        )
        raw["synthetic"]["query_tune"] = 32
        raw["synthetic"]["query_cert"] = 48
        raw["synthetic"]["query_test"] = 16
        raw["m_prime_sweep"]["candidates"] = [4, 8]
        raw["m_prime_sweep"]["threshold_grid"] = [0.8, 0.9]
        raw["m_prime_sweep"]["tune_lower_bound_target"] = 0.2
        raw["m_prime_sweep"]["max_saturation_fraction"] = 1.0
        raw["certification"]["target"] = 0.2
        config_path = directory / "config.json"
        config_path.write_text(json.dumps(raw))
        return load_config(config_path)

    def test_rejects_certification_records_during_selection(self):
        with self.assertRaisesRegex(ValueError, "query_tune records only"):
            _validate_tune_only(
                [
                    {"split": "query_tune"},
                    {"split": "query_cert"},
                ]
            )

    def test_sweep_freezes_selection_before_fresh_certification(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config = self._small_config(directory)
            output = directory / "sweep"
            paths = run_mprime_sweep(config, output)

            selection = json.loads(paths["selection.json"].read_text())
            selected_config = json.loads(paths["selected_config.json"].read_text())
            result = json.loads(paths["sweep_result.json"].read_text())
            manifest = json.loads(
                (paths["selected_run"] / "manifest.json").read_text()
            )
            policy = json.loads(
                (paths["selected_run"] / "tri_predict_policy.json").read_text()
            )

            fingerprint_input = dict(selection)
            stored_fingerprint = fingerprint_input.pop("selection_fingerprint")
            self.assertEqual(stored_fingerprint, fingerprint(fingerprint_input))
            self.assertEqual(result["selection_fingerprint"], stored_fingerprint)
            self.assertTrue(result["selection_written_before_certification"])
            self.assertEqual(selection["data_scope"], "query_tune_only")
            self.assertIn(selection["selected"]["m_prime"], [4, 8])
            self.assertEqual(
                selected_config["retrieval"]["m_prime"],
                selection["selected"]["m_prime"],
            )
            self.assertEqual(
                selected_config["tri_predict"]["target"],
                selection["selected"]["threshold"],
            )
            self.assertEqual(policy["m_prime"], selection["selected"]["m_prime"])
            self.assertEqual(policy["target"], selection["selected"]["threshold"])
            self.assertEqual(
                result["selected_policy_fingerprint"], policy["fingerprint"]
            )
            self.assertNotEqual(
                selection["tune_query_id_hash"],
                manifest["splits"]["query_cert"]["id_hash"],
            )
            self.assertTrue(paths["sweep_report.md"].is_file())


if __name__ == "__main__":
    unittest.main()
