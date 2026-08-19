import json
import tempfile
import unittest
from pathlib import Path

from tri_rag_harness.attribution import MODES, run_attribution
from tri_rag_harness.config import load_config
from tri_rag_harness.synthetic import generate_synthetic_dataset


ROOT = Path(__file__).resolve().parents[1]


class AttributionTests(unittest.TestCase):
    def test_attribution_artifacts_separate_lid_rank_and_remaining_error(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            raw = json.loads(
                (ROOT / "configs" / "synthetic_mvp.json").read_text()
            )
            raw["synthetic"]["query_tune"] = 16
            raw["synthetic"]["query_cert"] = 20
            raw["synthetic"]["query_test"] = 12
            config_path = directory / "config.json"
            config_path.write_text(json.dumps(raw))
            paths = run_attribution(
                load_config(config_path), directory / "attribution"
            )
            artifact = json.loads(paths["attribution.json"].read_text())
            self.assertEqual(
                artifact["status"], "diagnostic_only_not_deployable"
            )
            self.assertEqual(set(artifact["modes"]), set(MODES))
            for split in ("query_tune", "query_cert", "query_test"):
                self.assertEqual(
                    set(artifact["aggregates"][split]), set(MODES)
                )
                self.assertIn(
                    "remaining_mae_with_actual_beta",
                    artifact["attribution"][split],
                )
            rows = paths["attribution_per_query.jsonl"].read_text().splitlines()
            self.assertEqual(len(rows), 48)

    def test_synthetic_seed_namespaces_make_fresh_split_ids_distinct(self):
        config = load_config(ROOT / "configs" / "synthetic_mvp.json")
        first = generate_synthetic_dataset(config.synthetic, seed=1)
        second = generate_synthetic_dataset(config.synthetic, seed=2)
        self.assertTrue(set(first.queries.ids).isdisjoint(second.queries.ids))
        self.assertTrue(set(first.corpus.ids).isdisjoint(second.corpus.ids))


if __name__ == "__main__":
    unittest.main()
