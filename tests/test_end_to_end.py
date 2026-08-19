import json
import tempfile
import unittest
from pathlib import Path

from tri_rag_harness.config import load_config
from tri_rag_harness.run import run_harness


ROOT = Path(__file__).resolve().parents[1]
TIMING_KEYS = {
    "pilot_search_ms",
    "expansion_search_ms",
    "policy_compute_ms",
    "rerank_ms",
    "total_retrieval_ms",
}


class EndToEndTests(unittest.TestCase):
    def _small_config(self, directory: Path):
        raw = json.loads((ROOT / "configs" / "synthetic_mvp.json").read_text())
        raw["synthetic"]["query_tune"] = 48
        raw["synthetic"]["query_cert"] = 64
        raw["synthetic"]["query_test"] = 32
        path = directory / "config.json"
        path.write_text(json.dumps(raw))
        return load_config(path)

    def test_complete_run_is_auditable_disjoint_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config = self._small_config(directory)
            paths_one = run_harness(config, directory / "run-one")
            paths_two = run_harness(config, directory / "run-two")
            for name in (
                "manifest.json",
                "policy.json",
                "per_query.jsonl",
                "certification.json",
                "aggregates.json",
                "timings.json",
                "report.md",
            ):
                self.assertTrue((directory / "run-one" / name).is_file())
            manifest = json.loads(paths_one["manifest.json"].read_text())
            manifest_two = json.loads(paths_two["manifest.json"].read_text())
            self.assertTrue(manifest["dataset"]["queries_are_external"])
            manifest.pop("created_at_utc")
            manifest_two.pop("created_at_utc")
            self.assertEqual(manifest, manifest_two)
            split_hashes = [value["id_hash"] for value in manifest["splits"].values()]
            self.assertEqual(len(split_hashes), len(set(split_hashes)))
            policy_one = json.loads(paths_one["policy.json"].read_text())
            policy_two = json.loads(paths_two["policy.json"].read_text())
            self.assertEqual(policy_one, policy_two)
            certificate_one = json.loads(paths_one["certification.json"].read_text())
            certificate_two = json.loads(paths_two["certification.json"].read_text())
            self.assertEqual(certificate_one, certificate_two)
            aggregate_one = json.loads(paths_one["aggregates.json"].read_text())
            aggregate_two = json.loads(paths_two["aggregates.json"].read_text())
            self.assertEqual(aggregate_one, aggregate_two)
            rows_one = [json.loads(line) for line in paths_one["per_query.jsonl"].read_text().splitlines()]
            rows_two = [json.loads(line) for line in paths_two["per_query.jsonl"].read_text().splitlines()]
            stripped_one = [
                {key: value for key, value in row.items() if key not in TIMING_KEYS}
                for row in rows_one
            ]
            stripped_two = [
                {key: value for key, value in row.items() if key not in TIMING_KEYS}
                for row in rows_two
            ]
            self.assertEqual(stripped_one, stripped_two)
            query_ids = {
                split: {row["query_id"] for row in rows_one if row["split"] == split}
                for split in ("query_tune", "query_cert", "query_test")
            }
            self.assertTrue(query_ids["query_tune"].isdisjoint(query_ids["query_cert"]))
            self.assertTrue(query_ids["query_tune"].isdisjoint(query_ids["query_test"]))
            self.assertTrue(query_ids["query_cert"].isdisjoint(query_ids["query_test"]))
            for row in rows_one:
                fixed = [row["fixed_retentions"][str(m)] for m in config.retrieval.m_grid]
                self.assertTrue(all(left <= right for left, right in zip(fixed, fixed[1:])))
                expected = len(set(row["exact_top_k_ids"]) & set(row["reranked_top_k_ids"])) / config.retrieval.k_gt
                self.assertEqual(row["embedding_retention"], expected)
            report = paths_one["report.md"].read_text()
            expected_status = "PASS" if certificate_one["passed"] else "FAIL"
            self.assertIn(f"Stored artifact decision: **{expected_status}**", report)


if __name__ == "__main__":
    unittest.main()
