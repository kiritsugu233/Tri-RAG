import json
import tempfile
import unittest
from pathlib import Path

from tri_rag_harness.retrieval_benchmark import (
    METHOD_TRI_DOUBLE,
    METHOD_TRI_REUSE,
    load_retrieval_benchmark_config,
    run_retrieval_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]


class RetrievalBenchmarkTests(unittest.TestCase):
    def _small_config(self, directory: Path):
        raw = json.loads(
            (ROOT / "configs" / "retrieval_latency_100k_d768.json").read_text()
        )
        raw["run_name"] = "tiny_retrieval_latency"
        raw["dataset"].update(
            {
                "corpus_size": 128,
                "query_count": 4,
                "warmup_query_count": 1,
                "dimension": 16,
                "generation_batch_rows": 31,
            }
        )
        raw["projection"]["m_prime"] = 6
        raw["search"].update(
            {
                "k_gt": 3,
                "m_pilot": 6,
                "s_lid": 5,
                "min_lid_neighbors": 3,
                "m_grid": [6, 12, 24],
                "fixed_budget": 12,
                "corpus_block_rows": 17,
            }
        )
        raw["tri_predict"]["max_rank_samples"] = 32
        config_path = directory / "config.json"
        config_path.write_text(json.dumps(raw), encoding="utf-8")
        return load_retrieval_benchmark_config(config_path)

    def test_tiny_benchmark_reuses_one_scan_and_matches_double_scan_decisions(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config = self._small_config(directory)
            paths = run_retrieval_benchmark(config, directory / "run")
            summary = json.loads(paths["summary.json"].read_text())
            manifest = json.loads(paths["manifest.json"].read_text())
            compiled_policy = json.loads(
                paths["tri_predict_compiled_policy.json"].read_text()
            )
            rows = [
                json.loads(line)
                for line in paths["per_query.jsonl"].read_text().splitlines()
            ]
            self.assertEqual(len(rows), config.dataset.query_count * 4)
            reuse = summary[METHOD_TRI_REUSE]
            double = summary[METHOD_TRI_DOUBLE]
            self.assertEqual(
                reuse["work_per_query"]["projected_scan_count"], 1.0
            )
            self.assertEqual(
                double["work_per_query"]["projected_scan_count"], 2.0
            )
            self.assertEqual(
                reuse["work_per_query"]["projected_distance_evaluations"], 128.0
            )
            self.assertEqual(
                double["work_per_query"]["projected_distance_evaluations"], 256.0
            )
            self.assertEqual(
                summary["reuse_comparison"][
                    "projected_distance_evaluation_reduction_fraction"
                ],
                0.5,
            )
            self.assertEqual(
                manifest["policy_execution"]["implementation"],
                "compiled_float64_lid_decision_boundaries",
            )
            self.assertEqual(
                compiled_policy["reference_policy_fingerprint"],
                manifest["policy"]["fingerprint"],
            )
            self.assertEqual(
                summary["policy_execution"]["observed_equivalence_n"],
                config.dataset.query_count,
            )
            self.assertEqual(
                summary["policy_execution"]["observed_equivalence_mismatches"],
                0,
            )
            for query_index in range(config.dataset.query_count):
                reuse_row = next(
                    row
                    for row in rows
                    if row["query_index"] == query_index
                    and row["method"] == METHOD_TRI_REUSE
                )
                double_row = next(
                    row
                    for row in rows
                    if row["query_index"] == query_index
                    and row["method"] == METHOD_TRI_DOUBLE
                )
                self.assertEqual(reuse_row["chosen_m"], double_row["chosen_m"])
                self.assertEqual(
                    reuse_row["embedding_retention"],
                    double_row["embedding_retention"],
                )
            self.assertTrue(paths["report.md"].is_file())
            self.assertTrue(paths["memory.json"].is_file())
            self.assertTrue(paths["tri_predict_compiled_policy.json"].is_file())


if __name__ == "__main__":
    unittest.main()
