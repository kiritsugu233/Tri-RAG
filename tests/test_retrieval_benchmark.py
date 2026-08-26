import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tri_rag_harness.indexes import StreamingSearchResult
from tri_rag_harness.retrieval_benchmark import (
    METHOD_TRI_DOUBLE,
    METHOD_TRI_REUSE,
    _compare_search_results,
    load_retrieval_benchmark_config,
    run_retrieval_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]


class _FakeIndexFlatL2:
    def __init__(self, dimension):
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self):
        return len(self.vectors)

    def add(self, vectors):
        self.vectors = np.asarray(vectors, dtype=np.float32).copy()

    def search(self, queries, k):
        distances = np.sum(
            (
                np.asarray(queries, dtype=np.float32)[:, None, :]
                - self.vectors[None, :, :]
            )
            ** 2,
            axis=2,
            dtype=np.float32,
        )
        rows = np.argsort(distances, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(distances, rows, axis=1), rows.astype(np.int64)


class _FakeFaiss:
    __version__ = "test-double"
    IndexFlatL2 = _FakeIndexFlatL2


class RetrievalBenchmarkTests(unittest.TestCase):
    @staticmethod
    def _search_result(rows, distances):
        return StreamingSearchResult(
            rows=np.asarray(rows, dtype=np.int64),
            squared_distances=np.asarray(distances, dtype=np.float64),
            search_ms=0.0,
            distance_evaluations=10,
            scanned_vector_bytes=40,
        )

    def test_conformance_accepts_only_internal_cutoff_permutations(self):
        reference = self._search_result(
            [0, 1, 2, 3], [0.1, 0.2, 0.3000000, 0.3000002]
        )
        internal_swap = self._search_result(
            [0, 1, 3, 2], [0.1, 0.2, 0.3000001, 0.3000003]
        )
        comparison = _compare_search_results(
            reference, internal_swap, semantic_cutoffs=(2, 4)
        )
        self.assertTrue(comparison["accepted"])
        self.assertFalse(comparison["rows_equal"])
        self.assertTrue(comparison["rows_set_equal"])
        self.assertTrue(comparison["semantic_cutoffs_equal"])
        self.assertTrue(comparison["order_only_permutation_accepted"])

    def test_conformance_rejects_a_row_crossing_a_semantic_cutoff(self):
        reference = self._search_result([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4])
        crossing_swap = self._search_result(
            [0, 2, 1, 3], [0.1, 0.3, 0.2, 0.4]
        )
        comparison = _compare_search_results(
            reference, crossing_swap, semantic_cutoffs=(2, 4)
        )
        self.assertFalse(comparison["accepted"])
        self.assertFalse(comparison["semantic_cutoffs_equal"])
        self.assertTrue(comparison["distances_close"])

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

    def test_tiny_faiss_cpu_benchmark_conforms_to_numpy(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config = self._small_config(directory)
            paths = run_retrieval_benchmark(
                config,
                directory / "faiss-run",
                backend="faiss-cpu",
                faiss_module=_FakeFaiss(),
            )
            manifest = json.loads(paths["manifest.json"].read_text())
            summary = json.loads(paths["summary.json"].read_text())
            self.assertEqual(
                manifest["search"]["backend"], "faiss-cpu_index_flat_l2"
            )
            self.assertFalse(
                manifest["search"]["index_build"]["original"][
                    "gpu_resources_shared"
                ]
            )
            self.assertEqual(
                manifest["search"]["faiss_boundary_tie_overfetch"], 64
            )
            self.assertEqual(
                manifest["search"]["backend_validation"]["mismatches"], 0
            )
            self.assertEqual(
                len(manifest["search"]["backend_validation"]["checks"]), 2
            )
            self.assertTrue(
                manifest["search"]["backend_validation"][
                    "compiled_policy_decision_equal"
                ]
            )
            self.assertTrue(
                manifest["search"]["backend_validation"][
                    "reranked_top_k_rows_equal"
                ]
            )
            self.assertTrue(
                manifest["search"]["backend_validation"][
                    "embedding_retention_equal"
                ]
            )
            self.assertTrue(paths["gpu_memory.json"].is_file())
            self.assertGreater(
                summary[METHOD_TRI_REUSE]["latency_ms"]["backend_search_ms"][
                    "mean"
                ],
                0.0,
            )
            self.assertGreater(
                summary[METHOD_TRI_REUSE]["latency_ms"][
                    "backend_refinement_ms"
                ]["mean"],
                0.0,
            )
            alternate_paths = run_retrieval_benchmark(
                config,
                directory / "faiss-run-overfetch-32",
                backend="faiss-cpu",
                faiss_module=_FakeFaiss(),
                faiss_boundary_tie_overfetch=32,
            )
            alternate_manifest = json.loads(
                alternate_paths["manifest.json"].read_text()
            )
            self.assertEqual(
                alternate_manifest["search"]["faiss_boundary_tie_overfetch"],
                32,
            )
            self.assertNotEqual(
                alternate_manifest["reproducibility_fingerprint"],
                manifest["reproducibility_fingerprint"],
            )


if __name__ == "__main__":
    unittest.main()
