import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tri_rag_harness.indexes import ExactSquaredL2Index
from tri_rag_harness.real_dimension_sweep import (
    RealDimensionSweepError,
    _coordinate_work,
    _exact_projected_rankings,
    load_real_dimension_sweep_config,
    run_real_dimension_sweep,
)
from tri_rag_harness.utils import fingerprint, stable_id_hash, write_json


ROOT = Path(__file__).resolve().parents[1]


def _file_identity(path):
    value = path.read_bytes()
    return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def _jsonl(path, rows):
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class RealDimensionSweepTests(unittest.TestCase):
    def test_checked_in_protocol_freezes_common_cost_and_tune_scope(self):
        config = load_real_dimension_sweep_config(
            ROOT / "configs" / "real_scifact_fixed_dimension_tune.json"
        )
        self.assertEqual(config.evaluation_split, "query_tune")
        self.assertEqual(config.projection.seed, 27011)
        self.assertEqual(config.projection.candidates[0], 16)
        self.assertEqual(config.projection.candidates[-1], 768)
        self.assertFalse(config.projection.post_projection_normalize)
        self.assertEqual(config.search.m_pilot, 32)
        self.assertEqual(config.search.m_grid[-1], 5183)
        self.assertEqual(config.selection.target, 0.95)
        self.assertTrue(config.selection.include_query_projection)
        self.assertEqual(
            config.config_fingerprint,
            "3265e303c5249a6b90868f5234d333eca3f1fc4bc28c12cdb710382e2b71eabd",
        )

    def test_config_rejects_protected_scope_and_cost_formula_mutation(self):
        raw = json.loads(
            (ROOT / "configs" / "real_scifact_fixed_dimension_tune.json").read_text()
        )
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            raw["evaluation_split"] = "query_cert"
            protected = directory / "protected.json"
            protected.write_text(json.dumps(raw))
            with self.assertRaisesRegex(RealDimensionSweepError, "query_tune only"):
                load_real_dimension_sweep_config(protected)
            raw["evaluation_split"] = "query_tune"
            raw["selection"]["cost_formula"] = "m_prime + fixed_budget"
            changed = directory / "changed.json"
            changed.write_text(json.dumps(raw))
            with self.assertRaisesRegex(RealDimensionSweepError, "cost formula"):
                load_real_dimension_sweep_config(changed)

    def test_common_coordinate_work_includes_query_projection(self):
        work = _coordinate_work(
            corpus_size=100, dimension=10, m_prime=4, budget=20
        )
        self.assertEqual(work["query_projection"], 40)
        self.assertEqual(work["projected_full_scan"], 400)
        self.assertEqual(work["original_rerank"], 200)
        self.assertEqual(work["total"], 640)
        self.assertAlmostEqual(work["ratio_to_original_full_scan"], 0.64)

    def test_projected_ranking_matches_exact_index_and_stable_id_ties(self):
        corpus = np.asarray([[1.0], [-1.0], [0.0]])
        queries = np.asarray([[0.0], [0.75]])
        ids = ["doc:z", "doc:a", "doc:m"]
        rows, _ = _exact_projected_rankings(
            corpus, queries, ids, k=3, batch_size=1
        )
        reference = ExactSquaredL2Index(ids, corpus, batch_size=1).search(queries, 3)
        np.testing.assert_array_equal(rows, reference.rows)
        self.assertEqual(rows[0].tolist(), [2, 1, 0])

    def test_tiny_runner_is_tune_only_auditable_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            prepared = directory / "prepared"
            cache = directory / "cache"
            baseline = directory / "baseline"
            prepared.mkdir()
            cache.mkdir()
            baseline.mkdir()

            corpus_ids = ["fixture:doc:d0", "fixture:doc:d1", "fixture:doc:d2"]
            corpus = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
            tune_ids = [f"fixture:query:t{index:02d}" for index in range(64)]
            query_ids = tune_ids + ["fixture:query:cert", "fixture:query:test"]
            queries = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (66, 1))
            np.save(cache / "corpus_embeddings.f32.npy", corpus, allow_pickle=False)
            np.save(cache / "query_embeddings.f32.npy", queries, allow_pickle=False)
            (cache / "corpus_ids.json").write_text(json.dumps(corpus_ids))
            (cache / "query_ids.json").write_text(json.dumps(query_ids))
            query_rows = [
                {
                    "query_id": query_id,
                    "split": (
                        "query_tune"
                        if query_id in tune_ids
                        else "query_cert"
                        if query_id.endswith("cert")
                        else "query_test"
                    ),
                }
                for query_id in query_ids
            ]
            _jsonl(prepared / "queries.jsonl", query_rows)

            dataset_fingerprint = "a" * 64
            embedding_config_fingerprint = "c" * 64
            request_fingerprint = "d" * 64
            embedding_fingerprint = "e" * 64
            baseline_config_fingerprint = "f" * 64
            baseline_records = [
                {
                    "query_index": index,
                    "query_id": query_id,
                    "split": "query_tune",
                    "exact_top_k_ids": corpus_ids[:2],
                }
                for index, query_id in enumerate(tune_ids)
            ]
            _jsonl(baseline / "per_query.jsonl", baseline_records)
            write_json(baseline / "summary.json", {"n_queries": 64})
            (baseline / "report.md").write_text("fixture\n")
            result_artifacts = {
                name: _file_identity(baseline / name)
                for name in ("per_query.jsonl", "summary.json", "report.md")
            }
            split_hash = stable_id_hash(tune_ids)
            baseline_identity = {
                "config_fingerprint": baseline_config_fingerprint,
                "dataset_manifest_fingerprint": dataset_fingerprint,
                "embedding_manifest_fingerprint": embedding_fingerprint,
                "query_split_id_hash": split_hash,
                "artifacts": result_artifacts,
            }
            baseline_result_fingerprint = fingerprint(baseline_identity)
            write_json(
                baseline / "manifest.json",
                {
                    "kind": "real_original_exact_tune_manifest_v1",
                    "data_scope": "query_tune_only",
                    "config_fingerprint": baseline_config_fingerprint,
                    "dataset": {
                        "manifest_fingerprint": dataset_fingerprint,
                        "query_split": "query_tune",
                        "query_split_id_hash": split_hash,
                    },
                    "embedding": {
                        "manifest_fingerprint": embedding_fingerprint,
                    },
                    "result_artifacts": result_artifacts,
                    "result_fingerprint": baseline_result_fingerprint,
                },
            )
            raw = {
                "schema_version": 1,
                "benchmark": "real_fixed_dimension_tune_sweep_v1",
                "dataset_manifest_fingerprint": dataset_fingerprint,
                "embedding_config_fingerprint": embedding_config_fingerprint,
                "embedding_request_fingerprint": request_fingerprint,
                "embedding_manifest_fingerprint": embedding_fingerprint,
                "original_baseline_result_fingerprint": baseline_result_fingerprint,
                "evaluation_split": "query_tune",
                "projection": {
                    "family": "dense_gaussian_n0_variance_1_over_m_prime",
                    "seed": 7,
                    "candidates": [1, 2],
                    "candidate_coupling": "same_seed_nested_prefix_rescaled",
                    "post_projection_normalize": False,
                },
                "search": {
                    "normalized_inputs": True,
                    "distance": "squared_l2",
                    "arithmetic": "numpy_float64",
                    "stable_tie_break": "lexicographic_doc_id",
                    "query_batch_size": 8,
                    "k_ctx": 1,
                    "k_gt": 2,
                    "m_pilot": 2,
                    "m_grid": [2, 3],
                },
                "selection": {
                    "metric": "embedding_neighbor_retention_at_k_gt",
                    "alpha": 0.05,
                    "target": 0.5,
                    "statistic_role": "tune_selection_score_not_certificate",
                    "objective": "coordinate_multiply_adds_per_query",
                    "cost_formula": "(corpus_size + embedding_dimension) * m_prime + embedding_dimension * fixed_budget",
                    "include_query_projection": True,
                    "tie_break": [
                        "higher_lower_bound",
                        "smaller_m_prime",
                        "smaller_fixed_budget",
                    ],
                },
            }
            config_path = directory / "config.json"
            config_path.write_text(json.dumps(raw))
            config = load_real_dimension_sweep_config(config_path)
            dataset_manifest = {
                "fingerprint": dataset_fingerprint,
                "ids": {"corpus_id_hash": stable_id_hash(corpus_ids)},
                "splits": {"query_tune": {"n": 64, "id_hash": split_hash}},
            }
            embedding_manifest = {
                "fingerprint": embedding_fingerprint,
                "arrays": {
                    "corpus": {"array_fingerprint": "1" * 64},
                    "queries": {"array_fingerprint": "2" * 64},
                },
            }
            embedding_config = SimpleNamespace(
                config_fingerprint=embedding_config_fingerprint,
                model=SimpleNamespace(
                    embedding_dimension=2,
                    name="fixture/model",
                    revision="3" * 40,
                ),
            )
            validation = {
                "dataset_manifest": dataset_manifest,
                "embedding_manifest": embedding_manifest,
                "request_fingerprint": request_fingerprint,
            }
            with patch(
                "tri_rag_harness.real_dimension_sweep.load_text_embedding_config",
                return_value=embedding_config,
            ), patch(
                "tri_rag_harness.real_dimension_sweep.validate_text_embedding_cache",
                return_value=validation,
            ):
                first = run_real_dimension_sweep(
                    config,
                    prepared,
                    directory / "embedding-config.json",
                    cache,
                    baseline,
                    directory / "first",
                )
                second = run_real_dimension_sweep(
                    config,
                    prepared,
                    directory / "embedding-config.json",
                    cache,
                    baseline,
                    directory / "second",
                )
            for name in (
                "manifest.json",
                "per_query.jsonl",
                "selection.json",
                "selected_projection.json",
                "summary.json",
                "report.md",
            ):
                self.assertEqual(first[name].read_bytes(), second[name].read_bytes())
            records = [
                json.loads(line)
                for line in first["per_query.jsonl"].read_text().splitlines()
            ]
            self.assertEqual(len(records), 128)
            self.assertEqual({record["split"] for record in records}, {"query_tune"})
            self.assertFalse(any("cert" in record["query_id"] for record in records))
            selection = json.loads(first["selection.json"].read_text())
            self.assertEqual(selection["data_scope"], "query_tune_only")
            self.assertIn(selection["selected"]["m_prime"], [1, 2])
            self.assertEqual(selection["selection_target"], 0.5)
            frozen = json.loads(first["selected_projection.json"].read_text())
            self.assertFalse(frozen["post_projection_normalized"])


if __name__ == "__main__":
    unittest.main()
