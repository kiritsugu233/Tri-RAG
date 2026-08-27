import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from tri_rag_harness.real_original_baseline import (
    RealOriginalBaselineError,
    _ndcg_at_k,
    load_real_original_baseline_config,
    run_real_original_baseline,
)
from tri_rag_harness.text_embeddings import (
    TextEmbeddingError,
    build_or_load_text_embedding_cache,
    load_text_embedding_config,
)
from tri_rag_harness.utils import fingerprint, stable_id_hash, write_json


ROOT = Path(__file__).resolve().parents[1]


def _file_identity(path):
    value = path.read_bytes()
    return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


class _FixtureProvider:
    def encode(self, texts, *, batch_size, role):
        del batch_size
        if role == "corpus":
            assert len(texts) == 3
            return np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        assert role == "queries" and len(texts) == 4
        return np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

    def metadata(self):
        return {
            "provider": "original_baseline_fixture",
            "model": {"snapshot_fingerprint": "f" * 64},
            "runtime": {"device": "cpu"},
        }


class RealOriginalBaselineTests(unittest.TestCase):
    @staticmethod
    def _write_jsonl(path, rows):
        path.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    def _fixture(self, directory):
        prepared = directory / "prepared"
        prepared.mkdir()
        corpus = [
            {
                "doc_id": f"fixture:doc:d{index}",
                "source_doc_id": f"d{index}",
                "title": "",
                "text": f"Document {index}",
            }
            for index in range(3)
        ]
        split_names = [
            "query_tune",
            "query_tune",
            "query_cert",
            "query_test",
        ]
        queries = [
            {
                "query_id": f"fixture:query:q{index}",
                "source_query_id": f"q{index}",
                "split": split,
                "text": f"Query {index}",
            }
            for index, split in enumerate(split_names)
        ]
        qrels = [
            {
                "doc_id": "fixture:doc:d0",
                "query_id": "fixture:query:q0",
                "relevance": 2,
                "split": "query_tune",
            },
            {
                "doc_id": "fixture:doc:d1",
                "query_id": "fixture:query:q0",
                "relevance": 1,
                "split": "query_tune",
            },
            {
                "doc_id": "fixture:doc:d1",
                "query_id": "fixture:query:q1",
                "relevance": 1,
                "split": "query_tune",
            },
            {
                "doc_id": "fixture:doc:d2",
                "query_id": "fixture:query:q2",
                "relevance": 1,
                "split": "query_cert",
            },
            {
                "doc_id": "fixture:doc:d0",
                "query_id": "fixture:query:q3",
                "relevance": 1,
                "split": "query_test",
            },
        ]
        splits = {
            split: [row["query_id"] for row in queries if row["split"] == split]
            for split in ("query_tune", "query_cert", "query_test")
        }
        self._write_jsonl(prepared / "corpus.jsonl", corpus)
        self._write_jsonl(prepared / "queries.jsonl", queries)
        self._write_jsonl(prepared / "qrels.jsonl", qrels)
        write_json(prepared / "splits.json", splits)
        artifact_names = (
            "corpus.jsonl",
            "queries.jsonl",
            "qrels.jsonl",
            "splits.json",
        )
        manifest = {
            "schema_version": 1,
            "counts": {
                "corpus": len(corpus),
                "queries": len(queries),
                "qrels": len(qrels),
            },
            "ids": {
                "corpus_id_hash": stable_id_hash(
                    [row["doc_id"] for row in corpus]
                ),
                "query_id_hash": stable_id_hash(
                    [row["query_id"] for row in queries]
                ),
            },
            "splits": {
                split: {
                    "n": len(ids),
                    "id_hash": stable_id_hash(ids),
                }
                for split, ids in splits.items()
            },
            "artifacts": {
                name: _file_identity(prepared / name) for name in artifact_names
            },
        }
        manifest["fingerprint"] = fingerprint(manifest)
        write_json(prepared / "dataset_manifest.json", manifest)

        embedding_config_raw = json.loads(
            (
                ROOT / "configs" / "real_scifact_e5_base_v2_embeddings.json"
            ).read_text()
        )
        embedding_config_raw["dataset_manifest_fingerprint"] = manifest[
            "fingerprint"
        ]
        embedding_config_raw["model"]["embedding_dimension"] = 2
        embedding_config_raw["model"]["snapshot_allow_patterns"] = [
            "config.json"
        ]
        embedding_config_raw["encoding"]["batch_size"] = 2
        embedding_config_raw["runtime"]["required_packages"] = {
            "fixture-runtime": "1.0"
        }
        embedding_config_path = directory / "embedding-config.json"
        embedding_config_path.write_text(json.dumps(embedding_config_raw))
        embedding_config = load_text_embedding_config(embedding_config_path)
        cache = directory / "cache"
        cache_result = build_or_load_text_embedding_cache(
            embedding_config,
            prepared,
            cache,
            provider=_FixtureProvider(),
        )

        baseline_raw = {
            "schema_version": 1,
            "benchmark": "real_original_exact_v1",
            "dataset_manifest_fingerprint": manifest["fingerprint"],
            "embedding_config_fingerprint": (
                embedding_config.config_fingerprint
            ),
            "embedding_request_fingerprint": cache_result[
                "request_fingerprint"
            ],
            "embedding_manifest_fingerprint": cache_result["fingerprint"],
            "evaluation_split": "query_tune",
            "search": {
                "normalized_inputs": True,
                "distance": "squared_l2",
                "arithmetic": "numpy_float64",
                "stable_tie_break": "lexicographic_doc_id",
                "query_batch_size": 2,
                "cutoffs": [1, 2],
                "k_ctx": 1,
                "k_gt": 2,
            },
        }
        baseline_path = directory / "baseline.json"
        baseline_path.write_text(json.dumps(baseline_raw))
        baseline_config = load_real_original_baseline_config(baseline_path)
        return {
            "prepared": prepared,
            "embedding_config_path": embedding_config_path,
            "cache": cache,
            "baseline_path": baseline_path,
            "baseline_config": baseline_config,
        }

    def test_tune_only_exact_baseline_is_auditable_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            fixture = self._fixture(directory)
            first = run_real_original_baseline(
                fixture["baseline_config"],
                fixture["prepared"],
                fixture["embedding_config_path"],
                fixture["cache"],
                directory / "first",
            )
            second = run_real_original_baseline(
                fixture["baseline_config"],
                fixture["prepared"],
                fixture["embedding_config_path"],
                fixture["cache"],
                directory / "second",
            )
            for name in ("manifest.json", "per_query.jsonl", "summary.json", "report.md"):
                self.assertEqual(first[name].read_bytes(), second[name].read_bytes())
            records = [
                json.loads(line)
                for line in first["per_query.jsonl"].read_text().splitlines()
            ]
            self.assertEqual(
                [record["query_id"] for record in records],
                ["fixture:query:q0", "fixture:query:q1"],
            )
            self.assertTrue(all(record["split"] == "query_tune" for record in records))
            self.assertEqual(
                records[0]["exact_top_k_ids"],
                ["fixture:doc:d0", "fixture:doc:d1"],
            )
            self.assertEqual(
                records[1]["exact_top_k_ids"],
                ["fixture:doc:d1", "fixture:doc:d0"],
            )
            self.assertEqual(records[0]["metrics"]["1"]["evidence_recall"], 0.5)
            summary = json.loads(first["summary.json"].read_text())
            self.assertEqual(summary["n_queries"], 2)
            self.assertEqual(summary["metrics"]["1"]["mean_evidence_hit"], 1.0)
            self.assertEqual(summary["metrics"]["1"]["mean_evidence_recall"], 0.75)
            manifest = json.loads(first["manifest.json"].read_text())
            self.assertEqual(manifest["data_scope"], "query_tune_only")
            self.assertEqual(manifest["dataset"]["query_split"], "query_tune")
            self.assertNotIn("timings.json", manifest["result_artifacts"])

    def test_config_rejects_certification_or_test_scope(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            fixture = self._fixture(directory)
            raw = json.loads(fixture["baseline_path"].read_text())
            raw["evaluation_split"] = "query_cert"
            changed = directory / "changed.json"
            changed.write_text(json.dumps(raw))
            with self.assertRaisesRegex(
                RealOriginalBaselineError, "query_tune only"
            ):
                load_real_original_baseline_config(changed)
            output = directory / "protected-output"
            with self.assertRaisesRegex(
                RealOriginalBaselineError, "query_tune only"
            ):
                run_real_original_baseline(
                    replace(
                        fixture["baseline_config"],
                        evaluation_split="query_cert",
                    ),
                    fixture["prepared"],
                    fixture["embedding_config_path"],
                    fixture["cache"],
                    output,
                )
            self.assertFalse(output.exists())

    def test_graded_ndcg_penalizes_reversed_relevance(self):
        actual = _ndcg_at_k(
            ["fixture:doc:low", "fixture:doc:high"],
            {"fixture:doc:high": 2, "fixture:doc:low": 1},
            2,
        )
        expected = (1.0 + 3.0 / np.log2(3.0)) / (
            3.0 + 1.0 / np.log2(3.0)
        )
        self.assertAlmostEqual(actual, expected)

    def test_tampered_embedding_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            fixture = self._fixture(directory)
            path = fixture["cache"] / "query_embeddings.f32.npy"
            values = np.load(path)
            values[0, 0] += np.float32(0.25)
            np.save(path, values, allow_pickle=False)
            output = directory / "output"
            with self.assertRaisesRegex(
                TextEmbeddingError, "embedding cache artifact mismatch"
            ):
                run_real_original_baseline(
                    fixture["baseline_config"],
                    fixture["prepared"],
                    fixture["embedding_config_path"],
                    fixture["cache"],
                    output,
                )
            self.assertFalse(output.exists())

    def test_checked_in_config_freezes_tune_scope_and_real_identities(self):
        config = load_real_original_baseline_config(
            ROOT / "configs" / "real_scifact_original_exact_tune.json"
        )
        self.assertEqual(config.evaluation_split, "query_tune")
        self.assertEqual(config.search.k_ctx, 5)
        self.assertEqual(config.search.k_gt, 10)
        self.assertEqual(config.search.cutoffs, [1, 5, 10])
        self.assertEqual(
            config.config_fingerprint,
            "ff675fed06fc6506ed68a83426a021ee53a701f06af4144351b2172c2dbc19f6",
        )
        self.assertEqual(
            config.embedding_manifest_fingerprint,
            "2ec53ce38e226129ba0feffcd28ba1da1081e0627ad8e54f4a60e430c341e914",
        )


if __name__ == "__main__":
    unittest.main()
