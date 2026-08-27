import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

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


class _FakeProvider:
    def __init__(self, dimension=4, *, zero=False):
        self.dimension = dimension
        self.zero = zero
        self.calls = []

    def encode(self, texts, *, batch_size, role):
        self.calls.append((list(texts), batch_size, role))
        if self.zero:
            return np.zeros((len(texts), self.dimension), dtype=np.float32)
        rows = []
        for index, text in enumerate(texts):
            rows.append(
                [
                    len(text) + 1,
                    (sum(text.encode("utf-8")) % 997) + 1,
                    index + 1,
                    3,
                ][: self.dimension]
            )
        return np.asarray(rows, dtype=np.float32)

    def metadata(self):
        return {
            "provider": "offline_test_double",
            "model": {"snapshot_fingerprint": "f" * 64},
            "runtime": {"device": "cpu"},
        }


class TextEmbeddingTests(unittest.TestCase):
    @staticmethod
    def _write_dataset(directory):
        directory.mkdir()
        corpus = [
            {
                "doc_id": "fixture:doc:d0",
                "source_doc_id": "d0",
                "title": " Title zero ",
                "text": " Body zero ",
            },
            {
                "doc_id": "fixture:doc:d1",
                "source_doc_id": "d1",
                "title": "",
                "text": "Body one",
            },
            {
                "doc_id": "fixture:doc:d2",
                "source_doc_id": "d2",
                "title": "Title two",
                "text": "",
            },
        ]
        queries = [
            {
                "query_id": f"fixture:query:q{index}",
                "source_query_id": f"q{index}",
                "split": split,
                "text": f" Claim {index} ",
            }
            for index, split in enumerate(
                ["query_tune", "query_cert", "query_test", "query_test"]
            )
        ]
        qrels = [
            {
                "query_id": row["query_id"],
                "doc_id": "fixture:doc:d0",
                "relevance": 1,
                "split": row["split"],
            }
            for row in queries
        ]
        splits = {
            "query_tune": [queries[0]["query_id"]],
            "query_cert": [queries[1]["query_id"]],
            "query_test": [queries[2]["query_id"], queries[3]["query_id"]],
        }

        def jsonl(path, rows):
            path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )

        jsonl(directory / "corpus.jsonl", corpus)
        jsonl(directory / "queries.jsonl", queries)
        jsonl(directory / "qrels.jsonl", qrels)
        write_json(directory / "splits.json", splits)
        artifact_names = (
            "corpus.jsonl",
            "queries.jsonl",
            "qrels.jsonl",
            "splits.json",
        )
        corpus_ids = [row["doc_id"] for row in corpus]
        query_ids = [row["query_id"] for row in queries]
        manifest = {
            "schema_version": 1,
            "counts": {
                "corpus": len(corpus),
                "queries": len(queries),
                "qrels": len(qrels),
            },
            "ids": {
                "corpus_id_hash": stable_id_hash(corpus_ids),
                "query_id_hash": stable_id_hash(query_ids),
            },
            "artifacts": {
                name: _file_identity(directory / name) for name in artifact_names
            },
        }
        manifest["fingerprint"] = fingerprint(manifest)
        write_json(directory / "dataset_manifest.json", manifest)
        return manifest

    @staticmethod
    def _write_config(path, dataset_fingerprint, *, query_prefix="query: "):
        raw = json.loads(
            (
                ROOT
                / "configs"
                / "real_scifact_e5_base_v2_embeddings.json"
            ).read_text()
        )
        raw["dataset_manifest_fingerprint"] = dataset_fingerprint
        raw["model"]["embedding_dimension"] = 4
        raw["model"]["snapshot_allow_patterns"] = ["config.json"]
        raw["formatting"]["query_prefix"] = query_prefix
        raw["encoding"]["batch_size"] = 2
        raw["runtime"]["required_packages"] = {"fixture-runtime": "1.0"}
        path.write_text(json.dumps(raw), encoding="utf-8")
        return load_text_embedding_config(path)

    def test_fake_provider_builds_normalized_cache_and_reuses_it(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            dataset = directory / "dataset"
            manifest = self._write_dataset(dataset)
            config = self._write_config(
                directory / "embedding.json", manifest["fingerprint"]
            )
            output = directory / "cache"
            provider = _FakeProvider()
            created = build_or_load_text_embedding_cache(
                config, dataset, output, provider=provider
            )
            self.assertFalse(created["reused"])
            self.assertEqual(len(provider.calls), 2)
            self.assertEqual(provider.calls[0][2], "corpus")
            self.assertEqual(provider.calls[1][2], "queries")
            self.assertEqual(
                provider.calls[0][0],
                [
                    "passage: Title zero\nBody zero",
                    "passage: Body one",
                    "passage: Title two",
                ],
            )
            self.assertEqual(
                provider.calls[1][0],
                ["query: Claim 0", "query: Claim 1", "query: Claim 2", "query: Claim 3"],
            )
            corpus = np.load(output / "corpus_embeddings.f32.npy")
            queries = np.load(output / "query_embeddings.f32.npy")
            self.assertEqual(corpus.shape, (3, 4))
            self.assertEqual(queries.shape, (4, 4))
            self.assertEqual(corpus.dtype, np.float32)
            np.testing.assert_allclose(np.linalg.norm(corpus, axis=1), 1.0, atol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(queries, axis=1), 1.0, atol=1e-6)
            cache_manifest = json.loads(created["manifest"].read_text())
            self.assertEqual(
                cache_manifest["dataset"]["manifest_fingerprint"],
                manifest["fingerprint"],
            )
            self.assertEqual(cache_manifest["model"]["revision"], config.model.revision)
            reused = build_or_load_text_embedding_cache(config, dataset, output)
            self.assertTrue(reused["reused"])
            self.assertEqual(reused["fingerprint"], created["fingerprint"])

    def test_changed_dataset_artifact_is_rejected_before_provider_call(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            dataset = directory / "dataset"
            manifest = self._write_dataset(dataset)
            config = self._write_config(
                directory / "embedding.json", manifest["fingerprint"]
            )
            with (dataset / "corpus.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            provider = _FakeProvider()
            with self.assertRaisesRegex(
                TextEmbeddingError, "dataset artifact identity mismatch"
            ):
                build_or_load_text_embedding_cache(
                    config, dataset, directory / "cache", provider=provider
                )
            self.assertEqual(provider.calls, [])

    def test_changed_request_refuses_existing_cache(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            dataset = directory / "dataset"
            manifest = self._write_dataset(dataset)
            first = self._write_config(
                directory / "first.json", manifest["fingerprint"]
            )
            output = directory / "cache"
            build_or_load_text_embedding_cache(
                first, dataset, output, provider=_FakeProvider()
            )
            changed = self._write_config(
                directory / "changed.json",
                manifest["fingerprint"],
                query_prefix="changed-query: ",
            )
            with self.assertRaisesRegex(
                TextEmbeddingError, "request fingerprint does not match"
            ):
                build_or_load_text_embedding_cache(changed, dataset, output)

    def test_tampered_array_refuses_cache_reuse(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            dataset = directory / "dataset"
            manifest = self._write_dataset(dataset)
            config = self._write_config(
                directory / "embedding.json", manifest["fingerprint"]
            )
            output = directory / "cache"
            build_or_load_text_embedding_cache(
                config, dataset, output, provider=_FakeProvider()
            )
            array_path = output / "query_embeddings.f32.npy"
            values = np.load(array_path)
            values[0, 0] += np.float32(0.25)
            np.save(array_path, values, allow_pickle=False)
            with self.assertRaisesRegex(
                TextEmbeddingError, "embedding cache artifact mismatch"
            ):
                build_or_load_text_embedding_cache(config, dataset, output)

    def test_zero_provider_output_does_not_publish_partial_cache(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            dataset = directory / "dataset"
            manifest = self._write_dataset(dataset)
            config = self._write_config(
                directory / "embedding.json", manifest["fingerprint"]
            )
            output = directory / "cache"
            with self.assertRaisesRegex(ValueError, "cannot normalize zero vectors"):
                build_or_load_text_embedding_cache(
                    config, dataset, output, provider=_FakeProvider(zero=True)
                )
            self.assertFalse(output.exists())

    def test_checked_in_config_pins_model_and_dataset(self):
        config = load_text_embedding_config(
            ROOT / "configs" / "real_scifact_e5_base_v2_embeddings.json"
        )
        self.assertEqual(config.model.name, "intfloat/e5-base-v2")
        self.assertEqual(
            config.model.revision,
            "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        )
        self.assertEqual(config.model.embedding_dimension, 768)
        self.assertEqual(config.formatting.corpus_prefix, "passage: ")
        self.assertEqual(config.formatting.query_prefix, "query: ")
        self.assertFalse(config.encoding.allow_tf32)
        self.assertEqual(config.encoding.attention_implementation, "eager")
        self.assertEqual(config.encoding.cublas_workspace_config, ":4096:8")
        self.assertEqual(
            config.config_fingerprint,
            "705153fdd5110981e1bb0f37c7007064b851c50af722a34f41e6c2050e077af7",
        )
        self.assertEqual(
            config.dataset_manifest_fingerprint,
            "4a73586d3a29a0567287e501ac3c06c998af661cdc74dbc589e7525a7924f903",
        )


if __name__ == "__main__":
    unittest.main()
