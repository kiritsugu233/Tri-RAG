import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tri_rag_harness.config import ConfigError, load_config
from tri_rag_harness.embeddings import load_embedding_array, normalize_rows
from tri_rag_harness.projection import (
    CacheMetadataMismatch,
    dense_gaussian_projection,
    project_rows,
    projection_metadata,
    validate_projection_cache_metadata,
)


ROOT = Path(__file__).resolve().parents[1]


class ConfigProjectionTests(unittest.TestCase):
    def test_default_config_validates(self):
        config = load_config(ROOT / "configs" / "synthetic_mvp.json")
        self.assertEqual(config.retrieval.m_grid[0], config.retrieval.m_pilot)

    def test_invalid_projection_scale_related_budget_rejected(self):
        raw = json.loads((ROOT / "configs" / "synthetic_mvp.json").read_text())
        raw["retrieval"]["m_grid"] = [5, 20]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(raw))
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_projection_entries_have_expected_scale(self):
        m_prime = 32
        matrix = dense_gaussian_projection(m_prime, 4000, seed=41)
        self.assertAlmostEqual(float(np.mean(matrix)), 0.0, delta=0.004)
        self.assertAlmostEqual(float(np.std(matrix)), 1.0 / np.sqrt(m_prime), delta=0.002)

    def test_projected_vectors_are_not_renormalized(self):
        rng = np.random.default_rng(9)
        vectors = normalize_rows(rng.normal(size=(80, 24)))
        matrix = dense_gaussian_projection(8, 24, seed=5)
        projected = project_rows(vectors, matrix)
        norms = np.linalg.norm(projected, axis=1)
        self.assertGreater(float(np.std(norms)), 0.05)
        self.assertFalse(np.allclose(norms, 1.0))

    def test_model_free_embedding_array_ingestion(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            vectors = np.arange(12, dtype=np.float32).reshape(3, 4)
            np.save(directory / "vectors.npy", vectors)
            (directory / "ids.json").write_text(json.dumps(["a", "b", "c"]))
            table = load_embedding_array(directory / "vectors.npy", directory / "ids.json")
            self.assertEqual(table.ids.tolist(), ["a", "b", "c"])
            np.testing.assert_array_equal(table.vectors, vectors)

    def test_projection_metadata_change_invalidates_cache(self):
        cached = projection_metadata(
            dimension=32,
            m_prime=16,
            seed=1,
            normalization=True,
            embedding_model="model@1",
            corpus_hash="corpus-a",
        )
        expected = projection_metadata(
            dimension=32,
            m_prime=16,
            seed=2,
            normalization=True,
            embedding_model="model@1",
            corpus_hash="corpus-a",
        )
        with self.assertRaises(CacheMetadataMismatch):
            validate_projection_cache_metadata(cached, expected)
        validate_projection_cache_metadata(cached, dict(cached))

    def test_corpus_or_model_change_invalidates_projection_cache(self):
        cached = projection_metadata(
            dimension=32,
            m_prime=16,
            seed=1,
            normalization=True,
            embedding_model="model@1",
            corpus_hash="corpus-a",
        )
        for model, corpus_hash in (("model@2", "corpus-a"), ("model@1", "corpus-b")):
            expected = projection_metadata(
                dimension=32,
                m_prime=16,
                seed=1,
                normalization=True,
                embedding_model=model,
                corpus_hash=corpus_hash,
            )
            with self.assertRaises(CacheMetadataMismatch):
                validate_projection_cache_metadata(cached, expected)


if __name__ == "__main__":
    unittest.main()
