import unittest

import numpy as np

from tri_rag_harness.embeddings import normalize_rows
from tri_rag_harness.indexes import ExactSquaredL2Index, StreamingExactSquaredL2Index


class ExactIndexTests(unittest.TestCase):
    def test_exact_top_k_matches_brute_force(self):
        rng = np.random.default_rng(33)
        corpus = normalize_rows(rng.normal(size=(30, 12)))
        queries = normalize_rows(rng.normal(size=(7, 12)))
        ids = np.asarray([f"doc-{row:03d}" for row in range(len(corpus))])
        result = ExactSquaredL2Index(ids, corpus, batch_size=3).search(queries, 6)
        for query_row, query in enumerate(queries):
            distances = np.sum((corpus - query) ** 2, axis=1)
            expected = np.lexsort((ids, distances))[:6]
            np.testing.assert_array_equal(result.rows[query_row], expected)
            np.testing.assert_allclose(
                result.squared_distances[query_row], distances[expected], atol=1e-13
            )

    def test_ties_are_broken_by_stable_id(self):
        vectors = np.asarray([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
        ids = ["doc-b", "doc-a", "doc-c"]
        result = ExactSquaredL2Index(ids, vectors).search(np.asarray([[1.0, 0.0]]), 2)
        self.assertEqual(result.ids[0].tolist(), ["doc-a", "doc-b"])

    def test_candidate_overlap_equals_exact_rerank_overlap(self):
        rng = np.random.default_rng(8)
        corpus = normalize_rows(rng.normal(size=(40, 9)))
        query = normalize_rows(rng.normal(size=(1, 9)))[0]
        ids = np.asarray([f"doc-{row:03d}" for row in range(len(corpus))])
        exact = ExactSquaredL2Index(ids, corpus).search(query, 5).rows[0]
        candidate_rows = np.asarray([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
        candidate_distances = np.sum((corpus[candidate_rows] - query) ** 2, axis=1)
        order = np.lexsort((ids[candidate_rows], candidate_distances))
        candidate_retention = len(set(exact) & set(candidate_rows)) / 5
        rerank_retention = len(set(exact) & set(candidate_rows[order[:5]])) / 5
        self.assertEqual(candidate_retention, rerank_retention)

    def test_streaming_top_k_matches_brute_force_and_counts_scans(self):
        rng = np.random.default_rng(101)
        corpus = normalize_rows(rng.normal(size=(53, 11))).astype(np.float32)
        query = normalize_rows(rng.normal(size=(1, 11))).astype(np.float32)[0]
        norms = np.einsum("ij,ij->i", corpus, corpus)
        index = StreamingExactSquaredL2Index(
            corpus, squared_norms=norms, block_rows=7
        )
        result = index.search_one(query, 9)
        distances = np.maximum(
            float(np.dot(query, query)) + norms - 2.0 * (corpus @ query), 0.0
        )
        expected = np.lexsort((np.arange(len(corpus)), distances))[:9]
        np.testing.assert_array_equal(result.rows, expected)
        np.testing.assert_allclose(
            result.squared_distances, distances[expected], atol=1e-6
        )
        self.assertEqual(index.scan_calls, 1)
        self.assertEqual(result.distance_evaluations, len(corpus))
        self.assertEqual(result.scanned_vector_bytes, corpus.nbytes)


if __name__ == "__main__":
    unittest.main()
