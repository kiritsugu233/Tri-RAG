import unittest

import numpy as np

from tri_rag_harness.embeddings import normalize_rows
from tri_rag_harness.indexes import ExactSquaredL2Index


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


if __name__ == "__main__":
    unittest.main()
