import unittest

import numpy as np

from tri_rag_harness.embeddings import normalize_rows
from tri_rag_harness.indexes import (
    ExactSquaredL2Index,
    FaissBoundaryTieError,
    FaissExactSquaredL2Index,
    FaissUnavailableError,
    StreamingExactSquaredL2Index,
)


class _FakeIndexFlatL2:
    def __init__(self, dimension):
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self):
        return len(self.vectors)

    def add(self, vectors):
        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("invalid fake FAISS vectors")
        self.vectors = np.concatenate([self.vectors, values])

    def search(self, queries, k):
        values = np.asarray(queries, dtype=np.float32)
        distances = np.sum(
            (values[:, None, :] - self.vectors[None, :, :]) ** 2,
            axis=2,
            dtype=np.float32,
        )
        rows = np.argsort(distances, axis=1, kind="stable")[:, :k]
        selected = np.take_along_axis(distances, rows, axis=1)
        return selected.astype(np.float32), rows.astype(np.int64)


class _FakeFaiss:
    __version__ = "test-double"
    IndexFlatL2 = _FakeIndexFlatL2


class _FakeGpuResources:
    pass


class _FakeGpuFaiss(_FakeFaiss):
    resources_created = 0

    @classmethod
    def StandardGpuResources(cls):
        cls.resources_created += 1
        return _FakeGpuResources()

    @staticmethod
    def index_cpu_to_gpu(resources, device, index):
        if not isinstance(resources, _FakeGpuResources) or device != 0:
            raise ValueError("invalid fake GPU transfer")
        return index


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

    def test_faiss_cpu_adapter_matches_streaming_contract(self):
        rng = np.random.default_rng(912)
        corpus = normalize_rows(rng.normal(size=(61, 13))).astype(np.float32)
        query = normalize_rows(rng.normal(size=(1, 13))).astype(np.float32)[0]
        norms = np.einsum("ij,ij->i", corpus, corpus)
        reference = StreamingExactSquaredL2Index(
            corpus, squared_norms=norms, block_rows=9
        ).search_one(query, 11)
        index = FaissExactSquaredL2Index(
            corpus, device="cpu", faiss_module=_FakeFaiss()
        )
        result = index.search_one(query, 11)
        np.testing.assert_array_equal(result.rows, reference.rows)
        np.testing.assert_allclose(
            result.squared_distances,
            reference.squared_distances,
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertEqual(result.distance_evaluations, len(corpus))
        self.assertEqual(result.scanned_vector_bytes, corpus.nbytes)
        self.assertGreater(result.backend_search_ms, 0.0)
        self.assertEqual(result.refinement_distance_evaluations, len(corpus))
        self.assertEqual(result.requested_neighbors, len(corpus))
        self.assertEqual(index.build_metrics.backend, "faiss_cpu_index_flat_l2")
        self.assertEqual(index.build_metrics.boundary_tie_overfetch, 64)

    def test_faiss_adapter_resolves_bounded_ties_and_refuses_unclosed_tie_band(self):
        corpus = np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=np.float32
        )
        cpu = FaissExactSquaredL2Index(
            corpus, device="cpu", faiss_module=_FakeFaiss()
        )
        result = cpu.search_one(np.asarray([1.0, 0.0], dtype=np.float32), 1)
        self.assertEqual(result.rows.tolist(), [0])
        unclosed = FaissExactSquaredL2Index(
            np.repeat(corpus[:1], 100, axis=0),
            device="cpu",
            faiss_module=_FakeFaiss(),
        )
        with self.assertRaises(FaissBoundaryTieError):
            unclosed.search_one(np.asarray([1.0, 0.0], dtype=np.float32), 1)

    def test_faiss_adapter_refuses_missing_gpu_support(self):
        corpus = np.asarray([[1.0, 0.0]], dtype=np.float32)
        with self.assertRaises(FaissUnavailableError):
            FaissExactSquaredL2Index(
                corpus,
                device="gpu",
                faiss_module=_FakeFaiss(),
                enable_torch_transfer_timing=False,
            )

    def test_faiss_gpu_indexes_can_share_one_resource_pool(self):
        rng = np.random.default_rng(914)
        original = rng.normal(size=(31, 12)).astype(np.float32)
        projected = rng.normal(size=(31, 5)).astype(np.float32)
        _FakeGpuFaiss.resources_created = 0
        first = FaissExactSquaredL2Index(
            original,
            device="gpu",
            faiss_module=_FakeGpuFaiss,
            enable_torch_transfer_timing=False,
        )
        second = FaissExactSquaredL2Index(
            projected,
            device="gpu",
            faiss_module=_FakeGpuFaiss,
            gpu_resources=first.gpu_resources,
            enable_torch_transfer_timing=False,
        )
        self.assertIs(first.gpu_resources, second.gpu_resources)
        self.assertEqual(_FakeGpuFaiss.resources_created, 1)
        self.assertFalse(first.build_metrics.gpu_resources_shared)
        self.assertTrue(second.build_metrics.gpu_resources_shared)

    def test_real_faiss_cpu_conformance_when_installed(self):
        try:
            import faiss  # type: ignore
        except ImportError:
            self.skipTest("real FAISS is not installed in the offline test environment")
        rng = np.random.default_rng(913)
        corpus = normalize_rows(rng.normal(size=(97, 17))).astype(np.float32)
        query = normalize_rows(rng.normal(size=(1, 17))).astype(np.float32)[0]
        norms = np.einsum("ij,ij->i", corpus, corpus)
        reference = StreamingExactSquaredL2Index(
            corpus, squared_norms=norms, block_rows=13
        ).search_one(query, 12)
        actual = FaissExactSquaredL2Index(
            corpus, device="cpu", faiss_module=faiss
        ).search_one(query, 12)
        np.testing.assert_array_equal(actual.rows, reference.rows)
        np.testing.assert_allclose(
            actual.squared_distances,
            reference.squared_distances,
            rtol=1e-4,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
