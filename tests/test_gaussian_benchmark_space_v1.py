"""Offline independent fixtures for the approved benchmark namespace."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import faiss
import h5py
import numpy as np

from mechanism.gaussian_benchmark_data_v1 import (BenchmarkData, fingerprint, native_distances,
    native_recall, select_roles, stable_id, strict_retention, valid_ids, verify_source)
from mechanism.gaussian_benchmark_space_v1 import (build_backend, decompose_loss, exact_candidates,
    oracle_frontier, project, projection_matrix, rerank, run_backend, top_ids)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gaussian_runner', ROOT/'scripts/run_gaussian_benchmark_space_v1.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class GaussianBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'tiny.hdf5'
        self.config = json.loads((ROOT/'configs/gaussian_benchmark_space_v1.json').read_text())
        rng = np.random.default_rng(17)
        self.x = rng.normal(size=(128, 8)).astype(np.float32)
        self.x /= np.linalg.norm(self.x, axis=1)[:, None]
        self.q = rng.normal(size=(12, 8)).astype(np.float32)
        self.q /= np.linalg.norm(self.q, axis=1)[:, None]
        truth = np.argsort(1-self.q@self.x.T, axis=1)[:, :4]
        with h5py.File(self.path, 'w') as f:
            f.attrs['distance'] = 'normalized'
            f['train'] = self.x
            f['learn'] = self.q[:9]
            f['test'] = self.q[9:]
            f['learn_neighbors'] = truth[:9]
            f['neighbors'] = truth[9:]
            f['distances'] = np.take_along_axis(1-self.q[9:]@self.x.T, truth[9:], axis=1)
        self.config['source'].update(n=128, dimension=8, learn=9, test=3)
        self.config.update(k=4, development_count=4)
        self.rehash()

    def tearDown(self):
        self.temp.cleanup()

    def rehash(self):
        self.config['source'].update(bytes=self.path.stat().st_size, sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_source_hash_and_size_fail_closed(self):
        src = self.config['source']
        self.assertEqual(verify_source(self.path, src['bytes'], src['sha256']), src['sha256'])
        for size, digest in [(src['bytes']+1, src['sha256']), (src['bytes'], '0'*64)]:
            with self.assertRaises(ValueError):
                verify_source(self.path, size, digest)

    def test_full_corpus_shape_and_metric_not_rewritten(self):
        for change in ('shape', 'metric'):
            cfg = copy.deepcopy(self.config)
            if change == 'shape':
                cfg['source']['n'] -= 1
            else:
                with h5py.File(self.path, 'r+') as f:
                    f.attrs['distance'] = 'euclidean'
                self.rehash()
                cfg = copy.deepcopy(self.config)
            with self.assertRaises(ValueError):
                BenchmarkData(self.path, cfg)

    def test_roles_stable_and_disjoint_with_prefixes(self):
        roles = select_roles(20, 5, 4, 13)
        self.assertEqual(roles, select_roles(20, 5, 4, 13))
        self.assertFalse(set(roles['development']) & set(roles['reserved_learn']))
        self.assertEqual(set(roles['development']) | set(roles['reserved_learn']), set(range(20)))
        self.assertNotEqual(stable_id('learn', 0), stable_id('test', 0))
        self.assertNotEqual(roles, select_roles(20, 5, 4, 14))

    def test_test_access_requires_bound_seal_and_reserved_closed(self):
        data = BenchmarkData(self.path, self.config)
        try:
            for role, seal in [('official_test', None), ('reserved_learn', None), ('official_test', {'points': [1]})]:
                with self.assertRaises(ValueError):
                    data.queries(role, seal)
            q, truth, threshold, ids = data.queries('development')
            self.assertEqual(q.shape, (4, 8))
            self.assertIsNone(threshold)
            self.assertTrue(all(':learn:' in i for i in ids))
            seal = {'source_sha256': data.source_sha, 'config_sha256': data.config_hash, 'points': [{'index': 'fixture'}]}
            self.assertEqual(data.queries('official_test', seal)[0].shape, (3, 8))
        finally:
            data.close()

    def test_native_threshold_recall_not_identity_recall(self):
        self.assertEqual(native_recall(np.array([.1005, .1009]), .1, 2), 1)
        self.assertEqual(strict_retention([4, 5], [0, 1], 2), 0)
        self.assertEqual(native_recall(np.array([.1005, .1011]), .1, 2), .5)
        with self.assertRaises(ValueError):
            native_recall([np.nan], .1, 1)

    def test_duplicate_negative_and_out_of_range_candidates_rejected(self):
        for ids in ([0, 0], [-1, 0], [0, 128], [0.0, 1.0]):
            with self.assertRaises(ValueError):
                valid_ids(ids, 128, 2)

    def test_native_distance_preserves_provided_norms(self):
        x = np.array([[2., 0.], [0., 2.], [-2., 0.]], np.float32)
        np.testing.assert_array_equal(native_distances(np.array([1., 0.]), x), [0, 1, 2])
        np.testing.assert_array_equal(x[:, 0], [2, 0, -2])

    def test_projection_scale_no_renormalization_and_seed_replay(self):
        matrix = projection_matrix(1024, 128, 41)
        self.assertLess(abs(float(matrix.var())-1/128), .00015)
        np.testing.assert_array_equal(matrix, projection_matrix(1024, 128, 41))
        x = np.eye(1024, dtype=np.float32)[:3]*2
        np.testing.assert_array_equal(project(x, matrix), matrix[:3]*2)
        self.assertFalse(np.allclose(np.linalg.norm(project(x, matrix), axis=1), 1))
        self.assertFalse(np.array_equal(matrix, projection_matrix(1024, 128, 42)))

    def test_exact_l2_against_direct_subtraction_with_block_ties(self):
        x = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [2, 0]], np.float32)
        q = np.array([[0, 0], [.2, .3]], np.float32)
        actual, distances = exact_candidates(x, q, 3, block_rows=2)
        for j in range(2):
            brute = ((x-q[j])**2).sum(axis=1)
            expected = np.lexsort((np.arange(len(x)), brute))[:3]
            np.testing.assert_array_equal(actual[j], expected)
            np.testing.assert_allclose(distances[j], brute[expected], atol=1e-6)

    def test_exact_native_and_rerank_against_full_reference(self):
        ids, _ = exact_candidates(self.x, self.q[:2], 6, 'native', 17)
        for j in range(2):
            brute = np.argsort(1-self.x@self.q[j])[:6]
            np.testing.assert_array_equal(ids[j], brute)
            reranked, _ = rerank(self.q[j], self.x, ids[j][::-1], 4)
            np.testing.assert_array_equal(reranked, brute[:4])

    def test_exact_candidate_retention_equals_rerank_and_is_monotone(self):
        matrix = projection_matrix(8, 4, 4)
        ids, _ = exact_candidates(project(self.x, matrix), project(self.q[:1], matrix), 32)
        gt = np.argsort(1-self.x@self.q[0])[:4]
        scores = []
        for m in [4, 8, 16, 32]:
            ranked, _ = rerank(self.q[0], self.x, ids[0, :m], 4)
            score = strict_retention(ranked, gt, 4)
            self.assertEqual(score, strict_retention(ids[0, :m], gt, 4))
            scores.append(score)
        self.assertEqual(scores, sorted(scores))

    def test_ann_can_recover_exact_projection_misses(self):
        d = decompose_loss([0, 1], [0, 3], [0, 1], 2)
        self.assertEqual(d, {'representation_loss': .5, 'ann_removed': 0, 'ann_recovered': .5, 'actual_loss': 0})
        d = decompose_loss([0, 1, 2, 3], [0, 1, 7, 8], [1, 2, 3, 8], 4)
        self.assertEqual(d['actual_loss'], d['representation_loss']+d['ann_removed']-d['ann_recovered'])

    def test_oracle_global_mean_not_per_query_threshold(self):
        result = oracle_frontier([[1], [.9]], [[2], [4]], .95)
        self.assertEqual(result['status'], 'randomized_lp_lower_bound')
        self.assertAlmostEqual(result['cost_lower_bound_seconds'], 3)
        self.assertEqual(oracle_frontier([[1], [.8]], [[2], [4]], .95)['status'], 'quality_infeasible')

    def test_oracle_dual_certificate_against_hand_randomized_solution(self):
        result = oracle_frontier([[0, 1], [0, 1]], [[0, 2], [0, 4]], .75)
        self.assertAlmostEqual(result['cost_lower_bound_seconds'], 2)
        self.assertAlmostEqual(result['primal_cost_seconds'], 2)
        self.assertAlmostEqual(result['duality_gap_seconds'], 0)
        with self.assertRaises(ValueError):
            oracle_frontier([[1]], [[-1]], .9)

    def test_fingerprint_binds_seed_metric_and_roles(self):
        base = fingerprint(self.config)
        for key, value in [('k', 3), ('development_count', 3), ('native_epsilon', 0)]:
            cfg = copy.deepcopy(self.config)
            cfg[key] = value
            self.assertNotEqual(base, fingerprint(cfg))
        cfg = copy.deepcopy(self.config)
        cfg['seeds']['projection'][0] += 1
        self.assertNotEqual(base, fingerprint(cfg))

    def test_actual_hnsw_cost_components_and_save_load_replay(self):
        self.config['limits']['offline_threads'] = 1
        self.config['hnsw']['ef_construction'] = 40
        index, build = build_backend(self.x, self.config, {'family': 'original_hnsw', 'degree': 8})
        faiss.omp_set_num_threads(1)
        action = {'M': 8, 'ef_search': 128}
        a = run_backend(index, self.q[0], self.x, action, 4)
        faiss.write_index(index, str(Path(self.temp.name)/'index'))
        b = run_backend(faiss.read_index(str(Path(self.temp.name)/'index')), self.q[0], self.x, action, 4)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[0], np.argsort(1-self.x@self.q[0])[:4])
        timing = a[3]
        self.assertAlmostEqual(timing['total_seconds'], sum(timing[name] for name in ['projection_seconds', 'search_seconds', 'selection_seconds', 'rerank_seconds']))
        self.assertTrue(all(v >= 0 for v in timing.values()))
        self.assertGreater(build, 0)

    def test_projected_hnsw_real_backend_and_paid_projection(self):
        self.config['limits']['offline_threads'] = 1
        matrix = projection_matrix(8, 4, 5)
        index, _ = build_backend(self.x, self.config, {'family': 'gaussian_hnsw', 'degree': 8}, project(self.x, matrix))
        faiss.omp_set_num_threads(1)
        result, _, candidates, timing = run_backend(index, self.q[0], self.x, {'M': 12, 'ef_search': 128}, 4, matrix)
        expected, _ = rerank(self.q[0], self.x, candidates, 4)
        np.testing.assert_array_equal(result, expected)
        self.assertGreater(timing['projection_seconds'], 0)

    def test_real_fastscan_backend_and_exact_refine(self):
        self.config['limits']['offline_threads'] = 1
        self.config['ivfpq'].update(nlist=2, subquantizers=2, training_count=128)
        index, _ = build_backend(self.x, self.config, {'family': 'ivfpq'})
        faiss.omp_set_num_threads(1)
        ids, distances, candidates, _ = run_backend(index, self.q[0], self.x, {'M': 16, 'nprobe': 2}, 4)
        expected, d = rerank(self.q[0], self.x, candidates, 4)
        np.testing.assert_array_equal(ids, expected)
        np.testing.assert_array_equal(distances, d)

    def test_resource_bounds_and_artifact_no_overwrite(self):
        limits = {'seconds': 10, 'rss_bytes': 20, 'artifact_bytes': 30, 'temporary_bytes': 40}
        self.assertIsNone(runner.stop_reason(10, 20, 30, 40, limits))
        for values, expected in [((11, 0, 0, 0), 'seconds'), ((0, 21, 0, 0), 'rss_bytes'), ((0, 0, 31, 0), 'artifact_bytes'), ((0, 0, 0, 41), 'temporary_bytes')]:
            self.assertEqual(runner.stop_reason(*values, limits), expected)
        p = Path(self.temp.name)/'receipt.json'
        runner.write_json(p, {'status': 'failed'})
        with self.assertRaises(FileExistsError):
            runner.write_json(p, {'status': 'complete'})

    def test_block_search_checks_resource_stop(self):
        def stop():
            raise TimeoutError('fixture limit')
        with self.assertRaises(TimeoutError):
            exact_candidates(self.x, self.q[:1], 4, check=stop)

    def test_watchdog_terminates_real_owned_child(self):
        import subprocess
        import sys
        import time
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
        try:
            time.sleep(.05)
            self.assertGreater(runner.process_rss(child.pid), 0)
            runner.terminate_owned(child)
            self.assertIsNotNone(child.poll())
        finally:
            runner.terminate_owned(child)

    def test_monitor_failure_stops_worker_and_records_failure(self):
        import subprocess
        import sys
        from unittest.mock import patch
        real_popen = subprocess.Popen
        children = []
        def spawn(*args, **kwargs):
            child = real_popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
            children.append(child)
            return child
        out = Path(self.temp.name)/'monitor-failure'
        with patch.object(runner.subprocess, 'run'), patch.object(runner.subprocess, 'check_output', side_effect=['fixture-commit', '']), patch.object(runner.subprocess, 'Popen', side_effect=spawn), patch.object(runner, 'process_rss', side_effect=RuntimeError('injected monitor failure')):
            self.assertEqual(runner.supervise(self.path, out, self.config), 2)
        self.assertIsNotNone(children[0].poll())
        receipt = json.loads((out/'execution.json').read_text())
        self.assertEqual(receipt['status'], 'monitor_failed')
        self.assertIn('injected monitor failure', receipt['monitor_error'])

    def test_tiny_end_to_end_records_and_seals(self):
        cfg = copy.deepcopy(self.config)
        cfg.update(dimensions=[4], budgets=[4, 8], targets=[.1], reference_audit_count=2,
                   development_repeats=1, test_repeats=1, block_rows=64, query_batch=2)
        cfg['seeds']['projection'] = [63013, 63017]
        cfg['limits']['offline_threads'] = 1
        cfg['hnsw'].update(original_degree=[4], projected_degree=4, ef_construction=20)
        cfg['ivfpq'].update(nlist=2, subquantizers=2, training_count=128, nprobe=[2], refine=[8])
        out = Path(self.temp.name)/'run'
        out.mkdir()
        runner.worker(self.path, out, cfg)
        result = json.loads((out/'result.json').read_text())
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['test']), 3)
        self.assertTrue((out/'deployment_seal.json').exists())
        for path in out.glob('test-*.jsonl'):
            for line in path.read_text().splitlines():
                record = json.loads(line)
                self.assertIn(':test:', record['query_id'])
                self.assertEqual(record['timing']['risk_seconds'], 0)
        self.assertEqual(len((out/'corpus_id_map.tsv').read_text().splitlines()), 129)


if __name__ == '__main__':
    unittest.main()
