"""CPU/offline scientific and ownership regressions for the approved B0 scope."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

from mechanism import tls_rag_step5_route6_real_geometry_v1 as geometry
from mechanism import tls_rag_step5_route6_real_audit_v1 as audit
from scripts import run_tls_rag_step5_route6_real_geometry_v1 as runner


class Response(io.BytesIO):
    def __init__(self, body, start, total, status=206):
        super().__init__(body)
        self.status = status
        self.headers = {'Content-Range': 'bytes %d-%d/%d' % (start, total-1, total)}


class RealGeometryTests(unittest.TestCase):
    def fixture(self, root):
        cfg = copy.deepcopy(runner.config())
        cfg.update(queries=4, discovery_queries=2, k=4, near_boundary=2, half_width=1,
                   far_fractions=[.2, .4, .7], far_samples=[2]*5, block=31)
        cfg['source'].update(rows=512, dimension=8)
        rng = np.random.default_rng(19)
        x, q = rng.normal(size=(512, 8)), rng.normal(size=(12, 8))
        x = (x/np.linalg.norm(x, axis=1)[:, None]).astype('float32')
        q = (q/np.linalg.norm(q, axis=1)[:, None]).astype('float32')
        path = root/'input.hdf5'
        with h5py.File(path, 'w') as f:
            f.attrs['distance'], f.attrs['dimension'] = 'normalized', 8
            f['train'], f['learn'] = x, q
            f['test'] = np.zeros((2, 8), dtype='float32')
            f['neighbors'] = np.zeros((2, 100), dtype='int64')
            f['learn_neighbors'] = np.zeros((12, 100), dtype='int64')
        cfg['source'].update(bytes=path.stat().st_size, sha256=audit.sha(path))
        return cfg, path, x, q

    def transport(self, root, **updates):
        body = b'original-prefix' + b'-remaining-body'
        old = root/'old.partial'
        old.write_bytes(body[:15])
        spec = dict(bytes=len(body), partial_bytes=15, sha256=hashlib.sha256(body).hexdigest(), urls=['https://example.invalid/fixed'])
        spec.update(updates)
        limits = dict(network_bytes=1024, prepare_seconds=10)
        return old, root/'new.hdf5', spec, limits, body

    def test_full_l2_and_ip_match_independent_direct_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, path, x, q = self.fixture(Path(temp))
            with h5py.File(path) as f:
                result = geometry.exact_geometry_search(f['train'], q[0], 17)
            direct = np.array([sum((float(a)-float(b))**2 for a, b in zip(row, q[0])) for row in x])
            np.testing.assert_allclose(result['d2'], direct, rtol=2e-15, atol=1e-15)
            np.testing.assert_array_equal(result['order'], np.lexsort((np.arange(len(x)), direct)))
            np.testing.assert_allclose(result['native_ip'], -(x.astype(float)@q[0].astype(float)))

    def test_stored_norm_ip_l2_counterexample_is_not_rewritten(self):
        q = np.array([1., 0.])
        x = np.array([[1.+1e-6, 0.], [1., 0.]])
        before = x.copy()
        result = geometry.exact_geometry_search(x, q, 1)
        self.assertEqual(result['order'].tolist(), [1, 0])
        self.assertEqual(result['ip_order'].tolist(), [0, 1])
        self.assertEqual(result['zero_rows'].tolist(), [1])
        np.testing.assert_array_equal(x, before)

    def test_exact_ties_keep_row_id_order(self):
        x = np.array([[0., 1.], [0., -1.], [0., 1.]])
        self.assertEqual(geometry.exact_geometry_search(x, [1., 0.])['order'].tolist(), [0, 1, 2])

    def test_nonfinite_and_bad_norm_fail_closed(self):
        for x in (np.array([[np.nan, 0.]]), np.array([[2., 0.]])):
            with self.assertRaises(ValueError):
                geometry.exact_geometry_search(x, [1., 0.])
        with self.assertRaises(ValueError):
            geometry.exact_geometry_search(np.eye(2), [0., 0.])

    def test_query_decomposition_exact_heterogeneous_and_correlated_residual(self):
        q = np.array([1., 0., 0.])
        x = np.array([[0., 1., 0.], [0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
        result = geometry.query_direction_decomposition(x, q)
        np.testing.assert_allclose(result['gram'], result['residual_gram']+np.outer(result['loading'], result['loading']))
        np.testing.assert_allclose(result['loading'], -result['d2']/2)
        self.assertEqual(result['residual_gram'][0, 2], -1.)
        self.assertGreaterEqual(np.linalg.eigvalsh(result['gram'])[0], -1e-14)
        self.assertEqual(result['d2'][-1], 0.)

    def test_query_direction_uses_actual_norms(self):
        x, q = np.array([[2., 3.], [4., -1.]]), np.array([2., 0.])
        result = geometry.query_direction_decomposition(x, q)
        np.testing.assert_allclose(result['loading'], [0., 2.])
        self.assertGreater(np.max(np.abs(result['loading']+result['d2']/2)), 1)
        with self.assertRaises(ValueError):
            geometry.query_direction_decomposition(x, np.zeros(2))

    def test_negative_signed_angles_are_retained(self):
        t = .4
        x = [[np.cos(t), np.sin(t)], [np.cos(t), -np.sin(t)]]
        local = geometry.query_direction_decomposition(x, [1., 0.])
        self.assertLess(geometry.correlation(local['gram'])[0, 1], -.8)
        self.assertEqual(local['gram'].shape, (2, 2))

    def test_gaussian_difference_variance_coefficient_algebra(self):
        # Independent analytic two-vector quadratic form: trace((vv'-uu')^2).
        u, v = np.array([1., 2., -1.]), np.array([2., 0., 3.])
        a = np.outer(v, v)-np.outer(u, u)
        expected = np.dot(u, u)**2+np.dot(v, v)**2-2*np.dot(u, v)**2
        self.assertAlmostEqual(np.trace(a@a), expected)
        self.assertGreater(expected, 0)

    def test_zero_and_singular_geometry_is_reported_without_jitter(self):
        local = geometry.query_direction_decomposition(np.tile([1., 0.], (3, 1)), [1., 0.])
        sample = dict(groups=['G', 'B', 'F0'], pair_inclusion=np.ones((3, 3)))
        report = geometry.describe_geometry(local, sample)
        self.assertEqual(report['zero_displacements'], 3)
        self.assertEqual(report['difference_variance_coefficient']['zero_off_diagonal_count'], 3)
        self.assertEqual(report['blocks']['G-B']['rho']['count'], 0)
        self.assertEqual(report['local_factor_diagnostics'], [])
        np.testing.assert_array_equal(local['gram'], np.zeros((3, 3)))
        json.dumps(report, allow_nan=False)

    def test_sampling_covers_population_and_full_gt(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, _, _, _ = self.fixture(Path(temp))
            sample = geometry.sample_geometry(np.arange(512)[::-1], 'q', cfg)
            self.assertEqual(sample['ranks'][:4], [1, 2, 3, 4])
            self.assertEqual(sample['groups'][:4], ['G']*4)
            self.assertEqual(sum(r['population'] for r in sample['strata'])+sample['gt']+sample['boundary'], 512)
            self.assertEqual(len(sample['rows']), len(set(sample['rows'])))
            self.assertEqual(len(sample['strata']), 5)
            self.assertGreater(sample['strata'][0]['population'], 2)

    def test_real_plan_count_and_boundary(self):
        cfg = runner.config()
        k = cfg['k']
        b = set(range(k+1, k+cfg['near_boundary']+1))
        for f in cfg['budget_factors']:
            b.update(range(k*f-16, k*f+17))
        self.assertEqual(len(b), 148)
        self.assertEqual(max(b), 1616)
        self.assertEqual(k+len(b)+sum(cfg['far_samples']), 312)

    def test_same_stratum_joint_inclusion_is_not_product(self):
        strata = [dict(label='F0', population=10, sample=2, inclusion=.2)]
        pi = geometry.pair_inclusion(['G', 'B', 'F0', 'F0'], strata)
        self.assertEqual(pi[0, 1], 1.)
        self.assertEqual(pi[0, 2], .2)
        self.assertAlmostEqual(pi[2, 3], 1/45)
        self.assertNotAlmostEqual(pi[2, 3], .2**2)
        self.assertEqual(pi[2, 2], .2)

    def test_sampling_replay_and_bad_permutation(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, _, _, _ = self.fixture(Path(temp))
            a = geometry.sample_geometry(np.arange(512), 'q', cfg)
            b = geometry.sample_geometry(np.arange(512), 'q', cfg)
            np.testing.assert_array_equal(a['rows'], b['rows'])
            bad = np.arange(512)
            bad[0] = bad[1]
            with self.assertRaises(ValueError):
                geometry.sample_geometry(bad, 'q', cfg)
            cfg['max_points'] = 4
            with self.assertRaises(ValueError):
                geometry.sample_geometry(np.arange(512), 'q', cfg)

    def test_source_complete_hash_and_metric_gates(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, path, _, _ = self.fixture(Path(temp))
            receipt = audit.validate_source(path, cfg)
            self.assertFalse(receipt['test_payload_read'])
            wrong = copy.deepcopy(cfg)
            wrong['source']['sha256'] = '0'*64
            with self.assertRaises(ValueError):
                audit.validate_source(path, wrong)
            wrong = copy.deepcopy(cfg)
            wrong['source']['metric'] = 'ip'
            with self.assertRaises(ValueError):
                audit.validate_source(path, wrong)

    def test_forbidden_query_dataset_has_no_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, path, _, _ = self.fixture(Path(temp))
            cfg['query_dataset'] = 'test'
            with self.assertRaises(ValueError):
                audit.validate_source(path, cfg)

    def test_duplicate_vectors_and_signed_zero_do_not_cross_roles(self):
        cfg = runner.config()
        cfg.update(queries=2, discovery_queries=1, block=2)
        learn = np.array([[1., 0.], [1., -0.], [0., 1.]], dtype='float32')
        roles = audit.freeze_roles(learn, cfg)
        self.assertEqual(roles['distinct_vector_groups'], 2)
        self.assertEqual(len({r['vector_sha256'] for r in roles['queries']}), 2)
        self.assertNotIn(1, [r['row'] for r in roles['queries']])
        self.assertFalse(roles['semantic_independence_verified'])
        self.assertEqual(audit.freeze_roles(learn, cfg), roles)

    def test_insufficient_learn_groups_stops(self):
        cfg = runner.config()
        with self.assertRaises(ValueError):
            audit.freeze_roles(np.tile([1., 0.], (200, 1)), cfg)

    def test_range_copy_preserves_old_partial_and_full_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            old, dest, spec, limits, body = self.transport(Path(temp))
            opener = mock.Mock(return_value=Response(body[15:], 15, len(body)))
            result = audit.prepare_source(old, dest, spec, limits, opener=opener)
            self.assertEqual(dest.read_bytes(), body)
            self.assertEqual(old.read_bytes(), body[:15])
            self.assertEqual(result['network_received_bytes'], len(body)-15)
            self.assertEqual(opener.call_args[0][0].headers['Range'], 'bytes=15-')
            with self.assertRaises(FileExistsError):
                audit.prepare_source(old, dest, spec, limits, opener=opener)

    def test_range_ignored_rejected_without_consuming_body(self):
        with tempfile.TemporaryDirectory() as temp:
            old, dest, spec, limits, body = self.transport(Path(temp))
            response = Response(body, 15, len(body), status=200)
            with self.assertRaises(RuntimeError):
                audit.prepare_source(old, dest, spec, limits, opener=lambda *a, **k: response)
            self.assertEqual(dest.read_bytes(), body[:15])
            self.assertEqual(old.read_bytes(), body[:15])

    def test_range_truncated_and_wrong_hash_remain_failed(self):
        for truncated in (True, False):
            with self.subTest(truncated=truncated), tempfile.TemporaryDirectory() as temp:
                old, dest, spec, limits, body = self.transport(Path(temp))
                data = body[15:-1] if truncated else b'x'*len(body[15:])
                with self.assertRaises(ValueError):
                    audit.prepare_source(old, dest, spec, limits, opener=lambda *a, **k: Response(data, 15, len(body)))
                self.assertEqual(old.read_bytes(), body[:15])
                self.assertTrue(dest.exists())

    def test_network_cap_and_deadline_fail_before_science(self):
        with tempfile.TemporaryDirectory() as temp:
            old, dest, spec, limits, _ = self.transport(Path(temp))
            limits['network_bytes'] = 1
            opener = mock.Mock()
            with self.assertRaises(ValueError):
                audit.prepare_source(old, dest, spec, limits, opener=opener)
            opener.assert_not_called()
        with tempfile.TemporaryDirectory() as temp:
            old, dest, spec, limits, _ = self.transport(Path(temp))
            def stop():
                raise TimeoutError('fixture')
            with self.assertRaises(TimeoutError):
                audit.prepare_source(old, dest, spec, limits, check=stop)

    def test_tiny_end_to_end_forbidden_payload_traps_and_saved_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, path, x, q = self.fixture(root)
            out = root/'audit'
            out.mkdir()
            getitem = h5py.Dataset.__getitem__
            def guarded(dataset, key, *args, **kwargs):
                if dataset.name not in ('/train', '/learn'):
                    raise AssertionError('closed payload: '+dataset.name)
                return getitem(dataset, key, *args, **kwargs)
            with mock.patch.object(h5py.Dataset, '__getitem__', guarded):
                result = audit.run_geometry_audit(path, out, cfg)
            self.assertEqual(result['queries_completed'], 4)
            self.assertEqual(result['query_role_counts'], {'geometry_discovery': 2, 'geometry_check': 2})
            self.assertFalse(result['cuda_called'])
            self.assertEqual(result['new_mc_samples'], 0)
            first = json.loads((out/'queries/000/record.json').read_text())
            saved = np.load(out/'queries/000/geometry.npz', allow_pickle=False)
            row = first['query']['row']
            rows = [r['row'] for r in first['corpus_id_map']]
            u = x[rows].astype(float)-q[row].astype(float)
            np.testing.assert_allclose(saved['gram'], u@u.T)
            expected = geometry.stable_order(np.sum((x.astype(float)-q[row])**2, axis=1))[:4]
            self.assertEqual(rows[:4], expected.tolist())
            self.assertTrue((out/'rho_margin.svg').read_text().endswith('</g></svg>'))
            with self.assertRaises(FileExistsError):
                audit.run_geometry_audit(path, out, cfg)

    def test_closed_product_index_uses_final_progress_and_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('source_freeze.json', 'execution.json', 'worker.log', 'progress.jsonl'):
                (root/name).write_text('initial')
            with (root/'progress.jsonl').open('a') as stream:
                stream.write('final closed progress')
            (root/'source').mkdir()
            (root/'source/data.hdf5').write_bytes(b'input')
            runner.write_delivery_index(root)
            index = json.loads((root/'delivery_index.json').read_text())
            self.assertEqual(len(index['files']), 4)
            for entry in index['files']:
                self.assertEqual(entry['sha256'], audit.sha(root/entry['path']))
                self.assertEqual(entry['bytes'], (root/entry['path']).stat().st_size)

    def monitor_limits(self, **updates):
        limits = dict(seconds=.3, rss_bytes=1 << 40, disk_bytes=1 << 30, artifact_bytes=1 << 30)
        limits.update(updates)
        return limits

    def test_real_owned_child_stops_and_keeps_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = runner.monitor_worker([sys.executable, '-c', 'import time;print("started",flush=True);time.sleep(30)'],
                                           root, self.monitor_limits(), dict(os.environ))
            self.assertEqual(result['status'], 'resource_stop')
            self.assertEqual(result['limit'], 'seconds')
            self.assertNotEqual(result['returncode'], 0)
            self.assertGreater(result['peak_rss_bytes'], 0)
            self.assertTrue((root/'delivery_index.json').exists())

    def test_monitor_failure_is_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch.object(runner, 'process_tree', side_effect=[[(os.getpid(), 1024)], RuntimeError('fixture monitor unavailable')]):
                result = runner.monitor_worker([sys.executable, '-c', 'import time;time.sleep(30)'],
                                               root, self.monitor_limits(), dict(os.environ))
            self.assertEqual(result['status'], 'monitor_failed')
            self.assertNotEqual(result['returncode'], 0)

    def test_stable_string_ids_bind_complete_source_and_dataset(self):
        a = audit.stable_id('a'*64, 'train', 2)
        self.assertLess(a, audit.stable_id('a'*64, 'train', 10))
        self.assertNotEqual(a, audit.stable_id('a'*64, 'learn', 2))
        self.assertNotEqual(a, audit.stable_id('b'*64, 'train', 2))
        with self.assertRaises(ValueError):
            audit.stable_id('a'*64, 'test', 2)


if __name__ == '__main__':
    unittest.main()
