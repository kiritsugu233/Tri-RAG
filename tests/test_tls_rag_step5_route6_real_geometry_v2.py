"""CPU/offline v2 comparison, fixed-input and owned-process regressions."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

from mechanism import tls_rag_step5_route6_real_geometry_v1 as reference
from mechanism import tls_rag_step5_route6_real_audit_v1 as products
from mechanism import tls_rag_step5_route6_real_geometry_v2 as geometry
from mechanism import tls_rag_step5_route6_real_audit_v2 as audit
from scripts import run_tls_rag_step5_route6_real_geometry_v1 as cpu_runner
from scripts import run_tls_rag_step5_route6_real_geometry_v2 as runner


class OfflineFixtureBackend:
    """Explicit CPU fixture; never an available production CLI backend."""
    def __init__(self, corpus, cfg, check):
        self.x, self.cfg, self.check = corpus[:].astype(float), cfg, check
        self.identity = dict(backend='cpu-offline-fixture')
        self.closed = False

    def points(self, rows):
        return self.x[rows]

    def close(self):
        self.closed = True

    def batch(self, queries):
        for query in queries:
            row = reference.exact_geometry_search(self.x, query, self.cfg['block'], check=self.check)
            row.update(order_audit=dict(fixture=True), costs=dict(query_order_seconds=0.),
                       batch_costs=dict(batch_distance_seconds=0., batch_queries=len(queries)))
            yield row


class GeometryV2Tests(unittest.TestCase):
    def fixture(self, root):
        cfg = copy.deepcopy(runner.config())
        cfg.update(queries=4, discovery_queries=2, k=4, near_boundary=2, half_width=1,
                   far_fractions=[.2, .4, .7], far_samples=[2]*5, block=31, query_batch=3,
                   old_completed_queries=2, old_scientific_commit='synthetic-fixture-only')
        rng = np.random.default_rng(19)
        x, q = rng.normal(size=(512, 8)), rng.normal(size=(12, 8))
        x = (x/np.linalg.norm(x, axis=1)[:, None]).astype('float32')
        q = (q/np.linalg.norm(q, axis=1)[:, None]).astype('float32')
        source = root/'input.hdf5'
        with h5py.File(source, 'w') as f:
            f.attrs['distance'], f.attrs['dimension'] = 'normalized', 8
            f['train'], f['learn'] = x, q
            f['test'] = np.zeros((2, 8), dtype='float32')
            f['neighbors'] = np.zeros((2, 100), dtype='int64')
            f['learn_neighbors'] = np.zeros((12, 100), dtype='int64')
        cfg['source'].update(rows=512, dimension=8, bytes=source.stat().st_size,
                             sha256=products.sha(source))
        old = root/'old_cpu_fixture'
        old.mkdir()
        products.run_geometry_audit(source, old, cfg)
        products.json_write(old/'source_freeze.json', dict(commit=cfg['old_scientific_commit']))
        products.json_write(old/'execution.json', dict(status='resource_stop', fixture=True))
        products.json_write(old/'failure.json', dict(type='synthetic-fixture-only'))
        cpu_runner.write_delivery_index(old)
        cfg.update(old_index_sha256=products.sha(old/'delivery_index.json'),
                   roles_sha256=products.sha(old/'roles.json'))
        return cfg, source, old

    def test_transitive_overlap_clusters_and_stable_ties(self):
        e = geometry.ENVELOPE
        values = np.array([9., 0., 1.5*e, 3*e, 20.])
        exact = np.array([9., e, .5*e, 2*e, 20.])
        calls = []
        def refine(rows):
            calls.append(rows.tolist())
            return exact[rows]
        order, report = geometry.repair_order(values, refine)
        self.assertEqual(calls, [[1, 2, 3]])
        np.testing.assert_array_equal(order, np.lexsort((np.arange(5), exact)))
        self.assertEqual(report['largest_cluster'], 3)
        self.assertFalse(report['roundoff_certified'])
        values = np.array([2., 0., 0., 2.])
        order, _ = geometry.repair_order(values, lambda rows: values[rows])
        self.assertEqual(order.tolist(), [1, 2, 0, 3])

    def test_disjoint_intervals_need_no_refinement(self):
        refine = mock.Mock(side_effect=AssertionError('unexpected refinement'))
        order, report = geometry.repair_order(np.array([2., 0., 1.]), refine)
        self.assertEqual(order.tolist(), [1, 2, 0])
        self.assertEqual(report['refined_rows'], 0)

    def test_envelope_violation_and_nonfinite_refinement_fail_closed(self):
        for exact in (np.array([1., 0.]), np.array([np.nan, 0.]), np.array([0.])):
            with self.subTest(exact=exact), self.assertRaises(ArithmeticError):
                geometry.repair_order(np.zeros(2), lambda rows: exact)

    def test_frozen_envelope_and_score_type(self):
        for values in (np.array([0., np.nan]), np.zeros((2, 2)), np.array([1, 2])):
            with self.assertRaises(ValueError):
                geometry.repair_order(values, lambda rows: values[rows])
        with self.assertRaises(ValueError):
            geometry.repair_order(np.zeros(2), lambda rows: np.zeros(len(rows)), envelope=1.)

    def test_exact_two_envelope_link_and_zero_duplicates(self):
        e = geometry.ENVELOPE
        original = np.array([0., 2*e, 9., 9.])
        exact = np.array([e, e, 9., 9.])
        order, report = geometry.repair_order(original, lambda rows: exact[rows])
        self.assertEqual(order.tolist(), [0, 1, 2, 3])
        self.assertEqual(report['clusters'], 2)

    def test_cpu_direct_difference_preserves_stored_norm_and_signed_ip(self):
        x = np.array([[1.+1e-6, 0.], [1., 0.]])
        q = np.array([1., 0.])
        before = x.copy()
        l2, ip = geometry.cpu_scores(x, q, 'l2'), geometry.cpu_scores(x, q, 'ip')
        self.assertEqual(reference.stable_order(l2).tolist(), [1, 0])
        self.assertEqual(reference.stable_order(ip).tolist(), [0, 1])
        np.testing.assert_array_equal(before, x)
        with self.assertRaises(ValueError):
            geometry.cpu_scores(x, q, 'cosine')

    def test_repair_check_stops_without_changing_envelope(self):
        def stop():
            raise TimeoutError('fixture')
        with self.assertRaises(TimeoutError):
            geometry.repair_order(np.zeros(3), lambda rows: np.zeros(len(rows)), check=stop)

    def test_fixed_reference_detects_mutation_and_unsafe_index(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, _, old = self.fixture(Path(temp))
            roles, refs, receipt = audit.load_cpu_reference(old, cfg)
            self.assertEqual(len(refs), 2)
            self.assertEqual(receipt['roles_sha256'], cfg['roles_sha256'])
            (old/'failure.json').write_text('changed')
            with self.assertRaises(ValueError):
                audit.load_cpu_reference(old, cfg)
            index = json.loads((old/'delivery_index.json').read_text())
            index['files'][0]['path'] = '../unsafe'
            (old/'delivery_index.json').write_text(json.dumps(index))
            cfg['old_index_sha256'] = products.sha(old/'delivery_index.json')
            with self.assertRaises(ValueError):
                audit.load_cpu_reference(old, cfg)

    def test_fixed_roles_hash_and_commit_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, _, old = self.fixture(Path(temp))
            for key in ('roles_sha256', 'old_index_sha256', 'old_scientific_commit'):
                wrong = copy.deepcopy(cfg)
                wrong[key] = '0'*64
                with self.subTest(key=key), self.assertRaises(ValueError):
                    audit.load_cpu_reference(old, wrong)

    def test_cpu_fixture_end_to_end_reuses_roles_handles_batch_tail_and_payload_traps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, source, old = self.fixture(root)
            out = root/'v2_fixture'
            out.mkdir()
            before = {p:products.sha(p) for p in old.rglob('*') if p.is_file()}
            instances = []
            def factory(*args):
                instances.append(OfflineFixtureBackend(*args))
                return instances[-1]
            getitem = h5py.Dataset.__getitem__
            def guarded(dataset, key, *args, **kwargs):
                if dataset.name not in ('/train', '/learn'):
                    raise AssertionError('closed payload: '+dataset.name)
                return getitem(dataset, key, *args, **kwargs)
            with mock.patch.object(h5py.Dataset, '__getitem__', guarded):
                result = audit.run_geometry_audit(source, old, out, cfg, backend_factory=factory,
                                                  runtime_identity=dict(fixture=True))
            self.assertEqual(result['queries_completed'], 4)
            self.assertEqual(result['cpu_reference_matches'], 2)
            self.assertEqual(result['query_role_counts'], dict(geometry_discovery=2, geometry_check=2))
            self.assertFalse(result['cuda_called'])
            self.assertEqual(result['new_mc_samples'], 0)
            self.assertEqual((out/'roles.json').read_bytes(), (old/'roles.json').read_bytes())
            self.assertTrue(instances[0].closed)
            self.assertEqual(json.loads((out/'batches/001.json').read_text())['queries'], 1)
            self.assertEqual(before, {p:products.sha(p) for p in before})
            first = json.loads((out/'queries/000/record.json').read_text())
            self.assertTrue(first['old_cpu_comparison']['exact_selected_ids_and_ranks'])
            self.assertTrue((out/'rho_margin.svg').exists())
            with self.assertRaises(FileExistsError):
                audit.run_geometry_audit(source, old, out, cfg, backend_factory=factory)

    def test_reference_comparison_rejects_rank_and_matrix_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, _, old = self.fixture(Path(temp))
            _, refs, _ = audit.load_cpu_reference(old, cfg)
            saved = next(iter(refs.values()))
            record = copy.deepcopy(saved['record'])
            with np.load(saved['geometry_path']) as data:
                local = {key:data[key].copy() for key in data.files}
            record['corpus_id_map'][0]['rank'] += 1
            with self.assertRaises(ArithmeticError):
                audit.compare_cpu_reference(record, local, saved)
            record = saved['record']
            local['gram'][0, 0] += 1e-8
            with self.assertRaises(ArithmeticError):
                audit.compare_cpu_reference(record, local, saved)

    def test_compute_deadline_before_source_open(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = runner.config()
            cfg['limits']['compute_seconds'] = 0
            with self.assertRaises(TimeoutError):
                audit.run_geometry_audit('absent', 'absent', Path(temp), cfg,
                                         backend_factory=OfflineFixtureBackend)

    def test_device_receipt_is_source_and_config_bound(self):
        cfg = runner.config()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'device.json'
            products.json_write(path, dict(sources=runner.source_map(), config=cfg, result=dict(status='passed')))
            self.assertEqual(runner.validate_device_receipt(path, cfg)['result']['status'], 'passed')
            wrong = copy.deepcopy(cfg)
            wrong['comparison_envelope'] *= 2
            with self.assertRaises(RuntimeError):
                runner.validate_device_receipt(path, wrong)

    def test_frozen_real_configuration_keeps_scientific_inputs(self):
        old, new = cpu_runner.config(), runner.config()
        for key in ('queries', 'discovery_queries', 'role_seed', 'sample_seed', 'k', 'budget_factors',
                    'near_boundary', 'half_width', 'far_fractions', 'far_samples', 'max_points',
                    'norm_tolerance', 'projection', 'query_dataset'):
            self.assertEqual(old[key], new[key])
        for key in ('bytes', 'sha256', 'rows', 'dimension', 'metric', 'revision'):
            self.assertEqual(old['source'][key], new['source'][key])
        self.assertEqual(new['limits']['gpu_bytes'], 40*2**30)
        self.assertEqual(new['limits']['compute_seconds'], 1200)

    def limits(self, **updates):
        result = dict(seconds=.3, rss_bytes=1<<40, gpu_bytes=1<<40,
                      artifact_bytes=1<<30, disk_bytes=1<<30)
        result.update(updates)
        return result

    def test_actual_owned_child_is_stopped_and_closed_index_is_current(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch.object(runner.resource_helpers, 'gpu_process_memory', return_value=0):
                result = runner.monitor_worker([sys.executable, '-c',
                    'import time;print("fixture",flush=True);time.sleep(30)'], root, self.limits(), dict(os.environ))
            self.assertEqual(result['status'], 'resource_stop')
            self.assertEqual(result['limit'], 'seconds')
            self.assertNotEqual(result['returncode'], 0)
            self.assertGreater(result['peak_rss_bytes'], 0)
            index = json.loads((root/'delivery_index.json').read_text())
            for row in index['files']:
                self.assertEqual(products.sha(root/row['path']), row['sha256'])

    def test_gpu_limit_and_monitor_failure_fail_closed(self):
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                effect = RuntimeError('fixture GPU monitor unavailable') if failure else None
                with mock.patch.object(runner.resource_helpers, 'gpu_process_memory',
                                       side_effect=effect, return_value=1024):
                    result = runner.monitor_worker([sys.executable, '-c', 'import time;time.sleep(30)'],
                        Path(temp), self.limits(seconds=10, gpu_bytes=1), dict(os.environ))
                self.assertEqual(result['status'], 'monitor_failed' if failure else 'resource_stop')
                if not failure:
                    self.assertEqual(result['limit'], 'gpu_bytes')
                self.assertNotEqual(result['returncode'], 0)


if __name__ == '__main__':
    unittest.main()
