"""Offline route4 P1 contracts, actual FAISS kernels and bounded child execution."""
import contextlib
import importlib.util
import inspect
import io
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from mechanism import tls_rag_step5_route4_screen_v1 as screen
from mechanism.tls_rag_step5_route4_risk_v1 import (
    offline_pair_risk, retention_loss_upper_bound, geometric_counterexamples, free_budget_oracle,
)
from tri_rag_harness.tri_law import tri_law_probability

_spec = importlib.util.spec_from_file_location('route4_runner', screen.ROOT / screen.NEW_PATHS[-1])
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


class Route4P1Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name) / 'fixture'
        cls.root.mkdir()
        cls.s = screen.config()
        cls.s.update(n=1024, d=8, nlist=8, nprobe=4, pq_subquantizers=2, pq_bits=4,
                     dimensions=[2, 4], budgets=[10, 16, 32],
                     design_seeds=[57021, 57022], role_sizes=dict(advice=4, select=4, report=4))
        rng = np.random.default_rng(89)
        cls.X = rng.normal(size=(1024, 8))
        cls.X /= np.linalg.norm(cls.X, axis=1)[:, None]
        cls.ids = ['d%04d' % j for j in range(len(cls.X))]
        roles = {}
        for name in ('advice', 'select', 'report'):
            q = rng.normal(size=(4, 8))
            q /= np.linalg.norm(q, axis=1)[:, None]
            roles[name] = [dict(query_id=name + str(i), query=v.tolist()) for i, v in enumerate(q)]
        cls.roles = roles
        with patch.object(runner, 'load_verified_inputs', return_value=(cls.X, cls.ids, roles)), contextlib.redirect_stdout(io.StringIO()):
            cls.manifest = runner.prepare(cls.root, cls.s)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def runtime(self, kind='gaussian64', M=16, m=4):
        spec = dict(kind=kind, M=M)
        if kind.startswith(('gaussian', 'pca')):
            spec['m'] = m
        return screen.Runtime(self.root, spec, 57021 if kind.startswith('gaussian') else 0, self.s)

    def test_approved_scope_and_all_comparators(self):
        s = screen.config()
        self.assertEqual(len(screen.NEW_PATHS), 5)
        self.assertEqual(len(screen.specifications(s)), 157)
        self.assertEqual(s['limits']['seconds'], 2700)
        self.assertEqual(s['n'], 20000)
        self.assertEqual(s['role_sizes'], dict(advice=32, select=32, report=32))
        for key in ('formal_certification', 'fresh_independence', 'predictor_training', 'adapter_training', 'ssh', 'route5', 'multiview'):
            self.assertFalse(s[key])

    def test_gaussian_scale_prefix_and_no_post_normalization(self):
        a = screen.gaussian_projection(384, 32, 57021)
        b = screen.gaussian_projection(384, 64, 57021)
        raw = np.random.Generator(np.random.PCG64(57021)).standard_normal((192, 384))
        # Verify the common draw exactly, without an inexact division roundtrip.
        np.testing.assert_array_equal(a, raw[:32] / np.sqrt(32))
        np.testing.assert_array_equal(b, raw[:64] / np.sqrt(64))
        self.assertAlmostEqual(a.std(), 1 / np.sqrt(32), delta=.006)
        rt = self.runtime()
        np.testing.assert_allclose(rt.projected, self.X @ rt.Pi.T, rtol=0, atol=1e-14)
        self.assertGreater(np.std(np.linalg.norm(rt.projected, axis=1)), .1)

    def test_hash_selection_order_duplicates_and_exact_ties(self):
        values = ['q' + str(i) for i in range(30)]
        self.assertEqual(screen.hashed_ids(values, 57003, 12), screen.hashed_ids(values[::-1], 57003, 12))
        with self.assertRaises(ValueError):
            screen.hashed_ids(['x', 'x'], 1, 1)
        x = np.array([[1., 0.], [-1., 0.], [0., 2.]])
        ids = ['z', 'a', 'b']
        out = screen.rerank_original(np.zeros(2), x, np.arange(3), 2, ids)
        self.assertEqual(out.tolist(), [1, 0])
        self.assertEqual(screen.screen_exact(np.zeros(2), x, np.array([], dtype=int), 10, ids).tolist(), [])
        np.testing.assert_array_equal(screen.screen_exact(np.zeros(2), x, np.arange(3), 10, ids), np.arange(3))

    def test_pool_and_training_interfaces_exclude_projection_queries_and_truth(self):
        self.assertEqual(list(inspect.signature(screen.candidate_pool).parameters), ['q', 'coarse', 'inverted_ids'])
        self.assertEqual(list(inspect.signature(screen.build_independent_pool).parameters), ['X', 's'])
        rt = self.runtime()
        q = np.asarray(self.roles['select'][0]['query'])
        C, I, D = rt.candidates(q)
        # Actual PQ preassigned traversal returns precisely the independent C.
        pq = self.runtime('pq')
        _, found = pq.pq.search_preassigned(q[None].astype(np.float32), len(C), I, D)
        self.assertEqual(set(found[0]), set(C))
        with self.assertRaises(TypeError):
            screen.candidate_pool(q, rt.coarse, rt.inverted, truth=[])
        for seed in self.s['design_seeds']:
            other = screen.Runtime(self.root, dict(kind='gaussian64', m=2, M=10), seed, self.s)
            np.testing.assert_array_equal(C, other.candidates(q)[0])

    def test_real_kernels_exact_rerank_loss_and_full_cost_accounting(self):
        q = np.asarray(self.roles['select'][1]['query'])
        G = screen.rerank_original(q, self.X, np.arange(len(self.X)), 10, self.ids)
        for kind in ('original64', 'original32', 'gaussian64', 'gaussian32', 'pca64', 'pca32', 'pq'):
            rt = self.runtime(kind, M=10 if kind == 'original64' else 16)
            out = rt(q)
            ledger = screen.loss_ledger(G, out['C'], out['E'], out['final'])
            self.assertEqual(ledger['total_missed'], len(set(G) - set(out['final'])))
            direct = sorted(out['E'], key=lambda j: (np.sum((self.X[j] - q)**2), self.ids[j]))[:10]
            self.assertEqual(out['final'].tolist(), direct)
            self.assertEqual(out['full_ns'], sum(out[k] for k in ('route_ns', 'query_projection_ns', 'screen_ns', 'rerank_ns')))
            self.assertGreater(out['route_ns'], 0)
            self.assertGreater(out['rerank_ns'], 0)
            expected = len(out['E']) * 8 * 8 + (len(out['C']) * 8 * 4 if kind == 'original32' else 0)
            self.assertEqual(out['logical_original_bytes'], expected)
            with self.assertRaises(TypeError):
                rt(q, truth=G)

    def test_rank_count_implication_exhaustive_and_small_pools(self):
        G, k, M = {0, 1}, 2, 3
        count = 0
        for order in itertools.permutations(range(6)):
            for C in [set(range(6)), {0, 2, 3, 4}, {2, 3, 4, 5}, {0, 1}]:
                E = set([r for r in order if r in C][:M])
                H = G & C
                self.assertEqual(len(G - E), len(G - C) + len(H - E))
                for i in H - E:
                    preceding = set(order[:order.index(i)]) & (C - G)
                    self.assertGreaterEqual(len(preceding), M - len(H) + 1)
                count += 1
        self.assertEqual(count, 2880)
        self.assertEqual(retention_loss_upper_bound([], 10, 0, 10, 100)['filtering'], 0)
        self.assertEqual(retention_loss_upper_bound([2, 3], 10, 2, 10, 2)['filtering'], 0)
        self.assertAlmostEqual(retention_loss_upper_bound([.4, .2], 10, 2, 10, 100)['filtering'], .6 / 90)

    def test_shared_gaussian_geometry_not_independent_marginals(self):
        value = geometric_counterexamples()
        self.assertEqual((value['marginal'], value['opposite_joint'], value['orthogonal_joint']), ('1/3', '1/3', '1/6'))
        rng = np.random.default_rng(57071)
        columns = rng.normal(scale=1 / np.sqrt(2), size=(100000, 2, 3))
        norms = (columns**2).sum(axis=1)
        one = 2 * norms[:, 1] < norms[:, 0]
        two = 2 * norms[:, 2] < norms[:, 0]
        self.assertAlmostEqual(one.mean(), 1 / 3, delta=.006)
        self.assertAlmostEqual((one & two).mean(), 1 / 6, delta=.006)
        self.assertAlmostEqual(tri_law_probability(2, 0, 2), 1 / 3)

    def test_full_competitors_and_conservative_domain_without_changing_tri(self):
        x = np.array([[1., 0.], [2., 0.], [0., 2.], [-1., 0.], [0., 10.]])
        out = offline_pair_risk(np.zeros(2), x, [0], np.arange(5), [2], [1, 3], block=2)
        self.assertEqual(out['full_pairs'], 4)
        self.assertEqual(out['conservative_pairs'], 1)
        mu = 0 + tri_law_probability(4, 0, 2) + 1 + tri_law_probability(100, 0, 2)
        self.assertAlmostEqual(out['dimensions']['2']['mu'][0], mu)
        zero = offline_pair_risk(np.zeros(2), np.array([[0., 0.], [1., 0.]]), [0], [0, 1], [2], [1])
        self.assertEqual(zero['conservative_pairs'], 1)
        with self.assertRaises(ValueError):
            tri_law_probability(1, 0, 2)

    def test_oracle_matches_bruteforce_same_global_quality_budget(self):
        cost = np.array([[1., 10.], [10., 1.], [5., 2.]])
        missed = np.array([[1, 0], [0, 0], [0, 1]])
        value = free_budget_oracle(cost, missed, 10, .04, .1)
        feasible = [(sum(cost[j, a] for j, a in enumerate(actions)), actions)
                    for actions in itertools.product(range(2), repeat=3)
                    if sum(missed[j, a] for j, a in enumerate(actions)) <= 1]
        self.assertEqual(value['mean_cost'], min(feasible)[0] / 3)
        self.assertFalse(value['deployable'])
        self.assertEqual(value['quality_budget'], 1)
        self.assertEqual(free_budget_oracle(np.ones((2, 2)), np.ones((2, 2)), 10, .005, .1)['status'], 'infeasible')

    def test_report_closed_until_unchanged_selection_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy(self.root / 'report_queries.json', root)
            with self.assertRaises(FileNotFoundError):
                screen.open_role(root, 'report', self.manifest)
            selected = dict(status='selected', gaussian=dict(kind='gaussian64', m=4, M=16))
            screen.save(root / 'selection.json', selected)
            screen.save(root / 'selection_seal.json', dict(selection=screen.fingerprint(selected), inputs=screen.fingerprint(self.manifest), sources=screen.sources()))
            self.assertEqual(screen.open_role(root, 'report', self.manifest), self.roles['report'])
            selected['gaussian']['M'] = 32
            (root / 'selection.json').write_text(json.dumps(selected))
            with self.assertRaises(ValueError):
                screen.open_role(root, 'report', self.manifest)
        with self.assertRaises(ValueError):
            screen.open_role(self.root, 'cert', self.manifest)

    def test_manifest_source_input_and_unfrozen_actions_fail_closed(self):
        with patch.object(screen, 'sources', return_value={'modified': 'source'}):
            with self.assertRaises(ValueError):
                self.runtime()
        for spec, seed in [(dict(kind='gaussian64', m=4, M=11), 57021), (dict(kind='gaussian64', m=4, M=16), 999)]:
            with self.assertRaises(ValueError):
                screen.Runtime(self.root, spec, seed, self.s)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'x'
            p.write_bytes(b'original')
            expected = screen.sha(p)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                screen.checked(p, expected)

    def test_actual_timing_worker_replay_and_no_truth_file_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'
            shutil.copytree(self.root, root)
            screen.save(root / 'source_freeze.json', dict(sources=screen.sources(), config=self.s,
                        supervisor_pid=os.getpid(), deadline_monotonic=time.monotonic() + 30))
            output = root / 'actual_worker.json'
            opened = []
            original = Path.open
            def tracked(path, *args, **kwargs):
                opened.append(path.name)
                return original(path, *args, **kwargs)
            with patch.object(runner, 'config', return_value=self.s), patch.object(Path, 'open', tracked):
                runner.timing_worker(root, 'select', dict(kind='gaussian32', m=4, M=16), 57021, output)
            value = json.loads(output.read_text())
            self.assertEqual(len(value['samples']), 20)
            for row in value['samples']:
                self.assertEqual(row['full_ns'], sum(row[k] for k in ('route_ns', 'query_projection_ns', 'screen_ns', 'rerank_ns', 'return_overhead_ns')))
                self.assertGreater(row['return_overhead_ns'], 0)
            self.assertFalse(any('reference' in n or 'truth' in n or 'quality_' in n for n in opened))
            self.assertTrue(value['no_truth_file_opened'])
            self.assertFalse(value['cold_ssd_claim'])
            rt = screen.Runtime(root, dict(kind='gaussian32', m=4, M=16), 57021, self.s)
            for q in self.roles['select']:
                out = rt(np.asarray(q['query']))
                self.assertEqual(value['results'][q['query_id']], dict(E=out['E'].tolist(), final=out['final'].tolist()))

    def test_shared_resource_deadline_terminates_real_child_and_archives_failure(self):
        s = dict(self.s)
        s['limits'] = dict(s['limits'], seconds=.05, free_bytes=0)
        real_popen = subprocess.Popen
        def child(cmd, **kwargs):
            return real_popen([os.sys.executable, '-c', 'import time; time.sleep(5)'], **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'bounded'
            with patch.object(runner, 'config', return_value=s), patch.object(runner.subprocess, 'Popen', side_effect=child), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    runner.supervisor(root)
            self.assertEqual(raised.exception.code, 2)
            status = json.loads((root / 'status.json').read_text())
            self.assertEqual(status['status'], 'resource_stop')
            self.assertEqual(status['reason'], 'cumulative_wall_limit')
            self.assertTrue(json.loads((root / 'archive_verified.json').read_text())['verified'])


if __name__ == '__main__':
    unittest.main()
