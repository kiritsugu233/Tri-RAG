"""Independent deterministic/analytic R6-M0 checks, no samples or network."""
import copy
from fractions import Fraction
from itertools import permutations, product
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mechanism import tls_rag_step5_route6_macro_reference_v1 as ref
from mechanism import tls_rag_step5_route6_macro_audit_v1 as audit
from scripts import run_tls_rag_step5_route6_macro_reference_v1 as runner
from tri_rag_harness.tri_law import tri_law_probability


class MacroReferenceTests(unittest.TestCase):
    def enclosed(self, result, truths):
        self.assertIsNone(result['stop_reason'])
        for row, truth in zip(result['events'], truths):
            lo, hi = row['discretization_bracket']
            self.assertLessEqual(lo, truth+1e-12)
            self.assertGreaterEqual(hi, truth-1e-12)
            self.assertEqual(row['status'], 'numerically_resolved')
            self.assertLessEqual(row['gap'], result['width_goal'])
        self.assertFalse(result['roundoff_certified'])

    def test_event_identity_all_small_permutations_and_stable_ties(self):
        N, k = 6, 3
        for order in permutations(range(N)):
            rank = np.argsort(order)
            counts = np.array([sum(rank[j] < rank[i] for i in range(k)) for j in range(k, N)])
            for M in range(k, N+1):
                omitted = sum(rank[i] >= M for i in range(k))
                for ell in range(1, k+1):
                    self.assertEqual(ref.omission_from_cross_counts(counts, k, M, ell), omitted >= ell)
        # Equal distances are extended by one stable ID total order.
        ids = ['c', 'a', 'b', 'd']
        order = sorted(range(4), key=lambda i: (1., ids[i]))
        rank = {i: j for j, i in enumerate(order)}
        counts = np.array([sum(rank[j] < rank[i] for i in range(2)) for j in range(2, 4)])
        self.assertEqual(ref.omission_from_cross_counts(counts, 2, 2, 1), any(rank[i] >= 2 for i in range(2)))

    def test_poisson_binomial_against_fraction_subset_enumeration(self):
        p = [Fraction(0), Fraction(1), Fraction(1, 3), Fraction(1, 5), Fraction(1, 10**12)]
        expected = [Fraction(0)]*(len(p)+1)
        for bits in product((0, 1), repeat=len(p)):
            weight = Fraction(1)
            for bit, probability in zip(bits, p):
                weight *= probability if bit else 1-probability
            expected[sum(bits)] += weight
        np.testing.assert_allclose(ref.poisson_binomial_pmf([float(v) for v in p]), [float(v) for v in expected], atol=0, rtol=2e-15)
        np.testing.assert_array_equal(ref.poisson_binomial_pmf([]), [1.])
        self.assertGreater(ref.poisson_binomial_pmf([1e-20]*3)[3], 0)

    def test_conditional_cdf_exponential_scale_and_factor_order(self):
        d2 = np.array([1., 3.])
        np.testing.assert_allclose(ref.conditional_distance_cdf(.7, d2, 2, 0), 1-np.exp(-.7/d2))
        a = ref.conditional_distance_cdf(.7, d2, 8, .5, 2.)
        b = ref.conditional_distance_cdf(.7, d2, 8, .5, 10.)
        self.assertTrue(np.all(a >= b))
        np.testing.assert_array_equal(ref.conditional_distance_cdf(0., d2, 8, .5), [0., 0.])
        np.testing.assert_array_equal(ref.conditional_distance_cdf(math.inf, d2, 8, .5), [1., 1.])

    def test_order_cdf_independent_subset_reference(self):
        self.assertAlmostEqual(ref.target_order_cdf([.2, .4, .7], 2), .2*.4*.3+.2*.6*.7+.8*.4*.7+.2*.4*.7)

    def test_heterogeneous_exponential_rational_fixture(self):
        truth = 1-Fraction(10, 13)-Fraction(5, 6)+Fraction(2, 3)
        self.assertEqual(truth, Fraction(5, 78))
        result = ref.macro_reference_interval([1., 1.5, 2., 3.], 2, 2, 0., [(2, 2)], width=.002, max_nodes=2048)
        self.enclosed(result, [float(truth)])

    def test_pair_reduces_to_unchanged_tri_law(self):
        for c in (0., .5):
            for m in (1, 2, 8, 32):
                result = ref.macro_reference_interval([1., 2.], 1, m, c, [(1, 1)], width=.1, max_nodes=4096)
                self.enclosed(result, [tri_law_probability(2., c, m)])

    def test_exchangeable_hypergeometric_tail_for_both_geometries(self):
        N, k = 5, 2
        events = [(2, 1), (2, 2), (4, 1), (5, 2)]
        truth = [sum(math.comb(k, r)*math.comb(N-k, M-r) for r in range(k-ell+1)
                     if 0 <= M-r <= N-k)/math.comb(N, M) for M, ell in events]
        for c in (0., .5):
            result = ref.macro_reference_interval([1.]*N, k, 2, c, events, width=.1, max_nodes=8192)
            self.enclosed(result, truth)

    def test_refinement_nesting_and_explicit_infinite_tail(self):
        args = ([1., 1.5, 2., 3.], 2, 2, 0., [(2, 2)])
        coarse = ref.macro_reference_interval(*args, width=.0001, max_nodes=8)
        fine = ref.macro_reference_interval(*args, width=.0001, max_nodes=64)
        a, b = coarse['events'][0]['discretization_bracket']
        c, d = fine['events'][0]['discretization_bracket']
        self.assertLessEqual(a, c)
        self.assertGreaterEqual(b, d)
        self.assertIn([fine['partition']['distance'][-1][0], 'infinity'], fine['partition']['distance'])
        self.assertEqual(fine['stop_reason'], 'cdf_node_limit')
        self.assertEqual(fine['nodes'], 64)

    def test_crossed_outer_endpoints_bound_equal_factor_reference(self):
        # No monotonicity of theta(r) is assumed; the two groups move oppositely.
        args = ([1., 1.], 1, 2, .5, [(1, 1)])
        low = ref.conditional_omission_bracket(*args, r_gt=1., r_out=5., width=.005, max_nodes=4096)
        high = ref.conditional_omission_bracket(*args, r_gt=5., r_out=1., width=.005, max_nodes=4096)
        middle = ref.conditional_omission_bracket(*args, r_gt=3., r_out=3., width=.005, max_nodes=4096)
        self.enclosed(middle, [.5])
        self.assertLess(low['events'][0]['discretization_bracket'][1], middle['events'][0]['discretization_bracket'][0])
        self.assertGreater(high['events'][0]['discretization_bracket'][0], middle['events'][0]['discretization_bracket'][1])

    def test_outer_budget_preserves_tail_mass_and_nested_intersections(self):
        args = ([1., 2.], 1, 2, .5, [(1, 1)])
        coarse = ref.macro_reference_interval(*args, width=.0001, max_nodes=50)
        fine = ref.macro_reference_interval(*args, width=.0001, max_nodes=200)
        a, b = coarse['events'][0]['discretization_bracket']
        c, d = fine['events'][0]['discretization_bracket']
        self.assertLessEqual(a, c)
        self.assertGreaterEqual(b, d)
        self.assertLessEqual(c, tri_law_probability(2, .5, 2))
        self.assertGreaterEqual(d, tri_law_probability(2, .5, 2))
        self.assertAlmostEqual(sum(cell['mass'] for cell in fine['partition']), 1)
        self.assertEqual(fine['partition'][-1]['r'][1], 'infinity')
        self.assertEqual(fine['events'][0]['status'], 'unresolved')

    def test_impossible_event_is_exact_without_nodes(self):
        for c in (0., .5):
            result = ref.macro_reference_interval([1., 2., 3.], 1, 8, c, [(3, 1)])
            self.enclosed(result, [0.])
            self.assertEqual(result['nodes'], 0)

    def test_invalid_domains_and_numeric_guard(self):
        for p in ([np.nan], [-.1], [1.1]):
            with self.assertRaises(ValueError):
                ref.poisson_binomial_pmf(p)
        for d2, k, m, c in (([1., 0.], 1, 2, 0), ([1., 2.], True, 2, 0), ([1., 2.], 1, 0, 0), ([1., 2.], 1, 2, .2)):
            with self.assertRaises(ValueError):
                ref.macro_reference_interval(d2, k, m, c, [(1, 1)])
        with self.assertRaises(ref.NumericFailure):
            ref.Audit().mass(np.array([.5, .6]))
        with patch.object(ref, 'conditional_distance_cdf', return_value=np.array([.5, np.nan])):
            with self.assertRaises(ValueError):
                ref.macro_reference_interval([1., 2.], 1, 2, 0., [(1, 1)])

    def test_deadline_keeps_initial_valid_enclosure(self):
        def stop():
            raise ref.BudgetStop('scientific_deadline')
        result = ref.macro_reference_interval([1., 2.], 1, 2, 0., [(1, 1)], check=stop)
        self.assertEqual(result['stop_reason'], 'scientific_deadline')
        self.assertEqual(result['events'][0]['discretization_bracket'], [0., 1.])

    def test_geometry_scale_alignment_and_tv_diagnostic(self):
        d2 = np.array([1., 2., 3.])
        ideal = np.sqrt(d2[:, None]*d2[None, :])*(.5*np.eye(3)+.5)
        value = audit.geometry_model_diagnostic(.03125*ideal, d2, .5, 8)
        self.assertEqual(value['common_scale'], .03125)
        self.assertLess(value['tv_diagnostic'], 1e-12)
        self.assertFalse(value['roundoff_certified'])
        with self.assertRaises(ValueError):
            audit.geometry_model_diagnostic(np.ones((3, 3)), d2, .5, 8)

    def input_fixture(self, root):
        cfg = copy.deepcopy(runner.config())
        cfg.update(N=4, k=[1], m=[2], budget_factors=[1, 2, 4])
        paths = ['source_freeze.json']
        (root/'source_freeze.json').write_text(json.dumps(dict(commit=cfg['input_binding']['source_commit'], config=cfg['input_binding']['old_config'], sources=cfg['input_binding']['old_sources'])))
        for kind in ('independent', 'equicorrelated'):
            geometry = 'macro-k1-'+kind
            folder = root/('geometry/'+geometry)
            folder.mkdir(parents=True)
            d2 = 1+np.arange(4)/100
            c = 0 if kind == 'independent' else .5
            gram = .02*np.sqrt(d2[:, None]*d2[None, :])*((1-c)*np.eye(4)+c)
            np.save(folder/'gram.npy', gram)
            (folder/'ids.json').write_text(json.dumps(dict(query_id='route6:'+geometry+':query', row_to_id=['route6:'+geometry+':corpus:'+str(i) for i in range(4)])))
            paths += ['geometry/'+geometry+'/'+p for p in ('gram.npy', 'ids.json')]
            for prefix in ('bounds', 'units'):
                (root/prefix).mkdir(exist_ok=True)
            events = [dict(M=M, ell=1) for M in (1, 2, 4)]
            bounds = dict(geometry=geometry, m=2, bounds=events)
            unit = dict(geometry=geometry, m=2, path='gram', trials=65536, events=[dict(e, type='omission', id=str(i), geometry=geometry, m=2, path='gram') for i, e in enumerate(events)])
            for name, data in [('bounds/%s-m2.json'%geometry, bounds), ('units/%s-m2-gram.json'%geometry, unit)]:
                (root/name).write_text(json.dumps(data))
                paths.append(name)
        index = root/'index.json'
        index.write_text(json.dumps(dict(files=[dict(path=p, bytes=(root/p).stat().st_size, sha256=audit.sha(root/p)) for p in paths])))
        cfg['input_binding'].update(selected_paths=paths, index_sha256=audit.sha(index))
        return cfg, index

    def test_frozen_input_binding_and_missing_or_corrupt_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, index = self.input_fixture(root)
            models, receipt = audit.load_frozen_macro_inputs(root, index, cfg)
            self.assertEqual(len(models), 2)
            self.assertEqual(len(receipt['files']), 9)
            path = root/cfg['input_binding']['selected_paths'][-1]
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'SHA'):
                audit.load_frozen_macro_inputs(root, index, cfg)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                audit.load_frozen_macro_inputs(root, index, cfg)

    def test_frozen_index_and_source_identity_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, index = self.input_fixture(root)
            bad = copy.deepcopy(cfg)
            bad['input_binding']['source_commit'] = 'not-the-source'
            with self.assertRaisesRegex(ValueError, 'identity'):
                audit.load_frozen_macro_inputs(root, index, bad)
            index.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'index SHA'):
                audit.load_frozen_macro_inputs(root, index, cfg)

    def test_stable_ids_not_replaced_by_an_equal_length_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, index = self.input_fixture(root)
            path = root/'geometry/macro-k1-independent/ids.json'
            value = json.loads(path.read_text())
            value['row_to_id'][0] = value['row_to_id'][1]
            path.write_text(json.dumps(value))
            rows = json.loads(index.read_text())
            for row in rows['files']:
                if row['path'] == str(path.relative_to(root)):
                    row.update(bytes=path.stat().st_size, sha256=audit.sha(path))
            index.write_text(json.dumps(rows))
            cfg['input_binding']['index_sha256'] = audit.sha(index)
            with self.assertRaisesRegex(ValueError, 'ID'):
                audit.load_frozen_macro_inputs(root, index, cfg)

    def test_frozen_scope_and_tiny_entry(self):
        cfg = runner.config()
        self.assertEqual(cfg['max_nodes_per_model'], 32768)
        self.assertEqual(cfg['limits']['seconds'], 1200)
        self.assertEqual(cfg['width_goal'], 2.**-14)
        self.assertEqual(cfg['cpu_dependencies'], {'numpy': '1.26.4', 'scipy': '1.13.0'})
        self.assertFalse(runner.tiny_check()['cuda_called'])

    def test_exclusive_bounded_artifact_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            limits = dict(runner.config()['limits'], artifact_bytes=16, reserve_bytes=0)
            store = runner.helpers.ArtifactStore(tmp, limits)
            store.write('one', b'1')
            with self.assertRaises(FileExistsError):
                store.write('one', b'2')
            with self.assertRaises(runner.helpers.ResourceStop):
                store.write('big', b'0'*16)

    def test_real_owned_child_cpu_monitor_stops_and_reaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            limits = dict(runner.config()['limits'], seconds=.15, poll_seconds=.02, reserve_bytes=0, rss_bytes=2**40)
            value = runner.helpers.monitor_worker([sys.executable, '-c', 'import time; time.sleep(10)'], tmp, limits, dict(os.environ), gpu_monitor=False)
            self.assertEqual(value['status'], 'resource_stop')
            self.assertEqual(value['limit'], 'seconds')
            self.assertNotEqual(value['returncode'], 0)
            self.assertEqual(value['observed_peaks']['gpu_bytes'], 0)

    def test_delivery_index_measures_closed_progress_and_includes_source_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'source_freeze.json').write_text('{}')
            store = runner.helpers.ArtifactStore(root, dict(runner.config()['limits'], reserve_bytes=0))
            store.write('progress.jsonl', b'')
            with (root/'progress.jsonl').open('ab') as stream:
                store.append(stream, b'{"nodes":1}\n')
            runner.write_delivery_index(store)
            receipt = json.loads((root/'delivery_index.json').read_text())
            rows = {row['path']: row for row in receipt['files']}
            self.assertEqual(set(rows), {'source_freeze.json', 'progress.jsonl'})
            for path, row in rows.items():
                self.assertEqual(row['bytes'], (root/path).stat().st_size)
                self.assertEqual(row['sha256'], audit.sha(root/path))
            before = (root/'delivery_index.json').read_bytes()
            with self.assertRaises(FileExistsError):
                runner.write_delivery_index(store)
            self.assertEqual((root/'delivery_index.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
