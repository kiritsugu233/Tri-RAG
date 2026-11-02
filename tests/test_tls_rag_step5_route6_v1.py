"""Offline R6-A conformance; CUDA integration is separately required on A800."""
import copy
from itertools import permutations
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import f
from mechanism import tls_rag_step5_route6_bridge_v1 as bridge
from mechanism import tls_rag_step5_route6_a800_v1 as gpu
from scripts import run_tls_rag_step5_route6_v1 as runner
from tri_rag_harness.tri_law import tri_law_probability


class Route6Tests(unittest.TestCase):
    def setUp(self):
        self.cfg = runner.config()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def geometry(self, index=0):
        return gpu.build_geometry(gpu.geometry_specs()[index], self.cfg)

    def test_cross_inversion_witness_by_independent_permutation_enumeration(self):
        n, k = 6, 2
        for order in permutations(range(n)):
            rank = {point: i for i, point in enumerate(order)}
            inversions = sum(rank[j] < rank[i] for i in range(k) for j in range(k, n))
            for M in range(k, n+1):
                omitted = sum(rank[i] >= M for i in range(k))
                self.assertGreaterEqual(inversions, omitted*(M-k+omitted))

    def test_cross_bound_thresholds_and_impossible_events(self):
        self.assertEqual(bridge.cross_inversion_bound(3, 8, 2, 4, 1), 1)
        self.assertEqual(bridge.cross_inversion_bound(3, 8, 2, 4, 2), 3/8)
        self.assertEqual(bridge.cross_inversion_bound(3, 8, 2, 8, 1), 0)
        with self.assertRaises(ValueError):
            bridge.cross_inversion_bound(20, 8, 2, 4, 1)

    def test_marginal_interval_covers_dependent_even_and_odd_parity(self):
        even = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
        odd = 1-even
        for h in (1, 2, 3):
            lo, hi = bridge.marginal_bounds([.5]*3, h)
            for samples in (even, odd):
                risk = float(np.mean(samples.sum(axis=1) >= h))
                self.assertLessEqual(lo, risk)
                self.assertGreaterEqual(hi, risk)
        self.assertEqual(even.sum(axis=1).var(), odd.sum(axis=1).var())
        self.assertNotEqual(np.mean(even.sum(axis=1) >= 3), np.mean(odd.sum(axis=1) >= 3))

    def test_trimmed_marginal_bound_and_domain(self):
        self.assertAlmostEqual(bridge.marginal_bounds([1, .01, .01], 2)[1], .02)
        self.assertEqual(bridge.marginal_bounds([], 1), (0, 0))
        self.assertEqual(bridge.marginal_bounds([], 0), (1, 1))
        for p in ([np.nan], [-.1], [1.1]):
            with self.assertRaises(ValueError):
                bridge.marginal_bounds(p, 1)

    def test_boundary_split_against_exhaustive_joint_distribution(self):
        outcomes = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
        for h in (1, 2, 3):
            risk = np.mean(outcomes.sum(axis=1) >= h)
            eta = outcomes[:, 2].mean()
            for t in range(h):
                local = np.mean(outcomes[:, :2].sum(axis=1) >= h-t)
                self.assertGreaterEqual(bridge.boundary_remainder_bound(local, eta, h, t), risk)
        with self.assertRaises(ValueError):
            bridge.boundary_remainder_bound(.2, .1, 2, 2)

    def test_second_order_fixtures_with_same_or_exclusive_flips(self):
        same = bridge.second_order_bounds([.5, .5], np.array([[0, .5], [.5, 0]]), 2)
        exclusive = bridge.second_order_bounds([.5, .5], np.zeros((2, 2)), 2)
        self.assertEqual(same['variance_upper'], 1)
        self.assertEqual(exclusive['variance_upper'], 0)
        self.assertEqual(same['factorial2'], .5)
        self.assertEqual(exclusive['factorial2'], 0)
        self.assertEqual(bridge.cantelli_bound(1, 1, 2), .5)
        self.assertEqual(bridge.factorial_bound(1, 4, 3), 1/3)

    def test_inconsistent_joint_information_is_not_silently_repaired(self):
        result = bridge.second_order_bounds([1, 1], np.zeros((2, 2)), 2)
        self.assertEqual(result['status'], 'inconsistent_joint_constraints')
        self.assertEqual(result['cantelli'], 1)

    def test_scalar_beta_equivalence_to_independent_f_distribution(self):
        for m in (1, 2, 8, 64):
            for beta in (1.01, 2, 100):
                result = bridge.scalar_law(beta, 0, m)
                self.assertAlmostEqual(float(result['beta_argument']), 1/(1+beta))
                self.assertAlmostEqual(float(result['probability']), float(f.sf(beta, m, m)), places=12)
                self.assertGreaterEqual(float(result['power_upper'])+1e-14, float(result['probability']))

    def test_scalar_extreme_near_collinear_representable_tail(self):
        beta, rho = np.finfo(float).max, np.nextafter(1., 0.)
        result = bridge.scalar_law(beta, rho, 1)
        self.assertEqual(float(result['gamma']), 1.)
        self.assertGreater(float(result['probability']), 0)
        self.assertAlmostEqual(float(result['probability'])/tri_law_probability(beta, rho, 1), 1)
        self.assertEqual(float(bridge.scalar_law(2, 1, 8)['probability']), 0)
        for args in ((1, 0, 2), (2, 1.1, 2), (2, 0, True)):
            with self.assertRaises(ValueError):
                bridge.scalar_law(*args)

    def test_complete_scalar_grid_and_rho_symmetry(self):
        rows = gpu.scalar_grid_audit(self.cfg)
        self.assertEqual(len(rows), 210)
        self.assertTrue(all(row['absolute_difference'] <= 1e-12+1e-8*row['reference'] for row in rows))

    def test_conditional_enclosure_matches_hand_integrals(self):
        bins = 512
        result = bridge.conditional_marginal_bounds(np.array([2.]*4), np.zeros(4), 2, 4, bins)
        self.assertLessEqual(result['lower'], 1/48)
        self.assertGreaterEqual(result['lower'], 1/48-1/bins)
        self.assertGreaterEqual(result['upper'], 1/3)
        self.assertLessEqual(result['upper'], 1/3+1/bins)
        self.assertFalse(result['roundoff_certified'])

    def test_conditional_noncentral_branch_and_monotonicity_guard(self):
        result = bridge.conditional_marginal_bounds(np.array([2.]), np.array([1.]), 8, 1, 512)
        truth = tri_law_probability(2, 1/np.sqrt(2), 8)
        self.assertLessEqual(result['lower'], truth)
        self.assertGreaterEqual(result['upper'], truth)
        self.assertLessEqual(result['upper']-result['lower'], 1/512+1e-12)
        with self.assertRaises(ValueError):
            bridge.conditional_marginal_bounds(np.array([2.]), np.array([1.1]), 8, 1, 16)

    def test_confidence_envelope_keeps_zero_upper_and_family_correction(self):
        lo, hi = bridge.clopper_pearson_envelope([0, 10], [0, 10], 10, 20)
        self.assertEqual(lo[0], 0)
        self.assertGreater(hi[0], 0)
        self.assertEqual(hi[1], 1)
        self.assertLess(lo[1], 1)
        a, b = bridge.clopper_pearson_envelope(2, 8, 10, 20)
        self.assertLess(a, .2)
        self.assertGreater(b, .8)
        self.assertEqual(bridge.decision((0, .01), .01), 'pass')
        self.assertEqual(bridge.decision((.02, .1), .01), 'fail')
        self.assertEqual(bridge.decision((0, .1), .01), 'undecided')

    def test_all_16_geometries_unit_lifts_and_declared_ranks(self):
        geometries = [gpu.build_geometry(spec, self.cfg) for spec in gpu.geometry_specs()]
        self.assertEqual(len(geometries), 16)
        for geometry in geometries:
            np.testing.assert_allclose(np.linalg.norm(geometry['corpus'], axis=1), 1, atol=1e-12)
            self.assertAlmostEqual(np.linalg.norm(geometry['query']), 1)
            actual = geometry['corpus']-geometry['query']
            np.testing.assert_allclose(actual@actual.T, geometry['gram'], atol=1e-13)
            np.testing.assert_allclose(geometry['factor']@geometry['factor'].T, geometry['gram'], atol=1e-12)
            self.assertEqual(len(set(geometry['ids'])), geometry['spec']['N'])
        repeated = next(g for g in geometries if g['spec']['kind'] == 'repeated')
        np.testing.assert_array_equal(repeated['corpus'][1], repeated['corpus'][4])
        self.assertEqual(repeated['rank'], 2)

    def test_invalid_gram_or_dimension_fails_without_jitter(self):
        with self.assertRaises(ValueError):
            gpu.gram_factor(np.array([[1., 2.], [2., 1.]]), 2, 1e-11)
        with self.assertRaises(ValueError):
            gpu.gram_factor(np.eye(2), 1, 1e-11)
        cfg = copy.deepcopy(self.cfg)
        cfg['dimension'] = 2
        with self.assertRaises(ValueError):
            gpu.build_geometry(gpu.geometry_specs()[0], cfg)

    def test_seed_identity_and_batch_invariance(self):
        geometry = self.geometry()
        for path in ('coordinate', 'gram'):
            seed = gpu.seed_for(self.cfg, geometry['spec']['id'], 8, path)
            a = gpu.sample_shared_projection(np.random.default_rng(seed), geometry, 8, 10, path)
            rng = np.random.default_rng(seed)
            b = np.concatenate([gpu.sample_shared_projection(rng, geometry, 8, n, path) for n in (3, 2, 5)])
            np.testing.assert_array_equal(a, b)
            self.assertLess(abs(float(a.var())-1/8), .03)
        self.assertNotEqual(gpu.seed_for(self.cfg, 'a', 8, 'gram'), gpu.seed_for(self.cfg, 'a', 8, 'coordinate'))

    def test_full_catalogue_covers_60_units_and_both_paths(self):
        geometries = [gpu.build_geometry(spec, self.cfg) for spec in gpu.geometry_specs()]
        catalogue = gpu.build_catalogue(geometries, self.cfg)
        self.assertEqual(len({(event['geometry'], event['m']) for event in catalogue}), 60)
        self.assertEqual({event['path'] for event in catalogue}, {'coordinate', 'gram'})
        self.assertEqual(len({event['id'] for event in catalogue}), len(catalogue))
        self.assertTrue(any(event['type'] == 'joint' for event in catalogue))
        for geometry in geometries:
            self.assertEqual(gpu.boundary_indices(geometry, 8, self.cfg), gpu.boundary_indices(geometry, 8, self.cfg))

    def test_coordinate_projection_matches_direct_vector_reference(self):
        geometry = self.geometry()
        p = gpu.sample_shared_projection(np.random.default_rng(55), geometry, 8, 4, 'coordinate')
        actual, errors, _ = gpu.evaluate_projection(p, geometry, 'coordinate')
        for sample in range(4):
            direct = np.array([np.linalg.norm(p[sample]@x-p[sample]@geometry['query'])**2 for x in geometry['corpus']])
            self.assertTrue(np.all(np.abs(actual[sample]-direct) <= errors[sample]+1e-15))
        self.assertFalse(np.allclose(actual.sum(axis=1), 1))

    def test_gram_projection_preserves_shared_correlations(self):
        geometry = gpu.build_geometry(next(s for s in gpu.geometry_specs() if s['kind'] == 'repeated'), self.cfg)
        p = gpu.sample_shared_projection(np.random.default_rng(5), geometry, 8, 10, 'gram')
        squared, _, _ = gpu.evaluate_projection(p, geometry, 'gram')
        np.testing.assert_allclose(squared[:, 1:], np.repeat(squared[:, 1:2], 4, axis=1), atol=1e-14)

    def test_uncertainty_envelope_does_not_change_strict_event(self):
        geometry = self.geometry()
        events = gpu.unit_events(geometry, 2, 'coordinate', self.cfg)
        squared = np.array([[1., .5, 2, 3, 4], [1., 1., 2, 3, 4]])
        lo, hi, raw = gpu.event_observations(squared, np.zeros_like(squared), geometry, events)
        np.testing.assert_array_equal(raw['rank_lower'].ravel(), [2, 1])
        np.testing.assert_array_equal(raw['rank_upper'].ravel(), [2, 2])
        event = next(i for i, e in enumerate(events) if e['type'] == 'marginal' and e['j'] == 1)
        self.assertTrue(lo[0, event])
        self.assertFalse(lo[1, event])
        self.assertTrue(hi[1, event])
        self.assertFalse(np.any(lo & ~hi))

    def test_macro_events_match_independent_sorting(self):
        geometry = gpu.build_geometry(gpu.geometry_specs()[12], self.cfg)
        distances = np.arange(64, dtype=float)[None, ::-1]
        events = gpu.unit_events(geometry, 8, 'coordinate', self.cfg)
        lo, hi, _ = gpu.event_observations(distances, np.zeros_like(distances), geometry, events)
        for a, event in enumerate(events):
            if event['type'] == 'omission':
                ids = set(np.argsort(distances[0])[:event['M']])
                expected = sum(i not in ids for i in range(geometry['spec']['k'])) >= event['ell']
                self.assertEqual(bool(lo[0, a]), expected)
                self.assertEqual(bool(hi[0, a]), expected)

    def test_tiny_sampler_counts_and_replay(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.update(samples_per_path=19, batch=5)
        geometry = self.geometry()
        recordings = []
        result = gpu.sample_unit(geometry, 2, 'coordinate', cfg, 100, 'cpu', lambda offset, raw: recordings.append(raw), lambda: None)
        np.testing.assert_array_equal(np.concatenate([raw['sample_ids'] for raw in recordings]), np.arange(19))
        for event in result['events']:
            self.assertLessEqual(event['definite_count'], event['possible_count'])
            self.assertLessEqual(event['possible_count'], 19)
        repeated = gpu.sample_unit(geometry, 2, 'coordinate', cfg, 100, 'cpu', lambda *args: None, lambda: None)
        self.assertEqual(result['events'], repeated['events'])

    def test_resource_callback_interrupts_sampler(self):
        def stop():
            raise runner.ResourceStop('fixture')
        with self.assertRaises(runner.ResourceStop):
            gpu.sample_unit(self.geometry(), 2, 'coordinate', self.cfg, 100, 'cpu', lambda *args: None, stop)

    def test_gpu_absence_blocks_without_cpu_fallback(self):
        with patch.object(gpu, 'torch_backend', side_effect=gpu.GPUBlocked('fixture absent')):
            with self.assertRaises(gpu.GPUBlocked):
                gpu.evaluate_projection(np.ones((1, 2, 128)), self.geometry(), 'coordinate', 'cuda')

    def test_cpu_selftest_explicitly_not_gpu_acceptance(self):
        value = runner.tiny_check(self.cfg)
        self.assertEqual(value['device'], 'cpu')
        self.assertTrue(value['finite_integration_fixture_only'])
        self.assertEqual([p['samples'] for p in value['paths']], [8, 8])

    def test_artifact_store_safe_exclusive_and_prewrite_budget(self):
        limits = dict(self.cfg['limits'], artifact_bytes=8, reserve_bytes=0, write_chunk_bytes=2)
        store = runner.ArtifactStore(self.root, limits)
        store.write('a.bin', b'1234')
        self.assertEqual(store.used, 4)
        with self.assertRaises(FileExistsError):
            store.write('a.bin', b'123')
        with self.assertRaises(runner.ResourceStop):
            store.write('b.bin', b'12345')
        with self.assertRaises(ValueError):
            store.write('../unsafe', b'1')
        self.assertFalse((self.root/'b.bin').exists())

    def test_lock_rejects_source_environment_and_allocation_changes(self):
        saved = {'schema': 'fixture', 'environment': {'torch': 'fixed'}, 'sources': {'core': 'hash'},
                 'config_sha256': 'c', 'limits': dict(self.cfg['limits'])}
        lock = self.root/'lock.json'
        runner.immutable_json(lock, saved)
        with patch.object(runner, 'hardware_lock', return_value=copy.deepcopy(saved)):
            self.assertEqual(runner.validate_lock(lock, self.cfg), saved)
        for key in ('sources', 'environment'):
            changed = copy.deepcopy(saved)
            changed[key] = {'different': True}
            with patch.object(runner, 'hardware_lock', return_value=changed), self.assertRaises(gpu.GPUBlocked):
                runner.validate_lock(lock, self.cfg)
        actual = copy.deepcopy(saved)
        actual['limits']['gpu_bytes'] -= 1
        with patch.object(runner, 'hardware_lock', return_value=actual), self.assertRaises(gpu.GPUBlocked):
            runner.validate_lock(lock, self.cfg)

    def test_real_child_wall_limit_and_immutable_receipt(self):
        limits = dict(self.cfg['limits'], seconds=.08, poll_seconds=.01, reserve_bytes=0)
        receipt = runner.monitor_worker([sys.executable, '-c', 'import time; time.sleep(60)'], self.root, limits, os.environ.copy(), gpu_monitor=False)
        self.assertEqual(receipt['status'], 'resource_stop')
        self.assertEqual(receipt['limit'], 'seconds')
        self.assertLess(receipt['returncode'], 0)
        with self.assertRaises(FileExistsError):
            runner.immutable_json(self.root/'execution.json', {})

    def test_real_monitor_error_kills_owned_child(self):
        children = []
        real_popen = subprocess.Popen
        def child(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            children.append(process)
            return process
        with patch.object(runner, 'process_tree', side_effect=[[(os.getpid(), 1)], RuntimeError('injected monitor failure')]), patch.object(runner.subprocess, 'Popen', side_effect=child):
            receipt = runner.monitor_worker([sys.executable, '-c', 'import time; time.sleep(60)'], self.root, dict(self.cfg['limits'], reserve_bytes=0), os.environ.copy(), gpu_monitor=False)
        self.assertEqual(receipt['status'], 'monitor_failed')
        self.assertIsNotNone(children[0].poll())

    def test_real_artifact_limit_preserves_overshoot_and_stops_child(self):
        limits = dict(self.cfg['limits'], artifact_bytes=100, seconds=2, poll_seconds=.01, reserve_bytes=0)
        code = "import os; from pathlib import Path; import time; os.chdir("+repr(str(self.root))+"); Path('payload').write_bytes(b'x'*10000); time.sleep(60)"
        receipt = runner.monitor_worker([sys.executable, '-c', code], self.root, limits, os.environ.copy(), gpu_monitor=False)
        self.assertEqual(receipt['limit'], 'artifact_bytes')
        self.assertGreater(receipt['artifact_bytes'], limits['artifact_bytes'])
        self.assertTrue((self.root/'payload').exists())

    def test_partial_sample_file_retains_presealed_layout(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.update(single_m=[2], samples_per_path=8, batch=3, conditional_bins=32)
        calls = []
        def stop_after_first_batch():
            calls.append(1)
            if len(calls) == 3:
                raise runner.ResourceStop('injected partial stop')
        with patch.object(gpu, 'geometry_specs', return_value=[gpu.geometry_specs()[0]]):
            with self.assertRaises(runner.ResourceStop):
                runner.run_core(self.root, cfg, 'cpu', dict(cfg['limits'], reserve_bytes=0), stop_after_first_batch)
        layout_path = next(self.root.glob('samples/*.layout.json'))
        layout = json.loads(layout_path.read_text())
        binary = Path(str(layout_path)[:-len('.layout.json')])
        self.assertEqual(binary.stat().st_size // layout['record_bytes'], 3)
        self.assertEqual(layout['declared_rows'], 8)
        self.assertFalse((self.root/'result.json').exists())

    def test_tiny_core_binary_counts_s2_independent_check_and_hashes(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.update(single_m=[2], macro_m=[8], samples_per_path=8, batch=3, conditional_bins=32)
        specs = [gpu.geometry_specs()[0], gpu.geometry_specs()[12]]
        with patch.object(gpu, 'geometry_specs', return_value=specs):
            runner.run_core(self.root, cfg, 'cpu', dict(cfg['limits'], reserve_bytes=0), lambda: None)
        self.assertEqual(len(json.loads((self.root/'result.json').read_text())['units']), 2)
        catalogue = json.loads((self.root/'catalogue.json').read_text())
        self.assertEqual(len(catalogue['events']), catalogue['family_size'])
        for path in self.root.glob('samples/*.layout.json'):
            layout = json.loads(path.read_text())
            dtype = np.dtype([tuple(row) if len(row) == 2 else (row[0], row[1], tuple(row[2])) for row in layout['dtype']])
            binary = Path(str(path)[:-len('.layout.json')])
            raw = np.fromfile(binary, dtype=dtype)
            self.assertEqual(len(raw), layout['declared_rows'])
            np.testing.assert_array_equal(raw['sample_ids'], np.arange(8))
            # Independently unpack recorded event bits and recompute saved counts.
            mask = np.unpackbits(raw['event_definite'], axis=1, bitorder='little')[:, :len(layout['event_ids'])]
            records = json.loads((self.root/'units'/binary.name.replace('.bin', '.json')).read_text())['events']
            np.testing.assert_array_equal(mask.sum(axis=0), [event['definite_count'] for event in records])
        for path in self.root.glob('bounds/*.json'):
            value = json.loads(path.read_text())
            self.assertTrue(value['independent_gram_checks'])
            self.assertTrue(all('coordinate-stream' in row['S2_extra_input'] for row in value['bounds']))
        for row in json.loads((self.root/'delivery_index.json').read_text())['files']:
            self.assertEqual(runner.sha(self.root/row['path']), row['sha256'])


if __name__ == '__main__':
    unittest.main()
