"""CPU/offline tests for approved route2 v2, including compiled FAISS and workers."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from mechanism.tls_rag_step5_ann_continuation_v2 import (
    config, nested_folds, Inputs, Native, BoundarySession, Runtime, observed_features,
    export_model, Predictor, predictors, fit_models, predictions, calibrate, decide, replay, paired_quality,
    sources, sha, save, fingerprint,
)
from tri_rag_harness.tri_law import tri_law_probability


def zero_margins():
    return {f: {k: {'0.5': 0., '0.8': 0., '0.95': 0.} for k in ('retention', 'hit')}
            for f in ('S', 'G', 'T', 'T0')}


def constant_model(nf, value):
    return dict(roots=[0], feature=[-2], threshold=[-2.], left=[-1], right=[-1],
                value=[0.], initial=value, rate=.05, n_features=nf)


class ContinuationV2Contracts(unittest.TestCase):
    def test_only_approved_cpu_scope(self):
        s = config()
        self.assertEqual(s['active_routes'], [2])
        self.assertEqual(len(s['pair_ranks']), 16)
        self.assertEqual(len(set(x for pair in s['pair_ranks'] for x in pair)), 11)
        self.assertEqual(s['limits']['seconds'], 5400)
        for name in ('scale_1m', 'adapter_training', 'ssh', 'formal_certification', 'fresh_independence'):
            self.assertFalse(s[name])

    def test_nested_query_isolation_all_checkpoints_share_identity(self):
        qids = ['q' + str(j) for j in range(768)]
        folds = nested_folds(qids, config())
        self.assertEqual(folds, nested_folds(list(reversed(qids)), config()))
        self.assertEqual(set(q for f in folds for q in f['outer']), set(qids))
        for f in folds:
            self.assertEqual({k: len(v) for k, v in f.items()}, dict(fit=384, cal=96, select=96, outer=192))
            self.assertEqual(sum(len(v) for v in f.values()), len(set(q for v in f.values() for q in v)))
        with self.assertRaises(ValueError):
            nested_folds(qids + [qids[0]], config())

    def test_all_seals_required_for_outer_gate(self):
        x = object.__new__(Inputs)
        x.folds = nested_folds(['q' + str(j) for j in range(768)], config())
        x.seals = {}
        x.binding = 'toy'
        with self.assertRaises(ValueError):
            x.fold_ids(0, 'outer')
        for f in range(4):
            v = dict(fold=f, sources=sources(), inputs='toy')
            x.register_seal(f, v)
            if f < 3:
                with self.assertRaises(ValueError):
                    x.fold_ids(0, 'outer')
        self.assertEqual(len(x.fold_ids(0, 'outer')), 192)
        with self.assertRaises(ValueError):
            x.register_seal(0, v)

    def test_exact_tri_features_share_raw_geometry_and_ignore_future(self):
        rng = np.random.default_rng(56001)
        q = rng.normal(size=384)
        x = rng.normal(size=(64, 384))
        d = np.sum((x - q)**2, axis=1)
        rows = np.argsort(d)
        d = d[rows]
        stat = dict(full_distances=512, upper_distances=80, expanded=25, neighbor_slots=600,
                    result_updates=100, discovered=433, frontier_size=256, frontier_min=float(d[10]), pending=0)
        g, _ = observed_features(d, rows, stat, None, q, x, 'G', config())
        t, _ = observed_features(d, rows, stat, None, q, x, 'T', config())
        np.testing.assert_array_equal(g, t[:len(g)])
        beta, rho, mask = g[-48:-32], g[-32:-16], g[-16:]
        self.assertTrue(mask.all())
        np.testing.assert_allclose(t[len(g):len(g) + 16], tri_law_probability(beta, rho, 128), rtol=0, atol=0)
        with self.assertRaises(TypeError):
            observed_features(d, rows, stat, None, q, x, 'T', config(), truth=[])
        with self.assertRaises(ValueError):
            observed_features(d[:10], rows[:10], stat, None, q, x, 'T', config())

    def test_degenerate_geometry_has_explicit_mask(self):
        x = np.zeros((64, 384))
        stat = dict(full_distances=512, upper_distances=0, expanded=20, neighbor_slots=512,
                    result_updates=64, discovered=513, frontier_size=64, frontier_min=0.)
        t, _ = observed_features(np.zeros(64), np.arange(64), stat, None, np.zeros(384), x, 'T', config())
        self.assertTrue(np.isfinite(t).all())
        self.assertTrue(np.all(t[59:75] == 0))

    def test_calibration_preserves_late_gain_not_only_next_step(self):
        stages = [dict(retention=.9, hit=0, features={'S': [0]}, stats={'done': 0, 'full_distances': 512}),
                  dict(retention=.9, hit=0, features={'S': [0]}, stats={'done': 0, 'full_distances': 1024}),
                  dict(retention=1., hit=0, features={}, stats={'done': 1, 'full_distances': 2048})]
        scores = {'q': [{f: dict(retention=0., hit=0.) for f in ('S', 'G', 'T', 'T0')} for _ in stages]}
        margins = calibrate({'q': stages}, ['q'], scores, config())
        self.assertAlmostEqual(margins['S']['retention']['0.95'], .1)

    def test_decision_uses_only_current_frozen_scores_and_calibration(self):
        p = dict(family='S', quantile=.5, minimum=512, limit=.002)
        s = dict(full_distances=512, done=0)
        score = {'S': dict(retention=0., hit=0.)}
        self.assertTrue(decide(score, p, zero_margins(), s, config()))
        self.assertFalse(decide(score, dict(p, disabled=True), zero_margins(), s, config()))
        self.assertFalse(decide(score, p, zero_margins(), dict(s, full_distances=256), config()))
        with self.assertRaises(TypeError):
            decide(score, p, zero_margins(), s, config(), truth=1)

    def test_replay_skips_coalesced_checkpoint(self):
        stages = [dict(observed=False, stats=dict(done=0, full_distances=512)),
                  dict(observed=True, stats=dict(done=1, full_distances=1024))]
        p = dict(family='S', quantile=.5, minimum=512, limit=.002)
        j, _ = replay(stages, [{}, {}], p, zero_margins(), config())
        self.assertEqual(j, 1)

    def test_zero_errors_do_not_certify_zero_risk(self):
        r = {str(j): dict(retention=1., hit=0) for j in range(256)}
        v = paired_quality(r, r, config())
        self.assertEqual(v['retention']['interval95'], [0., 0.])
        self.assertGreater(v['retention']['conservative_mean_loss_upper95'], .01)


@unittest.skipUnless(os.environ.get('STEP5_CONTINUATION_V2_NATIVE'), 'requires approved v2 native library')
class ContinuationV2Native(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import faiss
        cls.s = config()
        cls.n = Native(os.environ['STEP5_CONTINUATION_V2_NATIVE'], cls.s)
        rng = np.random.default_rng(56003)
        cls.x = rng.normal(size=(1500, 384))
        cls.x /= np.linalg.norm(cls.x, axis=1)[:, None]
        cls.x[-1] = cls.x[0]
        cls.q = rng.normal(size=(4, 384))
        cls.q /= np.linalg.norm(cls.q, axis=1)[:, None]
        cls.index = faiss.IndexHNSWFlat(384, 16)
        cls.index.hnsw.efConstruction = 200
        cls.index.hnsw.efSearch = 256
        cls.index.add(cls.x.astype(np.float32))

    def test_full_boundary_and_batch_resume_match_faiss_bitwise(self):
        for q in list(self.q) + [self.x[0]]:
            d, i = self.index.search(q.astype(np.float32)[None], 64)
            counts = []
            for boundary in (False, True):
                with BoundarySession(self.n, self.index, q, debug=True) as se:
                    for target in [1, 127, 256, 512, 1024, 2048, 4096, 0]:
                        stats = se.advance(target, boundary)
                        if boundary:
                            self.assertEqual(stats['pending'], 0)
                    sd, si, st = se.snapshot()
                    np.testing.assert_array_equal(sd, d[0])
                    np.testing.assert_array_equal(si, i[0])
                    counts.append(st)
            for key in ('full_distances', 'distance_sequence', 'state_hash', 'batch4_calls', 'scalar_calls'):
                self.assertEqual(counts[0][key], counts[1][key])

    def test_pending_drain_observation_and_snapshot_do_not_mutate(self):
        with BoundarySession(self.n, self.index, self.q[0], debug=True) as se:
            a = se.advance(512, False)
            _, _, st = se.snapshot()
            pending, frontier, visited = se.audit()
            _, _, again = se.snapshot()
            self.assertEqual(st['state_hash'], again['state_hash'])
            self.assertEqual(len(pending), a['pending'])
            self.assertEqual(len(pending), a['discovered'] - (a['full_distances'] - a['upper_distances']) - 1)
            self.assertTrue(visited[pending].all())
            b = se.advance(512, True)
            self.assertEqual(b['pending'], 0)
            self.assertLessEqual(b['full_distances'] - a['full_distances'], 63)

    def test_tree_export_matches_sklearn_with_float32_split_convention(self):
        from sklearn.ensemble import GradientBoostingRegressor
        rng = np.random.default_rng(56007)
        x = rng.normal(size=(500, 7))
        y = x[:, 0]**2 + x[:, 1] * .1
        model = GradientBoostingRegressor(n_estimators=64, max_depth=2, learning_rate=.05, random_state=3).fit(x, y)
        p = Predictor(export_model(model), self.n)
        xx = np.vstack([x, rng.normal(size=(100, 7))])
        np.testing.assert_allclose([p(z) for z in xx], model.predict(xx), atol=1e-12, rtol=0)

    def test_training_prediction_calibration_with_coalesced_states(self):
        rng = np.random.default_rng(56011)
        traces = {}
        for j in range(8):
            stages = []
            for t in range(4):
                active = t in (0, 2)
                stages.append(dict(features={f: rng.normal(size=n).tolist() for f, n in
                    [('S', 27), ('G', 75), ('T', 94), ('T0', 94)]} if active else {},
                    observed=t != 1, retention=[.7, .7, .9, 1.][t], hit=0,
                    stats=dict(done=t == 3, full_distances=[512, 512, 1024, 2048][t])))
            traces[str(j)] = stages
        models, checks = fit_models(traces, list(traces)[:4], self.s, self.n)
        self.assertEqual(checks['S']['retention']['fit_states'], 8)
        score = predictions(traces, ['4', '5'], predictors(models, self.n))
        self.assertEqual(score['4'][1], {})
        margins = calibrate(traces, ['4', '5'], score, self.s)
        policy = dict(family='T', quantile=.5, minimum=512, limit=.002)
        j, _ = replay(traces['4'], score['4'], policy, margins, self.s)
        self.assertIn(j, [0, 2, 3])

    def test_invalid_closed_state_and_future_args_rejected(self):
        with patch('sys.unraisablehook') as hook:
            with self.assertRaises(TypeError):
                BoundarySession(self.n, self.index, self.q[0], truth=[])
            hook.assert_not_called()
        se = BoundarySession(self.n, self.index, self.q[0])
        with self.assertRaises(ValueError):
            se.advance(512, 1)
        se.close()
        with self.assertRaises(ValueError):
            se.audit()

    def test_real_policy_selective_gate_and_independent_workers(self):
        import faiss
        from scripts.run_tls_rag_step5_ann_continuation_v2 import measure, quality_rows
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            np.save(root / 'corpus.npy', self.x)
            save(root / 'ids.json', ['d' + str(j) for j in range(len(self.x))])
            faiss.write_index(self.index, str(root / 'index.faiss'))
            import shutil
            shutil.copyfile(os.environ['STEP5_CONTINUATION_V2_NATIVE'], root / 'continuation_v2.dylib')
            paths = dict(index=str(root / 'index.faiss'), index_sha=sha(root / 'index.faiss'),
                         corpus=str(root / 'corpus.npy'), ids=str(root / 'ids.json'))
            runtime = Runtime(paths, self.n, self.s)
            traces = runtime.trace(self.q[0])
            first = traces[0]
            models = {}
            for f in ('S', 'G', 'T', 'T0'):
                x, _ = observed_features(first['distances'], first['rows'], first['stats'], None,
                                         self.q[0], self.x, f, self.s)
                models[f] = {name: constant_model(len(x), .2 if name == 'retention' else 0.) for name in ('retention', 'hit')}
            policy = dict(family='Ts', quantile=.5, limit=.002, minimum=512)
            bundle = dict(models=models, margins=zero_margins(), policies={'policy:Ts': policy})
            value = runtime.call(self.q[0], 'policy:Ts', policy, predictors(models, self.n), zero_margins())
            self.assertEqual(value['stats']['tri_calls'], 0)
            self.assertEqual(value['stats']['geometry_calls'], 0)
            self.assertTrue(value['stats']['done'])
            for m in ('unknown', 'fixed:300', 'path:300', 'policy:BAD'):
                with self.assertRaises(ValueError):
                    runtime.call(self.q[0], m)
            inputs = SimpleNamespace(paths=paths, rows={'q': dict(query=self.q[0].tolist(), qrels={})}, truth={'q': ['d0']})
            methods = ['original:256', 'natural', 'adj_gap', 'policy:Ts', 'paid-noop:T']
            expected = quality_rows(runtime, inputs, ['q'], methods, bundle)
            runtime.close()
            bp = root / 'models.json'
            save(bp, bundle)
            times = measure(root, 'synthetic', inputs, ['q'], methods, self.s, perf_counter(), expected, bp, repeats=1)
            self.assertEqual(len(times), len(methods))
            self.assertTrue(all(t['ms'] > 0 for t in times))


if __name__ == '__main__':
    unittest.main()
