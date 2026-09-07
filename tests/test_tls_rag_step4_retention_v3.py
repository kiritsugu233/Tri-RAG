import copy
import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from tri_rag_harness import tls_rag_step4_retention_v3 as r
from tri_rag_harness import tls_rag_step4_probe as v2
from test_tls_rag_step4_probe import fixture


def protocol_fixture():
    p = copy.deepcopy(r.load_protocol())
    p['stop_thresholds'] = [0.]
    p['residual_quantiles'] = [.5]
    p['acceptance'].update(minimum_retention=0., maximum_hit_drop=1., maximum_mean_original_distances_exclusive=12)
    return p


def provider_fixture(env, roles, parent_protocol):
    def provide(role, output):
        v2.prepare_role(env, v2.component_config(parent_protocol), roles[role], output, role)
        return output / f'{role}.features.jsonl'
    return provide


def write_bundle(bundle, parent_protocol, env, roles):
    bundle.mkdir()
    v2.save(bundle / 'roles.json', roles)
    v2.save_rows(bundle / 'corpus.jsonl', ({'passage_id': key, 'text': text} for key, text in
                                        zip(env.upstream.corpus_ids, env.upstream.corpus_texts)))
    v2.save_rows(bundle / 'queries.jsonl', ({'query_id': q.query_id, 'text': q.text} for q in env.queries))
    np.save(bundle / 'corpus.npy', env.upstream.corpus_embeddings)
    np.save(bundle / 'queries.npy', np.vstack([q.embedding for q in env.queries]))
    for role in v2.ROLES:
        v2.save(bundle / f'{role}.labels.json', {qid: list(env.upstream.corpus_ids) for qid in roles[role]})
    binding = {'schema': 'tls_rag_nfcorpus_probe_binding_v2',
        'protocol_fingerprint': v2.fingerprint(parent_protocol), 'embedding': parent_protocol['embedding'],
        'source_revisions': {'dataset': parent_protocol['dataset_revision'], 'qrels': parent_protocol['qrels_revision']},
        'source_code_sha256': {name: v2.sha256(v2.ROOT / name) for name in v2.CODE_FILES},
        'files': {p.name: v2.sha256(p) for p in bundle.iterdir()}}
    v2.save(bundle / 'binding.json', binding)
    return binding


class RetentionV3Tests(unittest.TestCase):
    def test_protocol_has_27_candidates_and_unchanged_acceptance(self):
        p = r.load_protocol()
        self.assertEqual(len(r.policy_grid(p)), 27)
        self.assertEqual(p['acceptance']['minimum_retention'], .9)
        self.assertEqual(p['acceptance']['maximum_hit_drop'], .02)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'protocol.json'
            path.write_text(json.dumps({**p, 'stop_thresholds': [0.]}))
            with patch.object(r, 'PROTOCOL_PATH', path), self.assertRaisesRegex(ValueError, 'protocol changed'):
                r.load_protocol()

    def features(self, root):
        parent, env, roles = fixture()
        path = provider_fixture(env, roles, parent)(v2.ROLES[0], Path(root))
        return parent, env, roles, r.parse_features(path, roles[v2.ROLES[0]], [3, 6, 12]), path

    def test_continuous_retention_targets_are_not_boolean_hit_labels(self):
        with tempfile.TemporaryDirectory() as root:
            _, env, _, prepared, _ = self.features(root)
            targets, records = r.retention_targets(prepared, env)
            self.assertTrue(any(row['retention'] == .5 for row in records))
            models = r.fit_models(prepared, targets, protocol_fixture())
            model = models[2, 0]
            scores = [model.predict(q.stages[0].states[2]).score for q in prepared.values()]
            self.assertAlmostEqual(np.mean(scores), np.mean([values[0] for values in targets.values()]), places=2)
            self.assertEqual(model.outcome, 'exact_top_k_retention_fraction')

    def test_low_retention_score_expands_even_with_valid_context(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, _, prepared, _ = self.features(root)
            state = next(iter(prepared.values())).stages[0].states[2]
            model = NS(predict=lambda state: NS(score=.65, valid=True))
            controller = r.Controller(r.Policy(2, .9, .5), {(2, 0): model},
                                      {(2, 0, .5): {'valid': True, 'margin': 0.}}, [3, 6, 12])
            action = controller.choose(state)
            self.assertEqual(action['action'], 'EXPAND_TO_NEXT_GRID_VALUE')
            self.assertEqual(action['reason'], 'retention_score_requires_expand')

    def test_stage_adaptation_margin_and_explicit_terminal_failure(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, _, prepared, _ = self.features(root)
            states = next(iter(prepared.values())).stages
            models = {(2, i): NS(predict=lambda state: NS(score=.96 if state.stage else .65, valid=True)) for i in range(3)}
            cal = {(2, i, .5): {'valid': True, 'margin': .02} for i in range(3)}
            controller = r.Controller(r.Policy(2, .9, .5), models, cal, [3, 6, 12])
            self.assertEqual(controller.choose(states[0].states[2])['action'], 'EXPAND_TO_NEXT_GRID_VALUE')
            self.assertEqual(controller.choose(states[1].states[2])['reason'], 'retention_score_stop')
            cal[2, 1, .5]['margin'] = .2
            self.assertEqual(controller.choose(states[1].states[2])['action'], 'EXPAND_TO_NEXT_GRID_VALUE')
            invalid = replace(states[2].states[2], base_state_valid=False)
            self.assertEqual(controller.choose(invalid)['reason'], 'terminal_budget_without_retention_assertion')

    def test_no_calibration_and_forbidden_or_nonfinite_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, _, prepared, _ = self.features(root)
            state = next(iter(prepared.values())).stages[0].states[2]
            controller = r.Controller(r.Policy(2, .9, .5), {}, {}, [3, 6, 12])
            self.assertEqual(controller.choose(state)['reason'], 'invalid_or_uncalibrated_expand')
            with self.assertRaises(TypeError):
                controller.choose({'state': state, 'realized_retention': 1.})
            with self.assertRaises(ValueError):
                controller.choose(replace(state, feature_values=state.feature_values + (('realized_retention', 1.),)))
            with self.assertRaises(ValueError):
                controller.choose(replace(state, feature_values=((state.feature_values[0][0], float('nan')),)+state.feature_values[1:]))

    def test_quantile_margin_and_insufficient_calibration(self):
        state = NS(base_state_valid=True, required_risk_profile_valid=True, evidence_plan_valid=True)
        prepared = {f'q{i}': NS(stages=[NS(states={2: state})]) for i in range(8)}
        targets = {f'q{i}': [.1 * i] for i in range(8)}
        model = NS(predict=lambda state: NS(score=.8, valid=True))
        cal, rows = r.calibrate(prepared, targets, {(2, 0): model}, protocol_fixture())
        self.assertAlmostEqual(cal[2, 0, .5]['margin'], .5)
        self.assertEqual(len(rows), 8)
        cal, _ = r.calibrate(dict(list(prepared.items())[:2]), targets, {(2, 0): model}, protocol_fixture())
        self.assertFalse(cal[2, 0, .5]['valid'])

    def test_duplicate_missing_or_extra_cache_states_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, roles, _, path = self.features(root)
            original = path.read_text().splitlines()
            path.write_text('\n'.join(original + [original[0]]) + '\n')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                r.parse_features(path, roles[v2.ROLES[0]], [3, 6, 12])
            path.write_text('\n'.join(original[1:]) + '\n')
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                r.parse_features(path, roles[v2.ROLES[0]], [3, 6, 12])

    def test_unchanged_gates_reject_v2_collapse_and_no_saving(self):
        p = r.load_protocol()
        summary = {'retention': .6453, 'hit_at_context': .6211, 'exact_hit_at_context': .6328, 'mean_original_distances': 32.}
        self.assertEqual(r.gates(summary, p), {'retention': False, 'context_hit_drop': True, 'original_distance_saving': True})
        self.assertFalse(r.gates({**summary, 'retention': .96, 'mean_original_distances': 512.}, p)['original_distance_saving'])

    def test_complete_pipeline_is_reproducible_and_closes_probe_before_labels(self):
        parent, env, roles = fixture()
        with tempfile.TemporaryDirectory() as root:
            manifests = []
            for name in ('a', 'b'):
                output = Path(root) / name
                opened = []
                def labels(role, ids):
                    self.assertTrue((output / f'{role}.phase_a_closed.json').exists())
                    self.assertTrue((output / f'{role}.decisions.jsonl').exists())
                    if role == 'query_probe':
                        self.assertIsNotNone(json.loads((output / 'selection.json').read_text())['selected_candidate'])
                    opened.append(role)
                    return {qid: list(env.upstream.corpus_ids) for qid in ids}
                result = r.run(env, roles, protocol_fixture(), provider_fixture(env, roles, parent), labels, output)
                self.assertEqual(opened, ['query_tune', 'query_probe'])
                self.assertEqual(result['status'], 'heldout_targets_met')
                self.assertFalse(result['certified'])
                self.assertEqual(set(result['matched_feature_deltas']), {'row3_minus_row2', 'row4_minus_row3'})
                manifests.append(json.loads((output / 'artifact_manifest.json').read_text()))
            self.assertEqual(*manifests)

    def test_no_tune_candidate_keeps_probe_features_and_labels_closed(self):
        parent, env, roles = fixture()
        p = protocol_fixture()
        p['acceptance']['minimum_retention'] = 1.1
        requested = []
        provider = provider_fixture(env, roles, parent)
        def guarded(role, out):
            requested.append(role)
            return provider(role, out)
        with tempfile.TemporaryDirectory() as root:
            result = r.run(env, roles, p, guarded,
                           lambda role, ids: {qid: [] for qid in ids}, Path(root) / 'out')
            self.assertEqual(result['status'], 'no_tune_candidate')
            self.assertFalse(result['probe_opened'])
            self.assertEqual(requested, list(v2.ROLES[:3]))

    def test_parent_binding_cache_reuse_and_opened_probe_rejection(self):
        parent, env, roles = fixture()
        parent['selection']['minimum_retention'] = 1.1
        v3_protocol = protocol_fixture()
        with tempfile.TemporaryDirectory() as root:
            previous = Path(root) / 'v2'
            previous.mkdir()
            binding = write_bundle(previous / 'bundle', parent, env, roles)
            v2.run_probe(env, roles, parent, lambda role, ids: {qid: list(env.upstream.corpus_ids) for qid in ids},
                         previous / 'evaluation', binding=binding)
            with patch.object(v2, 'load_protocol', return_value=parent):
                loaded, assigned, loader, provider, lineage = r.verify_parent(previous, v3_protocol)
                output = Path(root) / 'reuse'
                output.mkdir()
                with patch.object(v2, 'prepare_role', side_effect=AssertionError('must reuse cache')):
                    cached = provider('query_tune', output)
                self.assertEqual(v2.sha256(cached), v2.sha256(previous / 'evaluation/query_tune.features.jsonl'))
                self.assertEqual(assigned, roles)
                full_output = Path(root) / 'acceptance'
                with patch.object(r, 'load_protocol', return_value=v3_protocol), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(r.main(['--previous-run', str(previous), '--output', str(full_output)]), 0)
                self.assertEqual(json.loads((full_output / 'result.json').read_text())['status'], 'heldout_targets_met')
                cached_source = previous / 'evaluation/query_tune.features.jsonl'
                original = cached_source.read_bytes()
                cached_source.write_bytes(original + b'\n')
                with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
                    r.verify_parent(previous, v3_protocol)
                cached_source.write_bytes(original)
                (previous / 'evaluation/query_probe.decisions.jsonl').write_text('')
                with self.assertRaisesRegex(ValueError, 'already been opened'):
                    r.verify_parent(previous, v3_protocol)

    def test_existing_output_refused(self):
        parent, env, roles = fixture()
        with tempfile.TemporaryDirectory() as root, self.assertRaises(FileExistsError):
            r.run(env, roles, protocol_fixture(), None, None, Path(root))


if __name__ == '__main__':
    unittest.main()
