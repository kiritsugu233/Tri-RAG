import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tri_rag_harness import tls_rag_step4_probe as p
from tri_rag_harness import tls_rag_step4_nfcorpus as adapter


def fixture():
    protocol = copy.deepcopy(p.load_protocol())
    protocol['embedding']['dimension'] = 8
    protocol['retrieval'].update(m_prime=2, k_gt=2, k_ctx=2, budget_grid=[3, 6, 12])
    protocol['role_counts'] = {role: 8 for role in p.ROLES}
    protocol['selection'].update(minimum_retention=0., maximum_hit_drop=1.)
    rng = np.random.default_rng(991)
    corpus = rng.normal(size=(12, 8))
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    vectors = rng.normal(size=(32, 8))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    queries = {f'query-{i:03d}': f'question topic{i}' for i in range(32)}
    roles, audit = p.assign_roles(queries, queries, protocol['role_counts'], protocol['split_seed'])
    ids = [f'passage-{i:03d}' for i in range(12)]
    env = p.make_environment(ids, [f'document article {i}' for i in range(12)],
                             corpus, queries, vectors, roles, protocol)
    return protocol, env, roles


class ProbeTests(unittest.TestCase):
    def test_protocol_is_pinned_and_component_behavior_preserved(self):
        protocol = p.load_protocol()
        config = p.component_config(protocol)
        self.assertEqual(config.raw['calibration']['family_wise_alpha'], .2)
        self.assertEqual(config.raw['controller']['delta_gain'], .65)
        self.assertEqual(config.raw['controller']['tau_sufficient'], .15)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'protocol.json'
            modified = {**protocol, 'split_seed': 1}
            path.write_text(json.dumps(modified))
            with patch.object(p, 'PROTOCOL_PATH', path), self.assertRaisesRegex(ValueError, 'protocol changed'):
                p.load_protocol()

    def test_duplicate_texts_stay_in_one_family_and_order_independent(self):
        queries = {f'q{i}': f'distinct question {i}' for i in range(12)}
        queries.update({'q-repeat': 'DISTINCT   question 0!!!'})
        counts = {role: 2 for role in p.ROLES}
        first, audit = p.assign_roles(queries, queries, counts, 31)
        second, _ = p.assign_roles(dict(reversed(list(queries.items()))), reversed(list(queries)), counts, 31)
        self.assertEqual(first, second)
        selected = [qid for values in first.values() for qid in values]
        self.assertEqual(len(selected), len(set(selected)))
        self.assertFalse('q0' in selected and 'q-repeat' in selected)
        self.assertEqual(len({row['text_family_sha256'] for row in audit}), 8)

    def test_insufficient_units_fail_without_reusing_queries(self):
        with self.assertRaisesRegex(ValueError, 'insufficient'):
            p.assign_roles({'a': 'same text', 'b': 'SAME text'}, ['a', 'b'], {r: 1 for r in p.ROLES}, 1)

    def test_qrel_staging_keeps_zero_grade_query_and_rejects_unknown_doc(self):
        roles = {role: [f'q{i}'] for i, role in enumerate(p.ROLES)}
        rows = [(f'q{i}', 'd1', i) for i in range(4)]
        labels = adapter.partition_qrels(rows, roles, {'d1'})
        self.assertEqual(labels[p.ROLES[0]], {'q0': []})
        with self.assertRaisesRegex(ValueError, 'unknown corpus'):
            adapter.partition_qrels(rows, roles, {'other'})

    def test_qrel_parser_validates_schema_duplicates_and_grades(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'train.tsv'
            path.write_text('query-id\tcorpus-id\tscore\nq1\td1\t0\nq2\td1\t2\n')
            ids, rows = adapter.read_train_qrels(path)
            self.assertEqual(ids, {'q1', 'q2'})
            self.assertEqual(len(rows), 2)
            path.write_text(path.read_text() + 'q1\td1\t1\n')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                adapter.read_train_qrels(path)

    def test_source_allowlist_excludes_official_dev_test_labels(self):
        self.assertEqual(set(adapter.SOURCES), {'corpus.parquet', 'queries.parquet', 'train.tsv'})
        self.assertTrue(all('/resolve/main/' not in url for url in adapter.SOURCES.values()))
        self.assertFalse(any('/test.tsv' in url or '/dev.tsv' in url for url in adapter.SOURCES.values()))

    def test_projection_is_fixed_gaussian_and_not_renormalized(self):
        protocol, env, roles = fixture()
        expected = np.random.default_rng(24103).normal(scale=1 / np.sqrt(2), size=(2, 8))
        np.testing.assert_array_equal(env.upstream.projection_matrix, expected)
        np.testing.assert_array_equal(env.upstream.projected_corpus, env.upstream.corpus_embeddings @ expected.T)
        self.assertFalse(np.allclose(np.linalg.norm(env.upstream.projected_corpus, axis=1), 1.))

    def test_bad_embeddings_and_overlapping_roles_fail(self):
        protocol, env, roles = fixture()
        kwargs = dict(corpus_ids=env.upstream.corpus_ids, corpus_texts=env.upstream.corpus_texts,
            corpus_vectors=env.upstream.corpus_embeddings,
            queries={q.query_id: q.text for q in env.queries},
            query_vectors=np.vstack([q.embedding for q in env.queries]), roles=roles, protocol=protocol)
        with self.assertRaisesRegex(ValueError, 'normalized'):
            p.make_environment(**{**kwargs, 'corpus_vectors': env.upstream.corpus_embeddings * 2})
        bad = copy.deepcopy(roles)
        bad[p.ROLES[1]][0] = bad[p.ROLES[0]][0]
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            p.make_environment(**{**kwargs, 'roles': bad})

    def test_pipeline_closes_every_role_before_labels_and_reproduces(self):
        protocol, env, roles = fixture()
        # Only this success-path integration test gets a larger independent
        # bound-calibration set. Eight queries can split 4+4 across two bins,
        # below the unchanged minimum of eight per cell on another BLAS backend.
        rng = np.random.default_rng(1001)
        bound_ids = [f'pipeline-bound-{i:03d}' for i in range(32)]
        vectors = rng.normal(size=(32, protocol['embedding']['dimension']))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        retained = [q for q in env.queries if q.query_id not in roles[p.ROLES[1]]]
        queries = {q.query_id: q.text for q in retained}
        queries.update({qid: f'independent calibration subject{i}' for i, qid in enumerate(bound_ids)})
        roles[p.ROLES[1]] = bound_ids
        protocol['role_counts'][p.ROLES[1]] = len(bound_ids)
        env = p.make_environment(env.upstream.corpus_ids, env.upstream.corpus_texts,
            env.upstream.corpus_embeddings, queries,
            np.vstack([*[q.embedding for q in retained], *vectors]), roles, protocol)
        results = []
        with tempfile.TemporaryDirectory() as root:
            for name in ('a', 'b'):
                output = Path(root) / name
                opened = []
                def loader(role, ids):
                    self.assertTrue((output / f'{role}.phase_a_closed.json').is_file())
                    if role in p.ROLES[2:]:
                        self.assertTrue((output / f'{role}.decisions.jsonl').is_file())
                    if role == 'query_probe':
                        self.assertIsNotNone(json.loads((output / 'selection.json').read_text())['selected_candidate'])
                    opened.append(role)
                    return {qid: list(env.upstream.corpus_ids) for qid in ids}
                result = p.run_probe(env, roles, protocol, loader, output)
                tables = json.loads((output / 'calibration_tables.json').read_text())
                minimum = p.component_config(protocol).raw['calibration']['minimum_cell_query_count']
                self.assertEqual(minimum, 8)
                for table in tables:
                    for outcome in ('remaining_useful_evidence_event', 'current_context_sufficiency_event'):
                        cells = [c for c in table['cells'] if c['stage'] == 0 and c['outcome'] == outcome]
                        self.assertEqual(sum(c['trials'] for c in cells), 32)
                        self.assertTrue(all(c['trials'] >= minimum and c['valid']
                                            for c in cells if c['trials']), cells)
                self.assertEqual(opened, list(p.ROLES))
                self.assertEqual(result['status'], 'descriptive_probe_targets_met')
                self.assertFalse(result['certified'])
                results.append(json.loads((output / 'artifact_manifest.json').read_text()))
                rows = [json.loads(line) for line in (output / 'query_probe.metrics.jsonl').read_text().splitlines()]
                for row in rows:
                    self.assertEqual(row['work']['total_original_distances'], row['budget'])
                    self.assertEqual(row['work']['pilot_original_distances'] + row['work']['expansion_original_distances'] + row['work'].get('reference_original_distances', 0), row['budget'])
                for qid in roles['query_probe']:
                    fixed = sorted((r for r in rows if r['query_id'] == qid and r['method'].startswith('fixed-')), key=lambda r: r['budget'])
                    self.assertEqual([r['retention'] for r in fixed], sorted(r['retention'] for r in fixed))
                    self.assertEqual(fixed[-1]['retention'], 1.)
            self.assertEqual(results[0], results[1])

    def test_four_plus_four_underpowered_bins_keep_probe_closed(self):
        protocol, env, roles = fixture()
        opened = []
        # Deliberately separated test scores reproduce the observed 4+4 bins
        # without depending on platform-specific rounding near a score of one.
        # Real fitting, binning, bounds, controller and selection still execute.
        halves = {qid: index % 2 for ids in roles.values() for index, qid in enumerate(ids)}
        original_predict = p.s3.LinearScoreModel.predict
        def predict(model, state):
            if model.outcome == 'current_context_sufficiency_event':
                return p.s3.ScorePrediction(.25 + .5 * halves[state.query_id], True, 'valid')
            return original_predict(model, state)
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'run'
            def loader(role, ids):
                opened.append(role)
                return {qid: list(env.upstream.corpus_ids) for qid in ids}
            with patch.object(p.s3.LinearScoreModel, 'predict', predict):
                result = p.run_probe(env, roles, protocol, loader, output)
            self.assertEqual(opened, list(p.ROLES[:3]))
            self.assertEqual(result['status'], 'no_tune_candidate')
            self.assertIsNone(result['selected_candidate'])
            self.assertFalse(any(output.glob('query_probe.*')))
            for table in json.loads((output / 'calibration_tables.json').read_text()):
                for stage in range(3):
                    cells = [c for c in table['cells'] if c['stage'] == stage
                             and c['outcome'] == 'current_context_sufficiency_event']
                    self.assertEqual([c['trials'] for c in cells], [4, 4])
                    self.assertTrue(all(not c['valid'] and c['lower_limit'] == 0.
                                        and c['upper_limit'] == 1. for c in cells))
                self.assertEqual(result['summaries']['query_tune'][table['candidate_id']]['mean_original_distances'], 12.)

    def test_no_eligible_tune_candidate_keeps_probe_unopened(self):
        protocol, env, roles = fixture()
        protocol['selection']['minimum_retention'] = 1.1
        opened = []
        with tempfile.TemporaryDirectory() as root:
            def loader(role, ids):
                opened.append(role)
                return {qid: list(env.upstream.corpus_ids) for qid in ids}
            output = Path(root) / 'run'
            result = p.run_probe(env, roles, protocol, loader, output)
            self.assertEqual(result['status'], 'no_tune_candidate')
            self.assertEqual(opened, list(p.ROLES[:3]))
            self.assertFalse((output / 'query_probe.features.jsonl').exists())

    def test_proxy_gain_depends_on_future_context_not_candidate_coverage(self):
        from types import SimpleNamespace as NS
        prepared = {'q': NS(stages=[NS(decision_input=NS(context_ids=ids))
                                     for ids in (('a',), ('b',), ('c',))])}
        supervision = p.supervision(prepared, {'q': ['c']})['q']
        self.assertEqual([r['current_context_is_sufficient'] for r in supervision], [False, False, True])
        self.assertEqual([r['later_context_has_useful_evidence'] for r in supervision], [True, True, False])

    def test_bundle_identity_and_late_label_tampering_are_rejected(self):
        protocol, env, roles = fixture()
        with tempfile.TemporaryDirectory() as root:
            bundle = Path(root)
            p.save(bundle / 'roles.json', roles)
            p.save_rows(bundle / 'corpus.jsonl',
                ({'passage_id': key, 'text': text} for key, text in
                 zip(env.upstream.corpus_ids, env.upstream.corpus_texts)))
            p.save_rows(bundle / 'queries.jsonl',
                ({'query_id': q.query_id, 'text': q.text} for q in env.queries))
            np.save(bundle / 'corpus.npy', env.upstream.corpus_embeddings)
            np.save(bundle / 'queries.npy', np.vstack([q.embedding for q in env.queries]))
            for role in p.ROLES:
                p.save(bundle / f'{role}.labels.json', {qid: [] for qid in roles[role]})
            binding = {'schema': 'tls_rag_nfcorpus_probe_binding_v2',
                'protocol_fingerprint': p.fingerprint(protocol), 'embedding': protocol['embedding'],
                'source_revisions': {'dataset': protocol['dataset_revision'], 'qrels': protocol['qrels_revision']},
                'source_code_sha256': {name: p.sha256(p.ROOT / name) for name in p.CODE_FILES},
                'files': {path.name: p.sha256(path) for path in bundle.iterdir()}}
            p.save(bundle / 'binding.json', binding)
            loaded, assigned, loader, _ = p.load_bundle(bundle, protocol)
            self.assertEqual(assigned, roles)
            self.assertEqual(loader(p.ROLES[0], roles[p.ROLES[0]]), {qid: [] for qid in roles[p.ROLES[0]]})
            target = bundle / f'{p.ROLES[0]}.labels.json'
            target.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'changed after'):
                loader(p.ROLES[0], roles[p.ROLES[0]])
            with self.assertRaisesRegex(ValueError, 'hash/path mismatch'):
                p.load_bundle(bundle, protocol)

    def test_unknown_bundle_paths_rejected_before_access(self):
        protocol = p.load_protocol()
        with tempfile.TemporaryDirectory() as root:
            bundle = Path(root)
            p.save(bundle / 'binding.json', {'schema': 'tls_rag_nfcorpus_probe_binding_v2',
                'protocol_fingerprint': p.fingerprint(protocol),
                'source_revisions': {'dataset': protocol['dataset_revision'], 'qrels': protocol['qrels_revision']},
                'files': {'../forbidden.json': '0' * 64}})
            with self.assertRaisesRegex(ValueError, 'unexpected bundle file'):
                p.load_bundle(bundle, protocol)

    def test_preexisting_output_is_never_overwritten(self):
        protocol, env, roles = fixture()
        with tempfile.TemporaryDirectory() as root, self.assertRaises(FileExistsError):
            p.run_probe(env, roles, protocol, lambda *args: {}, Path(root))


if __name__ == '__main__':
    unittest.main()
