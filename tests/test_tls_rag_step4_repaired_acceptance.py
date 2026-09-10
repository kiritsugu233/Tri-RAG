"""Offline synthetic compatibility tests; no real roles, network or GPU."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tri_rag_harness import tls_rag_step4_repaired_acceptance as a
from tri_rag_harness import tls_rag_step4_probe as v2
from tri_rag_harness import tls_rag_step4_retention_v3 as v3
from test_tls_rag_step4_probe import fixture
from test_tls_rag_step4_retention_v3 import write_bundle


def overwrite(path, data):
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + '\n')


class RepairedAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v3_protocol = v3.load_protocol()
        cls.seed = tempfile.TemporaryDirectory()
        cls.parent_protocol, cls.env, cls.roles = fixture()
        cls.parent_protocol['selection']['minimum_retention'] = 1.1
        root = Path(cls.seed.name).resolve() / 'parent'
        root.mkdir()
        binding = write_bundle(root / 'bundle', cls.parent_protocol, cls.env, cls.roles)
        binding['source_code_sha256'] = dict(a.HISTORICAL_CODE_SHA256)
        overwrite(root / 'bundle/binding.json', binding)
        with contextlib.redirect_stdout(io.StringIO()):
            v2.run_probe(cls.env, cls.roles, cls.parent_protocol,
                         lambda role, ids: {qid: list(cls.env.upstream.corpus_ids) for qid in ids},
                         root / 'evaluation', binding)
        cls.template = root

    @classmethod
    def tearDownClass(cls):
        cls.seed.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.previous = self.root / 'parent'
        shutil.copytree(self.template, self.previous)
        self.protocol_patch = patch.object(v2, 'load_protocol', return_value=copy.deepcopy(self.parent_protocol))
        self.protocol_patch.start()
        self.addCleanup(self.protocol_patch.stop)
        self.v3_patch = patch.object(v3, 'load_protocol', return_value=copy.deepcopy(self.v3_protocol))
        self.v3_patch.start()
        self.addCleanup(self.v3_patch.stop)

    def parent(self):
        return a.validate_historical_parent(self.previous)

    def run_pipeline(self, name='out'):
        with contextlib.redirect_stdout(io.StringIO()):
            return a.run_acceptance(self.previous, self.root / name)

    def refresh_evaluation(self):
        directory = self.previous / 'evaluation'
        overwrite(directory / 'artifact_manifest.json',
                  {p.name: v2.sha256(p) for p in directory.iterdir() if p.name != 'artifact_manifest.json'})

    def test_pinned_historical_hashes_match_git_reference(self):
        for name, digest in a.HISTORICAL_CODE_SHA256.items():
            contents = subprocess.check_output(['git', 'show', a.HISTORICAL_COMMIT + ':' + name], cwd=v2.ROOT)
            self.assertEqual(hashlib.sha256(contents).hexdigest(), digest)
        self.assertNotEqual(a.HISTORICAL_CODE_SHA256['src/tri_rag_harness/tri_law.py'],
                            v2.sha256(v2.ROOT / 'src/tri_rag_harness/tri_law.py'))

    def test_old_loader_still_rejects_but_explicit_historical_validation_accepts(self):
        with self.assertRaisesRegex(ValueError, 'code identity mismatch'):
            v2.load_bundle(self.previous / 'bundle', self.parent_protocol)
        parent = self.parent()
        self.assertEqual(parent.roles, self.roles)
        self.assertEqual(a.load_environment(parent).upstream.corpus_ids, self.env.upstream.corpus_ids)

    def test_mixed_or_current_source_hashes_are_rejected(self):
        path = self.previous / 'bundle/binding.json'
        binding = a.read_json(path)
        for replacement in ('0' * 64, v2.sha256(v2.ROOT / 'src/tri_rag_harness/tri_law.py')):
            binding['source_code_sha256']['src/tri_rag_harness/tri_law.py'] = replacement
            overwrite(path, binding)
            with self.assertRaisesRegex(ValueError, 'unsupported historical source'):
                self.parent()

    def test_bundle_payload_and_old_feature_tampering_are_rejected(self):
        for relative in ('bundle/corpus.npy', 'evaluation/query_tune.features.jsonl'):
            path = self.previous / relative
            before = path.read_bytes()
            path.write_bytes(before + b'bad')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                self.parent()
            path.write_bytes(before)

    def test_model_metadata_and_evaluation_identity_mismatch_rejected(self):
        path = self.previous / 'bundle/binding.json'
        before = path.read_text()
        binding = a.read_json(path)
        binding['embedding']['revision'] = 'wrong'
        overwrite(path, binding)
        with self.assertRaisesRegex(ValueError, 'protocol/model'):
            self.parent()
        path.write_text(before)
        path = self.previous / 'evaluation/roles.json'
        overwrite(path, {})
        self.refresh_evaluation()
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.parent()

    def test_historical_status_and_selection_dependencies_are_verified(self):
        path = self.previous / 'evaluation/result.json'
        before = path.read_text()
        result = a.read_json(path)
        result['status'] = 'descriptive_probe_targets_met'
        overwrite(path, result)
        self.refresh_evaluation()
        with self.assertRaisesRegex(ValueError, 'unsuccessful v2 tune'):
            self.parent()
        path.write_text(before)
        path = self.previous / 'evaluation/selection.json'
        selection = a.read_json(path)
        selection['tune_metrics_sha256'] = '0' * 64
        overwrite(path, selection)
        self.refresh_evaluation()
        with self.assertRaisesRegex(ValueError, 'selection dependency'):
            self.parent()

    def test_changed_source_or_closure_stops_before_label_read(self):
        original_sources = a.current_sources
        calls = 0
        def changed_sources():
            nonlocal calls
            calls += 1
            return original_sources() if calls == 1 else {}
        with patch.object(a, 'current_sources', side_effect=changed_sources), self.assertRaisesRegex(ValueError, 'source/binding changed'):
            self.run_pipeline('source-change')
        self.assertFalse((self.root / 'source-change/evaluation/query_tune.opened_labels.json').exists())
        original_closed = a.check_closed
        def corrupt(out, role):
            (out / f'{role}.decisions.jsonl').write_text('corrupted')
            return original_closed(out, role)
        with patch.object(a, 'check_closed', side_effect=corrupt), self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.run_pipeline('closure-change')
        self.assertFalse((self.root / 'closure-change/evaluation/query_tune.opened_labels.json').exists())

    def test_unsafe_manifests_and_symlinks_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            a.checked_manifest(self.previous, {'../outside': '0' * 64})
        path = self.previous / 'evaluation/result.json'
        payload = path.read_bytes()
        path.unlink()
        other = self.root / 'outside.json'
        other.write_bytes(payload)
        path.symlink_to(other)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.parent()

    def test_duplicate_json_keys_rejected(self):
        path = self.root / 'bad.json'
        path.write_text('{"x":1,"x":2}')
        with self.assertRaisesRegex(ValueError, 'duplicate JSON'):
            a.read_json(path)

    def test_any_historical_probe_artifact_or_manifest_entry_rejected(self):
        path = self.previous / 'evaluation/query_probe.anything'
        path.write_text('')
        with self.assertRaisesRegex(ValueError, 'probe already'):
            self.parent()
        path.unlink()
        path = self.previous / 'evaluation/artifact_manifest.json'
        manifest = a.read_json(path)
        manifest['query_probe.features.jsonl'] = '0' * 64
        overwrite(path, manifest)
        with self.assertRaisesRegex(ValueError, 'manifest records'):
            self.parent()

    def test_late_input_changes_and_probe_opening_rejected(self):
        parent = self.parent()
        path = self.previous / 'bundle/query_probe.labels.json'
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed after'):
            parent.check()
        (self.previous / 'evaluation/query_probe.opened_labels.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'probe already'):
            parent.check()

    def test_rebuild_never_copies_old_numerical_features(self):
        # Hash-valid but unusable historical feature payload: the new provider
        # must compute from vectors instead of parsing or copying this cache.
        path = self.previous / 'evaluation/query_tune.features.jsonl'
        path.write_text('historical numerical cache not reusable\n')
        self.refresh_evaluation()
        parent = self.parent()
        env = a.load_environment(parent)
        out = self.root / 'features'
        out.mkdir()
        with patch.object(v2, 'prepare_role', wraps=v2.prepare_role) as prepare:
            rebuilt = a.rebuild_features(parent, env, 'query_tune', out)
        self.assertEqual(prepare.call_count, 1)
        self.assertNotEqual(rebuilt.read_bytes(), path.read_bytes())
        parsed = v3.parse_features(rebuilt, self.roles['query_tune'], [3, 6, 12])
        self.assertEqual(set(parsed), set(self.roles['query_tune']))

    def test_pipeline_rebuilds_all_roles_and_enforces_selection_and_closure(self):
        prepared, opened = [], []
        original_prepare, original_read = a.rebuild_features, a.read_json
        def prepare(parent, env, role, out):
            prepared.append(role)
            if role == 'query_probe':
                self.assertIsNotNone(original_read(out / 'selection.json')['selected_candidate'])
                self.assertTrue(parent.claim_path.is_dir())
            return original_prepare(parent, env, role, out)
        def read(path):
            if str(path).endswith('.labels.json'):
                role = Path(path).name.removesuffix('.labels.json')
                opened.append(role)
                a.check_closed(self.root / 'out/evaluation', role)
            return original_read(path)
        before = {p.relative_to(self.previous): v2.sha256(p) for p in self.previous.rglob('*') if p.is_file()}
        with patch.object(a, 'rebuild_features', side_effect=prepare), patch.object(a, 'read_json', side_effect=read):
            result = self.run_pipeline()
        self.assertEqual(prepared, list(v2.ROLES))
        self.assertEqual(opened, ['query_tune', 'query_probe'])
        self.assertEqual(result['schema'], a.RESULT_SCHEMA)
        self.assertEqual(result['status'], 'heldout_targets_met')
        self.assertFalse(result['old_feature_cache_reused'])
        self.assertFalse(result['certified'])
        self.assertFalse(result['latency_claim'])
        self.assertFalse(result['official_dev_test_opened'])
        binding = a.read_json(self.root / 'out/binding.json')
        self.assertEqual(binding['current_source_sha256'], a.current_sources())
        self.assertEqual(before, {p.relative_to(self.previous): v2.sha256(p)
                                  for p in self.previous.rglob('*') if p.is_file()})
        manifest = a.read_json(self.root / 'out/artifact_manifest.json')
        self.assertTrue(all(v2.sha256(self.root / 'out' / name) == digest for name, digest in manifest.items()))
        with self.assertRaisesRegex(ValueError, 'already reserved'):
            self.run_pipeline('different-output')

    def test_failing_tune_keeps_probe_unprepared_and_unreserved(self):
        protocol = copy.deepcopy(v3.load_protocol())
        protocol['acceptance']['minimum_retention'] = 1.1
        with patch.object(v3, 'load_protocol', return_value=protocol), patch.object(v3, 'PROTOCOL_FP', v2.fingerprint(protocol)):
            result = self.run_pipeline()
        self.assertEqual(result['status'], 'no_tune_candidate')
        self.assertFalse(result['probe_opened'])
        self.assertFalse((self.root / 'out/evaluation/query_probe.features.jsonl').exists())
        self.assertFalse(self.parent().claim_path.exists())

    def test_crash_after_probe_reservation_blocks_retry(self):
        original = a.rebuild_features
        def fail(parent, env, role, out):
            if role == 'query_probe':
                raise RuntimeError('simulated interruption')
            return original(parent, env, role, out)
        with patch.object(a, 'rebuild_features', side_effect=fail), self.assertRaisesRegex(RuntimeError, 'interruption'):
            self.run_pipeline()
        with self.assertRaisesRegex(ValueError, 'already reserved'):
            self.run_pipeline('retry')

    def test_concurrent_probe_reservation_blocks_before_prepare(self):
        parent = self.parent()
        original = a.check_selection
        def competing(out, previous, protocol):
            result = original(out, previous, protocol)
            parent.claim_path.mkdir(parents=True, exist_ok=True)
            return result
        with patch.object(a, 'check_selection', side_effect=competing), self.assertRaisesRegex(ValueError, 'concurrent/repeated'):
            self.run_pipeline()
        self.assertFalse((self.root / 'out/evaluation/query_probe.features.jsonl').exists())

    def test_reproducible_scientific_artifacts_and_cli(self):
        self.run_pipeline('first')
        # A second independent synthetic parent location represents a separate
        # disposable test fixture, never a real-data retry instruction.
        second_parent = self.root / 'second-parent-location/parent'
        second_parent.parent.mkdir()
        shutil.copytree(self.template, second_parent)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(a.main(['--previous-run', str(second_parent), '--output', str(self.root / 'second')]), 0)
        for name in ('binding.json', 'result.json', 'report.md', 'evaluation/artifact_manifest.json'):
            self.assertEqual((self.root / 'first' / name).read_bytes(), (self.root / 'second' / name).read_bytes(), name)

    def test_existing_or_nested_output_refused_without_mutation(self):
        existing = self.root / 'existing'
        existing.mkdir()
        with self.assertRaises(FileExistsError):
            a.run_acceptance(self.previous, existing)
        with self.assertRaisesRegex(ValueError, 'separate'):
            a.run_acceptance(self.previous, self.previous / 'new-output')
        self.assertFalse((self.previous / 'new-output').exists())

    def test_frozen_acceptance_and_policy_grid_are_unchanged(self):
        protocol = v3.load_protocol()
        self.assertEqual(len(v3.policy_grid(protocol)), 27)
        self.assertEqual(protocol['acceptance'], {'minimum_retention': .9, 'maximum_hit_drop': .02,
                                                 'maximum_mean_original_distances_exclusive': 512})
        row = {'retention': .90, 'exact_hit_at_context': .60, 'hit_at_context': .59, 'mean_original_distances': 511.}
        self.assertTrue(all(v3.gates(row, protocol).values()))
        self.assertFalse(v3.gates({**row, 'mean_original_distances': 512.}, protocol)['original_distance_saving'])


if __name__ == '__main__':
    unittest.main()
