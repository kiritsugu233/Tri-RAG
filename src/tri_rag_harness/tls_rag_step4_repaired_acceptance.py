"""Explicit historical-data compatibility for reviewed Tri-Law / retention v3.

The old loaders and scientific policy stay frozen. Historical provenance is
validated independently; numerical features are always rebuilt by current code.
This is descriptive held-out acceptance, not certification or serving latency.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from . import tls_rag_step4_probe as v2
from . import tls_rag_step4_retention_v3 as v3
from .tri_law import TRI_LAW_NUMERICAL_VERSION

HISTORICAL_COMMIT = 'c08cd3b6800888c8d2a5a5f8c58a77d883a84ede'
# SHA-256 of Git blob contents at HISTORICAL_COMMIT, not hashes from user input.
HISTORICAL_CODE_SHA256 = {
    "src/tri_rag_harness/tls_rag_step4_probe.py": "836d57790687e5748596ab11f038563c7b7e6f45d6a2cd2e9b4c3c2bc52d6636",
    "src/tri_rag_harness/tls_rag_step4_nfcorpus.py": "edc95097f118a3f613a198f5868266af958d456be6beb6545d060a23e836490b",
    "src/tri_rag_harness/tls_rag_step3.py": "4fc53c4d98537315f25d8a531da7a53731881bb8d1638ac6696e07307fdf2b62",
    "src/tri_rag_harness/tls_rag_step2.py": "a6be67c944237ea8f7649f18eb6c6e84f71c94b1546f62cdbb037da7540ed3dd",
    "src/tri_rag_harness/tri_law.py": "27f8cb31c9625d9bcc54316d5618f18f35fc5882486f877fe425e44d6a162d19",
    "src/tri_rag_harness/embeddings.py": "8a73aa75d0803c84b81618608a0ac689846aa57db0e4fa8c3c4967fa573840fe",
    "src/tri_rag_harness/indexes.py": "32a5d1988ecc604ea54f99fb4c784f9bb90a617c624c022bda12bf45b38efe15",
    "src/tri_rag_harness/projection.py": "91f0cd3d0493200a068b057a5637bc6d7999e351c198c5af8305332ec8fef1a3",
    "src/tri_rag_harness/utils.py": "044be5bbf946bf042387c3632d7c40c018c5291d3cc1132b5c7d5bd69a41562b",
    "configs/tls_rag_step3_synthetic_v1.json": "c98b417b825953a889f929183ca454c819ef530c00ab676a06bd7c278e90c0b8",
    "configs/tls_rag_step2_synthetic_v1.json": "257c032be82e945ab757b3c1342d1d87f6cd399667b786008122c358f62ac5aa"
}
BINDING_SCHEMA = 'tls_rag_repaired_retention_binding_v1'
RESULT_SCHEMA = 'tls_rag_repaired_retention_acceptance_v1'
CURRENT_FILES = (*v2.CODE_FILES,
    'src/tri_rag_harness/__init__.py', 'pyproject.toml',
    'scripts/check_code_protection.py',
    'configs/tls_rag_step4_nfcorpus_probe_v2.json',
    'configs/tls_rag_step4_retention_v3.json',
    'src/tri_rag_harness/tls_rag_step4_retention_v3.py',
    'src/tri_rag_harness/tls_rag_step4_repaired_acceptance.py',
    'scripts/run_tls_rag_step4_repaired_acceptance.sh')


def safe_path(path):
    """Reject symlinks (including broken ones) before reading or creating paths."""
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f'symlink path refused: {path}')
    return path


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(safe_path(path).read_text(), object_pairs_hook=unique)


def checked_manifest(directory, manifest):
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError('invalid file manifest')
    # Validate every name before accessing any file specified by the manifest.
    if any(not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', name)
           or not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)
           for name, digest in manifest.items()):
        raise ValueError('unsafe file manifest')
    result = {}
    for name, digest in manifest.items():
        path = safe_path(directory / name)
        if not path.is_file() or v2.sha256(path) != digest:
            raise ValueError(f'historical file hash mismatch: {name}')
        result[path] = digest
    return result


def require_unopened(previous):
    evaluation = safe_path(previous / 'evaluation')
    if any(path.name.startswith('query_probe.') for path in evaluation.iterdir()):
        raise ValueError('historical probe already prepared/opened; reuse refused')


@dataclass
class HistoricalParent:
    previous: Path
    binding: dict
    roles: dict
    snapshot: dict
    protocol: dict

    def check(self):
        require_unopened(self.previous)
        for path, digest in self.snapshot.items():
            if not safe_path(path).is_file() or v2.sha256(path) != digest:
                raise ValueError(f'historical input changed after validation: {path.name}')

    @property
    def claim_path(self):
        # Fixed beside the old run, not inside it or relative to a new output.
        # This persists across output names. It is an audit guard, not OS security.
        identity = v2.fingerprint({'roles': self.roles,
            'vectors': {n: self.binding['files'][n] for n in ('corpus.npy', 'queries.npy')}})
        return self.previous.parent / '.tls-rag-repaired-probe-claims' / identity


def validate_historical_parent(previous):
    previous = safe_path(previous).resolve()
    require_unopened(previous)
    protocol = v2.load_protocol()
    bundle, evaluation = previous / 'bundle', previous / 'evaluation'
    binding = read_json(bundle / 'binding.json')
    if (binding.get('schema') != 'tls_rag_nfcorpus_probe_binding_v2' or
        binding.get('protocol_fingerprint') != v2.fingerprint(protocol) or
        binding.get('embedding') != protocol['embedding'] or
        binding.get('source_revisions') != {'dataset': protocol['dataset_revision'],
                                            'qrels': protocol['qrels_revision']}):
        raise ValueError('historical bundle protocol/model/revision mismatch')
    if binding.get('source_code_sha256') != HISTORICAL_CODE_SHA256:
        raise ValueError('unsupported historical source binding; mixed/current hashes refused')
    names = {'corpus.jsonl', 'queries.jsonl', 'roles.json', 'corpus.npy', 'queries.npy'}
    names |= {f'{role}.labels.json' for role in v2.ROLES}
    if not isinstance(binding.get('files'), dict) or set(binding['files']) != names:
        raise ValueError('unexpected historical bundle file manifest')
    snapshot = checked_manifest(bundle, binding['files'])
    manifest = read_json(evaluation / 'artifact_manifest.json')
    required = {'result.json', 'selection.json', 'input_binding.json', 'roles.json', 'protocol.json',
                'score_models.json', 'calibration_tables.json', 'query_tune.metrics.jsonl',
                'query_tune.summary.json'}
    required |= {f'{role}.features.jsonl' for role in v2.ROLES[:3]}
    if not isinstance(manifest, dict) or not required <= set(manifest):
        raise ValueError('incomplete historical evaluation manifest')
    if any(name.startswith('query_probe.') for name in manifest):
        raise ValueError('historical manifest records prepared/opened probe')
    snapshot.update(checked_manifest(evaluation, manifest))
    result = read_json(evaluation / 'result.json')
    selection = read_json(evaluation / 'selection.json')
    if (result.get('schema') != 'tls_rag_nfcorpus_probe_result_v2' or
        result.get('status') != 'no_tune_candidate' or
        'selected_candidate' not in result or result['selected_candidate'] is not None or
        'selected_candidate' not in selection or selection['selected_candidate'] is not None or
        set(result.get('summaries', {})) != {'query_tune'} or
        result.get('official_dev_test_opened') is not False):
        raise ValueError('requires completed unsuccessful v2 tune with unopened probe')
    roles = read_json(bundle / 'roles.json')
    if (read_json(evaluation / 'input_binding.json') != binding or
        read_json(evaluation / 'protocol.json') != protocol or
        read_json(evaluation / 'roles.json') != roles):
        raise ValueError('historical evaluation/bundle identity mismatch')
    if not isinstance(roles, dict) or set(roles) != set(v2.ROLES):
        raise ValueError('invalid historical roles')
    for role in v2.ROLES:
        if (not isinstance(roles[role], list) or len(roles[role]) != protocol['role_counts'][role]
            or any(not isinstance(qid, str) or not qid for qid in roles[role])):
            raise ValueError('invalid historical role counts/IDs')
    flat = [qid for role in v2.ROLES for qid in roles[role]]
    if len(flat) != len(set(flat)):
        raise ValueError('historical roles are not disjoint')
    if (selection.get('protocol_fingerprint') != v2.fingerprint(protocol) or
        selection.get('role_fingerprint') != v2.fingerprint(roles)):
        raise ValueError('historical selection protocol/role mismatch')
    for key, name in (('score_models_sha256', 'score_models.json'),
                      ('calibration_tables_sha256', 'calibration_tables.json'),
                      ('tune_metrics_sha256', 'query_tune.metrics.jsonl')):
        if selection.get(key) != manifest[name]:
            raise ValueError('historical selection dependency mismatch')
    summary = read_json(evaluation / 'query_tune.summary.json')
    if (result['summaries']['query_tune'] != summary or
        v2.select_candidate(summary, v2.component_config(protocol), protocol) is not None):
        raise ValueError('historical no-candidate result disagrees with tune summary')
    for path in (bundle / 'binding.json', evaluation / 'artifact_manifest.json'):
        snapshot[path] = v2.sha256(safe_path(path))
    parent = HistoricalParent(previous, binding, roles, snapshot, protocol)
    if safe_path(parent.claim_path).exists():
        raise ValueError('probe already reserved by a repaired acceptance; do not retry on these IDs')
    parent.check()
    return parent


def load_environment(parent):
    parent.check()
    bundle = parent.previous / 'bundle'
    corpus = [json.loads(line) for line in (bundle / 'corpus.jsonl').read_text().splitlines()]
    query_rows = [json.loads(line) for line in (bundle / 'queries.jsonl').read_text().splitlines()]
    if any(set(row) != {'query_id', 'text'} or not isinstance(row['query_id'], str)
           or not row['query_id'] or not isinstance(row['text'], str) for row in query_rows):
        raise ValueError('invalid query rows')
    if any(set(row) != {'passage_id', 'text'} for row in corpus):
        raise ValueError('invalid corpus rows')
    queries = {row['query_id']: row['text'] for row in query_rows}
    if len(queries) != len(query_rows):
        raise ValueError('duplicate query rows')
    families = [v2.canonical_text(text) for text in queries.values()]
    if any(not text for text in families) or len(families) != len(set(families)):
        raise ValueError('empty or duplicate query text families')
    # The selected subset retains the original ordering by seeded text family.
    assigned, _ = v2.assign_roles(queries, queries, parent.protocol['role_counts'],
                                  parent.protocol['split_seed'])
    if assigned != parent.roles:
        raise ValueError('historical role assignment does not match frozen seed/order')
    return v2.make_environment([row['passage_id'] for row in corpus], [row['text'] for row in corpus],
        np.load(bundle / 'corpus.npy', allow_pickle=False), queries,
        np.load(bundle / 'queries.npy', allow_pickle=False), parent.roles, parent.protocol)


def current_sources():
    if TRI_LAW_NUMERICAL_VERSION != 2:
        raise ValueError('requires reviewed Tri-Law numerical version 2')
    return {name: v2.sha256(safe_path(v2.ROOT / name)) for name in CURRENT_FILES}


def check_closed(output, role):
    receipt = read_json(output / f'{role}.phase_a_closed.json')
    if receipt.get('role') != role or receipt.get('protocol_fingerprint') != v3.PROTOCOL_FP:
        raise ValueError('missing/mismatched decision closure')
    required = {f'{role}.features.jsonl', f'{role}.decisions.jsonl',
                'retention_models.json', 'retention_calibration.json'}
    if role == 'query_probe':
        required.add('selection.json')
    if not required <= set(receipt.get('files', {})):
        raise ValueError('incomplete decision closure')
    checked_manifest(output, receipt['files'])


def check_selection(output, parent, protocol):
    selected = read_json(output / 'selection.json')
    policy = selected.get('selected_policy')
    if (policy is None or policy not in [vars(p) for p in v3.policy_grid(protocol)] or
        selected.get('selected_candidate') != v3.Policy(**policy).policy_id or
        selected.get('role_fingerprint') != v2.fingerprint(parent.roles) or
        selected.get('protocol_fingerprint') != v2.fingerprint(protocol)):
        raise ValueError('probe requires frozen tune selection')
    for key, name in (('models_sha256', 'retention_models.json'),
                      ('calibration_sha256', 'retention_calibration.json'),
                      ('tune_metrics_sha256', 'query_tune.metrics.jsonl')):
        if selected.get(key) != v2.sha256(safe_path(output / name)):
            raise ValueError('tune selection dependency changed')
    return selected


def rebuild_features(parent, environment, role, output):
    if role not in v2.ROLES:
        raise ValueError('unknown feature role')
    parent.check()
    # Intentionally never parse/copy old feature files. Their hashes are provenance only.
    print(f'{role}: rebuilding all prefixes with reviewed Tri-Law', flush=True)
    v2.prepare_role(environment, v2.component_config(parent.protocol), parent.roles[role], output, role)
    return output / f'{role}.features.jsonl'


def run_acceptance(previous, output):
    output = safe_path(output).resolve()
    if output.exists():
        raise FileExistsError('use an absent repaired output; previous results are preserved')
    parent = validate_historical_parent(previous)
    if output == parent.previous or parent.previous in output.parents or output in parent.previous.parents:
        raise ValueError('repaired output must be separate from historical run')
    protocol = v3.load_protocol()
    environment = load_environment(parent)
    sources = current_sources()
    output.mkdir(parents=True, exist_ok=False)
    binding = {'schema': BINDING_SCHEMA, 'tri_law_numerical_version': 2,
        'retention_protocol_fingerprint': v2.fingerprint(protocol),
        'historical_source_reference_commit': HISTORICAL_COMMIT,
        'historical_binding': parent.binding,
        'historical_inputs_sha256': {str(path.relative_to(parent.previous)): digest
                                      for path, digest in parent.snapshot.items()},
        'current_source_sha256': sources, 'role_fingerprint': v2.fingerprint(parent.roles),
        'development_features': 'recomputed_from_verified_vectors_with_reviewed_code',
        'old_feature_cache_reused': False, 'official_dev_test_opened': False}
    binding_hash = v2.save(output / 'binding.json', binding)
    evaluation = output / 'evaluation'
    claimed = False

    def check_inputs():
        parent.check()
        if current_sources() != sources or v2.sha256(output / 'binding.json') != binding_hash:
            raise ValueError('current source/binding changed during acceptance')

    def provider(role, directory):
        nonlocal claimed
        check_inputs()
        if role == 'query_probe':
            check_selection(directory, parent, protocol)
            claim = safe_path(parent.claim_path)
            claim.parent.mkdir(parents=True, exist_ok=True)
            # Atomic reservation BEFORE feature preparation. Retain after any failure.
            try:
                claim.mkdir()
            except FileExistsError as exc:
                raise ValueError('probe already reserved; concurrent/repeated acceptance refused') from exc
            v2.save(claim / 'reservation.json', {'schema': 'tls_rag_repaired_probe_reservation_v1',
                'output': str(output), 'binding_sha256': binding_hash,
                'selection_sha256': v2.sha256(directory / 'selection.json')})
            claimed = True
            v2.save(output / 'probe_reservation.json', read_json(claim / 'reservation.json'))
        return rebuild_features(parent, environment, role, directory)

    def labels(role, ids):
        if role not in ('query_tune', 'query_probe') or tuple(ids) != tuple(parent.roles[role]):
            raise ValueError('invalid label role/IDs')
        check_inputs()
        check_closed(evaluation, role)
        if role == 'query_probe':
            if not claimed:
                raise ValueError('probe not reserved')
            check_selection(evaluation, parent, protocol)
        return read_json(parent.previous / 'bundle' / f'{role}.labels.json')

    result = v3.run(environment, parent.roles, protocol, provider, labels, evaluation,
                    lineage={'schema': BINDING_SCHEMA, 'binding_sha256': binding_hash,
                             'old_feature_cache_reused': False, 'tri_law_numerical_version': 2})
    check_inputs()
    final = {**result, 'schema': RESULT_SCHEMA, 'engine_schema': result['schema'],
             'binding_sha256': binding_hash, 'tri_law_numerical_version': 2,
             'old_feature_cache_reused': False}
    v2.save(output / 'result.json', final)
    (output / 'report.md').write_text('# Reviewed-source Step 4 acceptance\n\n'
        'Tri-Law numerical v2; frozen retention-v3 policy and targets.\n'
        'Historical vectors verified; all numerical features recomputed.\n\n' +
        (evaluation / 'report.md').read_text())
    v2.save(output / 'artifact_manifest.json', {p.relative_to(output).as_posix(): v2.sha256(p)
        for p in sorted(output.rglob('*')) if p.is_file()})
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_acceptance(args.previous_run, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'summaries'}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
