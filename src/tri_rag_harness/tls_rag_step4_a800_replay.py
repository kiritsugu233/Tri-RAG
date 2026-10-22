"""Explicit reconstructed-data replay, never fresh independent validation."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import numpy as np

from . import tls_rag_step4_nfcorpus as adapter
from . import tls_rag_step4_probe as v2
from . import tls_rag_step4_retention_v3 as v3
from . import tls_rag_step4_repaired_acceptance as guard
from .tri_law import TRI_LAW_NUMERICAL_VERSION

ROOT = v2.ROOT
CONFIG = ROOT / 'configs/tls_rag_step4_a800_replay_v1.json'
CONFIG_SHA256 = 'ef0af4726d026c5738d5596e831938221a71b4cdfecdbce6cd882ffd6503aa36'
BINDING_SCHEMA = 'tls_rag_a800_rebuilt_binding_v1'
RESULT_SCHEMA = 'tls_rag_a800_replay_result_v1'
CODE_FILES = tuple(dict.fromkeys((*guard.CURRENT_FILES,
    'src/tri_rag_harness/tls_rag_step4_a800_replay.py',
    'configs/tls_rag_step4_a800_replay_v1.json', 'requirements-tls-rag-probe.txt')))


def load_protocol():
    if v2.sha256(CONFIG) != CONFIG_SHA256:
        raise ValueError('replay protocol changed')
    protocol = guard.read_json(CONFIG)
    if (protocol['data_protocol_fingerprint'] != v2.fingerprint(v2.load_protocol()) or
        protocol['retention_protocol_fingerprint'] != v2.fingerprint(v3.load_protocol()) or
        TRI_LAW_NUMERICAL_VERSION != 2):
        raise ValueError('frozen dependency protocol/version changed')
    return protocol


def sources():
    return {name: v2.sha256(guard.safe_path(ROOT / name)) for name in CODE_FILES}


def root_path(path):
    # Only the explicitly supplied root may traverse AutoDL's storage mount link.
    # Descendants continue to use strict anti-symlink checks.
    return Path(path).resolve()


def environment_record():
    def command(args):
        proc = subprocess.run(args, text=True, capture_output=True)
        return {'exit_code': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}
    return {'python': sys.version, 'executable': sys.executable, 'prefix': sys.prefix,
        'conda_prefix': os.environ.get('CONDA_PREFIX'), 'platform': platform.platform(),
        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'git_head': command(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
        'git_status': command(['git', '-C', str(ROOT), 'status', '--short']),
        'threads': {key: os.environ.get(key) for key in
                    ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}


def validate_payload(bundle, protocol):
    roles = guard.read_json(bundle / 'roles.json')
    if set(roles) != set(v2.ROLES) or any(not isinstance(roles[r], list) or
            len(roles[r]) != protocol['role_counts'][r] for r in v2.ROLES):
        raise ValueError('role counts changed')
    corpus = [json.loads(line) for line in (bundle / 'corpus.jsonl').read_text().splitlines()]
    rows = [json.loads(line) for line in (bundle / 'queries.jsonl').read_text().splitlines()]
    if any(set(row) != {'query_id','text'} or not isinstance(row['query_id'],str)
           or not isinstance(row['text'],str) for row in rows):
        raise ValueError('invalid query rows')
    queries = {row['query_id']: row['text'] for row in rows}
    if len(queries) != len(rows):
        raise ValueError('duplicate query IDs')
    families = [v2.canonical_text(text) for text in queries.values()]
    if len(set(families)) != len(families) or not all(families):
        raise ValueError('duplicate/empty query families')
    assigned, _ = v2.assign_roles(queries, queries, protocol['role_counts'], protocol['split_seed'])
    if assigned != roles:
        raise ValueError('role seed/order mismatch')
    env = v2.make_environment([r['passage_id'] for r in corpus], [r['text'] for r in corpus],
        np.load(bundle / 'corpus.npy', allow_pickle=False), queries,
        np.load(bundle / 'queries.npy', allow_pickle=False), roles, protocol)
    return env, roles


OFFICIAL_ENDPOINT = 'https://huggingface.co'
ALLOWED_ENDPOINTS = (OFFICIAL_ENDPOINT, 'https://hf-mirror.com')


def transport_record(endpoint, embedding):
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError('undeclared or malformed TLS_RAG_HF_ENDPOINT')
    prefix = OFFICIAL_ENDPOINT + '/'
    if any(not url.startswith(prefix) for url in adapter.SOURCES.values()):
        raise ValueError('unexpected official source URL')
    return {'endpoint': endpoint,
        'source_urls': {name: endpoint + url[len(OFFICIAL_ENDPOINT):]
                        for name, url in adapter.SOURCES.items()},
        'model_logical_url': prefix + embedding['model_id'] + '/tree/' + embedding['revision'],
        'model_transport_url': endpoint + '/' + embedding['model_id'] + '/tree/' + embedding['revision']}


def declared_transport(embedding):
    endpoint = os.environ.get('TLS_RAG_HF_ENDPOINT', OFFICIAL_ENDPOINT)
    if os.environ.get('HF_ENDPOINT', endpoint) != endpoint:
        raise ValueError('HF_ENDPOINT conflicts with explicit TLS_RAG_HF_ENDPOINT')
    return transport_record(endpoint, embedding)


def prepare_rebuilt_bundle(output):
    replay = load_protocol()
    protocol = v2.load_protocol()
    output = root_path(output)
    if output.exists():
        raise FileExistsError('rebuild output must be absent; preserve prior attempts')
    transport = declared_transport(protocol['embedding'])
    # Validate optional packages before network or directory creation.
    from huggingface_hub import snapshot_download
    import torch
    import pyarrow.parquet as parquet
    from sentence_transformers import SentenceTransformer
    expected = {'numpy':'1.26.4','scipy':'1.13.0','sentence-transformers':'3.4.1',
                'transformers':'4.48.3','pyarrow':'19.0.1'}
    for name, version in expected.items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'dependency mismatch: {name}; do not change versions to bypass failures')
    source_hashes = sources()
    output.mkdir(parents=True, exist_ok=False)
    raw, bundle = output / 'source', output / 'bundle'
    raw.mkdir(); bundle.mkdir()
    registration = {'schema':'tls_rag_a800_rebuild_registration_v1', 'protocol':replay,
        'data_protocol':protocol, 'source_urls':adapter.SOURCES, 'transport':transport, 'source_code_sha256':source_hashes,
        'environment':environment_record(), 'historical_probe_observed':True,
        'independent_validation':False, 'embedding_device':'cpu'}
    v2.save(output / 'registration.json', registration)
    hashes = {}
    for name, url in adapter.SOURCES.items():
        print(f'Downloading pinned source: {name}', flush=True)
        hashes[name] = adapter.download(transport['source_urls'][name], raw / name)
    corpus = adapter.validate_text_rows(parquet.read_table(raw / 'corpus.parquet').to_pylist(), corpus=True)
    queries = adapter.validate_text_rows(parquet.read_table(raw / 'queries.parquet').to_pylist())
    eligible, qrels = adapter.read_train_qrels(raw / 'train.tsv')
    roles, audit = v2.assign_roles(queries, eligible, protocol['role_counts'], protocol['split_seed'])
    labels = adapter.partition_qrels(qrels, roles, set(corpus))
    selected = {qid:queries[qid] for qid in sorted(q for r in v2.ROLES for q in roles[r])}
    v2.save(bundle / 'roles.json', roles)
    v2.save(bundle / 'role_audit.json', audit)
    v2.save_rows(bundle / 'corpus.jsonl', ({'passage_id':k,'text':v} for k,v in corpus.items()))
    v2.save_rows(bundle / 'queries.jsonl', ({'query_id':k,'text':v} for k,v in selected.items()))
    for role in v2.ROLES:
        v2.save(bundle / f'{role}.labels.json', labels[role])
    del labels, qrels
    torch.manual_seed(protocol['embedding_seed'])
    torch.use_deterministic_algorithms(True)
    model_dir = output / 'model_snapshot'
    snapshot_download(repo_id=protocol['embedding']['model_id'], revision=protocol['embedding']['revision'],
                      endpoint=transport['endpoint'], local_dir=str(model_dir))
    model_hashes = {p.relative_to(model_dir).as_posix():v2.sha256(guard.safe_path(p))
                    for p in sorted(model_dir.rglob('*')) if p.is_file()
                    and '.cache' not in p.relative_to(model_dir).parts}
    if not model_hashes:
        raise ValueError('empty model snapshot')
    model = SentenceTransformer(str(model_dir), device='cpu', local_files_only=True, trust_remote_code=False)
    model.max_seq_length = protocol['embedding']['max_seq_length']
    model.float().eval()
    for name, texts in (('corpus',list(corpus.values())),('queries',list(selected.values()))):
        vectors = np.asarray(model.encode(texts, batch_size=32, show_progress_bar=True,
                    convert_to_numpy=True, normalize_embeddings=True), dtype=np.float64)
        if vectors.shape != (len(texts),protocol['embedding']['dimension']) or not np.all(np.isfinite(vectors)):
            raise ValueError('invalid rebuilt vectors')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError('zero vector')
        np.save(bundle / f'{name}.npy', vectors / norms)
    validate_payload(bundle, protocol)
    if sources() != source_hashes:
        raise ValueError('source changed during rebuild')
    v2.save(bundle / 'binding.json', {'schema':BINDING_SCHEMA, 'replay_protocol':replay,
        'data_protocol':protocol, 'source_urls':adapter.SOURCES, 'source_sha256':hashes,
        'transport':transport, 'model_sha256':model_hashes,
        'source_code_sha256':source_hashes, 'embedding':protocol['embedding'],
        'embedding_device':'cpu', 'historical_probe_observed':True, 'independent_validation':False,
        'registration_sha256':v2.sha256(output / 'registration.json'),
        'files':{p.name:v2.sha256(p) for p in sorted(bundle.iterdir())}})
    print(f'Rebuilt replay bundle: {bundle}', flush=True)


def checked_model_manifest(directory, manifest):
    """Check relative model files, including dotfiles, without weakening old guards."""
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError('invalid model manifest')
    for name, digest in manifest.items():
        if (not isinstance(name, str) or
            any(part in ('', '.', '..') or not re.fullmatch(r'[A-Za-z0-9_.-]+', part)
                for part in name.split('/')) or
            not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)):
            raise ValueError('unsafe model manifest')
    snapshot = {}
    for name, digest in manifest.items():
        path = guard.safe_path(directory / name)
        if not path.is_file() or v2.sha256(path) != digest:
            raise ValueError('model file hash mismatch: ' + name)
        snapshot[path] = digest
    return snapshot


def validate_rebuilt_bundle(previous):
    previous = root_path(previous)
    protocol = load_protocol()
    bundle = guard.safe_path(previous / 'bundle')
    binding = guard.read_json(bundle / 'binding.json')
    if (binding.get('schema') != BINDING_SCHEMA or binding.get('replay_protocol') != protocol or
        binding.get('data_protocol') != v2.load_protocol() or
        binding.get('source_code_sha256') != sources() or
        binding.get('embedding') != v2.load_protocol()['embedding'] or
        binding.get('embedding_device') != 'cpu' or
        binding.get('historical_probe_observed') is not True or
        binding.get('independent_validation') is not False):
        raise ValueError('replay provenance mismatch; historical bundles cannot be relabeled')
    names = {'roles.json','role_audit.json','corpus.jsonl','queries.jsonl','corpus.npy','queries.npy'}
    names |= {f'{r}.labels.json' for r in v2.ROLES}
    if set(binding.get('files',{})) != names or binding.get('source_urls') != adapter.SOURCES:
        raise ValueError('unexpected replay file/source manifest')
    transport = binding.get('transport', {})
    if transport != transport_record(transport.get('endpoint'), binding['embedding']):
        raise ValueError('transport provenance mismatch')
    snapshot = checked_model_manifest(previous / 'model_snapshot', binding.get('model_sha256'))
    snapshot.update(guard.checked_manifest(bundle, binding['files']))
    if set(binding.get('source_sha256',{})) != set(adapter.SOURCES):
        raise ValueError('missing pinned source files')
    snapshot.update(guard.checked_manifest(previous / 'source', binding['source_sha256']))
    registration = guard.read_json(previous / 'registration.json')
    snapshot[previous / 'registration.json'] = binding['registration_sha256']
    snapshot[bundle / 'binding.json'] = v2.sha256(bundle / 'binding.json')
    if (registration.get('source_code_sha256') != binding['source_code_sha256'] or
        registration.get('protocol') != protocol or registration.get('data_protocol') != binding['data_protocol'] or
        registration.get('source_urls') != adapter.SOURCES or registration.get('transport') != transport):
        raise ValueError('registration identity mismatch')
    def check():
        for path,digest in snapshot.items():
            if v2.sha256(guard.safe_path(path)) != digest:
                raise ValueError('replay input changed: '+path.name)
        if sources() != binding['source_code_sha256']:
            raise ValueError('source changed after binding')
    check()
    env,roles = validate_payload(bundle,binding['data_protocol'])
    return previous,binding,env,roles,check


def run_replay(previous, output):
    output = root_path(output)
    if output.exists():
        raise FileExistsError('replay output must be absent')
    previous,binding,env,roles,check = validate_rebuilt_bundle(previous)
    if previous == output or previous in output.parents or output in previous.parents:
        raise ValueError('output must be separate from input')
    # Never touch historical acceptance reservations. This explicitly observed
    # replay has its own persistent, atomic attempt reservation.
    claim = guard.safe_path(previous / 'a800-replay-attempt')
    try:
        claim.mkdir()
    except FileExistsError as exc:
        raise ValueError('replay already attempted; preserve reservation and results') from exc
    v2.save(claim / 'reservation.json', {'output':str(output),
        'binding_sha256':v2.sha256(previous / 'bundle/binding.json')})
    output.mkdir(parents=True,exist_ok=False)
    protocol = v3.load_protocol()
    engine = output / 'engine_replay'
    replay = load_protocol()
    provenance = {'schema':RESULT_SCHEMA,'historical_probe_observed':True,'independent_validation':False,
                  'binding_sha256':v2.sha256(previous / 'bundle/binding.json'),'protocol':replay}
    v2.save(output / 'provenance.json',provenance)
    def provide(role,directory):
        check()
        if role == 'query_probe':
            class Parent:
                pass
            parent = Parent(); parent.roles = roles
            guard.check_selection(directory,parent,protocol)
        v2.prepare_role(env,v2.component_config(binding['data_protocol']),roles[role],directory,role)
        return directory / f'{role}.features.jsonl'
    def labels(role,ids):
        check()
        if role not in ('query_tune','query_probe') or tuple(ids) != tuple(roles[role]):
            raise ValueError('unexpected label access')
        guard.check_closed(engine,role)
        return guard.read_json(previous / 'bundle' / f'{role}.labels.json')
    result = v3.run(env,roles,protocol,provide,labels,engine,lineage=provenance)
    check()
    status = {'heldout_targets_met':'replay_targets_met','heldout_targets_failed':'replay_targets_failed',
              'no_tune_candidate':'no_tune_candidate'}[result['status']]
    final = {**result,'schema':RESULT_SCHEMA,'status':status,'engine_status':result['status'],
        'historical_probe_observed':True,'independent_validation':False,'certified':False,'latency_claim':False,
        'selection_matches_historical':result['selected_candidate']==replay['historical_selected_candidate'],
        'historical_selected_candidate':replay['historical_selected_candidate'],
        'binding_sha256':provenance['binding_sha256']}
    v2.save(output / 'result.json',final)
    (output / 'report.md').write_text('# TLS-RAG A800 reconstructed-data replay\n\n'
        f'Status: {status}\n\nHistorical probe already observed. NOT independent validation.\n'
        'Models were refitted; vectors regenerated. No claim of byte-identical historical recovery.\n'
        'No certification or latency claim. engine_replay contains unchanged legacy engine terminology.\n\n'
        + json.dumps({k:v for k,v in final.items() if k!='summaries'},indent=2)+'\n')
    v2.save(output / 'artifact_manifest.json',{p.relative_to(output).as_posix():v2.sha256(p)
        for p in sorted(output.rglob('*')) if p.is_file()})
    return final


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare');prepare.add_argument('--output',type=Path,required=True)
    run=sub.add_parser('replay');run.add_argument('--input',type=Path,required=True);run.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':
        prepare_rebuilt_bundle(args.output)
    else:
        result=run_replay(args.input,args.output)
        print(json.dumps({k:v for k,v in result.items() if k!='summaries'},sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
