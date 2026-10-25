"""Approved development audit. Never loads the legacy all-role runtime.

The ledger is an audit boundary, not OS access control. Failed reservations cannot
be retried. Shared parent JSONL files are streamed and filtered by the reader;
unselected records are not passed to study stages or retained in memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
from time import perf_counter, time

import numpy as np

from tri_rag_harness.projection import dense_gaussian_projection, project_rows
from tri_rag_harness.tls_rag_step5_ann import stable_top
from tri_rag_harness.tls_rag_step5_protocol import source_binding, sha256, load_protocol
from tri_rag_harness.utils import canonical_json_bytes, fingerprint

if __name__ == '__main__':
    # Preserve exception/class identity when the executable module imports B.
    sys.modules.setdefault('mechanism.tls_rag_step5_audit', sys.modules[__name__])

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/tls_rag_step5_mechanism_v1.json'
SOURCES = ('mechanism/tls_rag_step5_audit.py', 'mechanism/tls_rag_step5_exact_policy.py',
           'configs/tls_rag_step5_mechanism_v1.json', 'tests/test_tls_rag_step5_mechanism.py',
           'docs/TLS_RAG_STEP5_MECHANISM_PROTOCOL.md', 'scripts/run_tls_rag_step5_mechanism.sh')
PARENTS = {'fit': 'cal_model_fit', 'calibration': 'cal_bound_fit',
           'selection': 'tune', 'evaluation': 'tune'}


def save(path, value, exclusive=False):
    path = Path(path)
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    if exclusive:
        with path.open('x') as f:
            f.write(data)
    else:
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_text(data)
        tmp.replace(path)


def rows(path):
    with Path(path).open() as f:
        for line in f:
            yield json.loads(line)


def sample_groups(parent_ids, cfg):
    ordered = {}
    seen = set()
    for role, ids in parent_ids.items():
        if len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError('invalid parent IDs')
        if seen.intersection(ids):
            raise ValueError('parent identities overlap')
        seen.update(ids)
        ordered[role] = sorted(ids, key=lambda q: (hashlib.sha256(canonical_json_bytes(
            ['step5-mechanism-v1', cfg['sample_seed'], role, q])).digest(), q))
    offsets, result = {}, {}
    for group, role in PARENTS.items():
        start = offsets.get(role, 0)
        n = cfg['samples'][group]
        result[group] = ordered[role][start:start+n]
        if len(result[group]) != n:
            raise ValueError('insufficient declared queries; no resize')
        offsets[role] = start + n
    return result


class ResourceStop(RuntimeError):
    pass


class Ledger:
    """One exclusive reservation for this study and parent input root."""
    def __init__(self, output, cfg, reserve_parent=True):
        self.root, self.cfg, self.started = Path(output), cfg, perf_counter()
        if self.root.exists():
            raise ValueError('existing output; no retry')
        self.root.parent.mkdir(parents=True, exist_ok=True)
        if reserve_parent:
            save(self.root.parent / 'mechanism-v1.reservation.json',
                 {'output': str(self.root), 'pid': os.getpid(), 'created_unix': time()}, True)
        self.root.mkdir()
        self.stage = 'preflight'
        self.evaluation_open = False
        self.seal_hash = None
        self.status('running')

    def status(self, status, **extra):
        save(self.root / 'status.json', {'status': status, 'stage': self.stage,
             'elapsed_seconds': perf_counter()-self.started, 'pid': os.getpid(),
             'formal_certification': False, **extra})

    def check(self):
        lim = self.cfg['limits']
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss /= 2**30 if sys.platform == 'darwin' else 2**20
        size = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        if (perf_counter()-self.started > lim['seconds'] or rss > lim['rss_gib']
                or size > lim['artifacts_gib'] * 2**30):
            raise ResourceStop('compute/RSS/artifact ceiling; no retry')
        disk = Path('/root/autodl-tmp')
        if disk.exists() and shutil.disk_usage(disk).free < lim['data_free_gib']*2**30:
            raise ResourceStop('data disk reserve breached')

    def begin(self, name):
        self.check()
        save(self.root / (name + '.reserved.json'), {'stage': name}, True)
        self.stage = name
        self.status('running')

    def freeze(self, payload):
        if self.seal_hash is not None:
            raise ValueError('selection already sealed')
        save(self.root / 'selection_seal.json', payload, True)
        self.seal_hash = sha256(self.root / 'selection_seal.json')
        return self.seal_hash

    def authorize_evaluation(self):
        if self.seal_hash is None or sha256(self.root / 'selection_seal.json') != self.seal_hash:
            raise ValueError('evaluation requires unchanged selection seal')
        save(self.root / 'evaluation_opened.json', {'selection_sha256': self.seal_hash}, True)
        self.evaluation_open = True

    def archive(self):
        manifest = {str(p.relative_to(self.root)): sha256(p) for p in sorted(self.root.rglob('*'))
                    if p.is_file() and p.name != 'artifacts.json'}
        save(self.root / 'artifacts.json', manifest)
        if any(sha256(self.root/n) != h for n, h in manifest.items()):
            raise ValueError('new artifact checksum verification failed')


class Reader:
    def __init__(self, bundle, archive, cfg, ledger):
        self.bundle, self.archive = Path(bundle).resolve(), Path(archive).resolve()
        self.cfg, self.ledger, self.access = cfg, ledger, []
        self.verified = set()
        self.manifest = json.loads((self.bundle/'artifact_manifest.json').read_text())
        if fingerprint(self.manifest) != cfg['bundle_binding']:
            raise ValueError('original bundle manifest binding changed')
        self.archive_maps = {}
        for name, expected in cfg['archive_manifests'].items():
            path = self.archive/name/'artifact_manifest.json'
            if sha256(path) != expected:
                raise ValueError('archived manifest identity changed: '+name)
            self.archive_maps[name] = json.loads(path.read_text())
        metadata = self.json_bundle('metadata.json')
        if metadata['sources'] != source_binding() or metadata['protocol'] != fingerprint(load_protocol()):
            raise ValueError('legacy scientific source/config mismatch; do not refresh')
        self.metadata = metadata
        setup = self.json_archive('indexes', 'setup.json')
        if setup['binding'] != cfg['archive_binding']:
            raise ValueError('index binding changed')
        for role in sorted(set(PARENTS.values())):
            for name in ['reservation.json', 'labels_opened.json']:
                receipt = self.json_archive(role, name)
                if receipt['binding'] != cfg['archive_binding'] or receipt['stage'] != role:
                    raise ValueError('archived role authorization/binding changed')
        self.ids = tuple(self.json_bundle('corpus_ids.json'))
        if len(self.ids) != cfg['corpus_size'] or len(set(self.ids)) != len(self.ids):
            raise ValueError('corpus ID mismatch')
        self.id_array = np.asarray(self.ids, dtype=str)
        self.row_by_id = {i: n for n, i in enumerate(self.ids)}
        self.corpus = self.array_bundle('corpus.npy', (len(self.ids), cfg['dimension']))
        self.projected = self.array_archive('indexes', 'projected.npy', (len(self.ids), cfg['m_prime']))
        self.parent_ids, self.queries = {}, {}
        for role in sorted(set(PARENTS.values())):
            ids = self.json_bundle(role+'.ids.json')
            if (fingerprint(ids) != metadata['roles'][role]['ids']
                    or len(ids) != metadata['role_counts'][role]):
                raise ValueError('parent IDs/family metadata identity mismatch')
            self.parent_ids[role] = ids
            self.queries[role] = self.array_bundle(role+'.npy', (len(ids), cfg['dimension']))
        # Opaque family hashes are inherited from the sealed producer; no claim of
        # new semantic independence or reconstruction of unavailable family IDs.
        for values in [self.corpus, *self.queries.values()]:
            for start in range(0, len(values), cfg['block_rows']):
                ledger.check()
                block = values[start:start+cfg['block_rows']]
                if not np.all(np.isfinite(block)) or not np.allclose(np.linalg.norm(block, axis=1), 1., rtol=0, atol=1e-12):
                    raise ValueError('input normalization/finite validation failed')
        self.groups = sample_groups(self.parent_ids, cfg)
        self.query_rows = {r: {q: j for j, q in enumerate(ids)} for r, ids in self.parent_ids.items()}
        self.matrix = dense_gaussian_projection(cfg['m_prime'], cfg['dimension'], cfg['projection_seed'])
        self.witness = None
        save(ledger.root/'sampling.json', {'groups': self.groups, 'sha256': fingerprint(self.groups),
            'parent_roles': {r: metadata['roles'][r] for r in self.parent_ids},
            'family_separation': 'inherited from sealed lexical-family producer; not new semantic independence'}, True)

    def _allowed(self, role, name):
        if role == 'bundle':
            names = {'metadata.json', 'corpus.npy', 'corpus_ids.json'}
            for r in set(PARENTS.values()):
                names.update({r+'.ids.json', r+'.npy', 'labels/'+r+'.json'})
            return name in names
        if role == 'indexes':
            return name in {'projected.npy', 'setup.json'}
        return role in set(PARENTS.values()) and name in {'truth.jsonl', 'retrieval.decisions.jsonl',
                                                           'reservation.json', 'labels_opened.json'}

    def path(self, role, name):
        if not self._allowed(role, name):
            raise ValueError('forbidden-role/path access: '+role+'/'+name)
        base = self.bundle if role == 'bundle' else self.archive/role
        p = base/name
        if p.is_symlink() or not p.resolve().is_relative_to(base):
            raise ValueError('unsafe input path')
        return p

    def verify(self, role, name):
        p = self.path(role, name)
        if (role, name) not in self.verified:
            expected = (self.manifest if role == 'bundle' else self.archive_maps[role])[name]
            digest = hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda: f.read(4*1024*1024), b''):
                    self.ledger.check()
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError('input SHA mismatch: '+str(p))
            self.verified.add((role, name))
            self.access.append({'role': role, 'name': name, 'sha256': expected})
        return p

    def json_bundle(self, name):
        return json.loads(self.verify('bundle', name).read_text())

    def json_archive(self, role, name):
        return json.loads(self.verify(role, name).read_text())

    def array_bundle(self, name, shape):
        return self._array(self.verify('bundle', name), shape)

    def array_archive(self, role, name, shape):
        return self._array(self.verify(role, name), shape)

    @staticmethod
    def _array(path, shape):
        v = np.load(path, mmap_mode='r', allow_pickle=False)
        if v.shape != shape or v.dtype != np.float64:
            raise ValueError('array geometry/dtype changed')
        return v

    def group_ids(self, group):
        if group == 'evaluation' and not self.ledger.evaluation_open:
            raise ValueError('evaluation references/labels closed')
        return self.groups[group]

    def query(self, group, qid):
        if qid not in self.group_ids(group):
            raise ValueError('query outside requested group')
        role = PARENTS[group]
        return self.queries[role][self.query_rows[role][qid]]

    def truth(self, group):
        wanted = set(self.group_ids(group))
        role = PARENTS[group]
        result = {}
        for row in rows(self.verify(role, 'truth.jsonl')):
            if row['query_id'] in wanted:
                if row['query_id'] in result or len(set(row['exact_top_k'])) != self.cfg['k_gt']:
                    raise ValueError('invalid/duplicate truth')
                if not set(row['exact_top_k']) <= self.row_by_id.keys():
                    raise ValueError('truth outside corpus')
                result[row['query_id']] = row
        if set(result) != wanted:
            raise ValueError('missing saved truth')
        return result

    def labels(self, group):
        wanted = set(self.group_ids(group))
        # Trusted reader filters the shared, already-open parent file. Never pass
        # the parent map to a fitter/selector/evaluator or controller.
        parent = self.json_bundle('labels/'+PARENTS[group]+'.json')
        if not wanted <= parent.keys():
            raise ValueError('missing label')
        return {q: parent[q] for q in wanted}

    def hnsw(self):
        wanted = set(self.group_ids('selection'))
        role, name = 'tune', 'retrieval.decisions.jsonl'
        p = self.path(role, name)
        digest, found = hashlib.sha256(), {}
        with p.open('rb') as f:
            for line in f:
                self.ledger.check()
                digest.update(line)
                row = json.loads(line)
                q = row['query_id']
                if q in wanted:
                    if q in found:
                        raise ValueError('duplicate ANN query')
                    stages = row['rows']['all']['stages']
                    found[q] = {s['budget']: {'raw_ids': s['raw_ids'], 'candidate_ids': s['candidate_ids']}
                                for s in stages if s['budget'] in self.cfg['anchors']}
        expected = self.archive_maps[role][name]
        if digest.hexdigest() != expected or set(found) != wanted:
            raise ValueError('ANN archive checksum/query mismatch')
        self.verified.add((role, name))
        self.access.append({'role': role, 'name': name, 'sha256': expected})
        return found


class ExactScan:
    """Direct float64 differences and stable string ties; optional truth ranks."""
    def __init__(self, ids, vectors, block_rows=8192, check=lambda: None):
        self.ids = np.asarray(ids, dtype=str)
        self.values = vectors
        self.block_rows, self.check = block_rows, check

    def search(self, q, k, truth_rows=None):
        if not 0 < k <= len(self.ids) or np.shape(q) != (self.values.shape[1],):
            raise ValueError('invalid exact query/k')
        keep_r, keep_d = np.empty(0, dtype=int), np.empty(0)
        self.last_selection_ms = 0.
        truth_rows = np.asarray([] if truth_rows is None else truth_rows, dtype=int)
        if len(truth_rows):
            diff = self.values[truth_rows]-q
            td = np.einsum('ij,ij->i', diff, diff)
            z, w = np.zeros(len(td), dtype=int), np.zeros(len(td), dtype=int)
        for start in range(0, len(self.ids), self.block_rows):
            self.check()
            end = min(start+self.block_rows, len(self.ids))
            block = np.asarray(self.values[start:end], dtype=np.float64)-q
            d = np.einsum('ij,ij->i', block, block)
            rr = np.arange(start, end)
            if not np.all(np.isfinite(d)):
                raise ValueError('nonfinite exact distances')
            if len(truth_rows):
                is_truth = np.isin(rr, truth_rows)
                for j in range(len(td)):
                    ahead = (d < td[j]) | ((d == td[j]) & (self.ids[rr] < self.ids[truth_rows[j]]))
                    w[j] += np.count_nonzero(ahead & is_truth)
                    z[j] += np.count_nonzero(ahead & ~is_truth)
            # Numeric partition first; stable_top resolves only its cutoff ties.
            selecting = perf_counter()
            take = min(k, len(d))
            idx = stable_top(self.ids[rr], d, take)
            pool_r = np.concatenate((rr[idx], keep_r))
            pool_d = np.concatenate((d[idx], keep_d))
            idx = stable_top(self.ids[pool_r], pool_d, min(k, len(pool_r)))
            keep_r, keep_d = pool_r[idx], pool_d[idx]
            self.last_selection_ms += (perf_counter()-selecting)*1000
        rank = None
        if len(truth_rows):
            rank = [{'id': str(self.ids[r]), 'original_rank': j+1, 'Z': int(z[j]),
                     'W': int(w[j]), 'rank': int(1+z[j]+w[j]),
                     'W_minus_j_minus_1': int(w[j]-j)} for j, r in enumerate(truth_rows)]
            for item in rank:
                if (item['rank'] <= k) != (item['id'] in self.ids[keep_r]):
                    raise ValueError('rank/inclusion identity failed')
        return keep_r, keep_d, rank


def decompose(truth, before, after):
    t, e, h = set(truth), set(before), set(after)
    missed = sorted((t & e)-h)
    rescued = sorted((t & h)-e)
    lost = sorted(t-e)
    if len(t-h) != len(lost)+len(missed)-len(rescued):
        raise ValueError('set identity failed')
    return {'before_retention': len(t & e)/len(t), 'after_retention': len(t & h)/len(t),
            'projection_or_prior_omissions': lost, 'omissions': missed, 'rescues': rescued}


def flat_prefix(flat, query, ids, anchors):
    """FAISS direct single-query arithmetic, stable IDs at any cutoff ties."""
    q = np.ascontiguousarray(np.asarray(query)[None], dtype=np.float32)
    k = max(anchors)
    dd, rr = flat.search(q, k+1)
    ties = {m: bool(dd[0,m-1] == dd[0,m]) for m in anchors}
    if any(ties.values()):
        # Include the entire final distance tie before resolving by string ID.
        radius = float(np.nextafter(np.float32(dd[0,k-1]), np.float32(np.inf)))
        _, dist, rows_ = flat.range_search(q, radius)
        idx = stable_top(np.asarray(ids)[rows_], dist, k)
        return rows_[idx], dist[idx], ties
    return rr[0,:k], dd[0,:k], ties


def phase_a(reader, ledger, cfg):
    import faiss
    if faiss.__version__ != cfg['faiss_version']:
        raise ValueError('FAISS version mismatch')
    faiss.omp_set_num_threads(1)
    ledger.begin('phase_a')
    started = perf_counter()
    exact = ExactScan(reader.ids, reader.projected, cfg['block_rows'], ledger.check)
    # 512 MB derived flat index in RAM; no HNSW build and no disk duplicate.
    flat = faiss.IndexFlatL2(cfg['m_prime'])
    for start in range(0, len(reader.ids), cfg['block_rows']):
        ledger.check()
        flat.add(np.ascontiguousarray(reader.projected[start:start+cfg['block_rows']], dtype=np.float32))
    setup_seconds = perf_counter()-started
    truth, ann = reader.truth('selection'), reader.hnsw()
    all_rows, cached = [], {}
    with (ledger.root/'phase_a.jsonl').open('x') as out:
        for n, qid in enumerate(reader.group_ids('selection')):
            q = reader.query('selection', qid)
            qp = project_rows(q[None], reader.matrix)[0]
            t = truth[qid]['exact_top_k']
            indices, ds, ranks = exact.search(qp, max(cfg['anchors']), [reader.row_by_id[i] for i in t])
            i32, d32, ties = flat_prefix(flat, qp, reader.id_array, cfg['anchors'])
            result = {'query_id': qid, 'ranks': ranks, 'anchors': {}}
            for m in cfg['anchors']:
                e = reader.id_array[indices[:m]].tolist()
                e32 = reader.id_array[i32[:m]].tolist()
                h, c = ann[qid][m]['raw_ids'], ann[qid][m]['candidate_ids']
                if any(len(x) != m or len(set(x)) != m for x in [e, e32, h, c]):
                    raise ValueError('candidate size/uniqueness mismatch')
                for r in ranks:
                    if (r['rank'] <= m) != (r['id'] in e):
                        raise ValueError('anchor rank identity failed')
                tie = ties[m]
                result['anchors'][str(m)] = {'E64': e, 'E32': e32, 'H': h, 'C': c,
                    'float32_cutoff_tie': tie,
                    'backend_attribution': 'combined_backend_precision_or_tie; no pure graph claim',
                    'E64_H': decompose(t, e, h), 'E64_E32': decompose(t, e, e32),
                    'E32_H': decompose(t, e32, h), 'H_C': decompose(t, h, c)}
            out.write(json.dumps(result, sort_keys=True, allow_nan=False)+'\n');out.flush()
            all_rows.append(result)
            cached[qid] = (indices, ds)
            if n == cfg['timing']['headroom_queries']-1:
                per = (perf_counter()-started)/(n+1)
                forecast = (perf_counter()-ledger.started)+per*(cfg['samples']['selection']-n-1+
                    cfg['samples']['fit']+cfg['samples']['calibration']+cfg['samples']['evaluation'])
                save(ledger.root/'throughput_forecast.json', {'observed_queries': n+1,
                    'seconds_per_query_including_setup': per, 'forecast_scan_seconds': forecast,
                    'excludes_future_features_and_optional_timing': True})
                if forecast > cfg['limits']['seconds']:
                    raise ResourceStop('first-64 forecast exceeds compute ceiling')
            if (n+1) % 16 == 0:
                ledger.status('running', completed_queries=n+1)
    summary = {'queries': len(all_rows), 'elapsed_seconds': perf_counter()-started,
               'flat_setup_seconds': setup_seconds, 'identity_checks_passed': True,
               'formal_certification': False, 'anchors': {}}
    for m in cfg['anchors']:
        items = [r['anchors'][str(m)] for r in all_rows]
        s = {}
        for edge in ['E64_H', 'E64_E32', 'E32_H', 'H_C']:
            s[edge] = {key: float(np.mean([x[edge][key] for x in items]))
                       for key in ['before_retention', 'after_retention']}
            for key in ['projection_or_prior_omissions', 'omissions', 'rescues']:
                s[edge]['mean_'+key] = float(np.mean([len(x[edge][key])/cfg['k_gt'] for x in items]))
                s[edge][key+'_quantiles'] = np.quantile(
                    [len(x[edge][key])/cfg['k_gt'] for x in items], [0,.1,.5,.9,1]).tolist()
            from mechanism.tls_rag_step5_exact_policy import interval
            s[edge]['paired_retention_change'] = interval(
                [x[edge]['after_retention'] for x in items],
                [x[edge]['before_retention'] for x in items], cfg=cfg)
        s['cutoff_ties'] = sum(x['float32_cutoff_tie'] for x in items)
        summary['anchors'][str(m)] = s
    w = [i['W_minus_j_minus_1'] for r in all_rows for i in r['ranks']]
    summary['W_surrogate_departure'] = {'fraction_nonzero': float(np.mean(np.asarray(w) != 0)),
        'quantiles': np.quantile(w, [0,.1,.5,.9,1]).tolist()}
    worst = sorted(all_rows, key=lambda r: (r['anchors']['64']['E64_H']['before_retention'], r['query_id']))[:10]
    summary['difficult_query_ids'] = [r['query_id'] for r in worst]
    save(ledger.root/'phase_a.complete.json', summary, True)
    del flat
    print(json.dumps({'checkpoint': 'phase_a_complete', 'summary': summary}), flush=True)
    return cached


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    subprocess.run([sys.executable, str(ROOT/'scripts/check_code_protection.py')], cwd=ROOT, check=True)
    subprocess.run(['git','diff','--exit-code','--quiet'], cwd=ROOT, check=True)
    subprocess.run(['git','diff','--cached','--exit-code','--quiet'], cwd=ROOT, check=True)
    cfg = json.loads(CONFIG.read_text())
    if not args.output.resolve().parent == args.bundle.resolve().parents[1]:
        raise ValueError('output must be new sibling under current input root')
    ledger = Ledger(args.output, cfg)
    def stop(signum, frame):
        raise ResourceStop('external signal '+str(signum))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        import scipy
        binding = {'config': cfg, 'sources': {n: sha256(ROOT/n) for n in SOURCES},
            'legacy_sources': source_binding(), 'code_commit': subprocess.check_output(
                ['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
            'environment': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                'platform': platform.platform(), 'single_thread': True, 'cloud_backed_input': True}}
        save(ledger.root/'binding.json', binding, True)
        reader = Reader(args.bundle, args.archive, cfg, ledger)
        cached = phase_a(reader, ledger, cfg)
        from mechanism.tls_rag_step5_exact_policy import phase_b
        result = phase_b(reader, ledger, cfg, cached, binding)
        save(ledger.root/'result.json', result, True)
        save(ledger.root/'input_access.json', reader.access)
        ledger.stage = 'complete'
        ledger.status('complete', conclusion=result['conclusion'])
        print(json.dumps({'checkpoint': 'complete', 'result': result}), flush=True)
    except BaseException as e:
        if 'reader' in locals():
            save(ledger.root/'input_access.json', reader.access)
        ledger.status('resource_stop' if isinstance(e, ResourceStop) else 'failed',
                      error_type=type(e).__name__, error=str(e))
        import traceback
        traceback.print_exc()
        raise
    finally:
        ledger.archive()


if __name__ == '__main__':
    main()
