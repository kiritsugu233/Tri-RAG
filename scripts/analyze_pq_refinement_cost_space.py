#!/usr/bin/env python3
"""Read-only C0 audit of saved S0 calls; no search or new scientific measurements.

All original inputs must match the recovered, closed inventory. Bootstrap
intervals describe the observed query cohort, not independent certification.
Only new analysis products are written. No original vectors/indexes are opened.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INDEX_SHA = '003ccbbca3c10ad6725462556d79554c8c0c4deb5329999219313c97bb0dad1a'
BOOTSTRAP_SEED = 2026110405
BOOTSTRAP_REPLICATES = 10000
GAMMAS = (.05, .10, .20)
FRACTIONS = (.25, .50, .75, 1.)
TARGETS = (
    ('opq64_ivf64', 'ivf/opq64/64', 'ivf/opq64/64/candidates.npy',
     'ivf/opq64/64/quality.json', 6400),
    ('opq64_ivf256', 'ivf/opq64/256', 'ivf/opq64/256/candidates.npy',
     'ivf/opq64/256/quality.json', 6400),
    ('pq64_flat', 'representations/pq64/cpu14',
     'representations/pq64/faiss_candidates.npy',
     'representations/pq64/faiss_quality.json', 3200),
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


class Verified:
    def __init__(self, recovery):
        self.recovery = Path(recovery).resolve()
        self.root = self.recovery / 'priority'
        receipt = json.loads((self.recovery / 'recovery-verification.json').read_text())
        if receipt['source_index_sha256'] != INDEX_SHA or receipt['full_run_downloaded']:
            raise ValueError('unexpected recovery identity/status')
        if sha(self.root / 'delivery_index.json') != INDEX_SHA:
            raise ValueError('closed source index changed')
        self.inventory = {r['path']: r for r in receipt['verified']}
        original = {r['path']: r for r in json.loads(
            (self.root / 'delivery_index.json').read_text())['files']}
        if any(original.get(p) != row for p, row in self.inventory.items()):
            raise ValueError('recovery receipt differs from original closed inventory')
        self.consumed = {}
        self.receipt = receipt

    def path(self, relative):
        if relative not in self.inventory:
            raise ValueError('not in verified selected snapshot: ' + relative)
        p = self.root / relative
        expected = self.inventory[relative]
        if p.is_symlink() or p.stat().st_size != expected['bytes'] or sha(p) != expected['sha256']:
            raise ValueError('recovered file changed: ' + relative)
        self.consumed[relative] = expected
        return p

    def json(self, relative):
        return json.loads(self.path(relative).read_text())

    def array(self, relative):
        return np.load(self.path(relative), mmap_mode='r', allow_pickle=False)


def quantiles(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.quantile(a, .95)), minimum=float(a.min()), maximum=float(a.max()))


def interval(values):
    return [float(x) for x in np.quantile(values, [.025, .975])]


def logical_addresses(ids, offset, row_bytes, page_bytes):
    """Virtual file-page model only; no physical requests/DRAM/I/O are measured."""
    start = offset + np.asarray(ids, np.int64) * row_bytes
    # A row can span several pages; do not assume row alignment or a single page.
    pages = np.concatenate([np.arange(a // page_bytes, (a + row_bytes - 1) // page_bytes + 1)
                            for a in start])
    unique = int(len(np.unique(pages)))
    return unique, unique * page_bytes


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def analyze(recovery, output):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT / 'runs'):
        raise ValueError('new analysis directory under canonical runs/ required')
    data = Verified(recovery)
    freeze = data.json('source_freeze.json')
    cfg = freeze['config']
    roles = data.json('roles.json')['queries']
    inputs = data.json('inputs.json')
    truth = data.array('published_truth.npy')
    cutoffs = data.array('published_cutoffs.npy')
    data.path('queries.npy')  # Identity only; query vectors are not used by C0.
    source = cfg['source']
    if (len(roles) != 128 or source['dimension'] != 640 or cfg['k'] != 100
            or source['layout']['offset'] != 6144 or inputs['test_payload_read']):
        raise ValueError('unexpected scientific/input scope')
    if inputs['source_sha256'] != source['sha256']:
        raise ValueError('source identity inconsistent')
    # Bind the actual executed implementation without importing it or running search.
    source_bindings = {}
    for p in ('mechanism/tls_rag_step5_route6_representation_v1.py',
              'mechanism/tls_rag_step5_route6_representation_v2.py',
              'mechanism/tls_rag_step5_route6_representation_screen_v1.py',
              'mechanism/tls_rag_step5_route6_representation_screen_v2.py'):
        original = subprocess.check_output(['git', 'show', freeze['commit'] + ':' + p], cwd=ROOT)
        if (ROOT / p).read_bytes() != original:
            raise ValueError('executed scientific source differs: ' + p)
        source_bindings[p] = hashlib.sha256(original).hexdigest()

    cohort = {}
    for role in ('geometry_discovery', 'geometry_check'):
        eligible = [i for i, r in enumerate(roles) if r['role'] == role]
        if len(eligible) != 64:
            raise ValueError('role count')
        cohort[role] = sorted(sorted(eligible, key=lambda i: fingerprint(
            [cfg['timing']['cohort_seed'], roles[i]['id']]))[:16])
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    # Common query resampling across all pipelines; repetitions remain paired.
    draws = {role: rng.integers(0, len(q), (BOOTSTRAP_REPLICATES, len(q)))
             for role, q in cohort.items()}
    queries = []; sensitivity = []; groups = []; calls = []; coverage = []
    grouped = {}
    candidate_matches = quality_matches = 0
    for name, prefix, candidate_file, quality_file, m in TARGETS:
        timings = data.json(prefix + '/timings.json')
        points = data.json(prefix + '/points.json')
        candidates = data.array(candidate_file)
        quality = {(r['query_index'], r['budget']): r for r in data.json(quality_file)}
        target = [r for r in timings if r['budget'] == m]
        if candidates.shape != (128, 6400) or len(target) != 96:
            raise ValueError('candidate/call coverage: ' + name)
        by_query = {}
        for r in target:
            qi = r['query_index']; ids = candidates[qi, :m]
            if (r['query_id'] != roles[qi]['id'] or r['role'] != roles[qi]['role']
                    or r['hardware_profile'] != 'cpu14'
                    or len(set(map(int, ids))) != m or ids.min() < 0 or ids.max() >= source['rows']):
                raise ValueError('identity/hardware/candidate input')
            if r['candidate_sha256'] != fingerprint(ids.tolist()):
                raise ValueError('timed candidate sequence differs from snapshot')
            candidate_matches += 1
            q = quality[qi, m]
            if r['returned'] != q['returned'] or r['distances'] != q['distances']:
                raise ValueError('timed rerank differs from saved quality')
            returned = np.asarray(r['returned'], np.int64)
            if len(returned) != cfg['k'] or not set(returned).issubset(set(ids)):
                raise ValueError('returned IDs outside candidate set')
            recall = float(np.mean(np.asarray(r['distances']) <= cutoffs[qi] + cfg['epsilon']))
            strict = len(set(returned).intersection(map(int, truth[qi]))) / cfg['k']
            if not np.isclose(recall, q['recall'], atol=1e-14, rtol=0) or not np.isclose(
                    strict, q['strict_id'], atol=1e-14, rtol=0):
                raise ValueError('saved metric does not replay')
            quality_matches += 1
            t, f, s = [float(r[k]) * 1000 for k in
                       ('seconds', 'rerank_seconds', 'search_transform_transfer_seconds')]
            if not (0 <= f <= t and 0 <= s <= t and s + f <= t + 1e-8):
                raise ValueError('inconsistent nested clocks')
            calls.append(dict(pipeline=name, role=r['role'], query_id=r['query_id'],
                query_index=qi, budget=m, repeat=r['repeat'], total_ms=t, rerank_ms=f,
                search_ms=s, wrapper_residual_ms=t-s-f, candidate_sha256=r['candidate_sha256']))
            by_query.setdefault(qi, []).append((r, t, f, s, recall, strict))
        for role, indices in cohort.items():
            present = sorted(qi for qi in by_query if roles[qi]['role'] == role)
            if present != indices:
                raise ValueError('cohort differs from frozen hash selection')
            qrows = []
            for qi in indices:
                records = by_query[qi]
                if sorted(r[0]['repeat'] for r in records) != [0, 1, 2]:
                    raise ValueError('missing/duplicate repeat')
                t, f, s = [np.array([r[k] for r in records]) for k in (1, 2, 3)]
                ids = candidates[qi, :m]
                p4, b4 = logical_addresses(ids, source['layout']['offset'], 2560, 4096)
                p64, b64 = logical_addresses(ids, source['layout']['offset'], 2560, 65536)
                row = dict(pipeline=name, role=role, query_id=roles[qi]['id'], query_index=qi,
                    budget=m, calls=3, total_ms=float(t.mean()), rerank_ms=float(f.mean()),
                    search_ms=float(s.mean()), wrapper_residual_ms=float((t-s-f).mean()),
                    rerank_fraction=float(f.mean()/t.mean()), total_repeat_range_ms=float(np.ptp(t)),
                    rerank_repeat_range_ms=float(np.ptp(f)), recall=records[0][4],
                    strict_id=records[0][5], candidate_sha256=records[0][0]['candidate_sha256'],
                    logical_candidate_vectors=m, logical_output_reread_vectors=100,
                    logical_candidate_payload_bytes=m*2560, logical_output_reread_payload_bytes=256000,
                    logical_total_gather_payload_bytes=(m+100)*2560,
                    unique_source_vectors=m, unique_virtual_pages_4k=p4,
                    virtual_page_coverage_bytes_4k=b4, unique_virtual_pages_64k=p64,
                    virtual_page_coverage_bytes_64k=b64,
                    source_address_order_hash=hashlib.sha256((6144+ids.astype('<i8')*2560).tobytes()).hexdigest())
                qrows.append(row); queries.append(row)
            t, f, s = [np.array([r[k] for r in qrows]) for k in ('total_ms', 'rerank_ms', 'search_ms')]
            bt, bf = [v[draws[role]].mean(axis=1) for v in (t, f)]
            grouped[name, role] = (t, f, s, bt, bf)
            point = next(p for p in points if p['role']==role and p['budget']==m and p['metric']=='recall')
            if not np.isclose(t.mean()/1000, point['seconds'], atol=1e-12, rtol=0):
                raise ValueError('saved point disagrees with query means')
            groups.append(dict(pipeline=name, role=role, budget=m, queries=16, calls=48,
                total_ms=quantiles(t), rerank_ms=quantiles(f), search_ms=quantiles(s),
                total_mean_ci95=interval(bt), rerank_mean_ci95=interval(bf),
                rerank_fraction_ratio_of_means=float(f.mean()/t.mean()),
                rerank_fraction_ci95=interval(bf/bt),
                per_query_fraction=quantiles(f/t), wrapper_residual_ms=quantiles(t-s-f),
                timing_cohort_recall=float(np.mean([r['recall'] for r in qrows])),
                timing_cohort_strict_id=float(np.mean([r['strict_id'] for r in qrows])),
                full64_recall=point['quality'],
                full64_strict_id=float(np.mean([quality[qi, m]['strict_id'] for qi, item
                    in enumerate(roles) if item['role'] == role])),
                max_free_f_speedup=float(t.mean()/(t.mean()-f.mean())),
                virtual_page_coverage_bytes_4k=quantiles([r['virtual_page_coverage_bytes_4k'] for r in qrows])))
            for gamma in GAMMAS:
                for cancel in FRACTIONS:
                    a = cancel*f - gamma*t
                    sensitivity.append(dict(pipeline=name, role=role, target_fraction=gamma,
                        assumed_cancelled_fraction_of_all_rerank=cancel,
                        mean_aux_allowance_ms=float(a.mean()),
                        aux_allowance_ci95_low=interval(cancel*bf-gamma*bt)[0],
                        aux_allowance_ci95_high=interval(cancel*bf-gamma*bt)[1],
                        queries_with_nonnegative_allowance=int((a >= 0).sum()), queries=16,
                        required_rerank_fraction_if_aux_free=float(gamma*t.mean()/f.mean()),
                        model='optimistic affine cost envelope, not measured cancellation'))
            coverage.append(dict(pipeline=name, role=role, budget=m, quality_queries=64,
                timed_queries=16, repetitions=3, calls=48, timing_coverage_fraction=.25,
                cohort_hash=fingerprint(indices), same_query_cohort_across_pipelines=True,
                physical_cpu_model_in_original_receipt='not recorded', faiss_threads=14,
                blas_threads=1, batch=1, precision='native FP32 clip(1-dot,0,2)'))
    comparisons = []
    for role in cohort:
        for left, right in [('opq64_ivf64','opq64_ivf256'), ('opq64_ivf64','pq64_flat')]:
            t1, f1, _, bt1, bf1 = grouped[left, role]
            t2, f2, _, bt2, bf2 = grouped[right, role]
            comparisons.append(dict(role=role, left=left, right=right, paired_queries=16,
                total_difference_ms=float(np.mean(t1-t2)), total_difference_ci95=interval(bt1-bt2),
                rerank_difference_ms=float(np.mean(f1-f2)), rerank_difference_ci95=interval(bf1-bf2),
                qualification='same cohort/hardware profile; distinct sequential jobs, not interleaved randomized trials'))
    result = dict(schema='pq-refinement-cost-C0-audit-v0', source_index_sha256=INDEX_SHA,
        source_commit=freeze['commit'], audit_script_sha256=sha(__file__),
        data_scope='saved observed development S0 records only', formal_certification=False,
        source=source, environment=freeze['environment'], allocation=freeze['allocation'],
        frozen_timing_config=cfg['timing'], source_bindings=source_bindings,
        input_files=list(data.consumed.values()), candidate_hash_matches=candidate_matches,
        timed_quality_matches=quality_matches, query_rows=len(queries), call_rows=len(calls),
        bootstrap=dict(unit='query with all three calls averaged first', replicates=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED, interval='95% percentile', roles_separate=True,
            common_draws_for_paired_statistics=True, includes_future_hardware_drift=False,
            independent_certification=False, not_simultaneous_intervals=True),
        groups=groups, paired_pipeline_comparisons=comparisons,
        sensitivity=sensitivity, gate1_status='unresolved_C1_actual_read_compute_not_measured',
        actual_physical_read_requests=None, actual_dram_bytes=None, actual_ssd_bytes=None,
        cancellable_gather_time_ms=None,
        missing=['full original vectors for actual callback replay',
                 'gather/copy vs distance/selection/full callback timings',
                 'real survivors address/batch costs and complete paired replay',
                 'auxiliary/control cost (no gate2 implementation in C0)',
                 'physical traffic counters if supported; otherwise explicitly unmeasured'])
    output.mkdir(parents=True)
    (output/'cost_space_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    for filename, rows in [('per_query.csv',queries),('calls.csv',calls),('sensitivity.csv',sensitivity),
                           ('coverage.csv',coverage),('paired_pipeline_comparisons.csv',comparisons)]:
        write_csv(output/filename, rows)
    print(json.dumps(dict(output=str(output), query_rows=len(queries), calls=len(calls),
                         status=result['gate1_status']), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(args.recovery, args.output)
