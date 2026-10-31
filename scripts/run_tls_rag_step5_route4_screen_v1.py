#!/usr/bin/env python3
"""Bounded route4 P1 runner; science subprocess tree shares one resource budget."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import faiss
import numpy as np

from mechanism.tls_rag_step5_route4_screen_v1 import (
    config, check_versions, sources, sha, fingerprint, save, checked, load_verified_inputs,
    gaussian_projection, build_independent_pool, projection_name, Runtime, specifications,
    spec_key, open_role, rerank_original, loss_ledger, measure_full_call,
)
from mechanism.tls_rag_step5_route4_risk_v1 import offline_pair_risk, free_budget_oracle, geometric_counterexamples


def store(root, name, value):
    p = Path(root) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as stream:
        np.save(stream, value, allow_pickle=False)


def event(root, stage, **data):
    value = dict(stage=stage, elapsed_seconds=time.monotonic() - START, **data)
    with (Path(root) / 'events.jsonl').open('a') as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(value, sort_keys=True), flush=True)


def assert_frozen(root):
    frozen = json.loads((root / 'source_freeze.json').read_text())
    if frozen['sources'] != sources() or frozen['config'] != config():
        raise ValueError('frozen source/config changed')
    if time.monotonic() >= frozen['deadline_monotonic']:
        raise ValueError('shared scientific deadline expired')
    os.kill(frozen['supervisor_pid'], 0)


def prepare(root, s):
    start = time.perf_counter()
    X, ids, roles = load_verified_inputs(s)
    store(root, 'corpus64.npy', X)
    store(root, 'corpus32.npy', X.astype(np.float32))
    save(root / 'ids.json', ids)
    for role, recs in roles.items():
        # The historical source packet contains all roles. No report retrieval
        # or newly computed report truth occurs until the seal opens its file.
        save(root / (role + '_queries.json'), recs)
    event(root, 'inputs_frozen', n=len(X), roles={r: [q['query_id'] for q in v] for r, v in roles.items()},
          historical_packet_loaded=True, fresh_independence=False)
    t = time.perf_counter()
    coarse, lists, pq = build_independent_pool(X, s)
    store(root, 'coarse.npy', coarse)
    with (root / 'lists.npz').open('xb') as f:
        np.savez(f, **{str(i): v for i, v in enumerate(lists['lists'])})
    faiss.write_index(pq, str(root / 'pq.faiss'))
    event(root, 'independent_pool_complete', seconds=time.perf_counter() - t,
          list_sizes=[len(v) for v in lists['lists']], before_any_gaussian=True)
    t = time.perf_counter()
    centered = X - X.mean(axis=0)
    _, eig = np.linalg.eigh(centered.T @ centered / len(X))
    pca = eig[:, ::-1].T
    del centered
    for m in s['dimensions']:
        for dtype, suffix in [(np.float64, '64'), (np.float32, '32')]:
            pi = pca[:m].astype(dtype)
            kind = 'pca' + suffix
            store(root, projection_name('matrix-' + kind, 0, m), pi)
            store(root, projection_name(kind, 0, m), X.astype(dtype, copy=False) @ pi.T)
    event(root, 'pca_complete', seconds=time.perf_counter() - t, corpus_only=True)
    for seed in s['design_seeds'] + s['report_seeds']:
        t = time.perf_counter()
        for m in s['dimensions']:
            for dtype, suffix in [(np.float64, '64'), (np.float32, '32')]:
                pi = gaussian_projection(s['d'], m, seed).astype(dtype)
                kind = 'gaussian' + suffix
                store(root, projection_name('matrix-' + kind, seed, m), pi)
                store(root, projection_name(kind, seed, m), X.astype(dtype, copy=False) @ pi.T)
        event(root, 'projection_bank_complete', seed=seed, seconds=time.perf_counter() - t)
    names = ['corpus64.npy', 'corpus32.npy', 'ids.json', 'coarse.npy', 'lists.npz', 'pq.faiss']
    names += [r + '_queries.json' for r in roles]
    names += [p.name for p in root.glob('*.npy') if p.name not in names]
    manifest = dict(schema='route4-p1-inputs-v1', config=s, sources=sources(),
                    normalized_original=True, projected_renormalization=False,
                    model_binding='historical MiniLM384; parent export SHA256 in config',
                    files={n: dict(bytes=(root / n).stat().st_size, sha256=sha(root / n)) for n in sorted(names)},
                    role_ids={r: [q['query_id'] for q in v] for r, v in roles.items()},
                    historical_packet_read_once_for_export=True, forbidden_roles_opened=False,
                    id_namespaces=dict(corpus='msmarco-passage', query='msmarco-query'),
                    external_query_provenance='checksum-bound historical external-query export; semantic independence not newly established',
                    formal_certification=False, fresh_independence=False)
    save(root / 'manifest.json', manifest)
    save(root / 'preparation.json', dict(seconds=time.perf_counter() - start,
         total_representation_bytes=sum(v['bytes'] for v in manifest['files'].values()),
         gaussian_bank_includes_unused_configurations=True, counterexample=geometric_counterexamples()))
    return manifest


def groups(s, role):
    specs = specifications(s)
    if role == 'advice':
        specs = [x for x in specs if x['kind'].startswith('gaussian')]
    result = {}
    for spec in specs:
        seeds = s['design_seeds'] if spec['kind'].startswith('gaussian') else [0]
        for seed in seeds:
            key = (spec['kind'], spec.get('m', 0), seed)
            result.setdefault(key, []).append(spec)
    return result


def quality(root, role, s, manifest, selected=None):
    queries = open_role(root, role, manifest)
    refs, reference = {}, Runtime(root, dict(kind='original64', M=s['k']), 0, s)
    allrows = np.arange(s['n'])
    for rec in queries:
        q = np.asarray(rec['query'])
        G = rerank_original(q, reference.X, allrows, s['k'], reference.ids)
        C = reference.candidates(q)[0]
        refs[rec['query_id']] = dict(G=G.tolist(), C=C.tolist(), g=len(set(G) & set(C)))
    save(root / ('reference_' + role + '.json'), refs)
    if selected is None:
        grid = groups(s, role)
    else:
        grid = {}
        for spec in (selected['gaussian'], selected['baseline']):
            seeds = s['report_seeds'] if spec['kind'].startswith('gaussian') else [0]
            for seed in seeds:
                grid[(spec['kind'], spec.get('m', 0), seed)] = [spec]
    summary = {}
    with (root / ('quality_' + role + '.jsonl')).open('x') as stream:
        for (_, _, seed), specs in grid.items():
            runtime = Runtime(root, specs[0], seed, s)
            for spec in specs:
                runtime.spec = spec
                key = spec_key(spec) + '@' + str(seed)
                item = dict(spec=spec, seed=seed, query_ids=[], missed=[], retention=[], upstream=[], output_hashes={})
                for rec in queries:
                    qid = rec['query_id']
                    out = runtime(np.asarray(rec['query']))
                    ref = refs[qid]
                    if out['C'].tolist() != ref['C']:
                        raise ValueError('pool depends on projection/method')
                    ledger = loss_ledger(ref['G'], ref['C'], out['E'].tolist(), out['final'].tolist())
                    result = dict(E=out['E'].tolist(), final=out['final'].tolist())
                    item['query_ids'].append(qid)
                    item['missed'].append(ledger['filtering_missed'])
                    item['upstream'].append(ledger['upstream_missed'])
                    item['retention'].append(ledger['retention'])
                    item['output_hashes'][qid] = fingerprint(result)
                    stream.write(json.dumps(dict(query_id=qid, key=key, **ledger, **result), separators=(',', ':')) + '\n')
                summary[key] = item
            del runtime
    save(root / ('quality_' + role + '_summary.json'), summary)
    event(root, role + '_quality_complete', configurations=len(summary), queries=len(queries),
          upstream_mean_loss=float(np.mean([s['k'] - r['g'] for r in refs.values()]) / s['k']))
    return summary, refs


def risk_diagnostics(root, s, manifest, refs, summary):
    runtime = Runtime(root, dict(kind='original64', M=s['k']), 0, s)
    values = []
    for rec in open_role(root, 'advice', manifest):
        qid, q = rec['query_id'], np.asarray(rec['query'])
        ref = refs[qid]
        t = time.perf_counter()
        value = offline_pair_risk(q, runtime.X, ref['G'], ref['C'], s['dimensions'], s['budgets'], s['risk_block'])
        value.update(query_id=qid, seconds=time.perf_counter() - t)
        values.append(value)
    save(root / 'risk_advice.json', values)
    table = []
    for m in s['dimensions']:
        for M in s['budgets']:
            bound = np.array([v['dimensions'][str(m)]['budgets'][str(M)]['filtering'] for v in values])
            actual = []
            for seed in s['design_seeds']:
                item = summary[spec_key(dict(kind='gaussian64', m=m, M=M)) + '@' + str(seed)]
                actual.append(np.asarray(item['missed']) / s['k'])
            a = np.asarray(actual)
            rng = np.random.Generator(np.random.PCG64(s['bootstrap_seed']))
            boot = [float(a[rng.integers(0, len(a), len(a))][:, rng.integers(0, a.shape[1], a.shape[1])].mean())
                    for _ in range(s['bootstrap_repeats'])]
            table.append(dict(m=m, M=M, mean_bound=float(bound.mean()),
                              nonvacuous_queries=int(np.sum(bound < np.array([v['g'] for v in refs.values()]) / s['k'])),
                              observed_mean_filter_loss=float(a.mean()),
                              crossed_query_seed_descriptive_ci=np.quantile(boot, [.025, .975]).tolist(),
                              actual_seed_means=a.mean(axis=1).tolist()))
    save(root / 'risk_summary.json', dict(table=table, seconds=sum(v['seconds'] for v in values),
         full_pairs=sum(v['full_pairs'] for v in values), conservative_pairs=sum(v['conservative_pairs'] for v in values),
         theory_is_expectation_over_independent_gaussian=True, fixed_seed_certification=False))
    event(root, 'risk_complete', seconds=sum(v['seconds'] for v in values))


def timing_worker(root, role, spec, seed, output):
    s = config()
    check_versions(s)
    assert_frozen(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if role == 'select':
        expected_seed = s['timing_seed'] if spec['kind'].startswith('gaussian') else 0
        if seed != expected_seed or spec not in specifications(s):
            raise ValueError('unfrozen select timing action')
    else:
        selected = json.loads((root / 'selection.json').read_text())
        allowed = [selected.get('gaussian'), selected['baseline']]
        seeds = s['report_seeds'] if spec['kind'].startswith('gaussian') else [0]
        if spec not in allowed or seed not in seeds:
            raise ValueError('unselected report timing action')
    # No truth/reference/quality files are opened by this worker.
    queries = open_role(root, role, manifest)
    start = time.perf_counter()
    runtime = Runtime(root, spec, seed, s)
    setup = time.perf_counter() - start
    before = resource.getrusage(resource.RUSAGE_SELF)
    result = measure_full_call(runtime, queries, s['timing_repeats'], s['schedule_seed'])
    after = resource.getrusage(resource.RUSAGE_SELF)
    result.update(spec=spec, seed=seed, setup_seconds=setup, loaded_query_file=role + '_queries.json',
                  input_manifest=fingerprint(manifest), sources=sources(),
                  maximum_rss_bytes=after.ru_maxrss if sys.platform == 'darwin' else after.ru_maxrss * 1024,
                  minor_faults=after.ru_minflt - before.ru_minflt, major_faults=after.ru_majflt - before.ru_majflt,
                  no_truth_file_opened=True, cold_ssd_claim=False)
    save(output, result)


def timings(root, role, s, summary, selected=None):
    specs = specifications(s) if selected is None else [selected['gaussian'], selected['baseline']]
    tasks = []
    for spec in specs:
        seeds = (s['report_seeds'] if role == 'report' else [s['timing_seed']]) if spec['kind'].startswith('gaussian') else [0]
        tasks += [(spec, seed) for seed in seeds]
    rng = np.random.Generator(np.random.PCG64(s['schedule_seed']))
    tasks = [tasks[int(j)] for j in rng.permutation(len(tasks))]
    save(root / ('timing_schedule_' + role + '.json'), [dict(spec=x, seed=y) for x, y in tasks])
    result = {}
    for j, (spec, seed) in enumerate(tasks):
        assert_frozen(root)
        key = spec_key(spec) + '@' + str(seed)
        output = root / 'timing' / role / (key + '.json')
        cmd = [sys.executable, __file__, 'timing', '--root', str(root), '--role', role,
               '--spec', json.dumps(spec), '--seed', str(seed), '--output', str(output)]
        subprocess.run(cmd, check=True)
        record = json.loads(output.read_text())
        if {q: fingerprint(v) for q, v in record['results'].items()} != summary[key]['output_hashes']:
            raise ValueError('measured outputs differ from quality replay: ' + key)
        by_q = {q: [] for q in summary[key]['query_ids']}
        for row in record['samples']:
            by_q[row['query_id']].append(row['full_ns'])
        result[key] = dict(spec=spec, seed=seed, query_ids=list(by_q),
                           samples=[by_q[q] for q in by_q],
                           mean_ns=float(np.mean(list(by_q.values()))),
                           p95_ns=float(np.quantile(list(by_q.values()), .95)))
        if j % 16 == 0 or j == len(tasks) - 1:
            event(root, role + '_timing_progress', finished=j + 1, total=len(tasks))
    save(root / ('timing_' + role + '_summary.json'), result)
    return result


def eligible(item, s):
    missed = np.asarray(item['missed'])
    return bool(missed.mean() / s['k'] <= s['maximum_mean_extra_loss'] + 1e-12 and
                missed.max() / s['k'] <= s['maximum_query_extra_loss'] + 1e-12)


def paired_cost(a, b, s):
    if a['query_ids'] != b['query_ids']:
        raise ValueError('unpaired timing IDs')
    A, B = np.asarray(a['samples']), np.asarray(b['samples'])
    rng = np.random.Generator(np.random.PCG64(s['bootstrap_seed']))
    boot = []
    for _ in range(s['bootstrap_repeats']):
        q = rng.integers(0, len(A), len(A))
        ar = rng.integers(0, A.shape[1], A.shape[1])
        br = rng.integers(0, B.shape[1], B.shape[1])
        boot.append(1 - float(A[q][:, ar].mean()) / float(B[q][:, br].mean()))
    return dict(saving=1 - a['mean_ns'] / b['mean_ns'],
                paired_query_repeat_descriptive_ci=np.quantile(boot, [.025, .975]).tolist(),
                p95_ratio=a['p95_ns'] / b['p95_ns'])


def select(root, s, quality_summary, times):
    baseline_keys = [k for k, t in times.items() if not t['spec']['kind'].startswith('gaussian') and eligible(quality_summary[k], s)]
    basekey = min(baseline_keys, key=lambda k: (times[k]['mean_ns'], k))
    good = []
    for key, value in times.items():
        spec = value['spec']
        if spec['kind'].startswith('gaussian') and all(eligible(quality_summary[spec_key(spec) + '@' + str(seed)], s)
                                                       for seed in s['design_seeds']):
            good.append(key)
    oracle = []
    for kind in ('gaussian64', 'gaussian32'):
        for m in s['dimensions']:
            keys = [spec_key(dict(kind=kind, m=m, M=M)) + '@' + str(s['timing_seed']) for M in s['budgets']]
            costs = np.asarray([np.mean(times[k]['samples'], axis=1) for k in keys]).T
            missed = np.asarray([quality_summary[k]['missed'] for k in keys]).T
            value = free_budget_oracle(costs, missed, s['k'], s['maximum_mean_extra_loss'], s['maximum_query_extra_loss'])
            value.update(kind=kind, m=m)
            if value['status'] == 'feasible':
                value['saving_vs_best_baseline'] = 1 - value['mean_cost'] / times[basekey]['mean_ns']
                value['budgets'] = [s['budgets'][a] for a in value['actions']]
            oracle.append(value)
    save(root / 'free_M_oracle.json', dict(diagnostic_only=True, values=oracle, baseline=basekey,
         note='Optimistically knows both truth and measured costs; fixed dimension within each allocation.'))
    selected = dict(status='no_candidate', baseline=times[basekey]['spec'], baseline_key=basekey,
                    eligible_gaussian_count=len(good), formal_certification=False, fresh_independence=False)
    if good:
        key = min(good, key=lambda k: (times[k]['mean_ns'], k))
        comparison = paired_cost(times[key], times[basekey], s)
        positive = comparison['saving'] >= s['minimum_saving'] and comparison['paired_query_repeat_descriptive_ci'][0] > 0 and comparison['p95_ratio'] <= 1
        selected.update(status='selected', gaussian=times[key]['spec'], gaussian_key=key,
                        select_comparison=comparison, select_economic_signal=positive)
    save(root / 'selection.json', selected)
    manifest = json.loads((root / 'manifest.json').read_text())
    save(root / 'selection_seal.json', dict(selection=fingerprint(selected), inputs=fingerprint(manifest), sources=sources()))
    event(root, 'selection_sealed', **selected)
    return selected


def precision_review(root, role, summary):
    # Exact saved membership differences, not just aggregate recall differences.
    path = root / ('quality_' + role + '.jsonl')
    rows = {}
    differences = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            key = (r['query_id'], r['key'].replace('gaussian32', 'gaussian').replace('gaussian64', 'gaussian')
                   .replace('pca32', 'pca').replace('pca64', 'pca'))
            if not r['key'].startswith(('gaussian', 'pca')):
                continue
            if key in rows:
                a = rows.pop(key)
                if a['E'] != r['E'] or a['final'] != r['final']:
                    differences.append(dict(query_id=r['query_id'], key=key[1],
                        first=a, second=r))
            else:
                rows[key] = r
    save(root / ('precision_' + role + '.json'), dict(disagreements=differences,
          unpaired_rows=len(rows), not_an_exact_floating_point_theorem=True))


def science(root):
    s = config()
    check_versions(s)
    assert_frozen(root)
    manifest = prepare(root, s)
    a, refs = quality(root, 'advice', s, manifest)
    risk_diagnostics(root, s, manifest, refs, a)
    precision_review(root, 'advice', a)
    q, _ = quality(root, 'select', s, manifest)
    precision_review(root, 'select', q)
    t = timings(root, 'select', s, q)
    selected = select(root, s, q, t)
    report = dict(status='closed_no_candidate')
    if selected['status'] == 'selected':
        r, _ = quality(root, 'report', s, manifest, selected)
        rt = timings(root, 'report', s, r, selected)
        comparisons = {}
        base = rt[spec_key(selected['baseline']) + '@0']
        for seed in s['report_seeds']:
            key = spec_key(selected['gaussian']) + '@' + str(seed)
            comparisons[str(seed)] = dict(quality_eligible=eligible(r[key], s), **paired_cost(rt[key], base, s))
        report = dict(status='measured', comparisons=comparisons,
                      baseline_quality_eligible=eligible(r[spec_key(selected['baseline']) + '@0'], s))
    assert_frozen(root)
    save(root / 'result.json', dict(status='completed', selection=selected, report=report,
         formal_certification=False, fresh_independence=False, no_predictor_trained=True,
         no_modern_ANN_speedup_claim=True, elapsed_science_seconds=time.monotonic() - START))
    event(root, 'completed', report_status=report['status'])


def descendant_processes(pid):
    rows = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss='], text=True)
    data = [tuple(map(int, line.split())) for line in rows.splitlines() if len(line.split()) == 3]
    chosen = {pid}
    while True:
        bigger = chosen | {p for p, parent, _ in data if parent in chosen}
        if bigger == chosen:
            break
        chosen = bigger
    return [(p, rss * 1024) for p, _, rss in data if p in chosen]


def descendants_rss(pid):
    return sum(rss for _, rss in descendant_processes(pid))


def terminate_owned_tree(proc):
    """Only the tracked child lineage; no process-group signals are needed.

    Freeze the science parent before enumerating its one active cost worker,
    so it cannot create another child while the supervisor stops the tree.
    """
    try:
        os.kill(proc.pid, signal.SIGSTOP)
    except ProcessLookupError:
        proc.wait()
        return
    pids = [pid for pid, _ in descendant_processes(proc.pid)]
    for pid in [p for p in pids if p != proc.pid] + [proc.pid]:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait(timeout=5)


def supervisor(root):
    s = config()
    check_versions(s)
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    save(root / 'source_freeze.json', dict(sources=sources(), config=s,
         supervisor_pid=os.getpid(), deadline_monotonic=started + s['limits']['seconds'],
         git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         user_consent='继续P1', consent_date='2026-11-01', exact_proposal='journal section34'))
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    maximum_rss, maximum_bytes, reason = 0, 0, None
    if shutil.disk_usage(root).free < s['limits']['free_bytes']:
        reason = 'insufficient_free_space_at_start'
        proc = None
    else:
        log = (root / 'science.log').open('x')
        proc = subprocess.Popen([sys.executable, __file__, 'science', '--root', str(root)],
                                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        last_disk = -100.
        while proc.poll() is None:
            elapsed = time.monotonic() - started
            rss = descendants_rss(os.getpid())
            maximum_rss = max(maximum_rss, rss)
            if elapsed - last_disk >= 5:
                size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
                maximum_bytes = max(maximum_bytes, size)
                last_disk = elapsed
            if elapsed >= s['limits']['seconds']:
                reason = 'cumulative_wall_limit'
            elif rss > s['limits']['rss_bytes']:
                reason = 'aggregate_RSS_limit'
            elif maximum_bytes > s['limits']['artifact_bytes']:
                reason = 'artifact_bytes_limit'
            elif shutil.disk_usage(root).free < s['limits']['free_bytes']:
                reason = 'free_space_limit'
            if reason:
                terminate_owned_tree(proc)
                break
            time.sleep(.5)
        log.close()
    state = 'resource_stop' if reason else ('completed' if proc.returncode == 0 else 'execution_failure')
    save(root / 'status.json', dict(status=state, reason=reason, exit_code=None if proc is None else proc.returncode,
         supervision_seconds=time.monotonic() - started, maximum_aggregate_rss_bytes=maximum_rss,
         maximum_monitored_artifact_bytes=maximum_bytes, stop_method='tracked_owned_PID_tree',
         formal_certification=False, fresh_independence=False))
    entries = {p.relative_to(root).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p))
               for p in sorted(root.rglob('*')) if p.is_file()}
    save(root / 'archive_index.json', dict(schema='route4-p1-archive-v1', files=entries))
    for name, entry in entries.items():
        checked(root / name, entry['sha256'])
    save(root / 'archive_verified.json', dict(index_sha256=sha(root / 'archive_index.json'),
         file_count=len(entries), bytes=sum(v['bytes'] for v in entries.values()), verified=True))
    print(json.dumps(dict(root=str(root), status=state, reason=reason), sort_keys=True), flush=True)
    if state != 'completed':
        raise SystemExit(2)


START = time.monotonic()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['run', 'science', 'timing'])
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--role', choices=['select', 'report'])
    p.add_argument('--spec')
    p.add_argument('--seed', type=int)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    if a.mode == 'run':
        supervisor(root)
    elif a.mode == 'timing':
        timing_worker(root, a.role, json.loads(a.spec), a.seed, a.output)
    else:
        try:
            science(root)
        except Exception:
            save(root / 'failure.json', dict(traceback=traceback.format_exc(), elapsed=time.monotonic() - START))
            raise


if __name__ == '__main__':
    main()
