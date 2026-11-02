#!/usr/bin/env python3
"""Source-bound benchmark runner with an external resource watchdog."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / 'configs/gaussian_benchmark_space_v1.json'


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(str(path))
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def tree_size(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def stop_reason(elapsed, rss, artifact, temporary, limits):
    for key, value in [('seconds', elapsed), ('rss_bytes', rss), ('artifact_bytes', artifact), ('temporary_bytes', temporary)]:
        if value > limits[key]:
            return key
    return None


def check_dependencies(config):
    import numpy, scipy, faiss, h5py
    versions = {name: mod.__version__ for name, mod in [('numpy', numpy), ('scipy', scipy), ('faiss', faiss), ('h5py', h5py)]}
    if versions != config['dependencies']:
        raise RuntimeError('dependency mismatch: '+str(versions))
    # Check compiled capabilities without opening scientific data.
    probe = faiss.index_factory(8, 'IVF2,PQ2x4fs', faiss.METRIC_INNER_PRODUCT)
    probe.cp.seed = 63007
    probe.pq.cp.seed = 63007
    return versions


def worker(source, out, config):
    import numpy as np
    import faiss
    from mechanism.gaussian_benchmark_data_v1 import BenchmarkData, fingerprint, native_distances, native_recall, strict_retention, stable_id
    from mechanism.gaussian_benchmark_space_v1 import projection_matrix, project, exact_candidates, rerank, decompose_loss, oracle_frontier, build_backend, run_backend
    out = Path(out)
    versions = check_dependencies(config)
    started = time.monotonic()
    def check():
        if time.monotonic()-started > config['limits']['seconds']:
            raise TimeoutError('worker wall clock limit')
    data = BenchmarkData(source, config)
    q, truth, _, query_ids = data.queries('development')
    x = np.empty(data.file['train'].shape, np.float32)
    norm_min, norm_max = float('inf'), 0.0
    for i in range(0, len(x), config['block_rows']):
        check()
        block = data.file['train'][i:i+config['block_rows']]
        if not np.all(np.isfinite(block)):
            raise ValueError('nonfinite source')
        norm = np.sum(block.astype(np.float64)**2, axis=1)
        norm_min, norm_max = min(norm_min, float(norm.min())), max(norm_max, float(norm.max()))
        x[i:i+len(block)] = block
    # Row numbers are only persisted with explicit stable ID mapping artifacts.
    with (out/'corpus_id_map.tsv').open('x') as stream:
        stream.write('row\tstable_id\n')
        for i in range(len(x)):
            stream.write(f'{i}\t{stable_id("corpus", i)}\n')
    write_json(out/'roles.json', {key: [stable_id('test' if key == 'official_test' else 'learn', i) for i in rows] for key, rows in data.roles.items()})
    threshold = np.array([native_distances(qi, x[gt])[-1] for qi, gt in zip(q, truth)])
    # Derived learn distances, never represented as official distance arrays.
    write_json(out/'source.json', {'source_sha256': data.source_sha, 'config_sha256': data.config_hash,
        'dependencies': versions, 'corpus_shape': list(x.shape), 'norm_squared_range': [norm_min, norm_max],
        'learn_threshold_provenance': 'native distance to supplied kth learn neighbor',
        'query_norm_squared_range': [float(np.sum(q.astype(float)**2, axis=1).min()), float(np.sum(q.astype(float)**2, axis=1).max())]})
    audit_n = config['reference_audit_count']
    audit_i, audit_d = exact_candidates(x, q[:audit_n], config['k'], 'native', config['block_rows'], check)
    l2_i, _ = exact_candidates(x, q[:audit_n], config['k'], 'l2', config['block_rows'], check)
    write_json(out/'reference_audit.json', [{'query_id': query_ids[j],
        'supplied_truth_overlap': strict_retention(audit_i[j], truth[j], config['k']),
        'native_l2_overlap': strict_retention(l2_i[j], audit_i[j], config['k']),
        'supplied_kth_distance': float(threshold[j]), 'computed_kth_distance': float(audit_d[j, -1])} for j in range(audit_n)])
    # Float32 tolerance is diagnostic, not a rewrite of the supplied benchmark labels.
    if not np.allclose(audit_d[:, -1], threshold[:audit_n], rtol=0, atol=1e-5):
        raise ValueError('supplied learn ground truth fails native reference audit')
    exact_summary = []
    exact_paths = {}
    for m in config['dimensions']:
        for seed in config['seeds']['projection']:
            check()
            matrix = projection_matrix(x.shape[1], m, seed)
            np.save(out/f'projection-{m}-{seed}.npy', matrix)
            t = time.perf_counter()
            z = project(x, matrix)
            projection_cost = time.perf_counter()-t
            all_ids = []
            for j in range(0, len(q), config['query_batch']):
                ids, _ = exact_candidates(z, project(q[j:j+config['query_batch']], matrix), max(config['budgets']), 'l2', config['block_rows'], check)
                all_ids.append(ids)
            exact = np.concatenate(all_ids)
            path = out/f'exact-{m}-{seed}.npy'
            np.save(path, exact)
            exact_paths[(m, seed)] = path
            quality = np.empty((len(q), len(config['budgets'])))
            with (out/f'exact-{m}-{seed}.jsonl').open('x') as stream:
                for j, query in enumerate(q):
                    for a, budget in enumerate(config['budgets']):
                        ids, distances = rerank(query, x, exact[j, :budget], config['k'])
                        score = native_recall(distances, threshold[j], config['k'], config['native_epsilon'])
                        quality[j, a] = score
                        stream.write(json.dumps({'query_id': query_ids[j], 'M': budget, 'native_recall': score,
                            'strict_retention': strict_retention(ids, truth[j], config['k'])})+'\n')
            exact_summary.append({'m': m, 'seed': seed, 'fixed_mean_quality': quality.mean(axis=0).tolist(),
                'maximum_mean_quality': float(quality.max(axis=1).mean()), 'projection_build_seconds': projection_cost,
                'projection_sha256': hashlib.sha256(matrix.tobytes()).hexdigest()})
            write_json(out/f'exact-summary-{m}-{seed}.json', exact_summary[-1])
            del z, exact
            gc.collect()
    qualifying = [m for m in config['dimensions'] if all(row['maximum_mean_quality'] >= min(config['targets']) for row in exact_summary if row['m'] == m)]
    if qualifying:
        first = config['dimensions'].index(min(qualifying))
        selected_m = config['dimensions'][first:first+2]
    else:
        selected_m = [max(config['dimensions'])]
    write_json(out/'representation_selection.json', {'dimensions': selected_m, 'criterion': 'both seeds maximum mean native recall >= primary target; otherwise largest negative control', 'exact': exact_summary})
    specs = [{'family': 'original_hnsw', 'degree': degree} for degree in config['hnsw']['original_degree']]
    specs += [{'family': 'ivfpq'}]
    specs += [{'family': 'gaussian_hnsw', 'degree': config['hnsw']['projected_degree'], 'm': m, 'seed': seed} for m in selected_m for seed in config['seeds']['projection']]
    summaries, builds = [], []
    def actions_for(spec):
        if spec['family'] == 'ivfpq':
            return [{'M': m, 'nprobe': n} for n in config['ivfpq']['nprobe'] for m in config['ivfpq']['refine']]
        return [{'M': config['k'] if spec['family'] == 'original_hnsw' else m, 'ef_search': m*factor}
                for m in config['budgets'] for factor in (config['hnsw']['effort_multipliers'] if spec['family'] == 'gaussian_hnsw' else [1])]
    def measure(index, matrix, actions, queries, gt, thresholds, ids, repeats, logpath, exact=None):
        costs = np.zeros((len(queries), len(actions)))
        quality = np.full_like(costs, np.nan)
        faiss.omp_set_num_threads(config['limits']['latency_threads'])
        with logpath.open('x') as stream:
            for repeat in range(repeats):
                for j in range(len(queries)):
                    check()
                    for ai in np.roll(np.arange(len(actions)), (repeat+j) % len(actions)):
                        result, distances, candidates, timing = run_backend(index, queries[j], x, actions[ai], config['k'], matrix)
                        score = native_recall(distances, thresholds[j], config['k'], config['native_epsilon'])
                        if repeat and quality[j, ai] != score:
                            raise RuntimeError('backend nondeterministic query quality')
                        quality[j, ai] = score
                        costs[j, ai] += timing['total_seconds']/repeats
                        record = {'query_id': ids[j], 'action_index': int(ai), 'repeat': repeat, 'action': actions[ai], 'native_recall': score,
                            'strict_retention': strict_retention(result, gt[j], config['k']), 'timing': timing,
                            'candidate_rows': candidates.tolist(), 'returned_rows': result.tolist()}
                        if exact is not None:
                            record['loss_decomposition'] = decompose_loss(gt[j], exact[j, :actions[ai]['M']], candidates, config['k'])
                        stream.write(json.dumps(record)+'\n')
        return quality, costs
    for spec in specs:
        check()
        key = fingerprint(spec)[:16]
        matrix = None
        z = None
        exact = None
        projection_seconds = 0.0
        if spec['family'] == 'gaussian_hnsw':
            matrix = np.load(out/f"projection-{spec['m']}-{spec['seed']}.npy")
            t = time.perf_counter()
            z = project(x, matrix)
            projection_seconds = time.perf_counter()-t
            exact = np.load(exact_paths[(spec['m'], spec['seed'])], mmap_mode='r')
        index, build_seconds = build_backend(x, config, spec, z, check)
        index_path = out/f'index-{key}.faiss'
        faiss.write_index(index, str(index_path))
        actions = actions_for(spec)
        quality, costs = measure(index, matrix, actions, q, truth, threshold, query_ids, config['development_repeats'], out/f'development-{key}.jsonl', exact)
        np.savez(out/f'actions-{key}.npz', quality=quality, costs=costs)
        summary = {'key': key, 'spec': spec, 'actions': actions, 'mean_quality': quality.mean(axis=0).tolist(),
            'mean_cost_seconds': costs.mean(axis=0).tolist(),
            'oracle': {str(target): oracle_frontier(quality, costs, target) for target in config['targets']}}
        summaries.append(summary)
        builds.append({'key': key, 'spec': spec, 'index_bytes': index_path.stat().st_size,
            'original_vector_bytes_retained': x.nbytes, 'projected_array_bytes_during_build': 0 if z is None else z.nbytes,
            'build_seconds': build_seconds, 'projection_seconds': projection_seconds})
        write_json(out/f'summary-{key}.json', summary)
        write_json(out/f'build-{key}.json', builds[-1])
        del index, z, exact
        gc.collect()
    points = []
    for family in ('original_hnsw', 'ivfpq', 'gaussian_hnsw'):
        for target in config['targets']:
            choices = [(s['mean_cost_seconds'][i], s['key'], i) for s in summaries if s['spec']['family'] == family
                       for i in range(len(s['actions'])) if s['mean_quality'][i] >= target]
            if choices:
                _, key, ai = min(choices)
                points.append({'family': family, 'target': target, 'key': key, 'action_index': ai})
    if not points:
        write_json(out/'result.json', {'status': 'no_qualified_point', 'summaries': summaries, 'official_test_opened': False})
        data.close()
        return
    seal = {'source_sha256': data.source_sha, 'config_sha256': data.config_hash, 'points': points,
            'development_summaries_sha256': fingerprint(summaries)}
    write_json(out/'deployment_seal.json', seal)
    test_q, test_gt, test_threshold, test_ids = data.queries('official_test', seal)
    test_summary = []
    for key in sorted(set(point['key'] for point in points)):
        summary = next(s for s in summaries if s['key'] == key)
        spec = summary['spec']
        matrix = np.load(out/f"projection-{spec['m']}-{spec['seed']}.npy") if spec['family'] == 'gaussian_hnsw' else None
        chosen = sorted(set(point['action_index'] for point in points if point['key'] == key))
        index = faiss.read_index(str(out/f'index-{key}.faiss'))
        quality, costs = measure(index, matrix, [summary['actions'][a] for a in chosen], test_q, test_gt, test_threshold, test_ids,
                                 config['test_repeats'], out/f'test-{key}.jsonl')
        test_summary.append({'key': key, 'action_indices': chosen, 'mean_quality': quality.mean(axis=0).tolist(),
                             'mean_cost_seconds': costs.mean(axis=0).tolist(), 'p95_query_cost_seconds': np.quantile(costs, .95, axis=0).tolist()})
        del index
        gc.collect()
    write_json(out/'result.json', {'status': 'complete', 'development': summaries, 'test': test_summary,
        'builds': builds, 'seal': seal, 'claim_scope': 'matched native recall frontier; no Tri or formal certification'})
    data.close()


def terminate_owned(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def process_rss(pid):
    if sys.platform != 'linux':
        try:
            return int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(pid)], text=True).strip() or 0)*1024
        except subprocess.CalledProcessError:
            return 0  # Process exited between poll and ps; permission failures propagate.
    try:
        status = Path(f'/proc/{pid}/status').read_text()
        return int(next(line.split()[1] for line in status.splitlines() if line.startswith('VmRSS:')))*1024
    except (FileNotFoundError, StopIteration):
        return 0


def supervise(source, out, config):
    subprocess.run([sys.executable, str(ROOT/'scripts/check_code_protection.py')], cwd=ROOT, check=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    temporary = out/'tmp'
    temporary.mkdir()
    from mechanism.gaussian_benchmark_data_v1 import fingerprint
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
        raise RuntimeError('tracked checkout must be clean before scientific run')
    write_json(out/'manifest.json', {'source_commit': source_commit, 'config': config, 'config_sha256': fingerprint(config),
        'source_path': str(Path(source).resolve()), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(temporary))
    start = time.monotonic()
    peak_rss = 0
    reason = None
    with (out/'worker.log').open('x') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--source', str(Path(source).resolve()),
                                    '--output', str(out.resolve()), '--worker'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        monitor_error = None
        try:
            while process.poll() is None:
                rss = process_rss(process.pid)
                peak_rss = max(peak_rss, rss)
                reason = stop_reason(time.monotonic()-start, rss, tree_size(out), tree_size(temporary), config['limits'])
                if reason:
                    break
                time.sleep(.25)
        except BaseException as exc:
            monitor_error = type(exc).__name__ + ': ' + str(exc)
        finally:
            terminate_owned(process)
            code = process.wait()
    status = 'monitor_failed' if monitor_error else ('resource_stop' if reason else ('complete' if code == 0 else 'failed'))
    write_json(out/'execution.json', {'status': status, 'limit': reason, 'returncode': code,
        'monitor_error': monitor_error, 'elapsed_seconds': time.monotonic()-start,
        'peak_observed_rss_bytes': peak_rss, 'artifact_bytes': tree_size(out),
        'monitor_interval_seconds': .25, 'overshoot_possible': True})
    print(json.dumps({'status': status, 'output': str(out)}), flush=True)
    return 0 if status == 'complete' else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    if args.worker:
        worker(args.source, args.output, config)
        return 0
    return supervise(args.source, args.output, config)


if __name__ == '__main__':
    raise SystemExit(main())
