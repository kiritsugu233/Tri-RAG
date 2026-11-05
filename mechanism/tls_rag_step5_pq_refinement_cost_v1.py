"""Approved C1 CPU cost replay; no predictor, sketch or new retrieval policy.

The executed S0 score/tie/search functions are called unchanged. Subsets and
prepaid inputs/scores are cost diagnostics, never quality-preserving policies.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np

from mechanism import tls_rag_step5_route6_representation_v1 as native
from mechanism import tls_rag_step5_route6_representation_v2 as stored

ROOT = Path(__file__).resolve().parents[1]
NEW_PATHS = (
    'mechanism/tls_rag_step5_pq_refinement_cost_v1.py',
    'configs/tls_rag_step5_pq_refinement_cost_v1.json',
    'tests/test_tls_rag_step5_pq_refinement_cost_v1.py',
    'scripts/run_tls_rag_step5_pq_refinement_cost_v1.py',
    'docs/PQ_REFINEMENT_COST_PROTOCOL_V1.md',
)
fingerprint = native.fingerprint


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def config():
    value = json.loads((ROOT / NEW_PATHS[1]).read_text())
    if (value['namespace'] != 'pq-refinement-cost-v1' or value['search_budget'] != 6400
            or value['k'] != 100 or value['nprobe'] != [64, 256]
            or value['timing']['warmup'] != 2 or value['timing']['repeats'] != 7
            or value['timing']['faiss_threads'] != 14 or value['timing']['blas_threads'] != 1
            or value['subset_counts'] != [1600, 3200, 4800]
            or value['gate2_enabled'] or value['tri_enabled']):
        raise ValueError('C1 configuration outside approved scope')
    return value


def source_map(cfg):
    for p, expected in cfg['old_source_bindings'].items():
        if sha(ROOT / p) != expected:
            raise ValueError('frozen original source changed: ' + p)
    return {p: sha(ROOT / p) for p in NEW_PATHS}


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def unique_json(path, value):
    if Path(path).exists():
        raise FileExistsError(path)
    atomic_json(path, value)


def validate_ids(raw, candidates, k):
    # Match the original validation, including its unique check and cost.
    ids = np.asarray(candidates, np.int64)
    if (ids.ndim != 1 or len(ids) < k or np.any(ids < 0)
            or np.any(ids >= len(raw)) or len(np.unique(ids)) != len(ids)):
        raise ValueError('invalid retrieval candidates')
    return ids


def recorded_rerank(raw, query, candidates, k, clock=time.perf_counter):
    """Same values/operation order as native.rerank, with disjoint block clocks."""
    start = clock(); ids = validate_ids(raw, candidates, k); after_ids = clock()
    values = raw[ids]; after_gather = clock()
    distances = native.original_distances(values, query); del values; after_score = clock()
    selected = native.stable_top(distances, k, ids); after_select = clock()
    final_values = raw[selected]; after_final_gather = clock()
    final = native.original_distances(final_values, query); del final_values; end = clock()
    return selected, final, dict(
        id_seconds=after_ids-start, gather_copy_seconds=after_gather-after_ids,
        distance_clip_seconds=after_score-after_gather,
        selection_seconds=after_select-after_score,
        output_gather_seconds=after_final_gather-after_select,
        output_score_seconds=end-after_final_gather, observed_inner_seconds=end-start,
        gather_batches=2, gather_rows=int(len(ids)+k), unique_source_rows=int(len(ids)),
        logical_gather_payload_bytes=int((len(ids)+k)*raw.shape[1]*4))


def subset_positions(candidates, count, pattern, seed):
    ids = np.asarray(candidates, np.int64)
    if count < 1 or count > len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError('subset shape/identity')
    if pattern == 'prefix':
        return np.arange(count, dtype=np.int64)
    if pattern == 'spread':
        return (np.arange(count, dtype=np.int64)*len(ids))//count
    if pattern == 'hash':
        order = sorted(range(len(ids)), key=lambda i: (fingerprint([seed, int(ids[i])]), i))
        # Preserve the upstream order among retained IDs.
        return np.sort(np.asarray(order[:count], np.int64))
    raise ValueError('unknown subset pattern')


def survivor_replay(candidates, positions):
    positions = np.asarray(positions, np.int64)
    if (positions.ndim != 1 or not len(positions) or np.any(positions < 0)
            or np.any(positions >= len(candidates)) or len(np.unique(positions)) != len(positions)):
        raise ValueError('invalid saved subset positions')
    # The actual subset ID access and validation stay inside the timed callback.
    return np.asarray(candidates)[positions]


def prepare_prepaid(raw, query, candidates, k):
    """Explicitly paid outside-call oracle state; never called a deployment gain."""
    tick = time.perf_counter(); ids = validate_ids(raw, candidates, k)
    values = np.array(raw[ids], dtype=np.float32, copy=True)
    scores = native.original_distances(values, query)
    selected = native.stable_top(scores, k, ids)
    output_scores = native.original_distances(raw[selected], query)
    return dict(ids=ids, values=values, scores=scores,
        query_sha256=hashlib.sha256(np.ascontiguousarray(query,np.float32).tobytes()).hexdigest(),
        raw_object_id=id(raw),
        row_to_position={int(row): j for j, row in enumerate(ids)},
        output_scores={int(row): score for row, score in zip(selected, output_scores)},
        paid_prepare_seconds=time.perf_counter()-tick,
        array_bytes=int(values.nbytes+ids.nbytes+scores.nbytes),
        python_lookup_bytes=int(sys.getsizeof({int(row): j for j, row in enumerate(ids)})),
        actual_resident_bytes_not_measured=True)


def prepaid_rerank(raw, query, candidates, k, cache, mode):
    ids = validate_ids(raw, candidates, k)
    if (not np.array_equal(ids, cache['ids']) or cache['raw_object_id']!=id(raw)
            or cache['query_sha256']!=hashlib.sha256(np.ascontiguousarray(query,np.float32).tobytes()).hexdigest()):
        raise ValueError('prepaid state belongs to another candidate sequence')
    scores = (native.original_distances(cache['values'], query)
              if mode == 'prepaid_input' else cache['scores'])
    selected = native.stable_top(scores, k, ids)
    if mode == 'prepaid_input':
        positions = [cache['row_to_position'][int(row)] for row in selected]
        result = native.original_distances(cache['values'][positions], query)
    elif mode == 'prepaid_scores':
        result = np.asarray([cache['output_scores'][int(row)] for row in selected], np.float32)
    else:
        raise ValueError('unknown oracle mode')
    return selected, result


def full_callback(raw, query, search, k, mode='baseline', positions=None, cache=None):
    """Actual upstream+callback clock; hashes, output comparison and I/O outside."""
    start = time.perf_counter(); candidates = search(query); after_search = time.perf_counter()
    parts = None
    if mode == 'baseline':
        returned, distances = native.rerank(raw, query, candidates, k)
    elif mode == 'observed':
        returned, distances, parts = recorded_rerank(raw, query, candidates, k)
    elif mode in ('prepaid_input', 'prepaid_scores'):
        returned, distances = prepaid_rerank(raw, query, candidates, k, cache, mode)
    elif mode.startswith('subset_'):
        selected_ids = survivor_replay(candidates, positions)
        returned, distances = native.rerank(raw, query, selected_ids, k)
    else:
        raise ValueError('unknown replay mode')
    end = time.perf_counter()
    return dict(candidates=candidates, returned=returned, distances=distances,
        seconds=end-start, search_seconds=after_search-start, rerank_seconds=end-after_search,
        parts=parts)


def address_counts(ids, returned, cfg):
    row_bytes = cfg['source']['dimension']*4
    start = cfg['source']['layout']['offset']+np.asarray(ids, np.int64)*row_bytes
    last = start+row_bytes-1
    result = dict(gather_batches=2, logical_vectors=int(len(ids)+len(returned)),
        unique_source_rows=int(len(np.unique(ids))), logical_payload_bytes=int((len(ids)+len(returned))*row_bytes),
        original_candidate_payload_bytes=int(len(ids)*row_bytes),
        output_reread_payload_bytes=int(len(returned)*row_bytes),
        actual_dram_bytes=None, actual_ssd_bytes=None, physical_requests=None)
    for page in (4096, 65536):
        # Native row=2560B fits at most two pages at either granularity.
        if row_bytes > page:
            raise ValueError('address model requires rows no larger than page')
        n = int(len(np.union1d(start//page, last//page)))
        result['unique_virtual_pages_'+str(page)] = n
    return result


def cpu_identity():
    model = None
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                model = line.split(':', 1)[1].strip(); break
    except OSError:
        pass
    limits={}
    for p in ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/cpu/cpu.cfs_quota_us',
              '/sys/fs/cgroup/cpu/cpu.cfs_period_us','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/cpu.max'):
        try:limits[p]=Path(p).read_text().strip()
        except OSError:pass
    return dict(platform=platform.platform(), machine=platform.machine(), cpu_model=model, cgroup_limits=limits,
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        threads_env={n: os.environ.get(n) for n in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS')},
        actual_dram_bytes=None, traffic_counter_status='not measured: no attributed DRAM-byte counter configured')


def process_observation():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    io = None
    try:
        io = {line.split(':')[0]: int(line.split(':')[1]) for line in Path('/proc/self/io').read_text().splitlines()}
    except (OSError, ValueError):
        pass
    return dict(ru_minflt=usage.ru_minflt, ru_majflt=usage.ru_majflt,
        rss_highwater_bytes=int(usage.ru_maxrss*(1 if sys.platform=='darwin' else 1024)),
        process_io=io, io_is_not_dram_bytes=True)


def verify_product(root, entry):
    relative = Path(entry['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('unsafe old product path')
    path = Path(root)/relative
    if path.is_symlink() or path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
        raise ValueError('old product identity: ' + entry['path'])
    return path


def load_cost_inputs(cfg, phase=lambda *a: None):
    import faiss
    import h5py
    import scipy
    actual = dict(numpy=np.__version__, faiss=faiss.__version__, h5py=h5py.__version__, scipy=scipy.__version__)
    if actual != cfg['versions']:
        raise ValueError('existing fixed versions changed: '+str(actual))
    if any(os.environ.get(n) != '1' for n in cfg['single_thread_env']):
        raise ValueError('BLAS/initial thread environment must be one')
    source_map(cfg); old = Path(cfg['source_run'])
    if sha(old/'delivery_index.json') != cfg['source_index_sha256']:
        raise ValueError('old closed source index changed')
    original = {r['path']: r for r in json.loads((old/'delivery_index.json').read_text())['files']}
    for entry in cfg['products']:
        if original.get(entry['path']) != entry:
            raise ValueError('configuration product differs from closed source inventory')
        verify_product(old, entry)
    phase('verify_original_source_once')
    begin = time.perf_counter()
    # Uses the original read-only layout guard, only the train mapping.
    raw = stored.source_mapping(dict(source=cfg['source']), verify_hash=True)
    source_verify_seconds = time.perf_counter()-begin
    raw.flags.writeable = False
    queries = np.load(old/'queries.npy', allow_pickle=False)
    roles = json.loads((old/'roles.json').read_text())['queries']
    if queries.shape != (128, 640) or queries.dtype != np.float32 or len(roles) != 128:
        raise ValueError('original query shape/type/role count')
    chosen = []
    for role in ('geometry_discovery', 'geometry_check'):
        eligible = [i for i, r in enumerate(roles) if r['role']==role]
        if len(eligible) != 64:
            raise ValueError('old role sizes')
        chosen += sorted(eligible, key=lambda i: fingerprint([cfg['cohort_seed'], roles[i]['id']]))[:16]
    chosen = sorted(chosen)
    if chosen != cfg['query_indices'] or len({roles[i]['id'] for i in chosen}) != 32:
        raise ValueError('frozen C0 CPU query identities changed')
    for i in chosen:
        if (not np.isfinite(queries[i]).all() or stored.canonical_vector_hash(queries[i]) != roles[i]['vector_sha256']
                or roles[i]['id'] != stored.stable_id(cfg['source']['sha256'], 'learn', roles[i]['row'])):
            raise ValueError('saved query identity')
    matrix = np.load(old/'ivf/opq64/matrix.npy', allow_pickle=False)
    mean = np.load(old/'ivf/opq64/mean.npy', allow_pickle=False)
    identity = json.loads((old/'ivf/opq64/model_identity.json').read_text())
    if identity['source'] != cfg['source']['sha256'] or identity['spec'] != cfg['opq_spec']:
        raise ValueError('original OPQ identity')
    if matrix.shape != (640, 640) or mean.shape != (640,) or matrix.dtype != np.float32 or mean.dtype != np.float64:
        raise ValueError('original OPQ transform')
    phase('load_existing_ivf')
    index = faiss.read_index(str(old/'ivf/opq64.faiss'))
    if (index.d != 640 or index.ntotal != cfg['source']['rows'] or index.nlist != 1024
            or index.metric_type != faiss.METRIC_INNER_PRODUCT or index.code_size != 64
            or index.pq.M != 64 or index.pq.nbits != 8 or index.max_codes != 0):
        raise ValueError('original IVF structure/metric')
    faiss.omp_set_num_threads(14)
    model = dict(matrix=matrix, mean=mean, metric='ip', spec=identity['spec'])
    candidates = {}; expected = {}
    for probe in cfg['nprobe']:
        prefix=old/'ivf/opq64'/str(probe)
        candidates[probe] = np.load(prefix/'candidates.npy', mmap_mode='r', allow_pickle=False)
        if candidates[probe].shape != (128, 6400):
            raise ValueError('original candidate snapshot shape')
        rs = [r for r in json.loads((prefix/'timings.json').read_text()) if r['budget']==6400]
        for qi in chosen:
            rows = [r for r in rs if r['query_index']==qi]
            if len(rows)!=3 or sorted(r['repeat'] for r in rows)!=[0,1,2]:
                raise ValueError('missing original CPU records')
            digest = fingerprint(candidates[probe][qi].tolist())
            if any(r['query_id']!=roles[qi]['id'] or r['role']!=roles[qi]['role']
                   or r['candidate_sha256']!=digest or r['hardware_profile']!='cpu14'
                   or r['returned']!=rows[0]['returned'] or r['distances']!=rows[0]['distances'] for r in rows):
                raise ValueError('original reference repeats disagree')
            expected[probe, qi] = rows[0]
    return dict(raw=raw, queries=queries, roles=roles, indices=chosen, model=model, index=index,
        candidates=candidates, expected=expected, source_verify_seconds=source_verify_seconds,
        versions=actual, binary_sha256=dict(numpy=sha(np.core._multiarray_umath.__file__),
            faiss={name:sha(module.__file__) for name,module in list(sys.modules.items())
                if name.startswith('faiss.') and getattr(module,'__file__','').endswith(('.so','.pyd'))}),
        original_source_stat=dict(bytes=Path(cfg['source']['path']).stat().st_size,
            mtime_ns=Path(cfg['source']['path']).stat().st_mtime_ns))


def compare_exact(result, ids, expected):
    if not np.array_equal(result['candidates'], ids):
        raise RuntimeError('upstream candidate sequence differs from sealed S0')
    if (result['returned'].tolist()!=expected['returned']
            or result['distances'].tolist()!=expected['distances']):
        raise RuntimeError('native callback differs from sealed S0; no tolerance/retry')


def summarize_paired(records, cfg):
    groups = []; by = {}
    for r in records:
        by.setdefault((r['nprobe'], r['role'], r['mode'], r['query_index']), []).append(r)
    query_rows=[]
    for key, rs in sorted(by.items()):
        if sorted(r['repeat'] for r in rs)!=list(range(cfg['timing']['repeats'])):
            raise ValueError('incomplete repeated timing group')
        probe, role, mode, qi=key
        row=dict(nprobe=probe,role=role,mode=mode,query_index=qi,query_id=rs[0]['query_id'],
            calls=len(rs), seconds=float(np.mean([r['seconds'] for r in rs])),
            rerank_seconds=float(np.mean([r['rerank_seconds'] for r in rs])),
            search_seconds=float(np.mean([r['search_seconds'] for r in rs])))
        if mode=='observed':
            row['parts']={p:float(np.mean([r['parts'][p] for r in rs])) for p in rs[0]['parts']}
        query_rows.append(row)
    lookup={(r['nprobe'],r['role'],r['mode'],r['query_index']):r for r in query_rows}
    rng=np.random.Generator(np.random.PCG64(cfg['bootstrap_seed']))
    for probe in cfg['nprobe']:
        for role in ('geometry_discovery','geometry_check'):
            indices=sorted({r['query_index'] for r in query_rows if r['nprobe']==probe and r['role']==role})
            if len(indices)!=16: raise ValueError('incomplete paired query cohort')
            draws=rng.integers(0,16,(cfg['bootstrap_replicates'],16))
            base=np.array([lookup[probe,role,'baseline',i]['seconds'] for i in indices])
            for mode in cfg['modes']:
                values=np.array([lookup[probe,role,mode,i]['seconds'] for i in indices])
                saving=base-values; bs=base[draws].mean(1); vs=values[draws].mean(1)
                groups.append(dict(nprobe=probe,role=role,mode=mode,queries=16,
                    mean_seconds=float(values.mean()), median_query_seconds=float(np.median(values)),
                    p95_query_seconds=float(np.quantile(values,.95)), mean_saving_seconds=float(saving.mean()),
                    saving_fraction=float(saving.mean()/base.mean()),
                    saving_fraction_ci95=np.quantile((bs-vs)/bs,[.025,.975]).tolist(),
                    mean_rerank_seconds=float(np.mean([lookup[probe,role,mode,i]['rerank_seconds'] for i in indices])),
                    aux_allowance=[dict(gamma=g, mean_seconds=float(np.mean(saving-g*base)),
                        ci95=np.quantile(bs-vs-g*bs,[.025,.975]).tolist()) for g in cfg['gammas']],
                    cost_diagnostic_only=mode!='baseline'))
    return dict(query_rows=query_rows, groups=groups, actual_dram_bytes=None,
        formal_certification=False, gate2_started=False, tri_implemented=False,
        judgement='cost decomposition only; no actual quality-preserving cancellation controller',
        intervals='query-cluster percentile; observed cohorts, not simultaneous or independent certification')


def run_science(cfg, output):
    output=Path(output); initial=source_map(cfg)
    def phase(name, **fields): atomic_json(output/'progress.json',dict(phase=name,**fields))
    phase('load_and_validate_inputs')
    data=load_cost_inputs(cfg,phase); raw=data['raw']
    unique_json(output/'input_receipt.json',dict(versions=data['versions'], source_verify_seconds=data['source_verify_seconds'],
        source_stat=data['original_source_stat'], query_indices=data['indices'], source_index_sha256=cfg['source_index_sha256'],
        mapped_mode='read-only', test_or_truth_payload_read=False, hardware=cpu_identity(),
        source_map=initial, binary_sha256=data['binary_sha256'], source_allocation_is_not_asserted_equal_to_S0=True))
    records=[]; preparations=[]; checks=0; total=0
    rng=np.random.Generator(np.random.PCG64(cfg['schedule_seed']))
    with (output/'calls.jsonl').open('x') as stream:
        for probe in cfg['nprobe']:
            data['index'].nprobe=probe
            for qi in rng.permutation(data['indices']):
                qi=int(qi); q=data['queries'][qi]; ids=data['candidates'][probe][qi]
                search=lambda query: stored.search_backend(data['model'],query,6400,index=data['index'])
                reference=full_callback(raw,q,search,100)
                compare_exact(reference,ids,data['expected'][probe,qi]); checks+=1
                prep_start=time.perf_counter(); cache=prepare_prepaid(raw,q,ids,100); positions={}
                for pattern in ('prefix','spread','hash'):
                    for count in cfg['subset_counts']:
                        positions['subset_'+pattern+'_'+str(count)]=subset_positions(ids,count,pattern,cfg['subset_seed'])
                preparations.append(dict(nprobe=probe,query_index=qi,paid_prepare_seconds=time.perf_counter()-prep_start,
                    prepaid_array_bytes=cache['array_bytes'],python_lookup_shallow_bytes=cache['python_lookup_bytes']))
                # Reference new subsets with the original function. No truth/recall labels are loaded.
                subset_reference={mode:native.rerank(raw,q,survivor_replay(ids,pos),100) for mode,pos in positions.items()}
                before=process_observation()
                for repeat in range(-cfg['timing']['warmup'],cfg['timing']['repeats']):
                    for mode in rng.permutation(cfg['modes']):
                        mode=str(mode)
                        result=full_callback(raw,q,search,100,mode,positions.get(mode),cache)
                        if mode in positions:
                            if not np.array_equal(result['candidates'],ids):raise RuntimeError('subset upstream changed')
                            rr,dd=subset_reference[mode]
                            if not np.array_equal(rr,result['returned']) or not np.array_equal(dd,result['distances']):
                                raise RuntimeError('subset native score/ties changed')
                        else:
                            compare_exact(result,ids,data['expected'][probe,qi])
                        checks+=1
                        if repeat>=0:
                            used=survivor_replay(ids,positions[mode]) if mode in positions else ids
                            row=dict(nprobe=probe,query_index=qi,query_id=data['roles'][qi]['id'],role=data['roles'][qi]['role'],
                                mode=mode,repeat=repeat,seconds=result['seconds'],search_seconds=result['search_seconds'],
                                rerank_seconds=result['rerank_seconds'],parts=result['parts'],
                                full_candidate_sha256=fingerprint(ids.tolist()),used_candidate_sha256=fingerprint(used.tolist()),
                                returned_sha256=fingerprint(result['returned'].tolist()),
                                distance_sha256=hashlib.sha256(result['distances'].tobytes()).hexdigest(),
                                address=address_counts(used,result['returned'],cfg))
                            stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');stream.flush()
                            records.append(row);total+=1
                unique_json(output/'query_observations'/('{}-{}.json'.format(probe,qi)),
                    dict(before=before,after=process_observation(),counting_scope='whole query modes, including audit reads; not per-call DRAM'))
                phase('replay',nprobe=probe,query_index=qi,formal_calls=total,conformance_calls=checks)
    unique_json(output/'preparations.json',preparations)
    unique_json(output/'summary.json',summarize_paired(records,cfg))
    if initial!=source_map(cfg):raise RuntimeError('scientific source changed during C1')
    source_path=Path(cfg['source']['path']); st=data['original_source_stat']
    if source_path.stat().st_size!=st['bytes'] or source_path.stat().st_mtime_ns!=st['mtime_ns']:
        raise RuntimeError('source metadata changed during read-only replay')
    return dict(status='complete',formal_calls=total,conformance_calls=checks,
                source_map_unchanged=True,source_stat_unchanged=True, no_automatic_gate2=True)
