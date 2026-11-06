"""One closed-array attribution audit; no retrieval, fitting or raw-vector access."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PATHS = ('mechanism/tls_rag_step5_pq_attribution_v1.py',
         'configs/tls_rag_step5_pq_attribution_v1.json',
         'tests/test_tls_rag_step5_pq_attribution_v1.py',
         'scripts/run_tls_rag_step5_pq_attribution_v1.py',
         'docs/PQ_REFINEMENT_ATTRIBUTION_PROTOCOL_V1.md')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def config():
    c = json.loads((ROOT/PATHS[1]).read_text())
    if (c['namespace'] != 'pq-attribution-v1' or c['formal_certification']
            or c['budgets'] != [1600, 3200, 4800, 6400] or c['boundary_ranks'] != [80, 120]
            or c['nprobe'] != [64, 256] or c['primary'] != 'gaussian_2026110401_32'
            or c['dimensions'] != [8,16,32] or c['seeds'] != [2026110401,2026110402,2026110403,2026110404]
            or c['limits'] != dict(seconds=1200, rss_bytes=4294967296, output_bytes=268435456)):
        raise ValueError('outside approved one-attribution contract')
    return c


def stable_order(scores, ids):
    scores = np.asarray(scores, float); ids = np.asarray(ids)
    if scores.shape != ids.shape or scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError('finite aligned score vector required')
    if len(np.unique(ids)) != len(ids): raise ValueError('duplicate IDs')
    distance = np.clip(1-scores, 0, 2).astype(np.float32)
    return np.lexsort((ids, distance))


def point_scores(query, base, summary, positions, kind):
    """No truth, true errors or original rows accepted by point estimator."""
    q = np.asarray(query, float); base = np.asarray(base, float); pos = np.asarray(positions)
    if not np.isfinite(q).all() or not np.isfinite(base).all(): raise ValueError('nonfinite input')
    if kind == 'gaussian':
        Q = np.linalg.norm(q)
        if Q <= 0: raise ValueError('zero query: point attribution undefined')
        R = summary['R']; h = R @ (q/Q); T = float(h @ h)
        if T < 1e-12 or not np.isfinite(T): raise ValueError('invalid conditional projection norm')
        correction = Q*(summary['sketch'][pos].astype(float) @ h)/T
    elif kind == 'sidepq':
        centers = summary['centers'].astype(float); codes = summary['codes'][pos]
        width, _, subdim = centers.shape
        if q.shape != (width*subdim,): raise ValueError('sidePQ dimensions')
        table = np.einsum('wcd,wd->wc', centers, q.reshape(width, subdim))
        correction = table[np.arange(width)[None,:], codes].sum(axis=1)
    else: raise ValueError('unknown point estimator')
    result = base + correction
    if not np.isfinite(result).all(): raise ValueError('nonfinite point scores')
    return result


def rank_diagnostics(scores, ids, true_distances, true_top, budgets, order=None):
    ids = np.asarray(ids); d = np.asarray(true_distances)
    if ids.shape != d.shape or not np.isfinite(d).all(): raise ValueError('native distance schema')
    order = stable_order(scores, ids) if order is None else np.asarray(order)
    if not np.array_equal(np.sort(order), np.arange(len(ids))): raise ValueError('invalid permutation')
    out=[]; gt=set(map(int,true_top))
    for M in budgets:
        if M < len(gt) or M > len(ids): raise ValueError('invalid diagnostic cutoff')
        pos=order[:M]; chosen=pos[np.lexsort((ids[pos],d[pos]))[:len(gt)]]
        returned=ids[chosen]
        out.append(dict(budget=M, candidate_retention=len(set(map(int,returned))&gt)/len(gt),
                        preserves_top100=returned.tolist()==list(map(int,true_top)),
                        returned=returned.tolist(), distances=d[chosen].astype(float).tolist()))
    return out


def boundary_pairs(order, ranks=(80,120)):
    a,b=ranks
    if a<1 or b<a or b>len(order): raise ValueError('boundary ranks outside candidates')
    window=np.asarray(order)[a-1:b];i,j=np.triu_indices(len(window),1)
    return window[i],window[j]


def relative_error_diagnostics(estimate, truth, pairs):
    e=np.asarray(estimate,float)-np.asarray(truth,float); i,j=pairs
    if not np.isfinite(e).all(): raise ValueError('nonfinite errors')
    denominator=float(np.mean(e[i]**2+e[j]**2));diff=e[i]-e[j]
    truegap=np.asarray(truth)[i]-np.asarray(truth)[j]
    estgap=np.asarray(estimate)[i]-np.asarray(estimate)[j]
    return dict(point_mse=float(np.mean(e**2)), point_bias=float(np.mean(e)),
                pair_bias=float(np.mean(diff)), pair_mse=float(np.mean(diff**2)),
                uncorrelated_error_reference=denominator,
                cancellation_ratio=None if denominator==0 else float(np.mean(diff**2)/denominator),
                perfect_point_estimate=bool(np.all(e==0)),
                strict_pair_inversions=int(np.count_nonzero(truegap*estgap<0)),pairs=len(i))


def conditional_covariance(residuals, query, m, T):
    e=np.asarray(residuals,float);q=np.asarray(query,float)
    if m<1 or T<=0 or np.linalg.norm(q)==0: raise ValueError('invalid conditional fixture')
    u=q/np.linalg.norm(q);v=e-(e@u)[:,None]*u
    return v@v.T/(m*T)


def proxy_cost_diagnostics(adc, radii, sketch, pairs, pair_squared_error, pair_uncorrelated_reference=None):
    """Fixed descriptive associations; never a selector and no truth-fed proxy."""
    i,j=pairs; start=time.perf_counter()
    proxies=dict(adc_gap=np.abs(adc[i]-adc[j]), residual_radius=radii[i]+radii[j])
    if sketch is not None: proxies['sketch_difference_norm']=np.linalg.norm(sketch[i].astype(float)-sketch[j].astype(float),axis=1)
    elapsed=time.perf_counter()-start
    result={}
    for name,x in proxies.items():
        rho=None if np.ptp(x)==0 or np.ptp(pair_squared_error)==0 else float(spearmanr(x,pair_squared_error).statistic)
        ratio_rho=None
        if pair_uncorrelated_reference is not None:
            valid=np.asarray(pair_uncorrelated_reference)>0
            y=np.asarray(pair_squared_error)[valid]/np.asarray(pair_uncorrelated_reference)[valid]
            if len(y)>1 and np.ptp(y)>0 and np.ptp(x[valid])>0:
                ratio_rho=float(spearmanr(x[valid],y).statistic)
        result[name]=dict(spearman_with_squared_pair_error=rho,
                          spearman_with_pair_cancellation_ratio=ratio_rho,mean=float(np.mean(x)))
    return dict(proxies=result, arithmetic_seconds=elapsed,
                pair_input_logical_bytes=int(sum(x.nbytes for x in proxies.values())),
                summary_gather_logical_bytes=0 if sketch is None else int(2*len(i)*sketch.shape[1]*sketch.dtype.itemsize),
                excludes_full_index_mapping_and_serving=True, cheap_deployability_established=False)


def verify_inputs(c, source):
    source=Path(source).resolve()
    if sha(source/'archive_index.json')!=c['science_index_sha256']: raise ValueError('wrong completed index')
    entries=json.loads((source/'archive_index.json').read_text())['files']; index={e['path']:e for e in entries}
    required=['source_freeze.json','result.json','summaries/actual_code_book.npz']
    required += ['geometry/%s-%s.npz'%(p,q) for p in c['nprobe'] for q in c['query_indices']]
    required += ['summaries/%s.npz'%name for name in c['summary_names']]
    for name in required:
        e=index[name]; f=source/name
        if f.stat().st_size!=e['bytes'] or sha(f)!=e['sha256']: raise ValueError('input identity: '+name)
    freeze=json.loads((source/'source_freeze.json').read_text())
    result=json.loads((source/'result.json').read_text())
    if freeze['commit']!=c['scientific_source_commit'] or result['status']!='complete': raise ValueError('not completed scientific source')
    if freeze['config']['query_indices']!=c['query_indices']: raise ValueError('cohort changed')
    s0=ROOT/c['s0_directory']
    for e in c['original_products']:
        f=s0/e['path']
        if f.stat().st_size!=e['bytes'] or sha(f)!=e['sha256']: raise ValueError('original '+e['path'])
    return dict(source=str(source),verified=[index[n] for n in required],old_science_unchanged=True)


def check_resources(start, output, c):
    import resource,sys
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    size=sum(f.stat().st_size for f in output.rglob('*') if f.is_file())
    if time.monotonic()-start>c['limits']['seconds'] or rss>c['limits']['rss_bytes'] or size>c['limits']['output_bytes']:
        raise RuntimeError('attribution resource ceiling; preserve and stop')
    return dict(rss_highwater_bytes=rss,output_bytes=size)


def run(c, source, output):
    receipt=verify_inputs(c,source)  # readiness does not consume the one science run
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic(); sources={p:sha(ROOT/p) for p in PATHS}
    write(output/'source_freeze.json',dict(config=c,sources=sources,fingerprint=fingerprint(dict(config=c,sources=sources)),inputs=receipt))
    try:
        _run(c,source,output,start)
        check_resources(start,output,c)
        if sources!={p:sha(ROOT/p) for p in PATHS}: raise RuntimeError('executed source changed')
        write(output/'status.json',dict(status='complete',elapsed_seconds=time.monotonic()-start,formal_certification=False))
    except Exception as exc:
        write(output/'status.json',dict(status='failed',error=str(exc),elapsed_seconds=time.monotonic()-start,formal_certification=False))
        raise
    finally:
        files=[dict(path=str(p.relative_to(output)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(output.rglob('*')) if p.is_file()]
        write(output/'archive_index.json',dict(files=files))


def _run(c, source, output, start):
    s0=ROOT/c['s0_directory'];queries=np.load(s0/'queries.npy',allow_pickle=False)
    roles=json.loads((s0/'roles.json').read_text())['queries']
    truth=np.load(s0/'published_truth.npy',allow_pickle=False);cutoffs=np.load(s0/'published_cutoffs.npy',allow_pickle=False)
    with np.load(source/'summaries/actual_code_book.npz',allow_pickle=False) as z:rows=z['rows']
    summaries={}
    for name in c['summary_names']:
        with np.load(source/'summaries'/(name+'.npz'),allow_pickle=False) as z:summaries[name]={k:z[k] for k in z.files}
    ranking=[];relative=[];proxies=[]
    for probe in c['nprobe']:
        for qi in c['query_indices']:
            with np.load(source/'geometry'/('%s-%s.npz'%(probe,qi)),allow_pickle=False) as z:a={k:z[k] for k in z.files}
            ids=a['ids'];pos=np.searchsorted(rows,ids)
            if not np.array_equal(rows[pos],ids): raise ValueError('summary row mapping')
            q=queries[qi];base=a['original_reconstruction_dot'];adc=a['actual_ADC_reconstructed_dot']
            scores=dict(adc_order=adc,adc_rescore=adc,reconstruction=base)
            costs={}
            for name,summary in summaries.items():
                t=time.perf_counter();scores[name]=point_scores(q,base,summary,pos,'gaussian' if name.startswith('gaussian') else 'sidepq'); costs[name]=time.perf_counter()-t
            for name,estimate in scores.items():
                identity=dict(nprobe=probe,query_index=qi,role=roles[qi]['role'],method=name)
                slack = None
                if name in summaries:
                    point_distance = np.clip(1-estimate,0,2).astype(np.float32).astype(float)
                    slack = np.quantile(point_distance-a['lower_'+name],[0,.5,.9,1]).tolist()
                for row in rank_diagnostics(estimate,ids,a['native_distances'],a['native_topk'],c['budgets'],np.arange(len(ids)) if name=='adc_order' else None):
                    returned=np.asarray(row['returned']); d=np.asarray(row['distances'])
                    row.update(benchmark_recall=float(np.mean(d<=cutoffs[qi]+.001)),strict_id=len(set(returned)&set(truth[qi]))/100)
                    ranking.append(dict(**identity,**row,point_arithmetic_seconds=costs.get(name,0),prepaid_reconstruction=True,stored_native_lower_to_point_slack_quantiles=slack))
                for label,order in [('adc_boundary',np.arange(len(ids))),('true_boundary_oracle',np.lexsort((ids,a['native_distances'])))]:
                    pairs=boundary_pairs(order,c['boundary_ranks'])
                    rel=relative_error_diagnostics(estimate,a['real_raw_dot'],pairs)
                    relative.append(dict(**identity,boundary=label,**rel))
                    if name.startswith('gaussian'):
                        i,j=pairs;e=estimate-a['real_raw_dot'];summary=summaries[name]
                        proxy=proxy_cost_diagnostics(adc,a['residual_radius'],summary['sketch'][pos],pairs,(e[i]-e[j])**2,e[i]**2+e[j]**2)
                        proxies.append(dict(**identity,boundary=label,**proxy))
            check_resources(start,output,c)
    write(output/'ranking.json',ranking);write(output/'relative_errors.json',relative);write(output/'proxies.json',proxies)
    qualified=[]
    for M in [1600,3200]:
        rs=[r for r in ranking if r['nprobe']==64 and r['budget']==M]
        primary=[r for r in rs if r['method']==c['primary']]
        controls={n:[r for r in rs if r['method']==n] for n in ['adc_order','sidepq_128']}
        if all(r['preserves_top100'] for r in primary) and all(any(not r['preserves_top100'] for r in controls[n] if r['role']==role) for n in controls for role in ['geometry_discovery','geometry_check']): qualified.append(M)
    write(output/'decision.json',dict(point_headroom_necessary_screen_passed=bool(qualified),qualifying_budgets=qualified,
        actual_net_gain_measured=False,cheap_identification_established=False,formal_certification=False,
        disposition='end_current_interface_research' if not qualified else 'point_headroom_only_review_relative_and_proxy_evidence_no_automatic_experiment',
        limitation='Point estimates use prepaid real reconstruction; local arithmetic excludes serving, mapping and physical traffic. Four seeds are not covariance certification.'))
