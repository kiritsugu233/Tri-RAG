#!/usr/bin/env python3
"""Read closed attribution outputs and validate bookkeeping; no policy execution."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def mean(a):return statistics.fmean(a)


def audit(run, previous, output):
    run=Path(run);previous=Path(previous);output=Path(output)
    if output.exists():raise FileExistsError(output)
    idx=read(run/'archive_index.json')
    for f in idx['files']:
        p=run/f['path']
        if p.stat().st_size!=f['bytes'] or sha(p)!=f['sha256']:raise ValueError('closed file changed')
    status=read(run/'status.json')
    if status['status']!='complete':raise ValueError('only closed complete attribution may be analyzed')
    r=read(run/'ranking.json');rel=read(run/'relative_errors.json');proxies=read(run/'proxies.json')
    if (len(r),len(rel),len(proxies))!=(4608,2304,1536):raise ValueError('fixed cohort coverage')
    old={}
    with (previous/'calls.jsonl').open() as f:
        for line in f:
            v=json.loads(line)
            if v['repeat']==0 and (v['mode']=='baseline' or v['mode'].startswith('prefix_')):
                old[v['nprobe'],v['query_index'],v['mode']]=v
    fixed=0;group=collections.defaultdict(list)
    for v in r:
        if len(v['returned'])!=100 or len(set(v['returned']))!=100:raise ValueError('ranking schema')
        if v['method']=='adc_order':
            mode='baseline' if v['budget']==6400 else 'prefix_'+str(v['budget'])
            ref=old[v['nprobe'],v['query_index'],mode]
            if v['returned']!=ref['returned'] or v['distances']!=ref['distances']:raise ValueError('original fixed-prefix reference')
            fixed+=1
        group[v['nprobe'],v['role'],v['method'],v['budget']].append(v)
    summary=[]
    for key,values in sorted(group.items()):
        if len(values)!=16:raise ValueError('role size')
        summary.append(dict(nprobe=key[0],role=key[1],method=key[2],budget=key[3],queries=16,
            mean_candidate_retention=mean(x['candidate_retention'] for x in values),
            queries_losing_top100=sum(not x['preserves_top100'] for x in values),
            benchmark_recall=mean(x['benchmark_recall'] for x in values),strict_id=mean(x['strict_id'] for x in values),
            mean_point_arithmetic_ms=1000*mean(x['point_arithmetic_seconds'] for x in values),
            prepaid_reconstruction=True))
    relative_groups=collections.defaultdict(list)
    for v in rel:relative_groups[v['nprobe'],v['role'],v['method'],v['boundary']].append(v)
    relative_summary=[]
    for key,values in sorted(relative_groups.items()):
        if len(values)!=16 or any(v['pairs']!=820 for v in values):raise ValueError('pair window coverage')
        den=mean(x['uncorrelated_error_reference'] for x in values)
        relative_summary.append(dict(nprobe=key[0],role=key[1],method=key[2],boundary=key[3],
            point_mse=mean(x['point_mse'] for x in values),pair_mse=mean(x['pair_mse'] for x in values),
            pooled_cancellation_ratio=None if den==0 else mean(x['pair_mse'] for x in values)/den,
            mean_query_ratio=mean(x['cancellation_ratio'] for x in values if x['cancellation_ratio'] is not None) if any(x['cancellation_ratio'] is not None for x in values) else None,
            queries_ratio_below_one=sum(x['cancellation_ratio'] is not None and x['cancellation_ratio']<1 for x in values),
            strict_inversions=sum(x['strict_pair_inversions'] for x in values),pair_occurrences=sum(x['pairs'] for x in values)))
    pg=collections.defaultdict(list)
    for v in proxies:
        for name,x in v['proxies'].items():pg[v['nprobe'],v['role'],v['method'],v['boundary'],name].append((v,x))
    proxy_summary=[]
    for key,values in sorted(pg.items()):
        if len(values)!=16:raise ValueError('proxy cohort')
        d=dict(nprobe=key[0],role=key[1],method=key[2],boundary=key[3],proxy=key[4],
            mean_all_proxy_arithmetic_ms=1000*mean(v['arithmetic_seconds'] for v,x in values),
            mean_all_proxy_summary_gather_bytes=mean(v['summary_gather_logical_bytes'] for v,x in values),
            actual_serving_cost_measured=False)
        for field in ['spearman_with_squared_pair_error','spearman_with_pair_cancellation_ratio']:
            valid=[x[field] for v,x in values if x[field] is not None]
            d[field]=None if not valid else mean(valid);d[field+'_valid_queries']=len(valid)
        proxy_summary.append(d)
    decision=read(run/'decision.json')
    qualified=[]
    for M in [1600,3200]:
        ss=[x for x in summary if x['nprobe']==64 and x['budget']==M]
        primary=[x for x in ss if x['method']=='gaussian_2026110401_32']
        controls=[x for x in ss if x['method'] in ['adc_order','sidepq_128']]
        if all(x['queries_losing_top100']==0 for x in primary) and all(x['queries_losing_top100']>0 for x in controls):qualified.append(M)
    if qualified!=decision['qualifying_budgets']:raise ValueError('necessary screen mismatch')
    output.mkdir(parents=True)
    report=dict(status='passed',closed_index_sha256=sha(run/'archive_index.json'),elapsed_seconds=status['elapsed_seconds'],
        ranking_rows=len(r),relative_rows=len(rel),proxy_rows=len(proxies),original_prefix_output_groups_verified=fixed,
        summary=summary,relative_summary=relative_summary,proxy_summary=proxy_summary,decision=decision,
        no_new_scientific_run=True,formal_certification=False,reader_sha256=sha(__file__))
    write(output/'audit_and_tables.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['summary','relative_summary','proxy_summary']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--previous',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();audit(args.run,args.previous,args.output)
