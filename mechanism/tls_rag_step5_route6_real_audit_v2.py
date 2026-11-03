"""B0 v2 fixed-source/fixed-role audit with batched FP64 device search."""
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

from mechanism import tls_rag_step5_route6_real_geometry_v1 as geometry
from mechanism import tls_rag_step5_route6_real_audit_v1 as products
from mechanism import tls_rag_step5_route6_real_geometry_v2 as device


def load_cpu_reference(old_root, cfg, check=lambda: None):
    root = Path(old_root).resolve()
    index_path = root/'delivery_index.json'
    if products.sha(index_path, check) != cfg['old_index_sha256']:
        raise ValueError('frozen CPU index SHA mismatch')
    index = json.loads(index_path.read_text())
    seen = set()
    for row in index['files']:
        name = row['path']
        if Path(name).is_absolute() or '..' in Path(name).parts or name in seen:
            raise ValueError('unsafe/duplicate frozen CPU product')
        seen.add(name)
        path = (root/name).resolve()
        if root not in path.parents or path.stat().st_size != row['bytes'] or products.sha(path, check) != row['sha256']:
            raise ValueError('frozen CPU product identity: '+name)
    if not {'roles.json', 'source_freeze.json', 'execution.json', 'failure.json'}.issubset(seen):
        raise ValueError('missing CPU evidence')
    freeze = json.loads((root/'source_freeze.json').read_text())
    execution = json.loads((root/'execution.json').read_text())
    if freeze['commit'] != cfg['old_scientific_commit'] or execution['status'] != 'resource_stop':
        raise ValueError('CPU source/status mismatch')
    if products.sha(root/'roles.json', check) != cfg['roles_sha256']:
        raise ValueError('fixed roles SHA')
    roles = json.loads((root/'roles.json').read_text())
    entries = roles['queries']
    if (len(entries) != cfg['queries'] or len({row['id'] for row in entries}) != cfg['queries']
            or len({row['vector_sha256'] for row in entries}) != cfg['queries']):
        raise ValueError('fixed distinct development query identities')
    for i, row in enumerate(entries):
        if (row['id'] != products.stable_id(cfg['source']['sha256'], 'learn', row['row'])
                or row['role'] != ('geometry_discovery' if i < cfg['discovery_queries'] else 'geometry_check')):
            raise ValueError('query role or stable source ID mismatch')
    records = {}
    for i in range(cfg['old_completed_queries']):
        name = 'queries/%03d/record.json' % i
        matrix = 'queries/%03d/geometry.npz' % i
        if name not in seen or matrix not in seen:
            raise ValueError('missing frozen complete CPU query')
        row = json.loads((root/name).read_text())
        if row['query'] != entries[i]:
            raise ValueError('CPU/reference query identity mismatch')
        records[entries[i]['id']] = dict(record=row, geometry_path=str(root/matrix))
    return roles, records, dict(old_index_sha256=cfg['old_index_sha256'],
        roles_sha256=cfg['roles_sha256'], old_reference_count=len(records),
        old_status=execution['status'], old_scientific_commit=freeze['commit'])


def compare_cpu_reference(record, local, old):
    before = old['record']
    if record['corpus_id_map'] != before['corpus_id_map'] or record['strata'] != before['strata']:
        raise ArithmeticError('full-corpus CPU/GPU selected ID/rank/stratum mismatch')
    errors = {}
    with np.load(old['geometry_path'], allow_pickle=False) as saved:
        for name in ('displacements', 'gram', 'd2', 'loading', 'residual_gram'):
            error = float(np.max(np.abs(local[name]-saved[name])))
            if error > 2.**-40:
                raise ArithmeticError('CPU/GPU saved local geometry mismatch: '+name)
            errors[name] = error
    return dict(status='matched', exact_selected_ids_and_ranks=True,
                max_geometry_errors=errors, scope='released CPU development prefix only; no new CPU science samples')


def run_geometry_audit(source, old_root, root, cfg, check=lambda: None,
                       progress=lambda row: None, backend_factory=device.CudaCorpus,
                       runtime_identity=None):
    root = Path(root)
    started = time.monotonic()
    def bounded():
        check()
        if time.monotonic()-started >= cfg['limits']['compute_seconds']:
            raise TimeoutError('v2 geometry compute deadline')
    bounded()
    roles, cpu_reference, old_receipt = load_cpu_reference(old_root, cfg, bounded)
    source_receipt = products.validate_source(source, cfg, bounded)
    products.json_write(root/'input_receipt.json', dict(source=source_receipt, cpu_reference=old_receipt,
                        source_is_readonly_reuse=True, network_received_bytes=0))
    products.json_write(root/'roles.json', roles)
    fingerprint = hashlib.sha256(json.dumps(dict(config=cfg, roles_sha256=cfg['roles_sha256'],
        runtime_identity=runtime_identity), sort_keys=True).encode()).hexdigest()
    products.json_write(root/'id_map.json', dict(source_sha256=cfg['source']['sha256'],
        stable_id_format='vibe:sha256:<full-source-sha>:<dataset>:<row-zero-padded-8>',
        corpus_rows=cfg['source']['rows'], query_rows=roles['queries'], selected_corpus_maps='queries/*/record.json'))
    (root/'queries').mkdir()
    (root/'batches').mkdir()
    records, matched = [], 0
    with h5py.File(source, 'r') as file:
        backend = backend_factory(file['train'], cfg, bounded)
        products.json_write(root/'device_corpus.json', backend.identity)
        try:
            for start in range(0, len(roles['queries']), cfg['query_batch']):
                bounded()
                identities = roles['queries'][start:start+cfg['query_batch']]
                queries = np.array([file['learn'][row['row']] for row in identities], dtype=np.float32)
                for row, query in zip(identities, queries):
                    if products.canonical_vector_hash(query) != row['vector_sha256']:
                        raise ValueError('selected query vector SHA changed')
                batch_stamp = time.perf_counter()
                stream = iter(backend.batch(queries))
                for offset, identity in enumerate(identities):
                    search = next(stream)
                    stamp = time.perf_counter()
                    sample = geometry.sample_geometry(search['order'], identity['id'], cfg)
                    points = backend.points(sample['rows'])
                    actual_d2 = device.cpu_scores(points, queries[offset], 'l2')
                    actual_ip = device.cpu_scores(points, queries[offset], 'ip')
                    l2_error = float(np.max(np.abs(actual_d2-search['d2'][sample['rows']])))
                    ip_error = float(np.max(np.abs(actual_ip-search['native_ip'][sample['rows']])))
                    if max(l2_error, ip_error) > cfg['comparison_envelope']:
                        raise ArithmeticError('selected point FP64 envelope audit failed')
                    local = geometry.query_direction_decomposition(points, queries[offset])
                    description = geometry.describe_geometry(local, sample)
                    query_dir = root/'queries'/('%03d' % (start+offset))
                    query_dir.mkdir()
                    products.save_npz(query_dir/'geometry.npz', **{name:local[name] for name in
                        ('displacements', 'gram', 'd2', 'loading', 'residual_gram')}, pair_inclusion=sample['pair_inclusion'])
                    maps = [dict(row=int(row), id=products.stable_id(cfg['source']['sha256'], 'train', int(row)),
                                 rank=rank, group=group) for row, rank, group in
                            zip(sample['rows'], sample['ranks'], sample['groups'])]
                    k = cfg['k']
                    l2, ip = search['order'][:k], search['ip_order'][:k]
                    differences = sorted(set(map(int, l2)) ^ set(map(int, ip)))
                    postprocess = time.perf_counter()-stamp
                    record = dict(query=identity, fingerprint=fingerprint, points=len(points), corpus_id_map=maps,
                        strata=sample['strata'], geometry=description, batch_index=start//cfg['query_batch'],
                        order_audit=search['order_audit'], selected_envelope_audit=dict(l2=l2_error, ip=ip_error),
                        rank_diagnostics=dict(fp64_ip_not_official_fp32=True,
                            l2_native_ip_topk_overlap=len(set(l2)&set(ip)),
                            symmetric_difference_ids=[products.stable_id(cfg['source']['sha256'], 'train', row) for row in differences],
                            l2_cutoff=float(search['d2'][l2[-1]]), next_distance=float(search['d2'][search['order'][k]]),
                            ties_at_k=int(np.sum(search['d2']==search['d2'][l2[-1]])),
                            zero_distance_count=len(search['zero_rows']), max_norm_error=search['max_norm_error']),
                        costs=dict(search['costs'], postprocess_seconds=postprocess,
                            elapsed_seconds=search['costs']['query_order_seconds']+postprocess+
                                search['batch_costs']['batch_distance_seconds']/len(identities),
                            elapsed_is_batch_distance_amortized=True))
                    if identity['id'] in cpu_reference:
                        record['old_cpu_comparison'] = compare_cpu_reference(record, local, cpu_reference[identity['id']])
                        matched += 1
                    else:
                        record['old_cpu_comparison'] = dict(status='no_previous_CPU_record')
                    products.json_write(query_dir/'record.json', record)
                    records.append(record)
                    progress(dict(query_index=start+offset, id=identity['id'], status='complete',
                                  elapsed_seconds=time.monotonic()-started))
                    common = search['batch_costs']
                    del sample, points, local, search
                try:
                    next(stream)
                except StopIteration:
                    pass
                else:
                    raise ArithmeticError('unexpected extra batch query')
                products.json_write(root/('batches/%03d.json' % (start//cfg['query_batch'])),
                    dict(common, first_query=start, queries=len(identities),
                         batch_wall_seconds=time.perf_counter()-batch_stamp,
                         counted_once=True, query_record_distance_cost_is_amortized=True))
        finally:
            backend.close()
    bounded()
    if products.sha(source, bounded) != cfg['source']['sha256']:
        raise ValueError('source changed during v2 audit')
    products.render_diagnostics(root, records)
    bounded()
    if len(records) != cfg['queries'] or matched != cfg['old_completed_queries']:
        raise ArithmeticError('incomplete query or CPU-reference coverage')
    result = dict(namespace=cfg['namespace'], status='complete', queries_completed=len(records),
        expected_queries=cfg['queries'], cpu_reference_matches=matched,
        query_role_counts={role:sum(r['query']['role']==role for r in records)
            for role in ('geometry_discovery', 'geometry_check')},
        source_sha256=cfg['source']['sha256'], roles_sha256=cfg['roles_sha256'], fingerprint=fingerprint,
        ip_l2_differing_queries=sum(r['rank_diagnostics']['l2_native_ip_topk_overlap']!=cfg['k'] for r in records),
        elapsed_seconds=time.monotonic()-started, cuda_called=backend.identity['backend'].startswith('cuda'),
        new_mc_samples=0, network_received_bytes=0, semantic_independence_verified=False,
        roundoff_certified=False, scientific_claim='complete development geometry audit; no macro omission, RAG gain or certification')
    products.json_write(root/'result.json', result)
    return result
