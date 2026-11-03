"""Frozen-source acquisition, learn-only roles and real-geometry audit for B0."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import h5py
import numpy as np

from mechanism import tls_rag_step5_route6_real_geometry_v1 as geometry


def sha(path, check=lambda: None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            check()
            block = stream.read(8*1024*1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def json_write(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def prepare_source(partial, destination, spec, limits, check=lambda: None,
                   opener=urllib.request.urlopen):
    """Copy the old prefix; never append to or rename the historical partial."""
    partial, destination = Path(partial), Path(destination)
    before = partial.stat()
    if before.st_size != spec['partial_bytes'] or before.st_size >= spec['bytes']:
        raise ValueError('unexpected historical partial size')
    start = time.monotonic()
    def bounded():
        check()
        if time.monotonic()-start >= limits['prepare_seconds']:
            raise TimeoutError('source preparation deadline')
    prefix = hashlib.sha256()
    received, failures = 0, []
    with partial.open('rb') as src, destination.open('xb') as dst:
        while True:
            bounded()
            data = src.read(8*1024*1024)
            if not data:
                break
            prefix.update(data)
            dst.write(data)
        offset = dst.tell()
        if spec['bytes']-offset > limits['network_bytes']:
            raise ValueError('remaining source exceeds network cap')
        response = None
        for url in spec['urls']:
            bounded()
            try:
                candidate = opener(urllib.request.Request(url, headers={'Range': 'bytes=%d-' % offset}), timeout=15)
                if (candidate.status != 206 or candidate.headers.get('Content-Range') !=
                        'bytes %d-%d/%d' % (offset, spec['bytes']-1, spec['bytes'])):
                    candidate.close()
                    raise ValueError('range response identity mismatch')
                response = candidate
                break
            except Exception as exc:
                failures.append(dict(url=url, error=type(exc).__name__, detail=str(exc)))
        if response is None:
            raise RuntimeError('no valid fixed-source range response: '+json.dumps(failures))
        transport = url  # Do not retain ephemeral signed redirect credentials.
        with response:
            while received < spec['bytes']-offset:
                bounded()
                data = response.read(min(1024*1024, spec['bytes']-offset-received))
                if not data:
                    raise ValueError('truncated range body')
                received += len(data)
                if received > limits['network_bytes']:
                    raise ValueError('network cap')
                dst.write(data)
            if response.read(1):
                raise ValueError('oversized range body')
    bounded()
    after = partial.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('historical partial changed during acquisition')
    actual = sha(destination, bounded)
    if destination.stat().st_size != spec['bytes'] or actual != spec['sha256']:
        raise ValueError('complete source size/SHA mismatch')
    return dict(source=str(destination), bytes=spec['bytes'], sha256=actual,
                historical_partial=str(partial), historical_partial_sha256=prefix.hexdigest(),
                historical_partial_bytes=before.st_size, historical_partial_unchanged=True,
                network_received_bytes=received, transport_url=transport,
                transport_open_failures=failures, elapsed_seconds=time.monotonic()-start)


def validate_source(path, cfg, check=lambda: None):
    source = cfg['source']
    if Path(path).stat().st_size != source['bytes'] or sha(path, check) != source['sha256']:
        raise ValueError('full source identity')
    with h5py.File(path, 'r') as file:
        metric = file.attrs.get('distance')
        if isinstance(metric, bytes):
            metric = metric.decode()
        if metric != source['metric'] or int(file.attrs.get('dimension', -1)) != source['dimension']:
            raise ValueError('native source metric/dimension')
        if file['train'].shape != (source['rows'], source['dimension']) or file['train'].dtype != np.dtype('float32'):
            raise ValueError('complete corpus shape/dtype')
        if (cfg['query_dataset'] != 'learn' or 'learn' not in file or len(file['learn'].shape) != 2
                or file['learn'].shape[1] != source['dimension'] or file['learn'].dtype != np.dtype('float32')
                or len(file['learn']) < cfg['queries']):
            raise ValueError('learn-only query source')
        metadata = {name: dict(shape=list(file[name].shape), dtype=str(file[name].dtype)) for name in file.keys()}
    return dict(sha256=source['sha256'], datasets=metadata, metric=metric,
                payload_allowlist=['train', 'learn'], test_payload_read=False,
                original_text_mapping_available=False)


def canonical_vector_hash(vector):
    value = np.asarray(vector, dtype='<f4').copy()
    value[value == 0] = 0.
    return hashlib.sha256(value.tobytes()).hexdigest()


def stable_id(source_sha, dataset, row):
    if dataset not in ('train', 'learn') or row < 0:
        raise ValueError('ID domain')
    return 'vibe:sha256:%s:%s:%08d' % (source_sha, dataset, row)


def freeze_roles(learn, cfg, check=lambda: None):
    groups = {}
    for start in range(0, len(learn), cfg['block']):
        check()
        chunk = np.asarray(learn[start:start+cfg['block']], dtype=np.float32)
        if (not np.all(np.isfinite(chunk)) or
                np.max(np.abs(np.linalg.norm(chunk.astype(float), axis=1)-1)) > cfg['norm_tolerance']):
            raise ValueError('learn finite/norm gate')
        for i, vector in enumerate(chunk):
            groups.setdefault(canonical_vector_hash(vector), start+i)
    if len(groups) < cfg['queries']:
        raise ValueError('insufficient distinct learn groups')
    selected = sorted(groups, key=lambda key: hashlib.sha256((str(cfg['role_seed'])+'|'+key).encode()).digest())[:cfg['queries']]
    rows = [dict(id=stable_id(cfg['source']['sha256'], 'learn', groups[key]), row=groups[key],
                 vector_sha256=key, role='geometry_discovery' if i < cfg['discovery_queries'] else 'geometry_check')
            for i, key in enumerate(selected)]
    return dict(queries=rows, inspected_learn_rows=len(learn), distinct_vector_groups=len(groups),
                whole_inspected_learn_pool_is_development=True, semantic_independence_verified=False,
                test_payload_read=False, certification=False)


def save_npz(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def render_heatmaps(path, records):
    """Standalone SVG, no plotting package or data downloads on the node."""
    import html
    keys = ('G-G', 'G-B', 'G-F', 'B-B', 'B-F', 'F-F')
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="760" viewBox="0 0 1080 760">',
           '<rect width="1080" height="760" fill="white"/>',
           '<g font-family="Arial,sans-serif" fill="#222">',
           '<text x="30" y="30" font-size="20">B0: signed displacement correlation vs distance margin</text>',
           '<text x="30" y="53" font-size="12">Equal query weights; design weights within each block. Color: log relative mass. No omission probability.</text>']
    for panel, key in enumerate(keys):
        histograms = [np.array(row['geometry']['blocks'][key]['rho_margin_histogram']) for row in records]
        normalized = [v/np.sum(v) for v in histograms if np.sum(v) > 0]
        hist = np.mean(normalized, axis=0) if normalized else np.zeros((40, 40))
        x, y = 60+(panel % 3)*350, 100+(panel//3)*325
        svg.append('<text x="%d" y="%d" font-size="16">%s</text>' % (x, y-15, html.escape(key)))
        maximum = float(np.max(hist))
        for i in range(40):
            for j in range(40):
                strength = np.log1p(1000*hist[i, j])/np.log1p(1000*maximum) if maximum else 0.
                shade = int(255-220*strength)
                svg.append('<rect x="%.1f" y="%.1f" width="6.3" height="6.3" fill="rgb(%d,%d,255)"/>' %
                           (x+j*6.2, y+(39-i)*6.2, shade, shade))
        svg += ['<path d="M%d %dv248h248" fill="none" stroke="#555"/>' % (x, y),
                '<text x="%d" y="%d" font-size="12">margin: 0 to 1</text>' % (x+65, y+270),
                '<text x="%d" y="%d" font-size="11">rho +1</text>' % (x-48, y+10),
                '<text x="%d" y="%d" font-size="11">-1</text>' % (x-25, y+245)]
    svg.append('</g></svg>')
    Path(path).write_text('\n'.join(svg))


def render_lines(path, title, series, ymin, ymax, footer):
    import html
    colors = ['#2459a6', '#c84438', '#368956', '#8c55a2']
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="460" viewBox="0 0 1000 460">',
           '<rect width="1000" height="460" fill="white"/>', '<g font-family="Arial,sans-serif" fill="#222">',
           '<text x="55" y="30" font-size="20">%s</text>' % html.escape(title),
           '<path d="M60 70v300h880" stroke="#555" fill="none"/>']
    for tick in range(6):
        value = ymin+(ymax-ymin)*tick/5
        y = 370-300*tick/5
        svg.append('<text x="5" y="%.1f" font-size="11">%.3g</text>' % (y+3, value))
    for number, (label, values) in enumerate(series):
        color = colors[number % len(colors)]
        coordinates = ['%.2f,%.2f' % (60+880*i/max(1, len(values)-1), 370-300*(v-ymin)/(ymax-ymin))
                       for i, v in enumerate(values) if v is not None and np.isfinite(v)]
        svg.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="1.5"/>' % (' '.join(coordinates), color))
        svg.append('<text x="%d" y="405" fill="%s" font-size="12">%s</text>' %
                   (60+number*220, color, html.escape(label)))
    svg += ['<text x="60" y="440" font-size="12">%s</text>' % html.escape(footer), '</g></svg>']
    Path(path).write_text('\n'.join(svg))


def render_diagnostics(root, records):
    import csv
    root = Path(root)
    render_heatmaps(root/'rho_margin.svg', records)
    render_lines(root/'query_residual.svg', 'Query-direction component: signed correlation before / after removal',
        [(block+' '+name, [r['geometry']['blocks'][block][name]['weighted_mean'] for r in records])
         for block in ('G-B', 'G-F') for name in ('rho', 'residual_rho')], -1., 1.,
        'x: frozen query order; discovery first, check second. Residual dependence remains general.')
    ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    curves = []
    for role in ('geometry_discovery', 'geometry_check'):
        spectra = [np.array(r['geometry']['local_correlation_eigenvalues']) for r in records if r['query']['role'] == role]
        values = [float(np.median([np.sum(e[-min(rank, len(e)):])/np.sum(e) for e in spectra if len(e)]))
                  if any(len(e) for e in spectra) else None for rank in ranks]
        curves.append((role, values))
    render_lines(root/'local_spectrum.svg', 'Boundary-enriched local correlation: median cumulative spectrum', curves, 0., 1.,
                 'x ranks: 1,2,4,8,16,32,64,128,256. This is not the full-corpus spectrum or a tail-risk test.')
    gaps = [(r['rank_diagnostics']['next_distance']-r['rank_diagnostics']['l2_cutoff']) /
            (r['rank_diagnostics']['next_distance']+r['rank_diagnostics']['l2_cutoff'])
            if r['rank_diagnostics']['next_distance']+r['rank_diagnostics']['l2_cutoff'] > 0 else 0. for r in records]
    render_lines(root/'original_boundary_gaps.svg', 'Original top-k boundary: normalized next / cutoff distance gap',
                 [('(D[k+1]-D[k])/(D[k+1]+D[k])', gaps)], 0., max(max(gaps), 1e-16),
                 'x: frozen query order. Exact ties have zero gap; no projected omission was measured.')
    if records:
        data = np.load(root/'queries/000/geometry.npz', allow_pickle=False)
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="550" viewBox="0 0 1000 550">',
               '<rect width="1000" height="550" fill="white"/>', '<g font-family="Arial,sans-serif" fill="#222">',
               '<text x="30" y="28" font-size="19">First frozen query: signed correlation and query-orthogonal residual</text>',
               '<text x="30" y="50" font-size="12">8x8 tile means; full signed matrices are saved. Blue negative, white zero, red positive.</text>']
        for panel, key in enumerate(('gram', 'residual_gram')):
            c = geometry.correlation(data[key])
            x0, y0, step = 40+panel*480, 90, 8
            count = (len(c)+step-1)//step
            cell = 390/max(1, count)
            for i in range(count):
                for j in range(count):
                    tile = c[i*step:(i+1)*step, j*step:(j+1)*step]
                    value = float(np.mean(tile[np.isfinite(tile)])) if np.any(np.isfinite(tile)) else 0.
                    fade = int(255*(1-min(1., abs(value))))
                    color = 'rgb(255,%d,%d)' % (fade, fade) if value >= 0 else 'rgb(%d,%d,255)' % (fade, fade)
                    svg.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s"/>' %
                               (x0+j*cell, y0+i*cell, cell+.1, cell+.1, color))
            svg.append('<text x="%d" y="510" font-size="14">%s; samples in original-rank order</text>' % (x0, key))
        svg.append('</g></svg>')
        (root/'signed_gram_example.svg').write_text('\n'.join(svg))
    with (root/'query_summary.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['query_id', 'role', 'points', 'boundary_gap', 'G_B_rho', 'G_B_residual_rho',
                         'G_F_rho', 'G_F_residual_rho', 'IP_L2_overlap', 'zero_distances', 'query_seconds'])
        for record, gap in zip(records, gaps):
            writer.writerow([record['query']['id'], record['query']['role'], record['points'], gap]+
                [record['geometry']['blocks'][block][key]['weighted_mean'] for block in ('G-B', 'G-F')
                 for key in ('rho', 'residual_rho')]+[record['rank_diagnostics']['l2_native_ip_topk_overlap'],
                    record['rank_diagnostics']['zero_distance_count'], record['costs']['elapsed_seconds']])


def run_geometry_audit(source, root, cfg, check=lambda: None, progress=lambda row: None):
    root = Path(root)
    started = time.monotonic()
    def bounded():
        check()
        if time.monotonic()-started >= cfg['limits']['compute_seconds']:
            raise TimeoutError('geometry compute deadline')
    receipt = validate_source(source, cfg, bounded)
    json_write(root/'input_receipt.json', receipt)
    records = []
    with h5py.File(source, 'r') as file:
        roles = freeze_roles(file['learn'], cfg, bounded)
        json_write(root/'roles.json', roles)
        fingerprint = hashlib.sha256(json.dumps(dict(config=cfg, roles=roles), sort_keys=True).encode()).hexdigest()
        json_write(root/'id_map.json', dict(source_sha256=cfg['source']['sha256'],
                   stable_id_format='vibe:sha256:<full-file-sha>:<dataset>:<row-zero-padded-8>',
                   corpus_rows=cfg['source']['rows'], internal_row_domain='train',
                   query_rows=roles['queries'], selected_corpus_maps='queries/*/record.json'))
        queries_dir = root/'queries'
        queries_dir.mkdir()
        for index, identity in enumerate(roles['queries']):
            bounded()
            stamp = time.perf_counter()
            q = file['learn'][identity['row']]
            search = geometry.exact_geometry_search(file['train'], q, cfg['block'], cfg['norm_tolerance'], bounded)
            sample_stamp = time.perf_counter()
            sample = geometry.sample_geometry(search['order'], identity['id'], cfg)
            sampling_seconds = time.perf_counter()-sample_stamp
            # HDF5 fancy indexing requires ascending rows; restore original rank order.
            selected_read_stamp = time.perf_counter()
            perm = np.argsort(sample['rows'])
            points = np.asarray(file['train'][sample['rows'][perm]])[np.argsort(perm)]
            selected_read_seconds = time.perf_counter()-selected_read_stamp
            gram_stamp = time.perf_counter()
            local = geometry.query_direction_decomposition(points, q)
            description = geometry.describe_geometry(local, sample)
            gram_seconds = time.perf_counter()-gram_stamp
            query_dir = queries_dir/('%03d' % index)
            query_dir.mkdir()
            storage_stamp = time.perf_counter()
            save_npz(query_dir/'geometry.npz', **{name: local[name] for name in
                     ('displacements', 'gram', 'd2', 'loading', 'residual_gram')},
                     pair_inclusion=sample['pair_inclusion'])
            storage_seconds = time.perf_counter()-storage_stamp
            k = cfg['k']
            l2, ip = search['order'][:k], search['ip_order'][:k]
            maps = [dict(row=int(row), id=stable_id(cfg['source']['sha256'], 'train', int(row)),
                         rank=rank, group=group) for row, rank, group in
                    zip(sample['rows'], sample['ranks'], sample['groups'])]
            differences = sorted(set(map(int, l2)) ^ set(map(int, ip)))
            record = dict(query=identity, fingerprint=fingerprint, points=len(points), corpus_id_map=maps,
                          strata=sample['strata'], geometry=description,
                          rank_diagnostics=dict(fp64_ip_not_official_fp32=True,
                            l2_native_ip_topk_overlap=len(set(l2)&set(ip)),
                            symmetric_difference_ids=[stable_id(cfg['source']['sha256'], 'train', row) for row in differences],
                            l2_cutoff=float(search['d2'][l2[-1]]),
                            next_distance=float(search['d2'][search['order'][k]]),
                            ties_at_k=int(np.sum(search['d2'] == search['d2'][l2[-1]])),
                            zero_distance_count=len(search['zero_rows']), max_norm_error=search['max_norm_error']),
                          costs=dict(search['costs'], sampling_seconds=sampling_seconds,
                                     selected_read_seconds=selected_read_seconds, storage_seconds=storage_seconds,
                                     geometry_seconds=gram_seconds, elapsed_seconds=time.perf_counter()-stamp))
            json_write(query_dir/'record.json', record)
            records.append(record)
            progress(dict(query_index=index, id=identity['id'], status='complete', elapsed_seconds=time.monotonic()-started))
    bounded()
    # Recheck the complete source, not just a cached initial hash.
    if sha(source, bounded) != cfg['source']['sha256']:
        raise ValueError('source changed during computation')
    summaries = {}
    for key in ('G-G', 'G-B', 'G-F', 'B-B', 'B-F', 'F-F'):
        summaries[key] = {name: geometry.distribution([r['geometry']['blocks'][key][name]['weighted_mean']
                              for r in records if r['geometry']['blocks'][key][name]['weighted_mean'] is not None])
                          for name in ('rho', 'error_correlation', 'residual_rho', 'margin')}
    render_diagnostics(root, records)
    bounded()
    result = dict(namespace=cfg['namespace'], status='complete', queries_completed=len(records),
                  expected_queries=cfg['queries'], query_role_counts={role:sum(r['query']['role'] == role for r in records)
                      for role in ('geometry_discovery', 'geometry_check')},
                  query_level_summaries=summaries, source_sha256=cfg['source']['sha256'], fingerprint=fingerprint,
                  ip_l2_differing_queries=sum(r['rank_diagnostics']['l2_native_ip_topk_overlap'] != cfg['k'] for r in records),
                  zero_distance_queries=sum(r['rank_diagnostics']['zero_distance_count'] > 0 for r in records),
                  elapsed_seconds=time.monotonic()-started, cuda_called=False, new_mc_samples=0,
                  semantic_independence_verified=False, roundoff_certified=False,
                  scientific_claim='real local geometry only; no macro omission, RAG gain or deployment certification')
    json_write(root/'result.json', result)
    return result
