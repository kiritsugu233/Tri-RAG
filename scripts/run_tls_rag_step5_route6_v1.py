#!/usr/bin/env python3
"""Scheme6 A: CPU self-test now; locked, supervised A800 execution later."""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))

from mechanism import tls_rag_step5_route6_bridge_v1 as bridge
from mechanism import tls_rag_step5_route6_a800_v1 as gpu
import numpy as np
import scipy

CONFIG_PATH = ROOT/'configs/tls_rag_step5_route6_v1.json'
SOURCES = ['mechanism/tls_rag_step5_route6_bridge_v1.py', 'mechanism/tls_rag_step5_route6_a800_v1.py',
           'configs/tls_rag_step5_route6_v1.json', 'scripts/run_tls_rag_step5_route6_v1.py',
           'docs/TLS_RAG_STEP5_ROUTE6_PROTOCOL_V1.md', 'src/tri_rag_harness/tri_law.py',
           'scripts/check_code_protection.py', 'docs/code_protection.json']


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def source_map():
    return {name: sha(ROOT/name) for name in SOURCES}


def config():
    value = json.loads(CONFIG_PATH.read_text())
    if value['stages_enabled'] != ['A']:
        raise ValueError('only stage A is implemented/authorized')
    return value


class ResourceStop(RuntimeError):
    pass


class ArtifactStore:
    """New run directory, exclusive files, bounded chunks and immutable receipts."""
    def __init__(self, root, limits):
        self.root, self.limits = Path(root), limits
        self.used = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        self.rows = []

    def write(self, relative, data):
        path = self.root/relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('unsafe artifact path')
        self.reserve(len(data))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f:
            self.append(f, data)
        self.rows.append({'path': relative, 'bytes': len(data), 'sha256': sha(path)})

    def reserve(self, size):
        if self.used+size > self.limits['artifact_bytes']:
            raise ResourceStop('artifact_bytes')
        if shutil.disk_usage(self.root).free-size < self.limits['reserve_bytes']:
            raise ResourceStop('reserve_bytes')

    def append(self, stream, data):
        self.reserve(len(data))
        for offset in range(0, len(data), self.limits['write_chunk_bytes']):
            chunk = data[offset:offset+self.limits['write_chunk_bytes']]
            stream.write(chunk)
            self.used += len(chunk)

    def json(self, relative, value):
        self.write(relative, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode())

    def array(self, relative, value):
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        self.write(relative, buffer.getvalue())


def cpu_dependencies(cfg):
    actual = {'numpy': np.__version__, 'scipy': scipy.__version__}
    if actual != cfg['cpu_dependencies']:
        raise RuntimeError('fixed CPU dependencies mismatch: '+str(actual))
    return actual


def memory_limit():
    for name in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        if Path(name).exists():
            text = Path(name).read_text().strip()
            if text.isdigit() and int(text) < 1 << 60:
                return int(text)
    if sys.platform == 'linux':
        raise gpu.GPUBlocked('finite cgroup memory allocation unavailable')
    raise gpu.GPUBlocked('A800 execution requires Linux')


def cpu_allocation():
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    path = Path('/sys/fs/cgroup/cpu.max')
    if path.exists():
        quota, period = path.read_text().split()
        if quota != 'max':
            return min(affinity, max(1, int(quota)//int(period)))
    quota_path = Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
    period_path = Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
    if quota_path.exists() and period_path.exists() and int(quota_path.read_text()) > 0:
        return min(affinity, max(1, int(quota_path.read_text())//int(period_path.read_text())))
    return affinity


def hardware_lock(cfg):
    cpu_dependencies(cfg)
    torch, precision = gpu.torch_backend()
    if sys.platform != 'linux':
        raise gpu.GPUBlocked('A800 scientific worker requires Linux')
    # Query a single visible device. Other GPUs are never used.
    identity = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'], text=True).strip().splitlines()
    properties = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    visible_uuid = str(getattr(properties, 'uuid', ''))
    if visible_uuid:
        records = [line for line in identity if visible_uuid.replace('GPU-', '') in line.replace('GPU-', '')]
        mapping = 'cuda_property_uuid'
    elif len(identity) == 1 and torch.cuda.device_count() == 1:
        # Older torch lacks the UUID property. A unique physical+visible GPU is unambiguous.
        normalize = lambda name: name.strip().removeprefix('NVIDIA ')
        records = identity if normalize(identity[0].split(',')[1]) == normalize(torch.cuda.get_device_name(0)) else []
        mapping = 'unique_physical_and_visible_gpu'
    else:
        records = []
        mapping = 'unresolved'
    if len(records) != 1:
        raise gpu.GPUBlocked('cannot match CUDA device UUID to nvidia-smi')
    environment = {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'torch_extension_sha256': sha(torch._C.__file__),
        'gpu_identity': records[0], 'gpu_mapping': mapping, 'precision': precision}
    limits = dict(cfg['limits'])
    limits['gpu_bytes'] = min(limits['gpu_bytes'], int(free*limits['gpu_free_fraction']))
    limits['rss_bytes'] = min(limits['rss_bytes'], int(memory_limit()*limits['ram_fraction']))
    limits['cpu_threads'] = min(limits['cpu_threads'], cpu_allocation())
    return {'schema': 'route6-A-environment-lock-v1', 'environment': environment,
            'sources': source_map(), 'config_sha256': bridge.fingerprint(cfg), 'limits': limits,
            'gpu_total_bytes': total, 'gpu_free_at_preflight_bytes': free}


def validate_lock(path, cfg):
    saved = json.loads(Path(path).read_text())
    actual = hardware_lock(cfg)
    for key in ('schema', 'environment', 'sources', 'config_sha256'):
        if saved.get(key) != actual[key]:
            raise gpu.GPUBlocked('environment/source lock mismatch: '+key)
    if any(saved['limits'][key] > actual['limits'][key] for key in ('gpu_bytes', 'rss_bytes', 'cpu_threads')):
        raise gpu.GPUBlocked('current allocation below preflight budget')
    if {k: v for k, v in saved['limits'].items() if k not in ('gpu_bytes', 'rss_bytes', 'cpu_threads')} != {k: v for k, v in cfg['limits'].items() if k not in ('gpu_bytes', 'rss_bytes', 'cpu_threads')}:
        raise gpu.GPUBlocked('resource lock changed nonhardware settings')
    if any(not 0 < saved['limits'][key] <= cfg['limits'][key] for key in ('gpu_bytes', 'rss_bytes', 'cpu_threads')):
        raise gpu.GPUBlocked('resource lock exceeds configured maxima')
    return saved


def immutable_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def tiny_check(cfg, device='cpu'):
    # Fixed tiny integration fixture, never a substitute for the 60-cell experiment.
    geometry = gpu.build_geometry(gpu.geometry_specs()[0], cfg)
    result = []
    for path in ('coordinate', 'gram'):
        matrices = gpu.sample_shared_projection(np.random.default_rng(100 if path == 'coordinate' else 101), geometry, 2, 8, path)
        squared, error, _ = gpu.evaluate_projection(matrices, geometry, path, device)
        ref, ref_error = gpu._evaluate_numpy(matrices, geometry, path)
        if np.any(np.abs(ref-squared) > error+ref_error):
            raise ArithmeticError('tiny device audit failed')
        events = gpu.unit_events(geometry, 2, path, cfg)
        lo, hi, _ = gpu.event_observations(squared, error, geometry, events)
        if np.any(lo & ~hi):
            raise ArithmeticError('event envelope inconsistent')
        result.append({'path': path, 'samples': 8, 'uncertain_bits': int((hi & ~lo).sum())})
    return {'status': 'passed', 'device': device, 'finite_integration_fixture_only': True, 'paths': result}


def record_dtype(geometry, events):
    k, n = geometry['spec']['k'], geometry['spec']['N']
    return np.dtype([('sample_ids', '<u8'), ('rank_lower', 'u1', (k,)), ('rank_upper', 'u1', (k,)),
        ('outside_definite', 'u1', (k, (n-k+7)//8)), ('outside_possible', 'u1', (k, (n-k+7)//8)),
        ('event_definite', 'u1', ((len(events)+7)//8,)), ('event_possible', 'u1', ((len(events)+7)//8,))])


def run_core(root, cfg, device, limits, check, store=None):
    root = Path(root)
    store = store or ArtifactStore(root, limits)
    t = time.perf_counter()
    geometries = [gpu.build_geometry(spec, cfg) for spec in gpu.geometry_specs()]
    catalogue = gpu.build_catalogue(geometries, cfg)
    # All events, including extra S2 joint inputs and both routes, exist before any Pi draw.
    store.json('catalogue.json', {'events': catalogue, 'family_size': len(catalogue), 'alpha': cfg['family_alpha'],
                                'config_sha256': bridge.fingerprint(cfg), 'source_map': source_map()})
    store.json('scalar_audit.json', gpu.scalar_grid_audit(cfg))
    # This eight-sample integration fixture is separate from scientific counts.
    store.json('device_self_test.json', tiny_check(cfg, device))
    for geometry in geometries:
        prefix = 'geometry/'+geometry['spec']['id']
        for name in ('query', 'corpus', 'gram', 'factor'):
            store.array(prefix+'/'+name+'.npy', geometry[name])
        store.json(prefix+'/ids.json', {'query_id': geometry['query_id'], 'row_to_id': geometry['ids'], 'fingerprint': geometry['fingerprint']})
    preparation = time.perf_counter()-t
    jobs = [(i, m) for i, g in enumerate(geometries) for m in (cfg['single_m'] if g['spec']['k'] == 1 else cfg['macro_m'])]
    order = np.random.default_rng(cfg['seeds']['order']).permutation(len(jobs)).tolist()
    store.json('job_order.json', {'jobs': [[geometries[jobs[i][0]]['spec']['id'], jobs[i][1]] for i in order], 'preparation_seconds': preparation})
    outcomes = []
    for job_index in order:
        check()
        gi, m = jobs[job_index]
        geometry = geometries[gi]
        key = geometry['spec']['id']+f'-m{m}'
        print(json.dumps({'stage': 'unit_started', 'unit': key}), flush=True)
        results = {}
        for path in ('coordinate', 'gram'):
            events = gpu.unit_events(geometry, m, path, cfg)
            dtype = record_dtype(geometry, events)
            relative = f'samples/{key}-{path}.bin'
            file = root/relative
            file.parent.mkdir(parents=True, exist_ok=True)
            store.reserve(dtype.itemsize*cfg['samples_per_path'])
            store.json(relative+'.layout.json', {'dtype': dtype.descr, 'declared_rows': cfg['samples_per_path'],
                'record_bytes': dtype.itemsize, 'partial_file_rows': 'file_bytes // record_bytes',
                'event_ids': [event['id'] for event in events], 'geometry_id_map': 'geometry/'+geometry['spec']['id']+'/ids.json'})
            with file.open('xb') as stream:
                def write_batch(offset, raw):
                    packed = np.empty(len(raw['sample_ids']), dtype=dtype)
                    for name in dtype.names:
                        packed[name] = raw[name]
                    store.append(stream, packed.tobytes())
                results[path] = gpu.sample_unit(geometry, m, path, cfg, len(catalogue), device, write_batch, check)
            store.rows.append({'path': relative, 'bytes': file.stat().st_size, 'sha256': sha(file)})
            store.json(f'units/{key}-{path}.json', results[path])
        analyzed = gpu.analyze_unit(geometry, m, cfg, results['coordinate']['events'])
        # The Gram stream is independent of coordinate-stream S2 input fitting.
        checks = []
        for event in results['gram']['events']:
            if event['type'] == 'omission':
                bounds = next(row for row in analyzed['bounds'] if row['M'] == event['M'] and row['ell'] == event['ell'])
                keys = ('cross_first_moment', 'target_markov_union', 'classical_marginal_union',
                        'conditional_marginal_union', 'boundary_s1', 'boundary_s2_coordinate_cp')
                checks.append({'event_id': event['id'], 'statistical_anomaly_candidates': [name for name in keys if bounds[name] < event['lower']],
                               'interval': [event['lower'], event['upper']], 'not_a_universal_proof': True})
        analyzed['independent_gram_checks'] = checks
        store.json('bounds/'+key+'.json', analyzed)
        outcomes.append({'unit': key, 'anomaly_candidates': sum(len(row['statistical_anomaly_candidates']) for row in checks)})
        print(json.dumps({'stage': 'unit_complete', 'unit': key}), flush=True)
    store.json('result.json', {'status': 'completed_finite_measurement', 'units': outcomes,
        'preparation_seconds': preparation, 'no_automatic_next_stage': True,
        'scientific_claim': 'finite numerical/statistical diagnostics; review anomalies and uncertainty before any application gate'})
    store.json('delivery_index.json', {'files': store.rows.copy(), 'all_declared_units_complete': True})


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


def process_tree(root_pid):
    output = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss='], text=True)
    rows = [tuple(map(int, line.split())) for line in output.splitlines() if line.strip()]
    children = {root_pid}
    while True:
        expanded = children | {pid for pid, parent, _ in rows if parent in children}
        if expanded == children:
            break
        children = expanded
    return [(pid, rss*1024) for pid, _, rss in rows if pid in children]


def gpu_process_memory(pids):
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
    memory = 0
    for line in output.splitlines():
        if not line.strip():
            continue
        pid, used = [part.strip() for part in line.split(',')]
        if int(pid) in pids:
            if not used.isdigit():
                raise RuntimeError('GPU process memory unavailable')
            memory += int(used)*(1 << 20)
    return memory


def sizes(root):
    root = Path(root)
    total = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    temporary = sum(p.stat().st_size for p in (root/'tmp').rglob('*') if p.is_file())
    return total, temporary


def monitor_worker(command, root, limits, environment, gpu_monitor=True):
    """Public small hook for real owned-child stop/failure tests, no scientific fallback."""
    root = Path(root)
    process_tree(os.getpid())  # fail before spawn if process monitoring is unavailable
    start = time.monotonic()
    reason, failure = None, None
    peaks = {'rss_bytes': 0, 'gpu_bytes': 0}
    with (root/'worker.log').open('xb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        try:
            while process.poll() is None:
                tree = process_tree(os.getpid())
                rss = sum(memory for _, memory in tree)
                gpu_bytes = gpu_process_memory({pid for pid, _ in tree}) if gpu_monitor else 0
                peaks['rss_bytes'], peaks['gpu_bytes'] = max(peaks['rss_bytes'], rss), max(peaks['gpu_bytes'], gpu_bytes)
                artifact, temporary = sizes(root)
                values = {'seconds': time.monotonic()-start, 'rss_bytes': rss, 'gpu_bytes': gpu_bytes,
                          'artifact_bytes': artifact, 'temporary_bytes': temporary}
                reason = next((key for key, value in values.items() if value > limits[key]), None)
                if not reason and shutil.disk_usage(root).free < limits['reserve_bytes']:
                    reason = 'reserve_bytes'
                if reason:
                    break
                time.sleep(limits['poll_seconds'])
        except BaseException as exc:
            failure = type(exc).__name__+': '+str(exc)
        finally:
            terminate_owned(process)
            code = process.wait()
    actual_size, temp = sizes(root)
    status = 'monitor_failed' if failure else ('resource_stop' if reason or code == 2 else ('complete' if code == 0 else 'worker_failed'))
    receipt = {'status': status, 'limit': reason, 'monitor_error': failure, 'returncode': code,
        'elapsed_seconds': time.monotonic()-start, 'observed_peaks': peaks, 'artifact_bytes': actual_size,
        'temporary_bytes': temp, 'limits': limits, 'sampling_and_shutdown_overshoot_possible': True,
        'artifact_bytes_excludes_final_execution_receipt': True}
    immutable_json(root/'execution.json', receipt)
    return receipt


def supervise(root, lockpath, cfg):
    subprocess.run([sys.executable, str(ROOT/'scripts/check_code_protection.py')], cwd=ROOT, check=True)
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
        raise RuntimeError('tracked checkout is dirty')
    lock = validate_lock(lockpath, cfg)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'tmp').mkdir()
    store = ArtifactStore(root, lock['limits'])
    store.json('source_freeze.json', {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'sources': source_map(), 'config': cfg, 'environment_lock': lock, 'parent_pid': os.getpid(),
        'deadline_unix': time.time()+lock['limits']['seconds']})
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', TMPDIR=str(root/'tmp'))
    command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--run-dir', str(root)]
    return monitor_worker(command, root, lock['limits'], env)


def worker(root, cfg):
    root = Path(root)
    freeze = json.loads((root/'source_freeze.json').read_text())
    if os.getppid() != freeze['parent_pid'] or freeze['sources'] != source_map() or freeze['config'] != cfg:
        raise RuntimeError('worker must belong to frozen supervisor')
    saved = freeze['environment_lock']
    actual = hardware_lock(cfg)
    if actual['environment'] != saved['environment']:
        raise gpu.GPUBlocked('worker environment mismatch')
    torch, _ = gpu.torch_backend()
    torch.set_num_threads(saved['limits']['cpu_threads'])
    torch.cuda.set_per_process_memory_fraction(saved['limits']['gpu_bytes']/saved['gpu_total_bytes'], 0)
    last_source_check = [0.0]
    def check():
        if time.monotonic()-last_source_check[0] >= 5:
            if source_map() != freeze['sources']:
                raise RuntimeError('scientific source changed during execution')
            last_source_check[0] = time.monotonic()
        if time.time() > freeze['deadline_unix']:
            raise ResourceStop('cumulative scientific deadline')
    try:
        run_core(root, cfg, 'cuda', saved['limits'], check)
        if source_map() != freeze['sources']:
            raise RuntimeError('scientific source changed before completion')
    except ResourceStop:
        traceback.print_exc()
        return 2
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('self-test', 'preflight', 'gpu-check', 'run', 'worker'))
    parser.add_argument('--output')
    parser.add_argument('--environment-lock')
    parser.add_argument('--run-dir')
    parser.add_argument('--stage', choices=('A',), default='A')
    args = parser.parse_args()
    cfg = config()
    cpu_dependencies(cfg)
    if args.action == 'self-test':
        value = tiny_check(cfg, 'cpu')
        print(json.dumps(value))
        if args.output:
            immutable_json(args.output, value)
        return 0
    if args.action == 'preflight':
        if not args.output:
            parser.error('preflight requires --output')
        value = hardware_lock(cfg)
        immutable_json(args.output, value)
        print(json.dumps({'status': 'locked', 'output': args.output}))
        return 0
    if args.action == 'gpu-check':
        if not args.environment_lock or not args.output:
            parser.error('gpu-check requires --environment-lock and --output')
        validate_lock(args.environment_lock, cfg)
        immutable_json(args.output, tiny_check(cfg, 'cuda'))
        return 0
    if not args.run_dir:
        parser.error('run/worker requires --run-dir')
    if args.action == 'worker':
        return worker(args.run_dir, cfg)
    if not args.environment_lock:
        parser.error('run requires --environment-lock')
    receipt = supervise(args.run_dir, args.environment_lock, cfg)
    print(json.dumps(receipt))
    return 0 if receipt['status'] == 'complete' else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except gpu.GPUBlocked as exc:
        print(json.dumps({'status': 'blocked', 'reason': str(exc), 'A800_test_passed': False}), file=sys.stderr)
        raise SystemExit(3)
