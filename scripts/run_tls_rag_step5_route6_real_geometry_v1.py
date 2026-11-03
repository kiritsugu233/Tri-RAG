"""Exclusive, CPU-only supervised R6-B0 real-geometry run."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from mechanism import tls_rag_step5_route6_real_audit_v1 as audit

CONFIG = ROOT/'configs/tls_rag_step5_route6_real_geometry_v1.json'
SOURCES = ['mechanism/tls_rag_step5_route6_real_geometry_v1.py',
           'mechanism/tls_rag_step5_route6_real_audit_v1.py',
           'configs/tls_rag_step5_route6_real_geometry_v1.json',
           'tests/test_tls_rag_step5_route6_real_geometry_v1.py',
           'scripts/run_tls_rag_step5_route6_real_geometry_v1.py',
           'docs/TLS_RAG_STEP5_ROUTE6_REAL_GEOMETRY_PROTOCOL_V1.md',
           'scripts/run_tests.sh', 'scripts/check_code_protection.py', 'docs/code_protection.json']
THREADS = ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def config():
    cfg = json.loads(CONFIG.read_text())
    if (cfg['namespace'] != 'tls-rag-route6-real-geometry-v1' or cfg['stage'] != 'B0'
            or cfg['query_dataset'] != 'learn' or cfg['projection'] is not None
            or cfg['new_mc_samples'] != 0 or cfg['cuda_called']):
        raise ValueError('frozen B0 scope')
    return cfg


def source_map():
    return {name: audit.sha(ROOT/name) for name in SOURCES}


def environment(cfg):
    versions = {name: importlib.metadata.version(name) for name in cfg['dependencies']}
    if versions != cfg['dependencies']:
        raise ValueError('existing dependency versions differ')
    return dict(python=sys.version, executable=sys.executable, versions=versions,
                threads={name: os.environ.get(name) for name in THREADS}, cuda_called=False)


def allocated_resources():
    def read(paths):
        for path in paths:
            if Path(path).is_file():
                value = Path(path).read_text().strip()
                if value.isdigit() and int(value) > 0:
                    return int(value)
        raise RuntimeError('finite cgroup resource unavailable')
    memory = read(['/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'])
    quota = Path('/sys/fs/cgroup/cpu.max')
    if quota.exists():
        q, p = quota.read_text().split()
        cores = int(q)//int(p) if q != 'max' else len(os.sched_getaffinity(0))
    else:
        cores = read(['/sys/fs/cgroup/cpu/cpu.cfs_quota_us'])//read(['/sys/fs/cgroup/cpu/cpu.cfs_period_us'])
    return dict(memory_bytes=memory, cpu_threads=cores)


def process_tree(pid):
    raw = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss='], text=True)
    rows = [tuple(map(int, line.split())) for line in raw.splitlines() if line.strip()]
    pids = {pid}
    while True:
        expanded = pids | {p for p, parent, _ in rows if parent in pids}
        if expanded == pids:
            return [(p, rss*1024) for p, _, rss in rows if p in pids]
        pids = expanded


def terminate_owned(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def storage_sizes(root):
    total, artifacts = 0, 0
    for path in Path(root).rglob('*'):
        if path.is_file():
            size = path.stat().st_size
            total += size
            if path.relative_to(root).parts[0] != 'source':
                artifacts += size
    return total, artifacts


def write_delivery_index(root):
    root = Path(root)
    rows = [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=audit.sha(p))
            for p in sorted(root.rglob('*')) if p.is_file()
            and p.relative_to(root).parts[0] != 'source' and p.name != 'delivery_index.json']
    audit.json_write(root/'delivery_index.json', dict(files=rows,
                     exclusions=['delivery_index.json', 'source/ (input bound by source_receipt/input_receipt)'],
                     closed_products=True))


def monitor_worker(command, root, limits, env):
    root = Path(root)
    process_tree(os.getpid())
    stamp, peak, reason, failure = time.monotonic(), 0, None, None
    with (root/'worker.log').open('xb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            while process.poll() is None:
                rss = sum(value for _, value in process_tree(os.getpid()))
                peak = max(peak, rss)
                total, artifacts = storage_sizes(root)
                values = dict(seconds=time.monotonic()-stamp, rss_bytes=rss,
                              disk_bytes=total, artifact_bytes=artifacts)
                reason = next((key for key, value in values.items() if value > limits[key]), None)
                if reason:
                    break
                time.sleep(.25)
        except BaseException as exc:
            failure = type(exc).__name__+': '+str(exc)
        finally:
            terminate_owned(process)
            code = process.wait()
    total, artifacts = storage_sizes(root)
    status = 'monitor_failed' if failure else ('resource_stop' if reason or code == 2 else
                                              ('complete' if code == 0 else 'worker_failed'))
    result = dict(status=status, limit=reason, monitor_error=failure, returncode=code,
                  elapsed_seconds=time.monotonic()-stamp, peak_rss_bytes=peak,
                  disk_bytes=total, artifact_bytes=artifacts, cuda_called=False,
                  gpu_memory_monitored=False, sampling_and_shutdown_overshoot_possible=True)
    audit.json_write(root/'execution.json', result)
    write_delivery_index(root)
    return result


def worker(root, cfg):
    root = Path(root)
    freeze = json.loads((root/'source_freeze.json').read_text())
    if (os.getppid() != freeze['parent_pid'] or freeze['config'] != cfg
            or source_map() != freeze['sources'] or environment(cfg) != freeze['environment']):
        raise RuntimeError('worker ownership/source/environment mismatch')
    last = [0.]
    def check():
        if os.getppid() != freeze['parent_pid'] or time.time() >= freeze['deadline_unix']-5:
            raise TimeoutError('owner/global deadline')
        if time.monotonic()-last[0] >= 5:
            if source_map() != freeze['sources']:
                raise RuntimeError('implementation source changed')
            last[0] = time.monotonic()
    try:
        (root/'source').mkdir()
        prepared = audit.prepare_source(cfg['source']['partial_path'], root/'source/data.hdf5',
                                        cfg['source'], cfg['limits'], check)
        audit.json_write(root/'source_receipt.json', prepared)
        print(json.dumps(dict(phase='source_verified', receipt=prepared)), flush=True)
        with (root/'progress.jsonl').open('x') as progress:
            def update(row):
                progress.write(json.dumps(row, allow_nan=False)+'\n')
                progress.flush()
                print(json.dumps(row, allow_nan=False), flush=True)
            audit.run_geometry_audit(root/'source/data.hdf5', root, cfg, check, update)
        check()
        if source_map() != freeze['sources']:
            raise RuntimeError('source changed at completion')
        return 0
    except Exception as exc:
        audit.json_write(root/'failure.json', dict(type=type(exc).__name__, detail=str(exc),
                         traceback=traceback.format_exc(), result_is_incomplete=True,
                         completed_record_count=len(list((root/'queries').glob('*/record.json')))))
        return 2 if isinstance(exc, TimeoutError) else 1


def supervise(root, cfg):
    if sys.platform != 'linux':
        raise RuntimeError('scientific B0 uses the existing Linux node')
    subprocess.run([sys.executable, str(ROOT/'scripts/check_code_protection.py')], cwd=ROOT, check=True)
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
        raise RuntimeError('tracked checkout dirty')
    env_lock, resources = environment(cfg), allocated_resources()
    if resources['memory_bytes'] < cfg['limits']['rss_bytes'] or resources['cpu_threads'] < 1:
        raise RuntimeError('insufficient resources')
    if any(os.environ.get(name) != '1' for name in THREADS):
        raise RuntimeError('set all five BLAS/thread variables to1 before launch')
    root = Path(root).resolve()
    if shutil.disk_usage(root.parent).free < cfg['limits']['free_bytes']:
        raise RuntimeError('insufficient free disk')
    root.mkdir(exist_ok=False)
    audit.json_write(root/'source_freeze.json', dict(config=cfg, sources=source_map(), environment=env_lock,
        allocated_resources=resources, parent_pid=os.getpid(), deadline_unix=time.time()+cfg['limits']['seconds'],
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    return monitor_worker([sys.executable, str(Path(__file__).resolve()), 'worker', '--run-dir', str(root)],
                          root, cfg['limits'], env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'run', 'worker'])
    parser.add_argument('--run-dir')
    args = parser.parse_args()
    cfg = config()
    if args.action == 'preflight':
        print(json.dumps(dict(environment=environment(cfg), resources=allocated_resources(), sources=source_map())))
        return 0
    if not args.run_dir:
        parser.error('exclusive --run-dir required')
    if args.action == 'worker':
        return worker(args.run_dir, cfg)
    result = supervise(args.run_dir, cfg)
    print(json.dumps(result))
    return 0 if result['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
