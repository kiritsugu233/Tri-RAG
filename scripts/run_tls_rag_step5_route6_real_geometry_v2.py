"""One approved A800 B0 v2: fixed-source FP64 geometry, not Gaussian MC."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from mechanism import tls_rag_step5_route6_real_geometry_v2 as device
from mechanism import tls_rag_step5_route6_real_audit_v2 as audit
from mechanism import tls_rag_step5_route6_real_audit_v1 as products
from scripts import run_tls_rag_step5_route6_real_geometry_v1 as cpu_runner
from scripts import run_tls_rag_step5_route6_v1 as resource_helpers

CONFIG = ROOT/'configs/tls_rag_step5_route6_real_geometry_v2.json'
SOURCES = ['mechanism/tls_rag_step5_route6_real_geometry_v2.py',
    'mechanism/tls_rag_step5_route6_real_audit_v2.py',
    'configs/tls_rag_step5_route6_real_geometry_v2.json',
    'tests/test_tls_rag_step5_route6_real_geometry_v2.py',
    'scripts/run_tls_rag_step5_route6_real_geometry_v2.py',
    'docs/TLS_RAG_STEP5_ROUTE6_REAL_GEOMETRY_PROTOCOL_V2.md',
    'mechanism/tls_rag_step5_route6_real_geometry_v1.py',
    'mechanism/tls_rag_step5_route6_real_audit_v1.py',
    'scripts/run_tls_rag_step5_route6_real_geometry_v1.py',
    'mechanism/tls_rag_step5_route6_a800_v1.py',
    'mechanism/tls_rag_step5_route6_bridge_v1.py',
    'scripts/run_tls_rag_step5_route6_v1.py', 'src/tri_rag_harness/tri_law.py',
    'scripts/check_code_protection.py', 'scripts/run_tests.sh', 'docs/code_protection.json']


def config():
    cfg = json.loads(CONFIG.read_text())
    if (cfg['namespace'] != 'tls-rag-route6-real-geometry-v2' or cfg['stage'] != 'B0'
            or cfg['query_dataset'] != 'learn' or cfg['projection'] is not None
            or cfg['comparison_envelope'] != device.ENVELOPE or cfg['query_batch'] != 16
            or cfg['block'] != 8192 or not cfg['cuda_called'] or cfg['new_mc_samples'] != 0
            or cfg['limits']['network_bytes'] != 0):
        raise ValueError('frozen v2 scope')
    return cfg


def source_map():
    return {name:products.sha(ROOT/name) for name in SOURCES}


def gpu_lock(cfg, allowed_pids=()):
    from mechanism.tls_rag_step5_route6_a800_v1 import torch_backend
    torch, precision = torch_backend()
    versions = {name:importlib.metadata.version(name) for name in cfg['dependencies']}
    if versions != cfg['dependencies']:
        raise RuntimeError('fixed dependency versions')
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version',
                                      '--format=csv,noheader'], text=True).strip().splitlines()
    if len(devices) != 1 or torch.cuda.device_count() != 1 or 'A800' not in devices[0]:
        raise RuntimeError('single physical/visible A800 required')
    applications = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                            '--format=csv,noheader,nounits'], text=True)
    pids = {int(row.strip()) for row in applications.splitlines() if row.strip()}
    if pids-{os.getpid(), *allowed_pids}:
        raise RuntimeError('device has unrelated compute processes')
    free, total = torch.cuda.mem_get_info(0)
    if free < cfg['limits']['gpu_bytes']:
        raise RuntimeError('insufficient device allocation')
    return dict(versions=versions, python=sys.version, executable=sys.executable,
                torch_extension_sha256=products.sha(torch._C.__file__), cuda_runtime=torch.version.cuda,
                physical_identity=devices[0], precision=precision,
                threads={name:os.environ.get(name) for name in cpu_runner.THREADS},
                device_total_bytes=total)


def monitor_worker(command, root, limits, env):
    root = Path(root)
    cpu_runner.process_tree(os.getpid())
    stamp, peak_rss, peak_gpu = time.monotonic(), 0, 0
    reason, failure = None, None
    with (root/'worker.log').open('xb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        try:
            while process.poll() is None:
                tree = cpu_runner.process_tree(os.getpid())
                rss = sum(value for _, value in tree)
                used = resource_helpers.gpu_process_memory({pid for pid, _ in tree})
                peak_rss, peak_gpu = max(peak_rss, rss), max(peak_gpu, used)
                total, artifacts = cpu_runner.storage_sizes(root)
                values = dict(seconds=time.monotonic()-stamp, rss_bytes=rss, gpu_bytes=used,
                              disk_bytes=total, artifact_bytes=artifacts)
                reason = next((k for k,v in values.items() if v > limits[k]), None)
                if reason:
                    break
                time.sleep(.25)
        except BaseException as exc:
            failure = type(exc).__name__+': '+str(exc)
        finally:
            cpu_runner.terminate_owned(process)
            code = process.wait()
    total, artifacts = cpu_runner.storage_sizes(root)
    result = dict(status='monitor_failed' if failure else ('resource_stop' if reason or code == 2 else
                       ('complete' if code == 0 else 'worker_failed')),
                  limit=reason, monitor_error=failure, returncode=code,
                  elapsed_seconds=time.monotonic()-stamp, peak_rss_bytes=peak_rss,
                  peak_gpu_bytes=peak_gpu, gpu_memory_monitored=True,
                  disk_bytes=total, artifact_bytes=artifacts,
                  sampling_and_shutdown_overshoot_possible=True)
    products.json_write(root/'execution.json', result)
    cpu_runner.write_delivery_index(root)
    return result


def worker(root, cfg):
    root = Path(root)
    freeze = json.loads((root/'source_freeze.json').read_text())
    if (os.getppid() != freeze['parent_pid'] or source_map() != freeze['sources']
            or cfg != freeze['config'] or gpu_lock(cfg, [freeze['parent_pid']]) != freeze['environment']):
        raise RuntimeError('worker ownership/source/device mismatch')
    last = [0.]
    def check():
        if os.getppid() != freeze['parent_pid'] or time.time() >= freeze['deadline_unix']-5:
            raise TimeoutError('v2 owner/global deadline')
        if time.monotonic()-last[0] >= 5:
            if source_map() != freeze['sources']:
                raise RuntimeError('source changed during v2')
            last[0] = time.monotonic()
    try:
        with (root/'progress.jsonl').open('x') as progress:
            def update(row):
                progress.write(json.dumps(row, allow_nan=False)+'\n');progress.flush()
                print(json.dumps(row, allow_nan=False), flush=True)
            audit.run_geometry_audit(cfg['source_path'], cfg['old_run'], root, cfg, check, update,
                                     runtime_identity=freeze['environment'])
        check()
        if source_map() != freeze['sources']:
            raise RuntimeError('source changed at v2 completion')
        return 0
    except Exception as exc:
        products.json_write(root/'failure.json', dict(type=type(exc).__name__, detail=str(exc),
            traceback=traceback.format_exc(), result_is_incomplete=True,
            completed_record_count=len(list((root/'queries').glob('*/record.json')))))
        return 2 if isinstance(exc, TimeoutError) else 1


def validate_device_receipt(path, cfg):
    saved = json.loads(Path(path).read_text())
    if saved['sources'] != source_map() or saved['config'] != cfg or saved['result']['status'] != 'passed':
        raise RuntimeError('missing/mismatched successful CUDA tiny check')
    return saved


def supervise(root, check_receipt, cfg):
    if sys.platform != 'linux':
        raise RuntimeError('v2 scientific run requires existing Linux node')
    subprocess.run([sys.executable, str(ROOT/'scripts/check_code_protection.py')], cwd=ROOT, check=True)
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
        raise RuntimeError('tracked checkout dirty')
    if any(os.environ.get(name) != '1' for name in cpu_runner.THREADS):
        raise RuntimeError('set the five BLAS thread limits to1')
    saved = validate_device_receipt(check_receipt, cfg)
    environment = gpu_lock(cfg)
    if saved['environment'] != environment:
        raise RuntimeError('device check environment changed')
    if cpu_runner.allocated_resources()['memory_bytes'] < cfg['limits']['rss_bytes']:
        raise RuntimeError('host resource allocation')
    audit.load_cpu_reference(cfg['old_run'], cfg)
    root = Path(root).resolve()
    if shutil.disk_usage(root.parent).free < cfg['limits']['free_bytes']:
        raise RuntimeError('insufficient free disk')
    root.mkdir(exist_ok=False)
    products.json_write(root/'source_freeze.json', dict(config=cfg, sources=source_map(),
        environment=environment, device_check_sha256=products.sha(check_receipt),
        parent_pid=os.getpid(), deadline_unix=time.time()+cfg['limits']['seconds'],
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return monitor_worker([sys.executable, str(Path(__file__).resolve()), 'worker', '--run-dir', str(root)],
                          root, cfg['limits'], env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['device-check', 'run', 'worker'])
    parser.add_argument('--run-dir')
    parser.add_argument('--device-receipt')
    args = parser.parse_args()
    cfg = config()
    if args.action == 'device-check':
        if not args.device_receipt:
            parser.error('exclusive --device-receipt required')
        environment = gpu_lock(cfg)
        products.json_write(args.device_receipt, dict(environment=environment, config=cfg,
                            sources=source_map(), result=device.device_self_test(cfg)))
        return 0
    if not args.run_dir:
        parser.error('exclusive --run-dir required')
    if args.action == 'worker':
        return worker(args.run_dir, cfg)
    if not args.device_receipt:
        parser.error('successful --device-receipt required')
    result = supervise(args.run_dir, args.device_receipt, cfg)
    print(json.dumps(result))
    return 0 if result['status']=='complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
