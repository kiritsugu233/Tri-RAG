"""Supervised CPU-only R6-M0 reference; existing environment, no new MC."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

from mechanism import tls_rag_step5_route6_macro_reference_v1 as reference
from mechanism import tls_rag_step5_route6_macro_audit_v1 as audit
from scripts import run_tls_rag_step5_route6_v1 as helpers

CONFIG = ROOT/'configs/tls_rag_step5_route6_macro_reference_v1.json'
THREAD_ENV = ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
SOURCES = ['mechanism/tls_rag_step5_route6_macro_reference_v1.py',
           'mechanism/tls_rag_step5_route6_macro_audit_v1.py',
           'configs/tls_rag_step5_route6_macro_reference_v1.json',
           'tests/test_tls_rag_step5_route6_macro_reference_v1.py',
           'scripts/run_tls_rag_step5_route6_macro_reference_v1.py',
           'docs/TLS_RAG_STEP5_ROUTE6_MACRO_REFERENCE_PROTOCOL_V1.md',
           'scripts/run_tls_rag_step5_route6_v1.py',
           'mechanism/tls_rag_step5_route6_bridge_v1.py',
           'mechanism/tls_rag_step5_route6_a800_v1.py',
           'src/tri_rag_harness/tri_law.py', 'scripts/check_code_protection.py',
           'scripts/run_tests.sh', 'docs/code_protection.json']


def config():
    cfg = json.loads(CONFIG.read_text())
    if cfg['stages_enabled'] != ['M0'] or cfg['namespace'] != 'tls-rag-route6-macro-reference-v1' or cfg['audit_tolerance'] != reference.AUDIT_TOL:
        raise ValueError('frozen M0 stage')
    return cfg


def source_map():
    return {p: helpers.sha(ROOT/p) for p in SOURCES}


def environment(cfg):
    versions = helpers.cpu_dependencies(cfg)
    return dict(versions, python=sys.version, executable=sys.executable,
                platform=platform.platform(), cpu_allocation=helpers.cpu_allocation(),
                thread_environment={name: os.environ.get(name) for name in THREAD_ENV},
                cuda_called=False, seeds={}, stochastic_components=[])


def tiny_check():
    result = reference.macro_reference_interval([1., 1.5, 2., 3.], 2, 2, 0., [(2, 2)], width=.01, max_nodes=512)
    lo, hi = result['events'][0]['discretization_bracket']
    if not lo <= 5/78 <= hi or result['events'][0]['status'] != 'numerically_resolved':
        raise RuntimeError('independent rational fixture failed')
    return dict(scope='CPU deterministic integration fixture, not the production 108 events', result=result,
                exact_reference='5/78', cuda_called=False)


def execute(root, cfg, freeze, store, check):
    models, receipt = audit.load_frozen_macro_inputs(freeze['measurement_dir'], freeze['index_path'], cfg)
    store.json('input_receipt.json', receipt)
    results = []
    store.write('progress.jsonl', b'')
    status = 'complete'
    failure = None
    with (root/'progress.jsonl').open('ab') as progress:
        try:
            for index, model in enumerate(models):
                check()
                print('model %d/12 %s m=%d' % (index+1, model['geometry'], model['m']), flush=True)
                def checkpoint(value):
                    payload = dict(model=model['geometry'], m=model['m'], reference=value)
                    store.append(progress, (json.dumps(payload, allow_nan=False)+'\n').encode())
                    progress.flush()
                result = reference.macro_reference_interval(model['d2'], model['k'], model['m'], model['c'],
                    model['events'], cfg['width_goal'], cfg['max_nodes_per_model'], check, checkpoint)
                record = dict(geometry=model['geometry'], m=model['m'], k=model['k'], c=model['c'],
                              geometry_diagnostic=model['geometry_diagnostic'], reference=result,
                              comparisons=audit.compare_reference_records(model, result))
                results.append(record)
                store.json('models/%s-m%d.json' % (model['geometry'], model['m']), record)
                print('nodes=%d reason=%s resolved=%d/9' % (result['nodes'], result['stop_reason'],
                      sum(row['status'] == 'numerically_resolved' for row in result['events'])), flush=True)
                if result['stop_reason'] and result['stop_reason'].startswith('numeric_failure'):
                    status = 'numeric_failure'
                    break
                if result['stop_reason'] == 'scientific_deadline':
                    status = 'resource_stop'
                    break
        except (reference.BudgetStop, helpers.ResourceStop) as exc:
            status, failure = 'resource_stop', str(exc)
        except Exception:
            status, failure = 'failed', traceback.format_exc()
    pending = [dict(geometry=model['geometry'], m=model['m'], M=M, ell=ell, status='not_started',
                    discretization_bracket=[0., 1.], roundoff_certified=False)
               for model in models[len(results):] for M, ell in model['events']]
    # Verify immutable inputs and source again; do not rewrite any old manifest.
    try:
        if source_map() != freeze['sources']:
            raise RuntimeError('source changed before completion')
        audit.load_frozen_macro_inputs(freeze['measurement_dir'], freeze['index_path'], cfg)
    except Exception:
        status, failure = 'failed', traceback.format_exc()
    rows = [row for group in results for row in group['comparisons']]
    result = dict(namespace=cfg['namespace'], status=status, failure=failure,
                  models_completed=len(results), events_computed=len(rows), pending=pending,
                  numerically_resolved=sum(row['status'] == 'numerically_resolved' for row in rows),
                  unresolved=sum(row['status'] == 'unresolved' for row in rows),
                  roundoff_certified=False, cuda_called=False, new_mc_samples=0,
                  scientific_claim='ideal structured-model reference, not RAG headroom or deployment certification')
    store.json('result.json', result)
    store.json('delivery_index.json', dict(files=store.rows.copy(), excludes=['delivery_index.json', 'worker.log', 'execution.json']))
    return 0 if status == 'complete' else (2 if status == 'resource_stop' else 1)


def worker(root, cfg):
    root = Path(root)
    freeze = json.loads((root/'source_freeze.json').read_text())
    if os.getppid() != freeze['parent_pid'] or freeze['sources'] != source_map() or freeze['config'] != cfg or freeze['environment'] != environment(cfg):
        raise RuntimeError('worker ownership/source/environment mismatch')
    checked = [0.]
    def check():
        if os.getppid() != freeze['parent_pid']:
            raise RuntimeError('supervisor exited')
        if time.time() >= freeze['deadline_unix']-5:
            raise reference.BudgetStop('scientific_deadline')
        if time.monotonic()-checked[0] >= 5:
            if source_map() != freeze['sources']:
                raise RuntimeError('source changed')
            checked[0] = time.monotonic()
    return execute(root, cfg, freeze, helpers.ArtifactStore(root, cfg['limits']), check)


def supervise(root, measurement_dir, index_path, cfg):
    subprocess.run([sys.executable, str(ROOT/'scripts/check_code_protection.py')], cwd=ROOT, check=True)
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
        raise RuntimeError('tracked checkout dirty')
    if sys.platform != 'linux':
        raise RuntimeError('M0 scientific run requires existing Linux node')
    if any(os.environ.get(name) != '1' for name in THREAD_ENV):
        raise RuntimeError('start CPU supervisor with all five BLAS/thread environment limits set to1')
    env_lock = environment(cfg)
    if helpers.memory_limit() < cfg['limits']['rss_bytes'] or env_lock['cpu_allocation'] < 1:
        raise RuntimeError('insufficient allocated resources')
    audit.load_frozen_macro_inputs(measurement_dir, index_path, cfg)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'tmp').mkdir()
    store = helpers.ArtifactStore(root, cfg['limits'])
    freeze = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  sources=source_map(), config=cfg, environment=env_lock, parent_pid=os.getpid(),
                  deadline_unix=time.time()+cfg['limits']['seconds'],
                  measurement_dir=str(Path(measurement_dir).resolve()), index_path=str(Path(index_path).resolve()))
    store.json('source_freeze.json', freeze)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES='', TMPDIR=str(root/'tmp'))
    return helpers.monitor_worker([sys.executable, str(Path(__file__).resolve()), 'worker', '--run-dir', str(root)],
                                  root, cfg['limits'], env, gpu_monitor=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['self-test', 'audit-inputs', 'run', 'worker'])
    parser.add_argument('--measurement-dir')
    parser.add_argument('--index')
    parser.add_argument('--run-dir')
    args = parser.parse_args()
    cfg = config()
    helpers.cpu_dependencies(cfg)
    if args.action == 'self-test':
        print(json.dumps(tiny_check(), allow_nan=False))
        return 0
    if args.action == 'worker':
        return worker(args.run_dir, cfg)
    if not args.measurement_dir or not args.index:
        parser.error('frozen measurement and index paths required')
    if args.action == 'audit-inputs':
        models, receipt = audit.load_frozen_macro_inputs(args.measurement_dir, args.index, cfg)
        print(json.dumps(dict(models=len(models), events=sum(len(m['events']) for m in models), receipt=receipt)))
        return 0
    if not args.run_dir:
        parser.error('new exclusive run directory required')
    receipt = supervise(args.run_dir, args.measurement_dir, args.index, cfg)
    print(json.dumps(receipt))
    return 0 if receipt['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
