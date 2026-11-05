"""CPU-only C1 conformance, accounting, isolation and real-worker tests."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

from mechanism import tls_rag_step5_pq_refinement_cost_v1 as cost

spec=importlib.util.spec_from_file_location('c1_runner',cost.ROOT/'scripts/run_tls_rag_step5_pq_refinement_cost_v1.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class Reads:
    def __init__(self, values):self.values=values;self.shape=values.shape;self.calls=[]
    def __len__(self):return len(self.values)
    def __getitem__(self, key):self.calls.append(np.asarray(key).copy());return self.values[key]


class CostReplayTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(8041)
        self.raw=rng.normal(size=(30,16)).astype(np.float32)
        self.raw/=np.linalg.norm(self.raw,axis=1)[:,None]
        self.q=rng.normal(size=16).astype(np.float32);self.q/=np.linalg.norm(self.q)
        self.ids=np.array([19,2,13,27,0,5,9,11,24,8,21,6],np.int32)

    def test_observed_matches_original_for_noncontiguous_rows(self):
        for raw in (self.raw,self.raw[:,::-1]):
            a,b=cost.native.rerank(raw,self.q,self.ids,4)
            c,d,parts=cost.recorded_rerank(raw,self.q,self.ids,4)
            np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
            self.assertEqual(d.dtype,np.float32);self.assertEqual(parts['gather_rows'],16)

    def test_original_and_observed_keep_two_gather_batches(self):
        trace=Reads(self.raw)
        rr,dd,parts=cost.recorded_rerank(trace,self.q,self.ids,4)
        self.assertEqual(len(trace.calls),2)
        np.testing.assert_array_equal(trace.calls[0],self.ids)
        np.testing.assert_array_equal(trace.calls[1],rr)
        self.assertEqual(parts['logical_gather_payload_bytes'],16*16*4)

    def test_ties_and_clip_boundaries_preserve_row_order(self):
        raw=np.zeros((9,4),np.float32);raw[:,0]=np.array([1,2,1,0,-2,-1,1,0,2],np.float32)
        q=np.array([1,0,0,0],np.float32);ids=np.array([8,6,4,2,0,1,3,5,7])
        expected=cost.native.rerank(raw,q,ids,5)
        actual=cost.recorded_rerank(raw,q,ids,5)
        np.testing.assert_array_equal(actual[0],np.array([0,1,2,6,8]))
        for a,b in zip(expected,actual[:2]):np.testing.assert_array_equal(a,b)
        self.assertTrue(np.all((actual[1]>=0)&(actual[1]<=2)))

    def test_zero_query_is_native_tie_case(self):
        rr,dd,_=cost.recorded_rerank(self.raw,np.zeros(16,np.float32),self.ids,4)
        np.testing.assert_array_equal(rr,np.sort(self.ids)[:4]);np.testing.assert_array_equal(dd,np.ones(4,np.float32))

    def test_validation_rejects_duplicates_bounds_and_short_sets(self):
        for ids in ([1,1,2,3],[-1,1,2,3],[1,2,3,30],[1,2],np.array([[1,2],[3,4]])):
            with self.subTest(ids=str(ids)):
                with self.assertRaises(ValueError):cost.recorded_rerank(self.raw,self.q,ids,4)

    def test_nonfinite_scores_rejected(self):
        q=self.q.copy();q[0]=np.nan
        with self.assertRaises(ValueError):cost.recorded_rerank(self.raw,q,self.ids,4)

    def test_disjoint_clock_partition(self):
        ticks=iter(range(7))
        _,_,parts=cost.recorded_rerank(self.raw,self.q,self.ids,4,lambda:float(next(ticks)))
        phases=['id_seconds','gather_copy_seconds','distance_clip_seconds','selection_seconds','output_gather_seconds','output_score_seconds']
        self.assertEqual([parts[p] for p in phases],[1.]*6)
        self.assertEqual(sum(parts[p] for p in phases),parts['observed_inner_seconds'])

    def test_subset_patterns_are_label_free_ordered_and_exact_size(self):
        for pattern in ['prefix','spread','hash']:
            p=cost.subset_positions(self.ids,7,pattern,77)
            self.assertEqual(len(p),7);self.assertTrue(np.all(np.diff(p)>0))
            np.testing.assert_array_equal(cost.survivor_replay(self.ids,p),self.ids[p])
            np.testing.assert_array_equal(p,cost.subset_positions(self.ids,7,pattern,77))
        np.testing.assert_array_equal(cost.subset_positions(self.ids,len(self.ids),'spread',77),np.arange(len(self.ids)))

    def test_invalid_subset_requests_rejected(self):
        for n in [0,13]:
            with self.assertRaises(ValueError):cost.subset_positions(self.ids,n,'prefix',1)
        for p in [[-1],[99],[1,1],[]]:
            with self.assertRaises(ValueError):cost.survivor_replay(self.ids,p)
        with self.assertRaises(ValueError):cost.subset_positions(self.ids,4,'truth',1)

    def test_prepaid_oracles_preserve_native_output(self):
        c=cost.prepare_prepaid(self.raw,self.q,self.ids,4)
        expected=cost.native.rerank(self.raw,self.q,self.ids,4)
        for mode in ['prepaid_input','prepaid_scores']:
            actual=cost.prepaid_rerank(self.raw,self.q,self.ids,4,c,mode)
            for a,b in zip(expected,actual):np.testing.assert_array_equal(a,b)
        self.assertGreater(c['array_bytes'],0);self.assertGreaterEqual(c['paid_prepare_seconds'],0)

    def test_prepaid_query_candidate_and_source_binding(self):
        c=cost.prepare_prepaid(self.raw,self.q,self.ids,4)
        for raw,q,ids in [(self.raw,self.q,self.ids[::-1]),(self.raw,-self.q,self.ids),(self.raw.copy(),self.q,self.ids)]:
            with self.assertRaises(ValueError):cost.prepaid_rerank(raw,q,ids,4,c,'prepaid_input')

    def test_whole_callback_pays_search_for_every_mode(self):
        calls=[]
        def search(q):calls.append(1);return self.ids.copy()
        cache=cost.prepare_prepaid(self.raw,self.q,self.ids,4)
        for mode in ['baseline','observed','prepaid_input','prepaid_scores','subset_prefix_7']:
            r=cost.full_callback(self.raw,self.q,search,4,mode,np.arange(7),cache)
            self.assertGreaterEqual(r['seconds'],r['search_seconds']+r['rerank_seconds']-1e-12)
        self.assertEqual(len(calls),5)

    def test_real_faiss_cpu_callback_equivalence(self):
        import faiss
        faiss.omp_set_num_threads(1)
        index=faiss.IndexFlatIP(16);index.add(self.raw)
        model=dict(spec=dict(kind='raw'),metric='ip')
        search=lambda q:cost.stored.search_backend(model,q,12,index=index)
        a=cost.full_callback(self.raw,self.q,search,4)
        b=cost.full_callback(self.raw,self.q,search,4,'observed')
        for field in ['candidates','returned','distances']:np.testing.assert_array_equal(a[field],b[field])

    def test_sealed_reference_mismatch_is_not_tolerated(self):
        r=cost.full_callback(self.raw,self.q,lambda q:self.ids,4)
        expected=dict(returned=r['returned'].tolist(),distances=r['distances'].tolist())
        cost.compare_exact(r,self.ids,expected)
        wrong=copy.deepcopy(expected);wrong['distances'][0]+=1e-9
        with self.assertRaises(RuntimeError):cost.compare_exact(r,self.ids,wrong)
        with self.assertRaises(RuntimeError):cost.compare_exact(r,self.ids[::-1],expected)

    def test_virtual_page_coverage_is_not_physical_traffic(self):
        cfg=dict(source=dict(dimension=640,layout=dict(offset=6144)))
        r=cost.address_counts(np.array([0,1,3]),np.array([0]),cfg)
        self.assertEqual(r['logical_payload_bytes'],4*2560);self.assertEqual(r['gather_batches'],2)
        self.assertIsNone(r['actual_dram_bytes']);self.assertIsNone(r['physical_requests'])
        pages=set()
        for row in [0,1,3]:pages.update(range((6144+row*2560)//4096,(6144+row*2560+2559)//4096+1))
        self.assertEqual(r['unique_virtual_pages_4096'],len(pages))

    def test_config_role_and_old_source_lock(self):
        c=cost.config();self.assertEqual(len(c['query_indices']),32);self.assertFalse(c['tri_enabled'])
        bad=copy.deepcopy(c);p=next(iter(bad['old_source_bindings']));bad['old_source_bindings'][p]='0'*64
        with self.assertRaises(ValueError):cost.source_map(bad)

    def test_product_inventory_rejects_changed_bytes_and_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';p.write_bytes(b'original');e=dict(path='a',bytes=8,sha256=cost.sha(p))
            self.assertEqual(cost.verify_product(d,e),p)
            p.write_bytes(b'mutated!')
            with self.assertRaises(ValueError):cost.verify_product(d,e)
            with self.assertRaises(ValueError):cost.verify_product(d,dict(e,path='../a'))

    def test_paired_statistics_use_query_groups(self):
        cfg=dict(nprobe=[64],timing=dict(repeats=3),modes=['baseline','observed'],
            bootstrap_seed=3,bootstrap_replicates=1000,gammas=[.05,.1,.2])
        rows=[]
        for role in ['geometry_discovery','geometry_check']:
            for qi in range(16):
                for mode,value in [('baseline',.020),('observed',.010)]:
                    for repeat in range(3):rows.append(dict(nprobe=64,role=role,mode=mode,query_index=qi,
                        query_id=str(qi),repeat=repeat,seconds=value,rerank_seconds=value/2,search_seconds=value/2,
                        parts=dict(id_seconds=.001)))
        r=cost.summarize_paired(rows,cfg)
        observed=[g for g in r['groups'] if g['mode']=='observed']
        self.assertEqual(len(r['query_rows']),64)
        for g in observed:
            self.assertEqual(g['queries'],16);self.assertAlmostEqual(g['saving_fraction'],.5)
            np.testing.assert_allclose(g['saving_fraction_ci95'],[.5,.5],rtol=0,atol=1e-14)
        with self.assertRaises(ValueError):cost.summarize_paired(rows[:-1],cfg)


class WorkerTests(unittest.TestCase):
    def test_real_worker_success_closes_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run';r=runner.supervise([sys.executable,'-c','print(7)'],root,os.environ.copy(),.03,.01)
            self.assertEqual(r['status'],'complete')
            archive=json.loads((root/'archive_index.json').read_text())
            for e in archive['files']:self.assertEqual(runner.digest(root/e['path']),e['sha256'])
            self.assertTrue(json.loads((root/'archive_verified.json').read_text())['verified'])
            with self.assertRaises(ValueError):runner.request_stop(root)

    def test_real_worker_failure_retained(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run';r=runner.supervise([sys.executable,'-c','import sys;print("failure");sys.exit(7)'],root,os.environ.copy(),.03,.01)
            self.assertEqual(r['status'],'compute_failed');self.assertEqual(r['returncode'],7)
            self.assertIn('failure',(root/'worker.log').read_text())

    def test_real_owned_manual_stop(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run'
            def stop():
                for _ in range(500):
                    if (root/'ownership.json').exists():runner.request_stop(root);return
                    time.sleep(.005)
            thread=threading.Thread(target=stop);thread.start()
            r=runner.supervise([sys.executable,'-c','import time;time.sleep(30)'],root,os.environ.copy(),.03,.01)
            thread.join();self.assertEqual(r['status'],'user_stopped');self.assertTrue(r['worker_exited'])

    def test_bad_stop_token_does_not_abort_compute(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run'
            def bad():
                for _ in range(500):
                    if (root/'ownership.json').exists():
                        runner.write(root/'stop_requested.json',dict(action='stop',owner_token='wrong'));return
                    time.sleep(.005)
            thread=threading.Thread(target=bad);thread.start()
            r=runner.supervise([sys.executable,'-c','import time;time.sleep(.2)'],root,os.environ.copy(),.03,.01)
            thread.join();self.assertEqual(r['status'],'complete')
            self.assertTrue((root/'stop_request_error.json').exists())

    def test_observation_failure_is_not_a_resource_abort(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run';original=runner.observation
            try:
                runner.observation=lambda *a:(_ for _ in ()).throw(OSError('observation unavailable'))
                r=runner.supervise([sys.executable,'-c','import time;time.sleep(.1)'],root,os.environ.copy(),.03,.01)
            finally:runner.observation=original
            self.assertEqual(r['status'],'complete');self.assertFalse(r['automatic_budget_abort'])

    def test_refuse_wrong_child_identity(self):
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'])
        try:
            with self.assertRaises(ValueError):runner.terminate_owned(child,
                dict(child_pid=child.pid,parent_pid=os.getpid(),uid=os.getuid(),owner_token='wrong'), 'token')
            self.assertIsNone(child.poll())
        finally:child.terminate();child.wait()

    def test_refuse_existing_run_and_symlink_archive(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):runner.supervise([sys.executable,'-c','pass'],d,os.environ.copy())
            root=Path(d)/'run';root.mkdir();(root/'link').symlink_to(Path(d)/'target')
            with self.assertRaises(ValueError):runner.close_evidence(root)


if __name__=='__main__':unittest.main()
