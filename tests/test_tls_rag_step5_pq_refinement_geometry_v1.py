"""Independent CPU fixtures for approved gate2 geometry, control and ownership."""
import copy
import importlib.util
import inspect
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
from scipy.stats import norm

from mechanism import tls_rag_step5_pq_refinement_geometry_v1 as g

spec=importlib.util.spec_from_file_location('geometry_runner',g.ROOT/g.PATHS[3])
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class Reads:
    def __init__(self,x):self.x=x;self.calls=[]
    def __len__(self):return len(self.x)
    def __getitem__(self,ids):self.calls.append(np.asarray(ids).copy());return self.x[ids]


class GeometryTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(391);self.rng=rng;self.c=g.config()
        self.c.update(k=5,scan_block=7,build_block=16,dimensions=[2,4],sketch_seeds=[17],
            side_pq_bytes=[],side_pq_seeds=[])
        n,d,M=48,8,2
        self.book=dict(rows=np.arange(n),codes=rng.integers(0,256,(n,M),dtype=np.uint8),
            labels=rng.integers(0,3,n,dtype=np.int32),offsets=np.arange(n),
            centers=rng.normal(0,.08,(M,256,d//M)).astype(np.float32),
            coarse=rng.normal(0,.1,(3,d)).astype(np.float32))
        self.A=np.diag(np.linspace(.95,1.05,d)).astype(np.float32)
        self.ctx=g.reconstruction_context(self.book,self.A,np.zeros(d),self.c)
        self.pos=np.arange(n);self.ids=self.pos.copy()
        xhat,_=g.decode_actual_ivf_original(self.book,self.pos,self.ctx['inverse'],self.ctx['mean'])
        self.raw=(xhat+rng.normal(0,.01,xhat.shape)).astype(np.float32)
        self.q=rng.normal(size=d).astype(np.float32);self.q/=np.linalg.norm(self.q)
        self.summaries,self.errors,_=g.build_residual_summaries(self.raw,self.book,self.ctx,[],self.c)

    def test_nonorthogonal_reconstruction_and_residual_identity(self):
        xhat,y=g.decode_actual_ivf_original(self.book,self.pos,self.ctx['inverse'],self.ctx['mean'])
        np.testing.assert_allclose(xhat@self.A.astype(float).T,y,rtol=0,atol=1e-15)
        np.testing.assert_allclose(xhat+self.errors,self.raw,rtol=0,atol=1e-16)
        wrong=y.astype(float)@self.A.astype(float)
        self.assertGreater(np.max(abs(wrong-xhat)),.001)

    def test_LUT_encloses_native_coarse_addition_and_dual_transform(self):
        xhat,_=g.decode_actual_ivf_original(self.book,self.pos,self.ctx['inverse'],self.ctx['mean'])
        scores,error,Q=g.base_scores(self.q,self.book,self.pos,self.ctx)
        self.assertTrue(np.all(abs(scores-xhat@self.q.astype(float))<=error))
        self.assertGreater(Q,0)

    def test_norm_radii_are_stored_FP32_and_outward(self):
        r=self.summaries['norm']['radii']
        self.assertEqual(r.dtype,np.float32)
        self.assertTrue(np.all(r.astype(float)>=np.linalg.norm(self.errors,axis=1)))
        x=np.array([[0.,0.],[1e-200,0.],[1.,1.]],float)
        self.assertTrue(np.all(g.outward_norm(x)>=np.linalg.norm(x,axis=1)))

    def test_invalid_or_overflow_norms_rejected(self):
        for x in [np.array([[np.nan,0]]),np.array([[np.inf,0]]),np.array([1,2]),np.array([[1e100,0]])]:
            with self.assertRaises(ValueError):g.outward_norm(x)
        with self.assertRaises(ValueError):g.outward_norm(np.zeros((1,2)),-1)

    def test_ill_conditioned_or_nonfinite_matrix_rejected(self):
        for A in [np.diag([1e-10]+[1.]*7),np.full((8,8),np.nan),np.zeros((8,7))]:
            with self.assertRaises((ValueError,np.linalg.LinAlgError)):g.reconstruction_context(self.book,A,np.zeros(8),self.c)

    def test_norm_bound_covers_actual_native_distances(self):
        lower,info=g.score_bounds(self.q,self.book,self.pos,self.ctx,self.summaries['norm'],self.c)
        actual=g.cost.native.original_distances(self.raw,self.q)
        self.assertTrue(np.all(lower<=actual));self.assertIsNone(info['fallback'])
        self.assertFalse(info['formal_certification'])

    def test_sidePQ_bound_covers_known_exact_residual_reconstruction(self):
        side=dict(kind='sidepq',codes=np.zeros((len(self.raw),2),np.uint8),
            centers=np.zeros((2,256,4),np.float32),radii=self.summaries['norm']['radii'],payload=6)
        a,_=g.score_bounds(self.q,self.book,self.pos,self.ctx,side,self.c)
        b,_=g.score_bounds(self.q,self.book,self.pos,self.ctx,self.summaries['norm'],self.c)
        np.testing.assert_array_equal(a,b)

    def test_Gaussian_dimensions_scaled_from_shared_standard_prefix(self):
        a=self.summaries['gaussian_17_2'];b=self.summaries['gaussian_17_4']
        standard=np.random.Generator(np.random.PCG64(17)).standard_normal((4,8))
        np.testing.assert_array_equal(a['R'],standard[:2]/np.sqrt(2))
        np.testing.assert_array_equal(b['R'],standard[:4]/np.sqrt(4))
        self.assertFalse(np.array_equal(a['R'],b['R'][:2]))
        for s in [a,b]:
            np.testing.assert_allclose(s['sketch'],self.errors@s['R'].T,rtol=1e-6,atol=1e-9)
            self.assertEqual(s['payload'],4+4*len(s['R']))

    def test_exact_conditional_noise_matches_independent_Gaussian_reference(self):
        # Independent projection draws; neither observed retention nor the
        # implementation's bound is used to manufacture the reference law.
        R=np.random.default_rng(714).normal(size=(30000,8,2))/np.sqrt(8)
        a,b=.3,.7;h=R[:,:,0];T=np.sum(h*h,axis=1)
        projected=a*h+b*R[:,:,1]
        ahat=np.sum(h*projected,axis=1)/T
        Z=(ahat-a)*np.sqrt(8*T)/b
        self.assertLess(abs(float(Z.mean())),.025)
        self.assertLess(abs(float(Z.var())-1),.035)
        self.assertLess(abs(float(np.mean(Z < norm.ppf(.025)))-.025),.004)

    def test_zero_query_and_small_T_fall_back_to_full_read(self):
        s=copy.deepcopy(self.summaries['gaussian_17_2']);s['R'][:]=0
        for q,summary,reason in [(np.zeros(8),self.summaries['norm'],'zero_query'),(self.q,s,'small_or_invalid_T')]:
            lower,info=g.score_bounds(q,self.book,self.pos,self.ctx,summary,self.c)
            self.assertTrue(np.isneginf(lower).all());self.assertEqual(info['fallback'],reason)
            r=g.scan_and_refine(self.raw,q,self.ids,self.book,self.pos,self.ctx,summary,self.c)
            self.assertEqual(len(r['used']),len(self.ids))

    def test_invalid_radius_fallback_nonfinite_query_rejection(self):
        s=copy.deepcopy(self.summaries['norm']);s['radii'][0]=-1
        lower,info=g.score_bounds(self.q,self.book,self.pos,self.ctx,s,self.c)
        self.assertEqual(info['fallback'],'invalid_radius');self.assertTrue(np.isneginf(lower).all())
        q=self.q.copy();q[0]=np.nan
        with self.assertRaises(ValueError):g.score_bounds(q,self.book,self.pos,self.ctx,self.summaries['norm'],self.c)

    def test_forced_all_reads_match_native_and_are_not_repeated(self):
        raw=Reads(self.raw);r=g.scan_with_bounds(raw,self.q,self.ids,np.full(len(self.ids),-np.inf),5,7)
        rr,dd=g.cost.native.rerank(self.raw,self.q,self.ids,5)
        np.testing.assert_array_equal(r['returned'],rr);np.testing.assert_array_equal(r['distances'],dd)
        first=np.concatenate(raw.calls[:-1]);self.assertEqual(len(first),len(np.unique(first)))
        self.assertEqual(set(first),set(self.ids));np.testing.assert_array_equal(raw.calls[-1],rr)
        self.assertEqual(r['gather_batches'],len(raw.calls));self.assertEqual(r['logical_vectors'],len(self.ids)+5)

    def test_norm_scan_preserves_topk_and_trace_addresses(self):
        raw=Reads(self.raw);r=g.scan_and_refine(raw,self.q,self.ids,self.book,self.pos,self.ctx,self.summaries['norm'],self.c)
        rr,dd=g.cost.native.rerank(self.raw,self.q,self.ids,5)
        np.testing.assert_array_equal(r['returned'],rr);np.testing.assert_array_equal(r['distances'],dd)
        emitted=[self.ids[t['positions']] for t in r['trace'] if t['positions']]
        np.testing.assert_array_equal(np.concatenate(emitted),np.concatenate(raw.calls[:-1]))
        self.assertGreaterEqual(r['aux_seconds'],0);self.assertGreaterEqual(r['scan_seconds'],0)

    def test_ties_clip_and_equal_bound_must_be_read(self):
        raw=np.array([[2.,0.],[1.,0.],[1.,0.],[-2.,0.],[0.,0.]],np.float32)
        ids=np.array([2,1,0,4,3]);q=np.array([1.,0.],np.float32)
        r=g.scan_with_bounds(raw,q,ids,np.zeros(5),2,1)
        rr,dd=g.cost.native.rerank(raw,q,ids,2)
        np.testing.assert_array_equal(r['returned'],rr);np.testing.assert_array_equal(r['distances'],dd)
        self.assertEqual(len(r['used']),5)

    def test_scan_rejects_duplicates_bad_bounds_and_bad_pilot(self):
        for ids,low in [(np.r_[self.ids[:-1],self.ids[0]],np.zeros(48)),(self.ids,np.zeros(47)),(self.ids,np.full(48,np.nan))]:
            with self.assertRaises(ValueError):g.scan_with_bounds(self.raw,self.q,ids,low,5,7)
        with self.assertRaises(ValueError):g.scan_with_bounds(self.raw,self.q,self.ids,np.zeros(48),5,7,[0]*5)

    def test_online_interface_excludes_unread_scores_and_true_topk(self):
        self.assertEqual(list(inspect.signature(g.scan_and_refine).parameters),
            ['raw','query','ids','book','pos','ctx','summary','c'])
        rr,_=g.cost.native.rerank(self.raw,self.q,self.ids,5)
        r=g.ideal_threshold_diagnostic(self.raw,self.q,self.ids,self.book,self.pos,self.ctx,self.summaries['norm'],self.c,rr)
        self.assertTrue(r['oracle_only']);self.assertEqual(set(self.ids[r['trace'][0]['positions']]),set(rr))

    def test_every_complete_callback_repeats_search_and_does_not_accept_changed_candidates(self):
        calls=[]
        def search(q):calls.append(1);return self.ids.copy()
        rr,_=g.cost.native.rerank(self.raw,self.q,self.ids,5)
        for mode in ['baseline','fullread','prefix_12','norm','oracle:norm']:
            r=g.paired_geometry_replay(self.raw,self.q,search,self.ids,self.book,self.pos,self.ctx,self.summaries,self.c,mode,rr)
            self.assertGreaterEqual(r['seconds'],r['rerank_seconds']);self.assertGreaterEqual(r['aux_seconds'],0)
        self.assertEqual(len(calls),5)
        with self.assertRaises(RuntimeError):g.paired_geometry_replay(self.raw,self.q,lambda q:self.ids[::-1],self.ids,self.book,self.pos,self.ctx,self.summaries,self.c,'baseline')

    def test_config_source_and_role_identities_are_frozen(self):
        c=g.config();self.assertEqual(len(c['query_indices']),32);self.assertEqual(len(c['modes']),37)
        self.assertEqual(c['authorization']['actual_user_consent'],'批准第14节八路径，按顺序准备并执行')
        bad=copy.deepcopy(c);bad['old_source_bindings'][next(iter(bad['old_source_bindings']))]='0'*64
        with self.assertRaises(ValueError):g.source_map(bad)

    def test_callback_pays_live_summary_ID_mapping(self):
        a=g.paired_geometry_replay(self.raw,self.q,lambda q:self.ids,self.ids,self.book,self.pos,self.ctx,self.summaries,self.c,'norm')
        b=g.paired_geometry_replay(self.raw,self.q,lambda q:self.ids,self.ids,self.book,self.pos[::-1],self.ctx,self.summaries,self.c,'norm')
        np.testing.assert_array_equal(a['returned'],b['returned'])

    def test_quality_is_separate_from_the_controller(self):
        rr,dd=g.cost.native.rerank(self.raw,self.q,self.ids,5);r=dict(returned=rr,distances=dd)
        a=g.quality_record(r,rr,rr,float(dd.max()),rr,5)
        self.assertTrue(a['preserves_candidate_topk']);self.assertEqual(a['candidate_topk_retention'],1)

    def test_geometry_audit_saves_each_candidate_and_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            data=dict(book=self.book,ctx=self.ctx,model=dict(spec=dict(kind='opq'),matrix=self.A,mean=np.zeros(8)))
            rr,summary=g.geometry_audit(self.raw,self.q,self.ids,self.pos,data,self.errors,self.summaries,self.c,Path(d)/'audit.npz')
            self.assertEqual(summary['rows'],48);self.assertEqual(len(rr),5)
            with np.load(Path(d)/'audit.npz') as arrays:self.assertEqual(len(arrays['ids']),48);self.assertIn('lower_norm',arrays)

    def test_summary_aggregates_queries_not_repeated_calls(self):
        c=dict(nprobe=[64],timing=dict(repeats=2),modes=['baseline','norm'],bootstrap_seed=4,bootstrap_replicates=100,gammas=[.05,.1,.2])
        records=[]
        for role in ['geometry_discovery','geometry_check']:
            for qi in range(16):
                for mode,seconds in [('baseline',.02),('norm',.015)]:
                    for repeat in range(2):records.append(dict(nprobe=64,role=role,mode=mode,query_index=qi,query_id=str(qi),repeat=repeat,
                        seconds=seconds,rerank_seconds=seconds-.01,search_seconds=.01,unique_read_rows=100,
                        quality=dict(recall=1.,strict_id=1.,candidate_topk_retention=1.,global_native_retention=1.,preserves_candidate_topk=True)))
        s=g.summarize_geometry(records,c);self.assertTrue(s['gate2_started']);self.assertFalse(s['formal_certification'])
        self.assertEqual(len(s['query_rows']),64)
        for r in s['groups']:
            self.assertEqual(r['queries_with_candidate_topk_loss'],0)
            if r['mode']=='norm':self.assertAlmostEqual(r['saving_fraction'],.25)


class RealFaissAndRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import faiss
        faiss.omp_set_num_threads(1)
        rng=np.random.default_rng(918)
        cls.raw=rng.normal(0,.2,(1024,8)).astype(np.float32)
        cls.index=faiss.IndexIVFPQ(faiss.IndexFlatL2(8),8,4,2,8)
        cls.index.cp.niter=2;cls.index.pq.cp.niter=2;cls.index.pq.cp.min_points_per_centroid=1
        cls.index.train(cls.raw);cls.index.add(cls.raw)

    def test_shared_production_progress_accepts_name_and_isolates_run_roots(self):
        with tempfile.TemporaryDirectory() as d:
            left=Path(d)/'left';right=Path(d)/'right'
            a=g.progress_callback(left);b=g.progress_callback(right)
            a('sketch_build',name='gaussian_example',rows=12)
            b('side_pq_build',name='sidepq_example')
            self.assertEqual(json.loads((left/'progress.json').read_text()),
                dict(phase='sketch_build',name='gaussian_example',rows=12))
            self.assertEqual(json.loads((right/'progress.json').read_text()),
                dict(phase='side_pq_build',name='sidepq_example'))

    def test_real_Gaussian_and_sidePQ_build_use_production_progress_callback(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);actual=g.progress_callback(root);observed=[]
            def phase(*args,**fields):
                actual(*args,**fields)
                observed.append(json.loads((root/'progress.json').read_text()))
            c=dict(g.config(),k=5,dimensions=[2,4],sketch_seeds=[2026110401],
                side_pq_bytes=[2],side_pq_seeds=[2026110501],
                side_pq_training=dict(niter=2,nredo=1),build_block=256)
            book=g.extract_actual_ivf_codes(self.index,np.arange(len(self.raw)))
            ctx=g.reconstruction_context(book,np.eye(8),np.zeros(8),c)
            summaries,_,_=g.build_residual_summaries(self.raw,book,ctx,np.arange(1024),c,phase=phase)
            self.assertEqual(set(summaries),{'norm','gaussian_2026110401_2','gaussian_2026110401_4','sidepq_2'})
            self.assertEqual([r['name'] for r in observed if r['phase']=='sketch_build'],
                ['gaussian_2026110401_2','gaussian_2026110401_4'])
            self.assertEqual([r['name'] for r in observed if r['phase']=='side_pq_build'],['sidepq_2'])
            self.assertEqual(observed[-1],dict(phase='side_pq_build',name='sidepq_2'))
            self.assertEqual(json.loads((root/'progress.json').read_text()),observed[-1])

    def test_actual_list_decode_matches_library_and_only_requested_rows(self):
        import faiss
        wanted=np.array([1,5,37,901],np.int64);book=g.extract_actual_ivf_codes(self.index,wanted)
        xhat,y=g.decode_actual_ivf_original(book,np.arange(4),np.eye(8),np.zeros(8))
        self.assertEqual(book['codes'].shape,(4,2))
        for i in range(4):
            expected=np.empty(8,np.float32)
            self.index.reconstruct_from_offset(int(book['labels'][i]),int(book['offsets'][i]),faiss.swig_ptr(expected))
            np.testing.assert_array_equal(y[i],expected);np.testing.assert_array_equal(xhat[i],expected)

    def test_wrong_or_duplicate_original_rows_rejected(self):
        for rows in [[1,1],[4,1],[-1],[1024]]:
            with self.assertRaises(ValueError):g.extract_actual_ivf_codes(self.index,rows)

    def test_real_side_PQ_training_and_full_synthetic_cli_closure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'tiny';out=runner.local_tiny(root)
            self.assertEqual(out['status'],'passed');self.assertFalse(out['real_gate2_started'])
            receipt=json.loads((root/'tiny.json').read_text());self.assertEqual(len(receipt['checks']),4)
            self.assertIn('sidepq_2',[r['name'] for r in receipt['checks']])
            index=json.loads((root/'archive_index.json').read_text())
            for e in index['files']:self.assertEqual(runner.controller.digest(root/e['path']),e['sha256'])
            with self.assertRaises(FileExistsError):runner.local_tiny(root)

    def test_real_owned_worker_success_and_failure_close_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            for code,state in [(0,'complete'),(7,'compute_failed')]:
                root=Path(d)/str(code)
                r=runner.controller.supervise([sys.executable,'-c','import sys;sys.exit({})'.format(code)],root,os.environ.copy(),.03,.01)
                self.assertEqual(r['status'],state);self.assertTrue((root/'archive_verified.json').exists())
                with self.assertRaises(ValueError):runner.controller.request_stop(root)

    def test_real_manual_stop_remains_scoped_and_retained(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run'
            def stop():
                for _ in range(500):
                    if (root/'ownership.json').exists():runner.controller.request_stop(root);return
                    time.sleep(.005)
            thread=threading.Thread(target=stop);thread.start()
            r=runner.controller.supervise([sys.executable,'-c','import time;time.sleep(20)'],root,os.environ.copy(),.03,.01)
            thread.join();self.assertEqual(r['status'],'user_stopped');self.assertTrue(r['worker_exited'])

    def test_tiny_cli_refuses_symlink_root(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'target';target.mkdir();link=Path(d)/'link';link.symlink_to(target)
            p=subprocess.run([sys.executable,str(g.ROOT/g.PATHS[3]),'tiny','--root',str(link)],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0);self.assertIn('symlink',p.stderr)


if __name__=='__main__':unittest.main()
