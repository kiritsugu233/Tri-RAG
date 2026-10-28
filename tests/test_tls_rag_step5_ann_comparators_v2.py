"""Offline fixed-pair, cache, role and native checks for the approved v2 screen."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.stats import f,norm,nct
from mechanism.tls_rag_step5_ann_comparators_v2 import (config,sources,sha,save,
    polarization_threshold,conditional_normal_threshold,conditional_student_table,
    table_lookup,unit_boundary,prepare_cache,Context,Native,OpenedInputs,seal_selection,
    fingerprint,rerank)


class ComparatorMathematics(unittest.TestCase):
    def test_only_approved_route_and_fixed_cpu_settings(self):
        s=config();self.assertEqual(s['active_routes'],[1]);self.assertEqual(s['deferred_routes'],[2,3])
        self.assertFalse(s['adapter_training']);self.assertFalse(s['formal_certification'])
        self.assertEqual(s['threads'],1);self.assertEqual(s['terminal_audit'],0)

    def test_polar_boundary_is_exact_f_tail(self):
        for m in (16,32,64):
            for a in (.001,.01,.05):
                T=.8;Sq=1.2;Sx=.9;u=polarization_threshold(T,Sq,Sx,m,a)
                ratio=(Sq+Sx-2*u)/(Sq+Sx+2*u)
                self.assertAlmostEqual(f.sf(ratio*(4-T)/T,m,m),a,places=11)

    def test_normal_boundary_is_exact_conditional_tail(self):
        for m in (16,32,64):
            for h in (0,.3,.8):
                cut=conditional_normal_threshold(h,.9,m,.01)
                z=(cut-h*.9)/np.sqrt((1-h*h)*.9/m)
                self.assertAlmostEqual(norm.cdf(z),.01,places=13)

    def test_fixed_pair_monte_carlo_three_rules(self):
        # Predetermined fixed-pair diagnostic: six binomial standard errors.
        rng=np.random.default_rng(config()['seeds']['math']);n=24000;m=32;a=.01;h=.6
        pq=rng.normal(size=(n,m))/np.sqrt(m)
        px=h*pq+np.sqrt(1-h*h)*rng.normal(size=(n,m))/np.sqrt(m)
        sq=np.sum(pq*pq,1);sx=np.sum(px*px,1);u=np.sum(pq*px,1)
        fc=f.isf(a,m,m);c=fc*(1-h)/(1+h)
        polar=u<(1-c)/(2*(1+c))*(sq+sx)
        normal=u<h*sq-norm.isf(a)*np.sqrt((1-h*h)*sq/m)
        delta=h*np.sqrt(m*sq/(1-h*h));t=nct.ppf(a,m-1,delta)
        student=u/np.sqrt(sq*sx)<t/np.hypot(t,np.sqrt(m-1))
        tolerance=6*np.sqrt(a*(1-a)/n)
        for result in (polar,normal,student):self.assertLess(abs(result.mean()-a),tolerance)

    def test_invalid_geometry_fails_instead_of_clipping(self):
        for T,Sq,Sx in [(0,1,1),(4,1,1),(1,0,1),(np.nan,1,1)]:
            with self.assertRaises(ValueError):polarization_threshold(T,Sq,Sx,32,.01)
        for h in (-.1,1,np.nan):
            with self.assertRaises(ValueError):conditional_normal_threshold(h,1,32,.01)
        for m,a in [(True,.01),(0,.01),(32,0),(32,.5),(32,np.nan)]:
            with self.assertRaises(ValueError):polarization_threshold(1,1,1,m,a)

    def test_student_table_monotone_floor_is_conservative(self):
        s=dict(config(),student_delta_min=-3.,student_delta_max=3.,student_delta_step=.125)
        tab=conditional_student_table(32,.01,s);self.assertTrue((np.diff(tab)>=0).all())
        for delta in np.linspace(-3,3,123):
            r=table_lookup(tab,delta,s);t=nct.ppf(.01,31,delta)
            self.assertLessEqual(r,t/np.hypot(t,np.sqrt(31))+1e-15)
        self.assertIsNone(table_lookup(tab,-3.01,s));self.assertIsNone(table_lookup(tab,3.01,s))
        with self.assertRaises(ValueError):table_lookup(tab,np.nan,s)
        with self.assertRaises(ValueError):conditional_student_table(1,.01,s)

    def test_unit_boundary_minimizes_over_actual_norm_range(self):
        for T in (0,.1,1.,2.,3.):
            nq=.9999999;lo=.999999;hi=1.000001
            cut=unit_boundary(nq,lo,hi,T);nx=np.linspace(lo,hi,1001)
            h=(nq+nx-T)/(2*np.sqrt(nq*nx))
            self.assertLessEqual(cut,float(h.min())+2e-15)
        for args in [(0,1,1,1),(1,2,1,1),(1,1,1,-1)]:
            with self.assertRaises(ValueError):unit_boundary(*args)

    def test_report_requires_frozen_selection_and_source_identity(self):
        x=object.__new__(OpenedInputs);x.binding='synthetic';x._seal=None
        x.groups=dict(development=['d'],report=['r']);x._rows={'d':{'query_id':'d'},'r':{'query_id':'r'}}
        with self.assertRaises(ValueError):x.records('report')
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'seal.json';binding=seal_selection(p,x,dict(best='polar:32:0.05:point'))
            x.open_report(p,binding);self.assertEqual(x.records('report'),[{'query_id':'r'}])
            with self.assertRaises(ValueError):x.open_report(p,'wrong')
            value=json.loads(p.read_text());value['sources']['injected']='x';p.write_text(json.dumps(value))
            with self.assertRaises(ValueError):x.open_report(p,binding)


@unittest.skipUnless(os.environ.get('STEP5_COMPARATORS_V2_NATIVE'),'requires locally compiled fixed-version v2 shim')
class ComparatorNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import faiss
        cls.s=config();cls.n=Native(os.environ['STEP5_COMPARATORS_V2_NATIVE'],cls.s)
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        rng=np.random.default_rng(35517);cls.x=rng.normal(size=(1200,384))
        cls.x/=np.linalg.norm(cls.x,axis=1)[:,None]
        cls.q=rng.normal(size=(4,384));cls.q/=np.linalg.norm(cls.q,axis=1)[:,None]
        cls.base=rng.normal(size=(128,384))/np.sqrt(128)
        cls.index=faiss.IndexHNSWFlat(384,16);cls.index.hnsw.efConstruction=200
        cls.index.hnsw.efSearch=256;cls.index.add(cls.x.astype(np.float32))
        faiss.write_index(cls.index,str(cls.root/'index.faiss'));cls.index_sha=sha(cls.root/'index.faiss')
        cls.receipt=prepare_cache(cls.x,cls.base,cls.root/'cache',cls.s,cls.index_sha)
        cls.contexts={m:Context(cls.root/'cache',m,cls.s,cls.index_sha) for m in (16,32,64)}

    @classmethod
    def tearDownClass(cls):
        cls.contexts.clear();cls.temp.cleanup()

    def test_auxiliary_geometry_normalized_before_projection_only(self):
        x=self.x.astype(np.float32).astype(np.float64);x/=np.linalg.norm(x,axis=1)[:,None]
        for m,ctx in self.contexts.items():
            expected=np.asarray(x@(self.base[:m]*np.sqrt(128/m)).T,dtype=np.float32)
            np.testing.assert_array_equal(ctx.xp,expected)
            np.testing.assert_allclose(ctx.norms[:,0],np.linalg.norm(expected.astype(np.float64),axis=1),rtol=1e-7)
            self.assertGreater(float(np.std(ctx.norms[:,0])),.01)
            np.testing.assert_allclose(ctx.xp,self.contexts[64].xp[:,:m]*np.sqrt(64/m),rtol=2e-6,atol=1e-7)
        self.assertFalse(self.receipt['post_projection_normalized'])

    def test_cache_refuses_changed_index_and_file(self):
        with self.assertRaises(ValueError):Context(self.root/'cache',32,self.s,'wrong-index')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);shutil.copytree(self.root/'cache',root/'cache')
            p=root/'cache/points_32.npy'
            with p.open('r+b') as stream:stream.seek(-1,2);stream.write(b'X')
            with self.assertRaises(ValueError):Context(root/'cache',32,self.s,self.index_sha)

    def test_native_noop_bitwise_matches_actual_faiss(self):
        D,I=self.index.search(self.q.astype(np.float32),64)
        for j,q in enumerate(self.q):
            d,i,s,_=self.n.search(self.index,q)
            np.testing.assert_array_equal(d,D[j]);np.testing.assert_array_equal(i,I[j])
            self.assertEqual(s['dot_calls'],0);self.assertGreater(s['full_distances'],64)

    def test_all_shadow_rules_preserve_path_and_compute_once(self):
        for q in self.q:
            d,i,_,_=self.n.search(self.index,q)
            for rule in ('chi','polar','normal','student','corr'):
                sd,si,s,e=self.n.search(self.index,q,self.contexts[32],rule=rule,mode=1,events=True)
                np.testing.assert_array_equal(sd,d);np.testing.assert_array_equal(si,i)
                self.assertGreater(s['dot_calls'],0);self.assertEqual(s['dot_dimensions'],32*s['dot_calls'])
                self.assertEqual(s['skipped'],0);self.assertGreater(len(e),64)

    def test_native_polar_cutoff_matches_guarded_python(self):
        ctx=self.contexts[32];q=self.q[0].astype(np.float32);q64=q.astype(np.float64)
        qp=np.asarray((q64/np.linalg.norm(q64))@ctx.matrix.T,dtype=np.float32)
        sq=float(qp.astype(np.float64)@qp.astype(np.float64))
        _,_,_,ev=self.n.search(self.index,q,ctx,rule='polar',mode=1,events=True)
        for row in ev[np.isfinite(ev[:,4])][:100]:
            rid=int(row[0]);T=row[2]*(1+self.s['original_distance_guard_relative'])+self.s['original_distance_guard_absolute']
            h=unit_boundary(float(q64@q64),ctx.value['norm_min'],ctx.value['norm_max'],T)
            nx=float(ctx.norms[rid,0]);cut=polarization_threshold(2-2*h,sq,nx*nx,32,.05)-self.s['dot_guard']*np.sqrt(sq)*nx
            self.assertAlmostEqual(cut,row[4],places=13)

    def test_active_batch_and_progressive_account_for_actual_work(self):
        for rule in ('chi','polar','normal','student','corr'):
            for skip in ('point','batch'):
                d,i,s,_=self.n.search(self.index,self.q[0],self.contexts[64],rule=rule,mode=2,skip=skip)
                self.assertTrue(np.isfinite(d).all());self.assertEqual(len(set(i)),64)
                self.assertGreater(s['dot_calls'],0);self.assertGreaterEqual(s['skipped'],0)
                if skip=='batch':self.assertEqual(s['partial_batches'],0)
            ctx=Context(self.root/'cache',64,self.s,self.index_sha,progressive=True)
            d,i,s,_=self.n.search(self.index,self.q[0],ctx,rule=rule,mode=2)
            self.assertEqual(len(set(i)),64);self.assertGreater(s['dot_dimensions'],0)
            self.assertLessEqual(s['dot_dimensions'],64*s['gated'])

    def test_future_information_and_invalid_inputs_forbidden(self):
        with self.assertRaises(TypeError):self.n.search(self.index,self.q[0],oracle=np.zeros(len(self.x)))
        for kwargs in [dict(mode=2),dict(mode=3),dict(alpha=.02,mode=1,context=self.contexts[32]),dict(k=0)]:
            with self.assertRaises(ValueError):self.n.search(self.index,self.q[0],**kwargs)

    def test_independent_worker_and_screen_keep_query_level_evidence(self):
        path=Path(__file__).parents[1]/'scripts/run_tls_rag_step5_ann_comparators_v2.py'
        loader=importlib.util.spec_from_file_location('comparator_test_runner',path)
        runner=importlib.util.module_from_spec(loader);loader.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);shutil.copyfile(os.environ['STEP5_COMPARATORS_V2_NATIVE'],root/'comparators.dylib')
            np.save(root/'corpus.npy',self.x);ids=['d'+str(i) for i in range(len(self.x))];save(root/'ids.json',ids)
            paths=dict(index=str(self.root/'index.faiss'),index_sha=self.index_sha,corpus=str(root/'corpus.npy'),ids=str(root/'ids.json'),cache=str(self.root/'cache'))
            records=[dict(query_id=str(j),query=q.tolist(),qrels={}) for j,q in enumerate(self.q[:2])]
            inputs=SimpleNamespace(truth={r['query_id']:[ids[0]] for r in records})
            screen=runner.Screen(inputs,paths,self.n,root,self.s,perf_counter())
            screen.verify(records,'synthetic');screen.measure(records,['noop','polar:32:0.05:point'],'synthetic')
            self.assertEqual(len(screen.timings),12);self.assertTrue((root/'query_records.jsonl').is_file())
            for row in screen.timings:
                ref,_=screen.r.call(records[int(row['query_id'])]['query'],row['method'])
                self.assertEqual(row['candidate_ids'],ref['candidate_ids']);self.assertGreater(row['ms'],0)
            screen.close_if_unused=True
            screen.r.close()


if __name__=='__main__':unittest.main()
