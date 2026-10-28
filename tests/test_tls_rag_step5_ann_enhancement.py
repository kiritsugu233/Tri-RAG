"""Offline route-1 contract checks; real native tests require the compiled shim."""
import copy
import importlib.util
import json
import os
import shutil
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.stats import f
from mechanism.tls_rag_step5_ann_enhancement import (config,critical_beta,split_groups,
    ordered,rerank,metrics,ObservedInputs,seal_selection,sources,save,Native,paired_ratio)
from tri_rag_harness.tri_law import tri_law_probability


class Route1Contracts(unittest.TestCase):
    def test_only_route1_enabled(self):
        s=config();self.assertEqual(s['active_routes'],[1]);self.assertEqual(s['deferred_routes'],[2,3])
        self.assertFalse(s['formal_certification']);self.assertTrue(s['observed_development_only'])

    def test_threshold_is_exact_inverse_of_unchanged_law(self):
        for alpha in config()['alphas']:
            fc=f.isf(alpha,128,128)
            self.assertAlmostEqual(float(critical_beta(fc,0)),fc,places=13)
            self.assertEqual(float(critical_beta(fc,1)),1)
            rho=np.array([0,.2,.5,.9,.99])
            np.testing.assert_allclose(tri_law_probability(critical_beta(fc,rho),rho,128),alpha,rtol=1e-10)

    def test_invalid_geometry_rejected_instead_of_clipped(self):
        for fc,rho in [(1,0),(2,1.0001),(2,np.nan),(2,np.inf)]:
            with self.assertRaises(ValueError):critical_beta(fc,rho)

    def test_role_partitions_are_fixed_disjoint_and_development_only(self):
        selection=['s'+str(i) for i in range(256)];evaluation=['e'+str(i) for i in range(512)]
        groups=split_groups(selection,evaluation,config())
        self.assertEqual({g:len(x) for g,x in groups.items()},dict(fit=128,cal=64,select=64,report=256))
        self.assertEqual(len({i for v in groups.values() for i in v}),512)
        self.assertEqual(groups,split_groups(selection[::-1],evaluation[::-1],config()))
        with self.assertRaises(ValueError):split_groups(selection,selection*2,config())

    def test_report_requires_bound_selection_and_sources(self):
        inputs=object.__new__(ObservedInputs);inputs.binding='opened-packet';inputs._report_seal=None
        inputs.groups={'select':['s'],'report':['r']};inputs._rows={'s':{'query_id':'s'},'r':{'query_id':'r'}}
        with self.assertRaises(ValueError):inputs.records('report')
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'seal.json';binding=seal_selection(path,inputs,{'best':None})
            inputs.open_report(path,binding);self.assertEqual(inputs.records('report'),[{'query_id':'r'}])
            with self.assertRaises(ValueError):inputs.open_report(path,'wrong')
            value=json.loads(path.read_text());value['sources']['injected']='wrong';path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):inputs.open_report(path,binding)

    def test_original_rerank_and_labels_do_not_change_geometry(self):
        x=np.array([[1.,0.],[0.,1.],[0.,1.]])
        pool,ctx,_=rerank(x,['a','z','b'],np.array([2,0,1]),np.array([0.,1.]),2,1)
        self.assertEqual(pool,['b','z','a']);self.assertEqual(ctx,['b'])
        self.assertEqual(metrics(pool,ctx,['b','a'],{'b':1}),dict(retention=1.,hit=1))
        self.assertEqual(metrics(pool,ctx,['b','a'],{}),dict(retention=1.,hit=0))
        with self.assertRaises(ValueError):rerank(x,['a','z','b'],[1,1],np.array([0.,1.]))

    def test_output_exclusive_and_pairing_required(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'out.json';save(path,{'result':'failed'})
            with self.assertRaises(FileExistsError):save(path,{})
        rows=[dict(query_id=str(q),method=m,ms=t) for q in range(3) for m,t in [('a',1.),('b',2.)]]
        result=paired_ratio(rows,'a','b',config());self.assertEqual(result['ratio'],.5)
        self.assertEqual(result['interval95'],[.5,.5])


@unittest.skipUnless(os.environ.get('STEP5_ROUTE1_NATIVE'),'requires locally compiled fixed-version shim')
class Route1Native(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import faiss
        cls.s=config();cls.native=Native(os.environ['STEP5_ROUTE1_NATIVE'],cls.s)
        rng=np.random.default_rng(838);cls.x=rng.normal(size=(1200,384)).astype(np.float32)
        cls.x/=np.linalg.norm(cls.x,axis=1)[:,None]
        cls.q=rng.normal(size=(6,384)).astype(np.float32);cls.q/=np.linalg.norm(cls.q,axis=1)[:,None]
        cls.index=faiss.IndexHNSWFlat(384,16);cls.index.hnsw.efConstruction=200;cls.index.hnsw.efSearch=256
        cls.index.add(cls.x)
        cls.matrix=rng.normal(size=(128,384))/np.sqrt(128)
        cls.p=np.ascontiguousarray(cls.x@cls.matrix.T,dtype=np.float32)

    def test_native_noop_bitwise_matches_faiss(self):
        D,I=self.index.search(self.q,64)
        for j,q in enumerate(self.q):
            d,i,s,e=self.native.search(self.index,q,events=True)
            np.testing.assert_array_equal(d,D[j]);np.testing.assert_array_equal(i,I[j])
            self.assertEqual(s['projected_distances'],0)
            self.assertGreater(s['full_distances'],64)

    def test_shadow_overhead_and_free_oracle_preserve_output(self):
        for q in self.q:
            d,i,s,e=self.native.search(self.index,q,events=True)
            o=np.full(len(self.x),np.nan,np.float32);o[e[:,0].astype(int)]=e[:,3]
            nd,ni,ns,_=self.native.search(self.index,q,mode=3,oracle=o)
            np.testing.assert_array_equal(d,nd);np.testing.assert_array_equal(i,ni)
            self.assertGreater(ns['oracle_skipped'],0);self.assertLess(ns['full_distances'],s['full_distances'])
            for mode in (1,4):
                sd,si,ss,se=self.native.search(self.index,q,self.p,q@self.matrix.T,mode=mode,rule='tri',events=mode==1)
                np.testing.assert_array_equal(d,sd);np.testing.assert_array_equal(i,si)
                self.assertGreater(ss['projected_distances'],0)

    def test_final_result_rejection_is_not_frontier_rejection(self):
        # The graph may still need nodes outside the current result pool.
        found=False
        for q in self.q:
            _,_,_,e=self.native.search(self.index,q,events=True)
            possible_bridge=(e[:,3]>=e[:,12]) & (e[:,3]<e[:,13])
            if possible_bridge.any():
                found=True;self.assertTrue((e[possible_bridge,8]==0).all())
        self.assertTrue(found)

    def test_independent_timing_worker_matches_correctness_candidates(self):
        import faiss
        path=Path(__file__).parents[1]/'scripts/run_tls_rag_step5_ann_enhancement.py'
        loader=importlib.util.spec_from_file_location('route1_test_runner',path)
        module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);shutil.copyfile(os.environ['STEP5_ROUTE1_NATIVE'],root/'route1.dylib')
            np.save(root/'subset_corpus.npy',self.x.astype(np.float64))
            ids=['d'+str(j) for j in range(len(self.x))];save(root/'subset_ids.json',ids)
            faiss.write_index(self.index,str(root/'subset_original.faiss'))
            records=[dict(query_id=str(j),query=q.tolist()) for j,q in enumerate(self.q[:2])]
            job=dict(sources=sources(),spec=self.s,scale=False,input_root=td,output_root=td,
                queries=records,group='synthetic',method='native_noop',repeat=0,result_path=str(root/'result.json'))
            save(root/'job.json',job);module.timing_worker(root/'job.json')
            measured=json.loads((root/'result.json').read_text())
            for row in measured['timings']:
                q=self.q[int(row['query_id'])]
                _,i,_,_=self.native.search(self.index,q)
                pool,ctx,_=rerank(self.x.astype(np.float64),ids,i,q.astype(np.float64))
                self.assertEqual(row['candidate_ids'],pool);self.assertEqual(row['context_ids'],ctx)
            self.assertGreater(measured['rss_highwater_bytes'],0)

    def test_deployable_gate_separates_pending_and_counts_audit(self):
        for rule in ('chi','pair','tri','angle'):
            d,i,s,e=self.native.search(self.index,self.q[0],self.p,self.q[0]@self.matrix.T,
                mode=2,rule=rule,alpha=.05,audit=16,events=True)
            self.assertTrue(np.isfinite(d).all());self.assertEqual(len(set(i)),64)
            self.assertLessEqual(s['audited'],16);self.assertEqual(s['deferred'],s['pending'])
            self.assertEqual(s['audited'],min(16,s['pending']))
            self.assertTrue((i>=0).all())

    def test_future_information_forbidden_in_active_mode_and_bad_storage_rejected(self):
        q=self.q[0]
        with self.assertRaises(ValueError):self.native.search(self.index,q,mode=2,oracle=np.zeros(len(self.x)))
        with self.assertRaises(ValueError):self.native.search(self.index,q,self.p.astype('float64'),q@self.matrix.T,mode=1)
        with self.assertRaises(ValueError):self.native.search(self.index,q,mode=3)
        with self.assertRaises(ValueError):self.native.search(self.index,q,self.p,q@self.matrix.T,mode=1,alpha=.02)


if __name__=='__main__':unittest.main()
