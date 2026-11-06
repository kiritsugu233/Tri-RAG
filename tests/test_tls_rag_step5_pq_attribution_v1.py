"""Independent CPU counterexamples, conditional law and diagnostic isolation."""
import inspect
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from mechanism import tls_rag_step5_pq_attribution_v1 as a


class AttributionTests(unittest.TestCase):
    def test_frozen_config_and_actual_authorization(self):
        c=a.config();self.assertEqual(len(c['query_indices']),32)
        self.assertEqual(c['authorization']['actual_user_consent'],'批准第19.4节八路径，仅执行这一次归因')
        self.assertFalse(c['formal_certification'])

    def test_stable_native_clipping_and_id_ties(self):
        np.testing.assert_array_equal(a.stable_order([2,3,.5,.5],[9,3,8,2]),[1,0,3,2])

    def test_gaussian_point_matches_explicit_contraction(self):
        q=np.array([1.,2.]);R=np.array([[1.,.2],[.4,-.7]])
        e=np.array([[.2,.5],[-.4,.2],[0.,.1]])
        z=(e@R.T).astype(np.float32);s=dict(R=R,sketch=z)
        base=np.array([.1,.2]);pos=np.array([2,0]);h=R@q/np.linalg.norm(q)
        want=base+np.array([sum(z[t,j]*h[j] for j in range(2)) for t in pos])*np.linalg.norm(q)/sum(h*h)
        np.testing.assert_allclose(a.point_scores(q,base,s,pos,'gaussian'),want,rtol=1e-7)

    def test_sidepq_point_matches_manual_decode(self):
        centers=np.arange(24).reshape(2,3,4)/100
        codes=np.array([[0,2],[1,0]],np.uint8);q=np.arange(8)/10
        s=dict(centers=centers,codes=codes)
        want=[q@np.r_[centers[0,x],centers[1,y]] for x,y in codes]
        np.testing.assert_allclose(a.point_scores(q,np.zeros(2),s,np.arange(2),'sidepq'),want,atol=1e-15)

    def test_point_estimator_has_no_truth_arguments(self):
        self.assertEqual(list(inspect.signature(a.point_scores).parameters),['query','base','summary','positions','kind'])
        self.assertNotIn('true',inspect.getsource(a.point_scores).split('"""')[2])

    def test_zero_query_and_degenerate_projection_fail(self):
        for q,R in [(np.zeros(2),np.eye(2)),(np.ones(2),np.zeros((2,2)))]:
            with self.assertRaises(ValueError):a.point_scores(q,[0.],dict(R=R,sketch=np.ones((1,2))),[0],'gaussian')

    def test_duplicate_ids_and_nonfinite_scores_fail(self):
        with self.assertRaises(ValueError):a.stable_order([0,1],[2,2])
        with self.assertRaises(ValueError):a.stable_order([float('nan')],[0])

    def test_native_subset_rerank_quality_and_oracle_isolation(self):
        r=a.rank_diagnostics([.9,.2,.8,.1],np.array([0,1,2,3]),[.5,.2,.4,.9],[1,2],[2,4])
        self.assertEqual(r[0]['candidate_retention'],.5)
        self.assertTrue(r[1]['preserves_top100'])

    def test_invalid_budget_and_order_rejected(self):
        with self.assertRaises(ValueError):a.rank_diagnostics([.5,.4],[0,1],[.5,.6],[0],[3])
        with self.assertRaises(ValueError):a.rank_diagnostics([.5,.4],[0,1],[.5,.6],[0],[1],order=[0,0])

    def test_boundary_pairs_exact_finite_window(self):
        i,j=a.boundary_pairs(np.arange(6400));self.assertEqual(len(i),820)
        self.assertEqual((i.min(),j.max()),(79,119))
        with self.assertRaises(ValueError):a.boundary_pairs(np.arange(3))

    def test_common_offset_cancels_but_does_not_prove_net_gain(self):
        true=np.arange(6.)/8;pairs=a.boundary_pairs(np.arange(6),(1,6))
        r=a.relative_error_diagnostics(true+2,true,pairs)
        self.assertEqual(r['cancellation_ratio'],0);self.assertEqual(r['point_mse'],4)
        self.assertEqual(r['strict_pair_inversions'],0)

    def test_opposite_errors_do_not_cancel(self):
        r=a.relative_error_diagnostics([1.,-1.],[0.,0.],(np.array([0]),np.array([1])))
        self.assertEqual(r['cancellation_ratio'],2)

    def test_perfect_estimate_is_not_missing_ratio_zero(self):
        r=a.relative_error_diagnostics([0,1],[0,1],(np.array([0]),np.array([1])))
        self.assertIsNone(r['cancellation_ratio']);self.assertTrue(r['perfect_point_estimate'])

    def test_conditional_shared_gaussian_covariance_against_sampling(self):
        rng=np.random.default_rng(2026110601);m=8;h=np.ones(m)/np.sqrt(m);e=np.array([[.4,.2,.5],[.1,-.3,.5]])
        R=rng.normal(0,1/np.sqrt(m),(16000,m,3));R[:,:,0]=h
        ahat=np.einsum('m,nmd,id->ni',h,R,e);errors=ahat-e[:,0]
        covariance=errors.T@errors/len(errors);expected=a.conditional_covariance(e,[1,0,0],m,1)
        np.testing.assert_allclose(covariance,expected,rtol=.06,atol=.0005)
        delta=errors[:,0]-errors[:,1]
        self.assertAlmostEqual(float(np.mean(delta**2)),.25/m,delta=.002)

    def test_proxy_constants_and_cost_are_not_free(self):
        p=a.proxy_cost_diagnostics(np.zeros(3),np.ones(3),np.ones((3,2),np.float32),
                                  (np.array([0,1]),np.array([1,2])),np.zeros(2))
        self.assertIsNone(p['proxies']['adc_gap']['spearman_with_squared_pair_error'])
        self.assertGreater(p['summary_gather_logical_bytes'],0)
        self.assertFalse(p['cheap_deployability_established'])
        r=a.proxy_cost_diagnostics(np.array([0.,1.,3.]),np.ones(3),None,
            (np.array([0,0,1]),np.array([1,2,2])),np.array([1.,2.,3.]),np.array([1.,8.,6.]))
        self.assertAlmostEqual(r['proxies']['adc_gap']['spearman_with_pair_cancellation_ratio'],-1.)

    def test_resource_stop_preserves_existing_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'evidence').write_text('retain')
            c={'limits':dict(seconds=0,rss_bytes=10**15,output_bytes=10**9)}
            with self.assertRaises(RuntimeError):a.check_resources(0,p,c)
            self.assertEqual((p/'evidence').read_text(),'retain')

    def test_missing_arrays_preflight_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            with self.assertRaises(FileNotFoundError):a.verify_inputs(a.config(),p)
            self.assertEqual(list(p.iterdir()),[])

    def test_fingerprint_covers_identity_and_missing_is_not_zero(self):
        self.assertNotEqual(a.fingerprint(dict(m=8)),a.fingerprint(dict(m=16)))
        with self.assertRaises(ValueError):a.fingerprint(dict(x=float('nan')))

    def test_real_run_pipeline_synthetic_closed_arrays_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';s0=root/'s0';output=root/'out'
            (source/'summaries').mkdir(parents=True);(source/'geometry').mkdir();s0.mkdir()
            rng=np.random.default_rng(119);n,d=6400,8
            ids=np.arange(n);q=np.ones(d)/np.sqrt(d);base=np.linspace(.9,.1,n)
            np.save(s0/'queries.npy',np.array([q,q]))
            np.save(s0/'published_truth.npy',np.tile(ids[:100],(2,1)))
            np.save(s0/'published_cutoffs.npy',np.array([.3,.3]))
            a.write(s0/'roles.json',dict(queries=[dict(role='geometry_discovery'),dict(role='geometry_check')]))
            np.savez(source/'summaries/actual_code_book.npz',rows=ids)
            name='gaussian_2026110401_32';R=rng.normal(size=(32,d))/np.sqrt(32)
            np.savez(source/'summaries'/(name+'.npz'),R=R,sketch=np.zeros((n,32),np.float32))
            np.savez(source/'summaries/sidepq_128.npz',codes=np.zeros((n,2),np.uint8),centers=np.zeros((2,256,4),np.float32))
            for qi in [0,1]:
                np.savez(source/'geometry'/('64-%s.npz'%qi),ids=ids,native_distances=(1-base).astype(np.float32),native_topk=ids[:100],original_reconstruction_dot=base,actual_ADC_reconstructed_dot=base,real_raw_dot=base,residual_radius=np.ones(n),lower_gaussian_2026110401_32=(1-base).astype(np.float32)-.1,lower_sidepq_128=(1-base).astype(np.float32)-.1)
            c=a.config();c.update(query_indices=[0,1],nprobe=[64],s0_directory=str(s0),summary_names=[name,'sidepq_128'],original_products=[])
            a.write(source/'source_freeze.json',dict(commit=c['scientific_source_commit'],config=dict(query_indices=[0,1])))
            a.write(source/'result.json',dict(status='complete'))
            entries=[dict(path=str(p.relative_to(source)),bytes=p.stat().st_size,sha256=a.sha(p)) for p in source.rglob('*') if p.is_file()]
            a.write(source/'archive_index.json',dict(files=entries));c['science_index_sha256']=a.sha(source/'archive_index.json')
            a.run(c,source,output)
            self.assertEqual(json.loads((output/'status.json').read_text())['status'],'complete')
            decision=json.loads((output/'decision.json').read_text())
            rr=json.loads((output/'ranking.json').read_text())
            slack=next(r['stored_native_lower_to_point_slack_quantiles'] for r in rr if r['method']==name)
            np.testing.assert_allclose(slack,.1,atol=5e-8,rtol=0)
            self.assertFalse(decision['point_headroom_necessary_screen_passed'])
            self.assertEqual(decision['disposition'],'end_current_interface_research')
            for item in json.loads((output/'archive_index.json').read_text())['files']:
                self.assertEqual(a.sha(output/item['path']),item['sha256'])
            with self.assertRaises(FileExistsError):a.run(c,source,output)
            (source/'geometry/64-0.npz').write_bytes(b'tampered')
            with self.assertRaises(ValueError):a.verify_inputs(c,source)
