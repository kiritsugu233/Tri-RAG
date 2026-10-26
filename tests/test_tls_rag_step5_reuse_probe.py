"""Offline CPU regressions for the scoped reuse probe; no scientific retuning."""
from types import SimpleNamespace
import unittest

import numpy as np

from mechanism.tls_rag_step5_audit import ExactScan
from mechanism.tls_rag_step5_reuse_probe import CASES, ExactDistanceSession, run_case
from tri_rag_harness.projection import dense_gaussian_projection, project_rows


def fixture(n=513):
    rng = np.random.default_rng(201)
    corpus = rng.normal(size=(n,24))
    corpus /= np.linalg.norm(corpus, axis=1)[:,None]
    query = rng.normal(size=24);query /= np.linalg.norm(query)
    ids = np.array(['id-'+str(i) for i in rng.permutation(n)])
    matrix = dense_gaussian_projection(12,24,202)
    reader = SimpleNamespace(id_array=ids, corpus=corpus,
        projected=project_rows(corpus,matrix),row_by_id={v:i for i,v in enumerate(ids)})
    return reader,query,matrix


class ReuseProbeTests(unittest.TestCase):
    def check_session(self, ids, vv, q, grid):
        reference=ExactScan(ids,vv,7)
        for selector in ('reselect','partition'):
            session=ExactDistanceSession(ids,vv,q,11,selector)
            for budget in grid:
                rr,dd=session.expand(budget)
                er,ed,_=reference.search(q,budget)
                np.testing.assert_array_equal(rr,er)
                np.testing.assert_array_equal(dd.view(np.uint64),ed.view(np.uint64))
            self.assertEqual(session.distance_evaluations,len(ids))
            self.assertEqual(session.scan_count,1)

    def test_random_prefixes_are_bitwise_reference_equal(self):
        r,q,m=fixture()
        self.check_session(r.id_array,r.projected,project_rows(q[None],m)[0],[1,32,64,128,256,513])

    def test_cutoff_ties_nonsorted_ids_zero_and_near_ties(self):
        vv=np.array([[1.,0.],[1.,0.],[0.,0.],[2.,0.],[1.,0.],
            [np.nextafter(1.,2.),0.],[np.nextafter(1.,0.),0.]])
        self.check_session(np.array(['z','c','x','b','a','e','f']),vv,np.zeros(2),list(range(1,8)))

    def test_every_distance_equal_through_all_expansions(self):
        ids=np.array(['x'+str(i) for i in range(70,-1,-1)])
        self.check_session(ids,np.ones((71,2)),np.zeros(2),[1,2,32,64,71])

    def test_cache_owns_query_and_returns_readonly_prefix_copies(self):
        ids=np.array(['z','y','x']);vv=np.array([[1.,0],[2.,0],[3.,0]])
        for selector in ('reselect','partition'):
            q=np.zeros(2);s=ExactDistanceSession(ids,vv,q,2,selector)
            q[:]=100
            rr,dd=s.expand(1)
            self.assertEqual(rr.tolist(),[0]);self.assertEqual(dd.tolist(),[1.])
            with self.assertRaises(ValueError):rr[0]=2
            # Even re-enabling writes changes only the returned copy.
            rr.flags.writeable=True;rr[0]=2
            rr,dd=s.expand(2)
            self.assertEqual(rr.tolist(),[0,1]);self.assertEqual(dd.tolist(),[1.,4.])
            other=ExactDistanceSession(ids,vv,q,2,selector)
            self.assertEqual(other.expand(1)[0].tolist(),[2])

    def test_bad_budgets_do_not_consume_state(self):
        r,q,m=fixture()
        for selector in ('reselect','partition'):
            s=ExactDistanceSession(r.id_array,r.projected,project_rows(q[None],m)[0],31,selector)
            s.expand(32)
            for budget in [32,16,0,514,True,64.]:
                with self.assertRaises(ValueError):s.expand(budget)
            self.assertEqual(len(s.expand(64)[0]),64)

    def test_bad_inputs_and_nonfinite_distances_fail(self):
        r,q,m=fixture()
        for kwargs in [dict(query=np.zeros(13)),dict(query=np.full(12,np.nan)),
                       dict(block_rows=0),dict(selector='unknown')]:
            args=dict(ids=r.id_array,vectors=r.projected,query=np.zeros(12));args.update(kwargs)
            with self.assertRaises(ValueError):ExactDistanceSession(**args)
        vv=r.projected.copy();vv[-1,0]=np.inf
        with self.assertRaises(ValueError):ExactDistanceSession(r.id_array,vv,np.zeros(12))

    def test_actual_vector_rows_read_once_and_not_on_expansion(self):
        r,q,m=fixture()
        class Counted:
            ndim=2
            shape=r.projected.shape
            def __init__(self):self.rows=0
            def __getitem__(self,key):
                self.rows+=key.stop-key.start
                return r.projected[key]
        for selector in ('reselect','partition'):
            vv=Counted()
            s=ExactDistanceSession(r.id_array,vv,np.zeros(12),31,selector)
            self.assertEqual(vv.rows,513)
            for b in [32,64,128,256]:s.expand(b)
            self.assertEqual(vv.rows,513)
            expected=513*4 if selector=='reselect' else 513+(513-32)+(513-64)+(513-128)
            self.assertEqual(s.selection_rows,expected)

    def test_all_complete_cases_match_every_prefix_and_reranking(self):
        r,q,m=fixture()
        qp=project_rows(q[None],m)[0]
        reference=ExactScan(r.id_array,r.projected,31)
        checked=[]
        def validate(prefixes,records,originals):
            for (rr,dd),record in zip(prefixes,records):
                expected,distances,_=reference.search(qp,len(rr))
                np.testing.assert_array_equal(rr,expected)
                np.testing.assert_array_equal(dd.view(np.uint64),distances.view(np.uint64))
                diff=r.corpus[rr]-q
                od=np.einsum('ij,ij->i',diff,diff)
                for row,d in zip(rr,od):
                    self.assertEqual(np.float64(originals[r.id_array[row]]).view(np.uint64),d.view(np.uint64))
                rank=sorted(r.id_array[rr],key=lambda i:(originals[i],i))
                self.assertEqual(record['reranked_ids'],rank)
                self.assertEqual(record['context_ids'],rank[:5])
                checked.append(1)
        for case in CASES:
            for endpoint in [32,64,128,256]:
                result=run_case(r,q,m,case,endpoint,block_rows=31,validate=validate)
                self.assertEqual(result['projected_distances'],513)
                self.assertEqual(result['original_distances'],endpoint)
                self.assertEqual(result['scans'],1)
        self.assertEqual(len(checked),52)


if __name__=='__main__':unittest.main()
