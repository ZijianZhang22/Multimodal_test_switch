import unittest
import numpy as np
from core import procrustes,sinkhorn,fit,retrieval,linear_cka

class Tests(unittest.TestCase):
    def setUp(self):
        self.r=np.random.default_rng(7)
    def test_known_rotation(self):
        x=self.r.normal(size=(100,8)); q=np.linalg.qr(self.r.normal(size=(8,8)))[0]
        np.testing.assert_allclose(procrustes(x,x@q),q,atol=1e-10)
    def test_balanced_marginals(self):
        p=sinkhorn(self.r.normal(size=(13,4))*.2,self.r.normal(size=(19,4))*.2)
        np.testing.assert_allclose(p.sum(1),1/13,atol=1e-6)
        np.testing.assert_allclose(p.sum(0),1/19,atol=1e-6)
    def test_permutation_does_not_change_ot(self):
        x=self.r.normal(size=(13,4))*.2; y=self.r.normal(size=(19,4))*.2
        order=self.r.permutation(len(y)); p=sinkhorn(x,y); pp=sinkhorn(x,y[order])
        np.testing.assert_allclose(p@y,pp@y[order],atol=1e-7)
    def test_unbalanced_and_orthogonal(self):
        x=self.r.normal(size=(30,6)); y=x.copy()
        w,_=fit(x[:10],y[:10],x,y,y,y,iterations=2,batch=15,tau=.5)
        np.testing.assert_allclose(w.T@w,np.eye(6),atol=1e-10)
        self.assertTrue(np.isfinite(w).all())
    def test_multi_caption(self):
        x=np.array([[1.,0],[1.,0],[0,1]]); y=np.eye(2)
        m,_=retrieval(x,y,np.array(['a','a','b']),np.array(['a','b']),k=0)
        self.assertEqual(m['R@1'],1.)
    def test_cka_rotation(self):
        x=self.r.normal(size=(100,8)); q=np.linalg.qr(self.r.normal(size=(8,8)))[0]
        self.assertAlmostEqual(linear_cka(x,x@q),1.)

if __name__=='__main__': unittest.main()
