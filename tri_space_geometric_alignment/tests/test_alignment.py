import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import unit, procrustes, balanced_sinkhorn, fit_joint, retrieval_metrics

class TestGeometry(unittest.TestCase):
    def test_procrustes(self):
        rng=np.random.default_rng(1)
        x=unit(rng.normal(size=(80,8)))
        q,_=np.linalg.qr(rng.normal(size=(8,8)))
        w=procrustes(x,x@q)
        self.assertGreater(np.mean(np.sum((x@w)*(x@q),axis=1)),.9999)

    def test_sinkhorn_marginals(self):
        rng=np.random.default_rng(2)
        x=unit(rng.normal(size=(24,8)))
        p=balanced_sinkhorn(x,x[::-1],eps=.15,n_iter=100)
        np.testing.assert_allclose(p.sum(1),np.ones(24)/24,atol=1e-5)
        np.testing.assert_allclose(p.sum(0),np.ones(24)/24,atol=1e-5)

    def test_joint_outputs(self):
        rng=np.random.default_rng(3);d=6
        s=unit(rng.normal(size=(85,d)))
        q,_=np.linalg.qr(rng.normal(size=(d,d)))
        z=fit_joint(s@q,s,s,np.arange(12),steps=2,batch=30)
        self.assertEqual(len(z.history),2)
        for w in [z.low_to_en,z.image_to_en,z.low_to_image]:
            np.testing.assert_allclose(w.T@w,np.eye(d),atol=1e-7)
        m=retrieval_metrics(s,s)
        self.assertEqual(m['left_to_right']['R1'],1.)
if __name__=='__main__': unittest.main()
