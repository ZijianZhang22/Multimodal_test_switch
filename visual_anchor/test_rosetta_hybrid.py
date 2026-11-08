import unittest
import numpy as np

from core import polar
from rosetta_ot_hybrid import hard_refine


class HardRefineTests(unittest.TestCase):
    def test_deterministic_orthogonal_and_finite(self):
        rng=np.random.default_rng(5)
        d=8
        lpool=rng.normal(size=(100,d))
        epool=rng.normal(size=(120,d))
        lpair=rng.normal(size=(30,d))
        epair=rng.normal(size=(30,d))
        init=polar(lpair.T@epair)
        args=dict(seed=17,steps=5,batch=32,match_weight=.25)
        a,ta=hard_refine(init,lpool,epool,lpair,epair,**args)
        b,tb=hard_refine(init,lpool,epool,lpair,epair,**args)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(a.T@a,np.eye(d),atol=1e-10)
        self.assertTrue(np.isfinite(a).all())
        self.assertEqual(len(ta),5)
        self.assertEqual(ta,tb)

    def test_zero_steps_preserves_warm_start(self):
        rng=np.random.default_rng(6)
        w=polar(rng.normal(size=(4,4)))
        x=rng.normal(size=(20,4)); y=rng.normal(size=(25,4))
        after,trace=hard_refine(w,x,y,x[:5],y[:5],steps=0)
        np.testing.assert_array_equal(after,w)
        self.assertEqual(trace,[])

    def test_bad_options_fail(self):
        x=np.eye(4)
        with self.assertRaises(ValueError):
            hard_refine(x,x,x,x,x,steps=-1)


if __name__=='__main__':
    unittest.main()
