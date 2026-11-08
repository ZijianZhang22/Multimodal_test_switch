import unittest
import numpy as np

from run_rosetta_baseline import summarize, ros_target

class RosettaBaselineTests(unittest.TestCase):
    def test_summary(self):
        rows = [
            {'method': m, 'metrics': {k: {'R@1': v, 'R@5': v, 'R@10': v}
                for k in ('L-E','E-L','L-I','I-L')}}
            for m,v in [('rosetta_zero',.1),('rosetta_few',.2)]
        ]
        s=summarize(rows)
        self.assertAlmostEqual(s['rosetta_zero']['L-I']['R@1']['mean'],.1)
        self.assertAlmostEqual(s['rosetta_few']['E-L']['R@5']['mean'],.2)

    def test_target_centering_matches_official(self):
        # Optional dependency: skip if not installed; no monkey-patching the library.
        try:
            import torch
            from unpaired_rosetta.linalg import center_and_normalize
        except ImportError:
            self.skipTest('Official Rosetta not installed')
        class Model: pass
        m=Model();m.mean_y=torch.tensor([[.25,-.25]],dtype=torch.float32)
        x=np.array([[1.,0.],[0.,1.]],dtype=np.float32)
        expected=center_and_normalize(torch.tensor(x),m.mean_y).numpy()
        np.testing.assert_allclose(ros_target(m,x,torch),expected)

if __name__=='__main__':
    unittest.main()
