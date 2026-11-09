"""Small checkpoint-resume tests; no dataset/model download required."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from run_rosetta3 import restore_checkpoint

class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "image_only_9.npz"
        self.d = 8
        self.v = np.eye(self.d)
        self.idx = np.array([0, 2, 5])
        self.w = np.eye(self.d, dtype=np.float32)
        np.savez(self.path, W=self.w, mean_x=np.zeros((1, self.d), np.float32),
                 mean_y=np.ones((1, self.d), np.float32), V_IE=self.v,
                 anchor_indices=self.idx, lambda_y=np.array(0.), lambda_z=np.array(1.))
    def tearDown(self):
        self.tmp.cleanup()
    def test_restore_weights_and_means(self):
        model = SimpleNamespace()
        restore_checkpoint(model, self.path, self.idx, 0, 1, self.v, self.d)
        np.testing.assert_allclose(model.weight.numpy(), self.w)
        np.testing.assert_allclose(model.mean_y.numpy(), np.ones((1, self.d)))
    def test_wrong_budget_or_weights_refused(self):
        with self.assertRaisesRegex(ValueError, "anchors"):
            restore_checkpoint(SimpleNamespace(), self.path, np.array([0, 2]), 0, 1, self.v, self.d)
        with self.assertRaisesRegex(ValueError, "weights"):
            restore_checkpoint(SimpleNamespace(), self.path, self.idx, 1, 1, self.v, self.d)
    def test_wrong_bridge_refused(self):
        with self.assertRaisesRegex(ValueError, "bridge"):
            restore_checkpoint(SimpleNamespace(), self.path, self.idx, 0, 1,
                               np.full_like(self.v, 0.1), self.d)

if __name__ == "__main__": unittest.main()
