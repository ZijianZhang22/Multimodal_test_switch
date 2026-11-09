"""Regression checks for centered CLIP-native A-E comparisons."""
import unittest
import numpy as np
import torch
from rosetta3 import WassersteinProcrustes3
from run_rosetta3 import tensor

PARAMS = dict(num_clusters=4, num_restarts=1, batch_size=32,
              num_iterations=2, num_workers=1, num_threads=1)

class CenteringTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8)
        self.x = tensor(rng.normal(size=(200, 8)) + np.arange(8) / 5)
        self.y = tensor(rng.normal(size=(150, 8)) + np.arange(8) / 7)
        self.px = tensor(rng.normal(size=(10, 8)) + 3)
        self.py = tensor(rng.normal(size=(10, 8)) - 3)

    def test_anchor_centering_modes(self):
        for name, mx, my in (
            ("paired", self.px.mean(0, keepdim=True), self.py.mean(0, keepdim=True)),
            ("source_pool", self.x.mean(0, keepdim=True), self.py.mean(0, keepdim=True)),
            ("pool", self.x.mean(0, keepdim=True), self.y.mean(0, keepdim=True))):
            model = WassersteinProcrustes3(anchor_centering=name, **PARAMS).fit_supervised(
                self.x, self.y, self.px, self.py)
            torch.testing.assert_close(model.mean_x, mx)
            torch.testing.assert_close(model.mean_y, my)

    def test_invalid_centering(self):
        with self.assertRaises(ValueError):
            WassersteinProcrustes3(anchor_centering="invalid", **PARAMS)

    def test_source_pool_ablation_no_english_dependence(self):
        model1 = WassersteinProcrustes3(lambda_y=0, lambda_z=1,
                                       init_mode="supervised", anchor_centering="source_pool", **PARAMS)
        model2 = WassersteinProcrustes3(lambda_y=0, lambda_z=1,
                                       init_mode="supervised", anchor_centering="source_pool", **PARAMS)
        z = tensor(np.random.default_rng(4).normal(size=(80, 8)))
        model1.fit(self.x, self.y, self.px, self.py, randomness=9, samples_z=z)
        model2.fit(self.x, self.y * 99, self.px, self.py, randomness=9, samples_z=z)
        self.assertTrue(torch.equal(model1.weight, model2.weight))

    def test_image_bank_centering_independent_of_model_mean_y(self):
        image_bank = tensor(np.random.default_rng(6).normal(size=(100, 8)) + 2)
        image_test = tensor(np.random.default_rng(7).normal(size=(20, 8)) + 2)
        bank_mean = image_bank.mean(0, keepdim=True)
        model_a = WassersteinProcrustes3(**PARAMS).fit_supervised(
            self.x, self.y, self.px, self.py)
        changed_y = self.py + 70
        model_b = WassersteinProcrustes3(**PARAMS).fit_supervised(
            self.x, self.y, self.px, changed_y)
        expected = model_a.transform_target(image_test, bank_mean)
        actual = model_b.transform_target(image_test, bank_mean)
        torch.testing.assert_close(expected, actual)
        english_centered = model_a.transform_target(image_test, model_a.mean_y)
        self.assertFalse(torch.equal(expected, english_centered))

if __name__ == "__main__":
    unittest.main()
