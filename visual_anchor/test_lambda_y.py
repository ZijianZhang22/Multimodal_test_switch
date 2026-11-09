"""Tests for 5-way A-E controlled refinement, require pinned official Rosetta."""
import unittest
import numpy as np
try:
    import torch
    from unpaired_rosetta import WassersteinProcrustes
    from rosetta3 import WassersteinProcrustes3
except ImportError as err:
    raise unittest.SkipTest(f"Official Rosetta not installed: {err}")

PARAMS = dict(num_clusters=5, num_restarts=1, batch_size=64,
              num_iterations=3, num_workers=1, num_threads=1)


def world():
    r = np.random.default_rng(4)
    d, n = 8, 150
    q = np.linalg.qr(r.normal(size=(d, d)))[0]
    lat = r.normal(size=(n, d)) * np.arange(1, d + 1)
    x = torch.tensor(lat, dtype=torch.float32)
    y = torch.tensor(lat @ q, dtype=torch.float32)
    z = torch.tensor((lat[::-1].copy() @ q), dtype=torch.float32)
    return x, y, z


class LambdaYTests(unittest.TestCase):
    def test_official_behavior_lambda_y_one_lambda_z_zero(self):
        x, y, z = world()
        for supervised in (False, True):
            px, py = (x[:24], y[:24]) if supervised else (None, None)
            expected = WassersteinProcrustes(**PARAMS).fit(x, y, px, py, randomness=7)
            actual = WassersteinProcrustes3(lambda_y=1, lambda_z=0, **PARAMS).fit(
                x, y, px, py, randomness=7)
            self.assertTrue(torch.equal(expected.weight, actual.weight))

    def test_lambda_y_zero_does_not_use_unpaired_english_pool(self):
        x, y, z = world()
        # No Y matching and no Y-based initialization: scrambling full Y pool
        # cannot affect model, provided the known pair anchors stay fixed.
        p1 = WassersteinProcrustes3(lambda_y=0, lambda_z=1,
                                    init_mode="supervised", anchor_centering="source_pool", **PARAMS).fit(
                                        x, y, x[:24], y[:24], randomness=7,
                                        samples_z=z)
        garbage = torch.randn(y.shape, generator=torch.Generator().manual_seed(99))
        p2 = WassersteinProcrustes3(lambda_y=0, lambda_z=1,
                                    init_mode="supervised", anchor_centering="source_pool", **PARAMS).fit(
                                        x, garbage, x[:24], y[:24], randomness=7,
                                        samples_z=z)
        self.assertTrue(torch.equal(p1.weight, p2.weight))
        self.assertEqual(len(p1.z_counts), PARAMS["num_iterations"])
        torch.testing.assert_close(p1.weight.T @ p1.weight, torch.eye(x.shape[1]),
                                   atol=1e-4, rtol=0)

    def test_anchor_only_has_no_unpaired_dependence(self):
        x, y, z = world()
        a = WassersteinProcrustes3(anchor_centering="paired").fit_supervised(x, y, x[:24], y[:24])
        b = WassersteinProcrustes3(anchor_centering="paired").fit_supervised(x * 100, y * 0, x[:24], y[:24])
        self.assertTrue(torch.equal(a.weight, b.weight))

    def test_invalid_lambda_y_and_pairless_supervised_init(self):
        x, y, z = world()
        with self.assertRaises(ValueError):
            WassersteinProcrustes3(lambda_y=-0.1, **PARAMS)
        with self.assertRaises(ValueError):
            WassersteinProcrustes3(init_mode="supervised", anchor_centering="source_pool", **PARAMS).fit(x,y)
        with self.assertRaises(ValueError):
            WassersteinProcrustes3(lambda_y=0, lambda_z=0, **PARAMS).fit(x,y)


if __name__ == "__main__":
    unittest.main()
