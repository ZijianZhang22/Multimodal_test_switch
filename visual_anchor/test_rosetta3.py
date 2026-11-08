import unittest

import numpy as np

try:
    import torch
    from unpaired_rosetta import WassersteinProcrustes
    from rosetta3 import WassersteinProcrustes3
except ImportError as error:  # the NumPy-only environment of run.py does not need these tests
    raise unittest.SkipTest(f"Rosetta environment missing: {error}")

from run_rosetta3 import check_overlap

D = 12
SMALL = dict(num_clusters=8, num_restarts=3, batch_size=10_000, num_iterations=8)


def world(seed=0, n=600, noise=0.05):
    """Anisotropic latent; X, Y and Z are rotations of it (Z already in Y's coordinates), disjoint samples."""
    r = np.random.default_rng(seed)
    qx, qy = (np.linalg.qr(r.normal(size=(D, D)))[0] for _ in range(2))
    scale = np.linspace(.3, 2.5, D)

    def latent(m):
        return r.normal(size=(m, D)) * scale

    def obs(lat, q):
        return torch.tensor(lat @ q + noise * r.normal(size=lat.shape), dtype=torch.float32)

    pair = latent(20)
    test = latent(200)
    return dict(x=obs(latent(n), qx), y=obs(latent(n), qy), z=obs(latent(n), qy),
                px=obs(pair, qx), py=obs(pair, qy), tx=obs(test, qx), ty=obs(test, qy),
                truth=torch.tensor(qx.T @ qy, dtype=torch.float32))


def r_at_1(model, w):
    sims = model.transform(w["tx"]) @ model.transform_target(w["ty"], model.mean_y).T
    return (sims.argmax(1) == torch.arange(len(sims))).float().mean().item()


class Rosetta3Tests(unittest.TestCase):
    def setUp(self):
        self.w = world()

    def official(self, pairs=True):
        w = self.w
        return WassersteinProcrustes(**SMALL).fit(
            w["x"], w["y"], w["px"] if pairs else None, w["py"] if pairs else None, randomness=3).weight

    def test_lambda_zero_is_official_bit_for_bit(self):
        w = self.w
        for pairs in (True, False):
            reference = self.official(pairs)
            for kwargs in ({"lambda_z": 0.0}, {"lambda_z": 0.7}):
                z = w["z"] if kwargs["lambda_z"] == 0 else None
                ours = WassersteinProcrustes3(**kwargs, **SMALL).fit(
                    w["x"], w["y"], w["px"] if pairs else None, w["py"] if pairs else None, randomness=3,
                    samples_z=z).weight
                self.assertTrue(torch.equal(ours, reference), f"pairs={pairs}, {kwargs}")

    def test_third_term_is_used_and_keeps_orthogonality(self):
        w = self.w
        noise = torch.randn(len(w["z"]), D, generator=torch.Generator().manual_seed(1))
        model = WassersteinProcrustes3(lambda_z=1.0, **SMALL).fit(
            w["x"], w["y"], w["px"], w["py"], randomness=3, samples_z=noise)
        self.assertFalse(torch.equal(model.weight, self.official()))
        torch.testing.assert_close(model.weight.T @ model.weight, torch.eye(D), atol=1e-4, rtol=0)
        self.assertEqual(len(model.z_counts), SMALL["num_iterations"])

    def test_row_order_of_z_does_not_matter(self):
        w = self.w
        order = torch.randperm(len(w["z"]), generator=torch.Generator().manual_seed(2))
        fits = [WassersteinProcrustes3(lambda_z=1.0, **SMALL).fit(
            w["x"], w["y"], w["px"], w["py"], randomness=3, samples_z=z).weight for z in (w["z"], w["z"][order])]
        torch.testing.assert_close(fits[0], fits[1], atol=1e-4, rtol=0)

    def test_recovers_known_rotation_with_third_pool(self):
        w = self.w
        base = WassersteinProcrustes3(**SMALL).fit(w["x"], w["y"], w["px"], w["py"], randomness=3)
        ours = WassersteinProcrustes3(lambda_z=1.0, **SMALL).fit(
            w["x"], w["y"], w["px"], w["py"], randomness=3, samples_z=w["z"])
        self.assertGreater(r_at_1(ours, w), .8)
        self.assertGreater(r_at_1(ours, w), r_at_1(base, w) - .05)
        self.assertLess(torch.linalg.norm(ours.weight - w["truth"]).item(), 1.0)

    def test_cached_initialization_is_bit_identical(self):
        w = self.w
        cache = {}
        for pairs in (True, False):
            px, py = (w["px"], w["py"]) if pairs else (None, None)
            for lam, z in ((0.0, None), (1.0, w["z"]), (0.0, None)):
                plain = WassersteinProcrustes3(lambda_z=lam, **SMALL).fit(
                    w["x"], w["y"], px, py, randomness=3, samples_z=z).weight
                cached = WassersteinProcrustes3(lambda_z=lam, init_cache=cache, **SMALL).fit(
                    w["x"], w["y"], px, py, randomness=3, samples_z=z).weight
                self.assertTrue(torch.equal(plain, cached), f"pairs={pairs}, lambda={lam}")
        self.assertEqual(len(cache), 2)

    def test_overlap_guard(self):
        d = {"e_bank_ids": np.array(["a", "b"]), "e_pool_ids": np.array(["b", "c"])}
        with self.assertRaises(ValueError):
            check_overlap(d, ["visual"], allow=False)
        self.assertEqual(check_overlap(d, ["visual"], allow=True), 1)
        self.assertEqual(check_overlap(d, ["rosetta", "extra_text"], allow=False), 1)


if __name__ == "__main__":
    unittest.main()
