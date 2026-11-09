"""Fast NumPy-only tests for matched Z pools and crossed paired bootstrap."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_ablation import crossed_bootstrap, group_hits, paired_matrix, paired_seed_sign_flip, analyze_result


class StatsTests(unittest.TestCase):
    def test_grouped_captions(self):
        ids, scores = group_hits([1, 0, 1, 1], ["b", "a", "b", "a"])
        self.assertEqual(ids.tolist(), ["a", "b"])
        np.testing.assert_allclose(scores, [0.5, 1.0])

    def test_crossed_bootstrap_zero_difference(self):
        point, interval = crossed_bootstrap(np.zeros((7, 33)), repetitions=100)
        self.assertEqual(point, 0.0)
        np.testing.assert_array_equal(interval, [0, 0])
        self.assertEqual(paired_seed_sign_flip(np.zeros((7, 33))), 1.0)

    def test_consistent_improvement(self):
        delta = np.ones((7, 25))
        point, interval = crossed_bootstrap(delta, repetitions=100)
        self.assertEqual(point, 1.0)
        np.testing.assert_array_equal(interval, [1, 1])
        self.assertLess(paired_seed_sign_flip(delta), 0.05)

    def test_pair_alignment_and_missing_seed_detection(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for seed in [0, 1, 2, 3, 4, 5]:
                for meth, hit in [("joint", [1, 0, 1]), ("text_only", [0, 0, 1])]:
                    np.savez(d / f"hits_{meth}_{seed}.npz", hits_L_E=hit, ids_L_E=["a", "b", "c"])
            ids, delta = paired_matrix(d, "joint", "text_only", "L-E", list(range(6)))
            self.assertEqual(ids.tolist(), ["a", "b", "c"])
            np.testing.assert_array_equal(delta[0], [1, 0, 0])
            bad = d / "hits_text_only_0.npz"
            np.savez(bad, hits_L_E=[1, 1, 1], ids_L_E=["bad", "b", "c"])
            with self.assertRaisesRegex(ValueError, "Non-matched"):
                paired_matrix(d, "joint", "text_only", "L-E", [0])

    def test_full_synthetic_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            methods = ["image_only", "pairs_only"]
            for s in range(6):
                for method in methods:
                    values = {}
                    for direction in ("L-E", "E-L", "L-I", "I-L"):
                        key = direction.replace("-", "_")
                        values["hits_" + key] = [1, 0, 1] if method == "image_only" else [0, 0, 1]
                        values["ids_" + key] = ["x", "y", "z"]
                    np.savez_compressed(folder / f"hits_{method}_{s}.npz", **values)
            (folder / "report.json").write_text(json.dumps({
                "config": {"budget": 25, "seeds": list(range(6))},
                "runs": [{"method": m} for m in methods]
            }))
            results = analyze_result(folder, comparisons=[("image_only", "pairs_only")], repetitions=100)
            self.assertEqual(len(results), 4)
            self.assertAlmostEqual(results[0]["delta_r1_pp"], 100 / 3, places=4)
            self.assertEqual(results[0]["n_seeds"], 6)


class ZControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from run_rosetta3 import ABLATIONS, third_pool, check_overlap
        except ImportError as e:
            raise unittest.SkipTest(f"torch/Rosetta missing: {e}")
        cls.ablation = ABLATIONS
        cls.third_pool = staticmethod(third_pool)
        cls.check_overlap = staticmethod(check_overlap)

    def test_joint_control_arm_parameters(self):
        for name, source in (("joint", "visual"), ("joint_caption", "bank_text"),
                             ("joint_noise", "gaussian"), ("joint_extra_english", "extra_text")):
            self.assertEqual(self.ablation[name], ("supervised", 1., 1., source))

    def test_equal_z_counts_and_matched_caption_indices(self):
        rng = np.random.default_rng(42)
        # Simulates Swahili image availability filtering: image-bank length differs from extra-English length.
        z = {"i_bank":rng.normal(size=(8,4)), "e_bank":rng.normal(size=(8,4)),
             "e_extra":rng.normal(size=(5,4))}
        outputs = {m:self.third_pool(m,z,np.eye(4),seed=6) for m in
                   ("visual","bank_text","gaussian","extra_text")}
        self.assertEqual({a.shape[0] for a in outputs.values()}, {5})
        # Use index fingerprints to ensure the visual and caption control used the SAME rows.
        paired = {"i_bank":np.arange(8)[:,None].repeat(4,1).astype(float),
                  "e_bank":np.arange(8)[:,None].repeat(4,1).astype(float),
                  "e_extra":rng.normal(size=(5,4))}
        np.testing.assert_array_equal(self.third_pool("visual",paired,np.eye(4),8),
                                      self.third_pool("bank_text",paired,np.eye(4),8))

    def test_overlap_ignored_only_for_unrelated_methods(self):
        d={"e_bank_ids":np.array(["a"]), "e_pool_ids":np.array(["a"])}
        with self.assertRaisesRegex(ValueError,"bank image IDs"):
            self.check_overlap(d,["joint_caption"],False)
        self.assertEqual(self.check_overlap(d,["joint_noise"],False),1)


if __name__ == '__main__':
    unittest.main()