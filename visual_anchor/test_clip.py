import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core import PCA, fit_space_pcas, procrustes
from prepare_real import ENCODERS, bridge_diagnostics, model_names_for, resolve_revisions


def rotation(r, d):
    return np.linalg.qr(r.normal(size=(d, d)))[0]


class SharedPCATests(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(0)
        self.base = r.normal(size=(300, 20)) * np.linspace(.2, 3, 20)
        self.gap = r.normal(size=20) * 5  # CLIP-like modality offset

    def test_offset_spaces_share_coordinates(self):
        e, i = PCA.fit_shared(8, [self.base, self.base + self.gap])
        np.testing.assert_allclose(e.proj, i.proj)
        np.testing.assert_allclose(e.transform(self.base), i.transform(self.base + self.gap), atol=1e-10)
        v = procrustes(i.transform(self.base + self.gap), e.transform(self.base))
        np.testing.assert_allclose(v, np.eye(8), atol=1e-8)  # the bridge collapses to the identity

    def test_separate_pca_does_not_share_coordinates(self):
        rot = rotation(np.random.default_rng(1), 20)
        e, i = PCA(8).fit(self.base), PCA(8).fit(self.base @ rot)
        self.assertGreater(np.abs(e.proj - i.proj).max(), 1e-3)

    def test_fit_space_pcas_default_is_unchanged(self):
        d = {f"{s}_fit": self.base + k for k, s in enumerate("lei")}
        old = {s: PCA(8).fit(d[s + "_fit"]) for s in "lei"}
        new = fit_space_pcas(d, 8)
        for s in "lei":
            np.testing.assert_array_equal(old[s].proj, new[s].proj)
            np.testing.assert_array_equal(old[s].mean, new[s].mean)

    def test_joint_clip_pca_uses_identical_mean_and_projection(self):
        from core import fit_space_pcas
        d = {"l_fit": self.base, "e_fit": self.base,
             "i_fit": self.base + self.gap}
        pcas = fit_space_pcas(d, 8, joint_ei=True)
        self.assertIs(pcas["e"], pcas["i"])
        np.testing.assert_array_equal(pcas["e"].mean, pcas["i"].mean)
        np.testing.assert_array_equal(pcas["e"].proj, pcas["i"].proj)

    def test_shared_needs_equal_widths(self):
        with self.assertRaises(ValueError):
            PCA.fit_shared(4, [self.base, self.base[:, :10]])


class RevisionLockTests(unittest.TestCase):
    def test_new_lock_roundtrip_and_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "model_revisions.json"
            clip = model_names_for("clip", "openai/clip-vit-base-patch32")
            self.assertEqual(clip["e"], clip["i"])
            self.assertEqual(clip["l"], ENCODERS["default"]["l"])  # L stays the monolingual encoder
            revs = resolve_revisions(lock, clip, lambda name: "sha-" + name)
            self.assertEqual(resolve_revisions(lock, clip, lambda name: 1 / 0), revs)  # no second lookup
            with self.assertRaises(ValueError):
                resolve_revisions(lock, ENCODERS["default"], lambda name: "x")

    def test_legacy_lock_only_for_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "model_revisions.json"
            lock.write_text(json.dumps({"l": "a", "e": "b", "i": "c"}))
            self.assertEqual(resolve_revisions(lock, ENCODERS["default"], lambda n: 1 / 0), {"l": "a", "e": "b", "i": "c"})
            with self.assertRaises(ValueError):
                resolve_revisions(lock, model_names_for("clip", "openai/clip-vit-base-patch32"), lambda n: "x")


class BridgeDiagnosticTests(unittest.TestCase):
    def test_clip_like_bridge_is_near_perfect(self):
        r = np.random.default_rng(2)
        latent = lambda n: r.normal(size=(n, 24)) * np.linspace(.3, 3, 24)
        gap = r.normal(size=24) * 4
        data = {}
        for name, n in (("bridge", 400), ("bank", 200), ("test", 100)):
            z = latent(n)
            ids = np.array([f"{name}{k}" for k in range(n)])
            data[f"e_{name}"], data[f"i_{name}"] = z + .05 * r.normal(size=z.shape), z + gap + .05 * r.normal(size=z.shape)
            data[f"e_{name}_ids"] = data[f"i_{name}_ids"] = ids
        data["e_fit"] = np.vstack([data["e_bridge"], data["e_bank"]])
        data["i_fit"] = np.vstack([data["i_bridge"], data["i_bank"]])
        out = bridge_diagnostics(data, "clip", dim=16)
        variants = out["variants"]
        self.assertGreater(variants["clip_native"]["R@1"], .9)
        self.assertGreater(variants["shared_pca_identity"]["R@1"], .9)
        self.assertLess(variants["shared_pca_map_distance_from_identity"], 1.0)
        self.assertIn("R@1", out)  # old top-level format kept
        self.assertEqual(set(bridge_diagnostics(data, "default", dim=16)["variants"]), {"separate_pca_procrustes"})


class ClipHelperTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            from transformers import CLIPConfig, CLIPModel
        except ImportError as error:
            self.skipTest(f"transformers missing: {error}")
        self.torch = torch
        cfg = CLIPConfig(
            text_config=dict(vocab_size=99, hidden_size=32, intermediate_size=37, num_hidden_layers=2,
                             num_attention_heads=4, max_position_embeddings=77, eos_token_id=2, bos_token_id=1,
                             pad_token_id=0),
            vision_config=dict(image_size=32, patch_size=8, hidden_size=32, intermediate_size=37, num_hidden_layers=2,
                               num_attention_heads=4),
            projection_dim=16)
        self.model = CLIPModel(cfg).eval()

    def test_text_and_image_features_are_clip_embeddings(self):
        import clip_encoders
        torch = self.torch

        def tokenizer(texts, **kwargs):  # stub: CLIP-style ids ending in the eos token
            ids = torch.tensor([[1] + [3 + (len(t) + k) % 90 for k in range(5)] + [2] for t in texts])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        class Processor:
            def __call__(self, images, return_tensors):
                return {"pixel_values": torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in images])}

        pics = [np.random.default_rng(k).normal(size=(3, 32, 32)) for k in range(3)]
        texts = ["a dog", "two cats", "a red car"]
        t = clip_encoders.encode_texts(self.model, tokenizer, texts, "cpu")
        i = clip_encoders.encode_images(self.model, Processor(), pics, "cpu")
        self.assertEqual(t.shape, (3, 16))
        self.assertEqual(i.shape, (3, 16))
        with torch.inference_mode():
            full = self.model(**tokenizer(texts), pixel_values=Processor()(pics, "pt")["pixel_values"])
        unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
        np.testing.assert_allclose(unit(t), full.text_embeds.numpy(), atol=1e-5)
        np.testing.assert_allclose(unit(i), full.image_embeds.numpy(), atol=1e-5)


class SupervisedBaselineTests(unittest.TestCase):
    def test_pairs_only_procrustes(self):
        try:
            import torch
            from rosetta3 import WassersteinProcrustes3
        except ImportError as error:
            self.skipTest(f"Rosetta environment missing: {error}")
        r = np.random.default_rng(3)
        q = rotation(r, 10)
        lat = r.normal(size=(500, 10)) * np.linspace(.3, 2, 10)
        x, y = torch.tensor(lat, dtype=torch.float32), torch.tensor(lat @ q, dtype=torch.float32)
        model = WassersteinProcrustes3(anchor_centering="paired").fit_supervised(x, y, x[:40], y[:40])
        np.testing.assert_allclose(model.weight.numpy(), q, atol=1e-4)
        with self.assertRaises(ValueError):
            WassersteinProcrustes3(anchor_centering="paired").fit_supervised(x, y, x[:0], y[:0])


if __name__ == "__main__":
    unittest.main()
