"""Unit tests for Swahili caption alignment; require no model downloads."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from prepare_swahili import (aligned_caption, build_text_fields,
                             independent_image_caption, load_swahili_csv,
                             norm_english, reconcile_missing_images)


class SwahiliTests(unittest.TestCase):
    def test_normalize_and_exact_translation(self):
        self.assertEqual(norm_english("A DOG, running!"),
                         norm_english("a dog running"))
        rows = {"abc": [("adogrunning", "A dog running.", "Mbwa anakimbia"),
                        ("adogplaying", "A dog playing.", "Mbwa anacheza")]}
        self.assertEqual(aligned_caption(rows, "abc", "A dog, running!"),
                         ("Mbwa anakimbia", True))
        self.assertEqual(independent_image_caption(rows, "abc", "A dog running."),
                         ("Mbwa anacheza", True))

    def test_nonmatching_caption_never_silently_falls_back(self):
        rows = {"abc": [("adog", "A dog.", "Mbwa")]}
        with self.assertRaisesRegex(ValueError, "No exact English-caption match"):
            aligned_caption(rows, "abc", "A cat.")
        self.assertEqual(aligned_caption(rows, "abc", "A cat.", True),
                         ("Mbwa", False))

    def test_missing_caption_id_refills_anchor_from_bridge_and_preserves_splits(self):
        keys = {
            "l_pair": ["missing"],
            "e_pair": ["missing"],
            "l_pool": ["pool"],
            "e_bridge": ["bridge1", "bridge2"],
            "i_bridge": ["bridge1", "bridge2"],
            "e_bank": ["bank"],
            "i_bank": ["bank"],
            "e_extra": ["extra"],
            "l_test": ["test"],
            "e_test": ["test"],
            "l_img_test": ["test"],
            "i_test": ["test"],
        }
        base = {}
        for key, ids in keys.items():
            base[key] = np.ones((len(ids), 4), dtype=np.float32)
            base[key + "_ids"] = np.asarray(ids)
        for dest, sources in {
            "l_fit": ("l_pair", "l_pool"),
            "e_fit": ("e_pair", "e_bridge", "e_bank"),
            "i_fit": ("i_bridge", "i_bank"),
            "e_pool": ("e_bridge",),
        }.items():
            base[dest] = np.concatenate([base[k] for k in sources])
            base[dest + "_ids"] = np.concatenate([base[k + "_ids"] for k in sources])
        sw = {
            key: [(norm_english(value), value, "Kiswahili")]
            for key, value in {
                "pool": "Pool caption", "bridge1": "Bridge caption",
                "bridge2": "Another bridge", "bank": "Bank caption",
                "extra": "Extra caption", "test": "Test caption",
            }.items()
        }
        en = {("bridge1", 0): "Bridge caption", ("test", 0): "Test caption"}
        output, audit = reconcile_missing_images(base, sw, en)
        self.assertEqual(output["l_pair_ids"].tolist(), ["bridge1"])
        self.assertEqual(output["e_pool_ids"].tolist(), ["bridge2"])
        self.assertEqual(audit["original_anchor_count"], 1)
        self.assertEqual(audit["repaired_anchor_count"], 1)
        self.assertEqual(audit["repaired_test_count"], 1)
        self.assertEqual(audit["replacement_anchor_ids_from_bridge"], ["bridge1"])

    def test_csv_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle,
                                        fieldnames=["image", "caption", "caption_SW"])
                writer.writeheader()
                writer.writerow(dict(image="abc.jpg", caption="A dog.",
                                     caption_SW="Mbwa"))
            parsed = load_swahili_csv(path)
            self.assertEqual(parsed["abc"][0][2], "Mbwa")

    def test_id_preservation_and_distinct_image_caption(self):
        base = {
            "l_pair_ids": np.array(["a"]),
            "l_pool_ids": np.array(["b"]),
            "l_fit_ids": np.array(["a", "b"]),
            "l_test_ids": np.array(["c"]),
            "l_img_test_ids": np.array(["c"]),
        }
        english = {("a", 0): "A dog", ("c", 0): "A cat"}
        sw = {
            "a": [("adog", "A dog", "Mbwa")],
            "b": [("abird", "A bird", "Ndege")],
            "c": [("acat", "A cat", "Paka"),
                  ("acatplaying", "A cat playing", "Paka anacheza")],
        }
        fields, audit = build_text_fields(base, english, sw)
        self.assertEqual(fields["l_fit"], ["Mbwa", "Ndege"])
        self.assertEqual(fields["l_test"], ["Paka"])
        self.assertEqual(fields["l_img_test"], ["Paka anacheza"])
        self.assertEqual(audit["test_distinct_img_caption"], 1)


if __name__ == "__main__":
    unittest.main()
