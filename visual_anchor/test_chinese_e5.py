"""Unit tests for Chinese/Swahili frozen E5 same-encoder control."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare_chinese_e5 import chinese_fields, check_shared_protocol


class ChineseE5Tests(unittest.TestCase):
    def test_fields_use_english_translated_indices(self):
        arrays = {name + "_ids": np.array(ids) for name, ids in {
            "l_pair": ["p", "p2"], "l_pool": ["b"],
            "l_test": ["t"], "l_img_test": ["t", "u"],
            "l_fit": ["p", "p2", "b"],
        }.items()}
        en = {(key, 0): "en0" for key in ["p", "p2", "b", "t", "u"]}
        en[("t", 1)] = "en1"
        goog = {(key, 0): f"zh-{key}-0" for key in ["p", "p2", "b", "t", "u"]}
        goog[("t", 1)] = "zh-t-1"
        human = {("t", 0): "human-t-0"}
        texts, counts = chinese_fields(arrays, goog, human, en)
        self.assertEqual(texts["l_pair"], ["zh-p-0", "zh-p2-0"])
        self.assertEqual(texts["l_pool"], ["zh-b-0"])
        self.assertEqual(texts["l_test"], ["human-t-0"])
        self.assertEqual(texts["l_img_test"], ["zh-t-1", "zh-u-0"])
        self.assertEqual(texts["l_fit"], ["zh-p-0", "zh-p2-0", "zh-b-0"])
        self.assertEqual(counts["img_caption_index_0_fallback"], 1)
        self.assertEqual(counts["img_caption_index_1"], 1)

    def test_missing_caption_fails_closed(self):
        ids = {key + "_ids": np.array(["im"]) for key in
               ("l_pair", "l_pool", "l_test", "l_img_test")}
        ids["l_fit_ids"] = np.array(["im", "im"])
        with self.assertRaisesRegex(ValueError, "Chinese caption"):
            chinese_fields(ids, {}, {}, {("im", 0): "en"})

    def test_validate_shared_preserves_ei_and_all_ids(self):
        rng = np.random.default_rng(9)
        groups = {
            "l_pair": ["p"], "e_pair": ["p"], "l_pool": ["l"],
            "e_bridge": ["br"], "i_bridge": ["br"],
            "e_bank": ["bk"], "i_bank": ["bk"], "e_extra": ["ex"],
            "l_test": ["t"], "e_test": ["t"],
            "l_img_test": ["t"], "i_test": ["t"],
        }
        sw = {}
        for key, ids in groups.items():
            sw[key] = rng.normal(size=(len(ids), 8)).astype(np.float32)
            sw[key + "_ids"] = np.array(ids)
        for dest, keys in {"l_fit": ["l_pair", "l_pool"],
                           "e_fit": ["e_pair", "e_bridge", "e_bank"],
                           "i_fit": ["i_bridge", "i_bank"],
                           "e_pool": ["e_bridge"]}.items():
            sw[dest] = np.concatenate([sw[k] for k in keys])
            sw[dest + "_ids"] = np.concatenate([sw[k + "_ids"] for k in keys])
        zh = dict(sw)
        for k in ("l_pair", "l_pool", "l_test", "l_img_test", "l_fit"):
            zh[k] = sw[k] + 1.0
        check_shared_protocol(zh, sw)
        bad = dict(zh)
        bad["i_test"] = bad["i_test"] + .001
        with self.assertRaisesRegex(ValueError, "Non-Chinese"):
            check_shared_protocol(bad, sw)
        bad = dict(zh)
        bad["l_pair_ids"] = np.array(["changed"])
        with self.assertRaisesRegex(ValueError, "Non-Chinese"):
            check_shared_protocol(bad, sw)


if __name__ == "__main__":
    unittest.main()