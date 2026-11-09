"""Paired, matched-ID E5 Chinese vs Swahili difference-in-differences.

Each method contrast is first evaluated within a language, then the two
languages are compared using matched training-seed indices and test image IDs.
The CI describes this dataset/encoder and is not a causal test of language.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_ablation import CONTRASTS, DIRECTIONS, paired_matrix, crossed_bootstrap, paired_seed_sign_flip


def language_comparisons(zh_folder, sw_folder, repetitions=2000):
    a = json.loads((zh_folder / "report.json").read_text())
    b = json.loads((sw_folder / "report.json").read_text())
    if a["config"]["budget"] != b["config"]["budget"]:
        raise ValueError("Chinese/Swahili budgets do not match")
    seeds = list(a["config"]["seeds"])
    if seeds != list(b["config"]["seeds"]):
        raise ValueError("Chinese/Swahili seeds do not match")
    keys = ("dim", "anchor_centering", "image_centering", "ablation_lambda_z", "bridge")
    for key in keys:
        if a["config"][key] != b["config"][key]:
            raise ValueError(f"Mismatch in {key}")
    for seed in seeds:
        with np.load(zh_folder / f"pairs_only_{seed}.npz", allow_pickle=False) as z, \
             np.load(sw_folder / f"pairs_only_{seed}.npz", allow_pickle=False) as s:
            if not np.array_equal(z["anchor_indices"], s["anchor_indices"]):
                raise ValueError("Anchor indices differ despite supposedly matched datasets")
    out = []
    for left, right in CONTRASTS:
        for direction in DIRECTIONS:
            ids_zh, dz = paired_matrix(zh_folder, left, right, direction, seeds)
            ids_sw, ds = paired_matrix(sw_folder, left, right, direction, seeds)
            if not np.array_equal(ids_zh, ids_sw):
                raise ValueError("Chinese and Swahili test image IDs differ")
            delta = dz - ds
            mean, interval = crossed_bootstrap(delta, repetitions=repetitions)
            out.append({
                "budget": a["config"]["budget"], "direction": direction,
                "left": left, "right": right,
                "zh_delta_pp": round(100 * dz.mean(), 5),
                "sw_delta_pp": round(100 * ds.mean(), 5),
                "difference_zh_minus_sw_pp": round(100 * mean, 5),
                "ci95_low_pp": round(100 * interval[0], 5),
                "ci95_high_pp": round(100 * interval[1], 5),
                "sign_flip_p_two_sided": paired_seed_sign_flip(delta),
                "n_seeds": len(seeds), "matched_test_image_ids": len(ids_zh),
            })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=Path("real_work_clip_native"))
    p.add_argument("--zh-prefix", required=True)
    p.add_argument("--sw-prefix", required=True)
    p.add_argument("--budgets", type=int, nargs="+", default=[10, 25, 50])
    p.add_argument("--repetitions", type=int, default=2000)
    p.add_argument("--output", default="language_diff_e5.csv")
    a = p.parse_args()
    rows = []
    for budget in a.budgets:
        zh = a.work / f"results_{a.zh_prefix}_budget_{budget}"
        sw = a.work / f"results_{a.sw_prefix}_budget_{budget}"
        if not (zh / "report.json").is_file() or not (sw / "report.json").is_file():
            raise FileNotFoundError(f"Cannot compare incomplete budget={budget}: {zh}, {sw}")
        rows.extend(language_comparisons(zh, sw, repetitions=a.repetitions))
    out = a.work / a.output
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("Matched-language difference in differences:", out)


if __name__ == "__main__":
    main()