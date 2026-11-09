"""Paired two-way bootstrap for CLIP Rosetta3: training seeds x held-out image IDs.

Requires per-query hit NPZ files written by the updated run_rosetta3.py.
We sample the SAME seed indices and SAME held-out image IDs for both methods.
Crossed (seed x image) resampling reflects both anchor variability and test data
variability; its CI is descriptive, not a pre-registration or causal proof.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

DIRECTIONS = ("L-E", "E-L", "L-I", "I-L")
CONTRASTS = (
    ("image_only", "pairs_only"),
    ("image_only", "noise_only"),
    ("image_only", "caption_only"),
    ("joint", "text_only"),
    ("joint", "joint_noise"),
    ("joint", "joint_caption"),
    ("joint", "joint_extra_english"),
)


def group_hits(hit, ids):
    """Mean over repeated query captions of the same image; return sorted image IDs."""
    hit = np.asarray(hit, dtype=np.float64)
    ids = np.asarray(ids).astype(str)
    if hit.ndim != 1 or ids.ndim != 1 or len(hit) != len(ids):
        raise ValueError("hits and IDs must be aligned vectors")
    if len(ids) == 0 or not np.isin(hit, [0.0, 1.0]).all():
        raise ValueError("Expected non-empty binary R@1 hit array")
    groups, inverse = np.unique(ids, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(groups))
    means = np.bincount(inverse, weights=hit, minlength=len(groups)) / counts
    return groups, means


def paired_matrix(folder, left, right, direction, seeds):
    """A single S x G matrix of paired per-image hit differences."""
    key = direction.replace("-", "_")
    matrix = []
    expected_ids = None
    for seed in seeds:
        left_file = folder / f"hits_{left}_{seed}.npz"
        right_file = folder / f"hits_{right}_{seed}.npz"
        if not left_file.is_file() or not right_file.is_file():
            raise FileNotFoundError(f"Missing R@1 hit files for seed={seed}: {left_file}, {right_file}")
        with np.load(left_file, allow_pickle=False) as a, np.load(right_file, allow_pickle=False) as b:
            ida, aa = group_hits(a["hits_" + key], a["ids_" + key])
            idb, bb = group_hits(b["hits_" + key], b["ids_" + key])
        if not np.array_equal(ida, idb):
            raise ValueError(f"Non-matched test image IDs: {left} vs {right}, seed={seed}")
        if expected_ids is None:
            expected_ids = ida
        elif not np.array_equal(expected_ids, ida):
            raise ValueError("Test image IDs differ between seeds; cannot perform paired two-way bootstrap")
        matrix.append(aa - bb)
    return expected_ids, np.stack(matrix)


def crossed_bootstrap(delta, repetitions=2000, seed=2026):
    """Resample seeds and image IDs independently; return paired ΔR@1 CI.

    Seed resampling is paired across methods; image resampling preserves any
    within-image correlation caused by multiple captions.
    """
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 2 or 0 in delta.shape or repetitions < 100:
        raise ValueError("Require non-empty S x G delta matrix and >=100 bootstrap repetitions")
    s, g = delta.shape
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for b in range(repetitions):
        seed_rows = rng.integers(0, s, size=s)
        image_cols = rng.integers(0, g, size=g)
        samples[b] = delta[np.ix_(seed_rows, image_cols)].mean()
    return float(delta.mean()), np.quantile(samples, [0.025, 0.975]).tolist()


def paired_seed_sign_flip(delta):
    """Two-sided paired sign-flip p-value for variability across anchor seeds.

    Tests symmetry of per-seed paired differences; it does not establish
    generalization to other languages or datasets.
    """
    per_seed = np.asarray(delta, dtype=float).mean(axis=1)
    n = len(per_seed)
    observed = abs(per_seed.mean())
    if n <= 16:
        signs = np.asarray(list(itertools.product([-1., 1.], repeat=n)))
        stats = np.abs(signs @ per_seed / n)
        return float((stats >= observed - 1e-12).mean())
    rng = np.random.default_rng(312)
    signs = rng.choice([-1., 1.], size=(20000, n))
    stats = np.abs(signs @ per_seed / n)
    return float((1 + (stats >= observed - 1e-12).sum()) / (len(stats) + 1))


def analyze_result(folder, comparisons=CONTRASTS, repetitions=2000):
    folder = Path(folder)
    report_file = folder / "report.json"
    report = json.loads(report_file.read_text())
    configured_seeds = list(report["config"]["seeds"])
    present = {row["method"] for row in report["runs"]}
    rows = []
    for left, right in comparisons:
        if left not in present or right not in present:
            continue
        for direction in DIRECTIONS:
            ids, delta = paired_matrix(folder, left, right, direction, configured_seeds)
            mean, ci = crossed_bootstrap(delta, repetitions=repetitions)
            rows.append({
                "budget": report["config"]["budget"],
                "direction": direction,
                "left": left,
                "right": right,
                "delta_r1_pp": round(mean * 100, 5),
                "ci95_low_pp": round(ci[0] * 100, 5),
                "ci95_high_pp": round(ci[1] * 100, 5),
                "sign_flip_p_two_sided": paired_seed_sign_flip(delta),
                "n_seeds": len(configured_seeds),
                "n_test_image_ids": len(ids),
                "per_seed_delta_pp": [round(float(v * 100), 5) for v in delta.mean(axis=1)],
                "ci_method": "two-way crossed paired seed-by-image bootstrap",
                "seed_significance_warning": len(configured_seeds) < 6,
                "result_dir": str(folder),
            })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=Path("real_work_clip_native"))
    p.add_argument("--pattern", default="results_clip_*_budget_*")
    p.add_argument("--output", default="paired_ablation.csv")
    p.add_argument("--repetitions", type=int, default=2000)
    a = p.parse_args()
    all_rows = []
    for folder in sorted(a.work.glob(a.pattern)):
        if not (folder / "report.json").is_file():
            continue
        rows = analyze_result(folder, repetitions=a.repetitions)
        if rows:
            all_rows.extend(rows)
            print(f"budget={rows[0]['budget']}, {len(rows)} paired comparisons: {folder}", flush=True)
    if not all_rows:
        raise SystemExit("No matched result folders with per-query hits. Re-run with the updated experiment runner.")
    out = a.work / a.output
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = [k for k in all_rows[0] if k not in ("per_seed_delta_pp", "ci_method", "seed_significance_warning")]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: r[k] for k in keys} for r in all_rows)
    json_file = out.with_suffix(".json")
    json_file.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False))
    print(f"Wrote {len(all_rows)} rows: {out}, {json_file}")
    if any(r["seed_significance_warning"] for r in all_rows):
        print("WARNING: Fewer than 6 seeds; confidence intervals are exploratory.")


if __name__ == "__main__":
    main()