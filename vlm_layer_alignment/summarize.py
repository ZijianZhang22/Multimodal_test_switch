"""Paired held-out CV-Bench accuracy/valid-output comparison across independent routes."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest


def read_predictions(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate example ID in {path}")
    return {r["id"]: r for r in rows}


def contrast(before, after, bootstrap=2000, seed=42):
    if set(before) != set(after):
        raise ValueError("Different test image IDs; paired evaluation is required")
    ids = sorted(before)
    if any(before[k]["answer"] != after[k]["answer"] or before[k]["task"] != after[k]["task"] for k in ids):
        raise ValueError("Target label or task changed")
    b = np.array([int(before[k]["correct"]) for k in ids], np.float64)
    a = np.array([int(after[k]["correct"]) for k in ids], np.float64)
    vb = np.array([int(before[k]["valid"]) for k in ids], np.float64)
    va = np.array([int(after[k]["valid"]) for k in ids], np.float64)
    delta = a - b
    # If an image is duplicated, resample by image IDs rather than question rows.
    # Protocol enforces unique image filenames per group, and sample IDs are unique.
    rng = np.random.default_rng(seed)
    boots = [float(delta[rng.integers(0, len(delta), len(delta))].mean()) for _ in range(bootstrap)]
    discordant_gain = int(((a == 1) & (b == 0)).sum())
    discordant_loss = int(((a == 0) & (b == 1)).sum())
    disagree = discordant_gain + discordant_loss
    p = float(binomtest(discordant_gain, disagree, .5).pvalue) if disagree else 1.0
    return dict(n=len(ids), before_pct=round(100*b.mean(), 3), after_pct=round(100*a.mean(), 3),
                delta_pp=round(100*delta.mean(), 3),
                ci95_low_pp=round(100*np.quantile(boots, .025), 3),
                ci95_high_pp=round(100*np.quantile(boots, .975), 3),
                mc_nemar_exact_p=p, improved=discordant_gain, worsened=discordant_loss,
                invalid_before=int((1-vb).sum()), invalid_after=int((1-va).sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs")
    p.add_argument("--language", default="zh")
    p.add_argument("--bootstrap", type=int, default=2000)
    a = p.parse_args()
    root = Path(a.root); lang = a.language
    rows = []
    frozen = root / f"frozen_{lang}"
    if (frozen / "selection_by_method.json").exists():
        selected = json.loads((frozen / "selection_by_method.json").read_text())
        before = read_predictions(frozen / f"test_baseline_{lang}.csv")
        for method, best in selected.items():
            after = read_predictions(frozen / f"test_{best['name']}.csv")
            rows.append({"route": f"frozen_{method}", "language": lang,
                         "config": best["name"], **contrast(before, after, a.bootstrap)})
    for label in ("lora_ce", "lora_alignment"):
        folder = root / f"{label}_{lang}"
        p0, p1 = folder / f"test_baseline_{lang}.csv", folder / f"test_lora_{lang}.csv"
        if p0.is_file() and p1.is_file():
            rows.append({"route": label, "language": lang, "config": str(folder),
                         **contrast(read_predictions(p0), read_predictions(p1), a.bootstrap)})
            p0en, p1en = folder / "test_baseline_en.csv", folder / "test_lora_en.csv"
            if p0en.exists() and p1en.exists():
                rows.append({"route": label, "language": "en", "config": str(folder),
                             **contrast(read_predictions(p0en), read_predictions(p1en), a.bootstrap)})
    if not rows:
        raise ValueError("No completed held-out predictions found")
    dest = root / f"vlm_alignment_comparison_{lang}.csv"
    with dest.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    print("Wrote", dest)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()