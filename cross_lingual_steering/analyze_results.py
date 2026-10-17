#!/usr/bin/env python3
"""Summarize EN/ZH vector geometry and paired visual accuracy with bootstrap CIs."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def cosine(x, y):
    x, y = x.double().flatten(), y.double().flatten()
    return float(torch.dot(x, y) / (x.norm() * y.norm()).clamp_min(1e-12))


def vector_report(vectors_file, out_dir):
    obj = torch.load(vectors_file, map_location="cpu", weights_only=True)
    rows = []
    for concept, ls in obj["vectors"].items():
        for layer, vecs in ls.items():
            rows.append({
                "concept": concept, "layer": int(layer),
                "cos_en_zh": cosine(vecs["en"], vecs["zh"]),
                "cos_en_split": cosine(vecs["en_a"], vecs["en_b"]),
                "cos_zh_split": cosine(vecs["zh_a"], vecs["zh_b"]),
                "norm_en": float(vecs["en"].norm()),
                "norm_zh": float(vecs["zh"].norm()),
            })
    write_csv(out_dir / "vector_geometry.csv", rows)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def paired_interval(delta, seed, n_boot=2000):
    """Bootstrap paired example-level accuracy differences, in percentage points."""
    v = np.asarray(delta, dtype=np.float64)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = v[rng.integers(0, len(v), size=len(v))].mean() * 100
    return float(v.mean() * 100), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def accuracy_report(results_file, out_dir, seed):
    with Path(results_file).open(encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))
    # Always compare identical (image, language, task) examples to their no-steering output.
    baseline = {}
    groups = defaultdict(list)
    for row in records:
        k = (row["id"], row["task"], row["prompt_lang"])
        if row["condition"] == "baseline":
            if k in baseline:
                raise ValueError(f"Duplicate baseline for {k}")
            baseline[k] = int(row["correct"])
        else:
            g = (row["split"], row["task"], row["prompt_lang"],
                 row["vector_lang"], row["condition"], row["layer"],
                 row["alpha"], row["mask_type"])
            groups[g].append(row)

    output = []
    for key, rows in sorted(groups.items()):
        differences = []
        correct = []
        seen = set()
        for r in rows:
            k = (r["id"], r["task"], r["prompt_lang"])
            if k not in baseline:
                raise ValueError(f"Missing matched baseline: {k}")
            if k in seen:
                raise ValueError(f"Duplicate intervention example: {k}, {key}")
            seen.add(k)
            c = int(r["correct"])
            correct.append(c)
            differences.append(c - baseline[k])
        gain, low, high = paired_interval(differences, seed)
        output.append(dict(zip(("split", "task", "prompt_lang", "vector_lang",
                                "condition", "layer", "alpha", "mask_type"), key),
                           n=len(rows), accuracy_pct=round(np.mean(correct) * 100, 3),
                           baseline_pct=round(np.mean([baseline[
                               (r["id"], r["task"], r["prompt_lang"])] for r in rows]) * 100, 3),
                           gain_pp=round(gain, 3), ci_low_pp=round(low, 3),
                           ci_high_pp=round(high, 3)))
    write_csv(out_dir / "paired_accuracy.csv", output)
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vectors", default="outputs/pilot/vectors.pt")
    p.add_argument("--results", default="outputs/pilot/results_val.csv")
    p.add_argument("--out", default="outputs/pilot/analysis")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    out = Path(a.out)
    geometry = vector_report(Path(a.vectors), out)
    gains = accuracy_report(Path(a.results), out, a.seed)
    print(json.dumps({"geometry": geometry, "accuracy": gains[:12],
                      "n_accuracy_rows": len(gains)}, indent=2, ensure_ascii=False))
    print(f"Full tables: {out / 'vector_geometry.csv'} and {out / 'paired_accuracy.csv'}")


if __name__ == "__main__":
    main()
