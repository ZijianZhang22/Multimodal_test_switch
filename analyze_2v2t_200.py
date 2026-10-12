#!/usr/bin/env python3
"""Matched significance tests for the six 2T/2V schedules in a 4-hop task."""

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

SCHEDULES = ["TTVV", "TVTV", "TVVT", "VTTV", "VTVT", "VVTT"]


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def holm_adjust(pvals):
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    out = [None] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, (idx, p) in enumerate(indexed):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        out[idx] = running
    return out


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("analysis_2v2t_200"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(args.results)
    if "schedule" not in df.columns:
        df["schedule"] = df.get("logical_schedule", df.get("presentation_schedule"))
    df = df[df["schedule"].isin(SCHEDULES)].copy()
    df["correct"] = df["correct"].astype(int)

    summary = (
        df.groupby("schedule", as_index=False)
        .agg(n=("correct", "size"), correct=("correct", "sum"), accuracy=("correct", "mean"))
    )
    summary["accuracy_pct"] = 100 * summary["accuracy"]
    summary = summary.set_index("schedule").reindex(SCHEDULES).reset_index()
    summary.to_csv(args.out_dir / "accuracy_by_schedule.csv", index=False)

    wide = df.pivot_table(index="problem_id", columns="schedule", values="correct", aggfunc="first")
    wide = wide[SCHEDULES].dropna()
    wide.to_csv(args.out_dir / "matched_correctness_matrix.csv")

    rows = []
    for a, b in itertools.combinations(SCHEDULES, 2):
        ga = wide[a].astype(int)
        gb = wide[b].astype(int)
        a0_b1 = int(((ga == 0) & (gb == 1)).sum())
        a1_b0 = int(((ga == 1) & (gb == 0)).sum())
        p = exact_mcnemar(a0_b1, a1_b0)
        rows.append({
            "schedule_a": a,
            "schedule_b": b,
            "n_matched": len(wide),
            "accuracy_a": ga.mean(),
            "accuracy_b": gb.mean(),
            "difference_b_minus_a": gb.mean() - ga.mean(),
            "a_wrong_b_right": a0_b1,
            "a_right_b_wrong": a1_b0,
            "discordant_pairs": a0_b1 + a1_b0,
            "mcnemar_exact_p": p,
        })

    pair = pd.DataFrame(rows)
    pair["holm_p"] = holm_adjust(pair["mcnemar_exact_p"].tolist())
    pair["significant_holm_0.05"] = pair["holm_p"] < 0.05
    pair.to_csv(args.out_dir / "pairwise_mcnemar_holm.csv", index=False)

    print("\n=== Accuracy by 2T/2V schedule ===")
    print(summary[["schedule", "correct", "n", "accuracy_pct"]].to_string(index=False))

    print("\n=== Pairwise matched exact McNemar tests ===")
    show = pair.sort_values("holm_p")[
        ["schedule_a", "schedule_b", "difference_b_minus_a",
         "a_wrong_b_right", "a_right_b_wrong", "mcnemar_exact_p",
         "holm_p", "significant_holm_0.05"]
    ].copy()
    show["difference_b_minus_a"] *= 100
    print(show.to_string(index=False))

    print("\nSaved:", args.out_dir)


if __name__ == "__main__":
    main()
