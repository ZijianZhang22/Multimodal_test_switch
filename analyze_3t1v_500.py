#!/usr/bin/env python3
"""Matched analysis for 4-hop single-visual logical-role experiment.

Conditions:
  TTTT  = all-text baseline
  VTTT  = Vision at logical role 1
  TVTT  = Vision at logical role 2
  TTVT  = Vision at logical role 3
  TTTV  = Vision at logical role 4

Primary confirmatory contrasts:
  1) VTTT vs TTTV  (same 3T+1V, same switch_count=1; role 1 vs role 4)
  2) TVTT vs TTVT  (same 3T+1V, same switch_count=2; role 2 vs role 3)

Also reports:
- Cochran's Q across the four 3T+1V schedules
- exact matched McNemar tests among all four 3T+1V schedules
- exact matched McNemar tests for each V@k vs TTTT baseline
- Holm correction within each comparison family
"""

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.contingency_tables import cochrans_q

BASELINE = "TTTT"
SINGLE_V = ["VTTT", "TVTT", "TTVT", "TTTV"]
ALL = [BASELINE] + SINGLE_V


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


def exact_mcnemar(a, b):
    # a,b are matched 0/1 correctness vectors.
    a0_b1 = int(((a == 0) & (b == 1)).sum())
    a1_b0 = int(((a == 1) & (b == 0)).sum())
    n = a0_b1 + a1_b0
    p = 1.0 if n == 0 else binomtest(
        min(a0_b1, a1_b0), n=n, p=0.5, alternative="two-sided"
    ).pvalue
    return a0_b1, a1_b0, p


def pair_row(wide, a_name, b_name):
    a = wide[a_name].astype(int)
    b = wide[b_name].astype(int)
    a0_b1, a1_b0, p = exact_mcnemar(a, b)
    return {
        "schedule_a": a_name,
        "schedule_b": b_name,
        "n_matched": len(wide),
        "accuracy_a": a.mean(),
        "accuracy_b": b.mean(),
        "difference_b_minus_a": b.mean() - a.mean(),
        "a_wrong_b_right": a0_b1,
        "a_right_b_wrong": a1_b0,
        "discordant_pairs": a0_b1 + a1_b0,
        "mcnemar_exact_p": p,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("analysis_3t1v_500"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(args.results)
    if "schedule" not in df.columns:
        df["schedule"] = df.get("logical_schedule", df.get("presentation_schedule"))
    df = df[df["schedule"].isin(ALL)].copy()
    df["correct"] = df["correct"].astype(int)

    summary = (
        df.groupby("schedule", as_index=False)
        .agg(n=("correct", "size"), correct=("correct", "sum"), accuracy=("correct", "mean"))
    )
    summary["accuracy_pct"] = 100 * summary["accuracy"]
    summary = summary.set_index("schedule").reindex(ALL).reset_index()
    summary.to_csv(args.out_dir / "accuracy_by_schedule.csv", index=False)

    wide = df.pivot_table(index="problem_id", columns="schedule", values="correct", aggfunc="first")
    wide = wide[ALL].dropna()
    wide.to_csv(args.out_dir / "matched_correctness_matrix.csv")

    # Omnibus test only among the four schedules with exactly one Vision fact.
    q = cochrans_q(wide[SINGLE_V].to_numpy())
    omnibus = pd.DataFrame([{
        "n_matched": len(wide),
        "n_single_visual_schedules": len(SINGLE_V),
        "cochrans_q_statistic": float(q.statistic),
        "df": len(SINGLE_V) - 1,
        "p_value": float(q.pvalue),
    }])
    omnibus.to_csv(args.out_dir / "cochrans_q_single_visual.csv", index=False)

    # All six pairwise comparisons among the four single-V schedules.
    role_rows = [
        pair_row(wide, a, b)
        for a, b in itertools.combinations(SINGLE_V, 2)
    ]
    role_pairs = pd.DataFrame(role_rows)
    role_pairs["holm_p"] = holm_adjust(role_pairs["mcnemar_exact_p"].tolist())
    role_pairs["significant_holm_0.05"] = role_pairs["holm_p"] < 0.05
    role_pairs.to_csv(args.out_dir / "pairwise_single_visual_mcnemar_holm.csv", index=False)

    # Each single-V role against the exact same all-text problems.
    baseline_rows = [pair_row(wide, BASELINE, s) for s in SINGLE_V]
    baseline_pairs = pd.DataFrame(baseline_rows)
    baseline_pairs["holm_p"] = holm_adjust(baseline_pairs["mcnemar_exact_p"].tolist())
    baseline_pairs["significant_holm_0.05"] = baseline_pairs["holm_p"] < 0.05
    baseline_pairs.to_csv(args.out_dir / "single_visual_vs_text_mcnemar_holm.csv", index=False)

    # Pre-specified primary contrasts: report raw exact p-values separately.
    primary = pd.DataFrame([
        pair_row(wide, "VTTT", "TTTV"),
        pair_row(wide, "TVTT", "TTVT"),
    ])
    primary.to_csv(args.out_dir / "prespecified_primary_contrasts.csv", index=False)

    print("\n=== Accuracy by schedule ===")
    print(summary[["schedule", "correct", "n", "accuracy_pct"]].to_string(index=False))

    print("\n=== Cochran's Q: four single-V schedules ===")
    print(omnibus.to_string(index=False))

    print("\n=== Pre-specified primary contrasts ===")
    show = primary.copy()
    show["difference_b_minus_a"] *= 100
    print(show[
        ["schedule_a","schedule_b","difference_b_minus_a",
         "a_wrong_b_right","a_right_b_wrong","mcnemar_exact_p"]
    ].to_string(index=False))

    print("\n=== All single-V pairwise tests (Holm corrected) ===")
    show2 = role_pairs.copy()
    show2["difference_b_minus_a"] *= 100
    print(show2.sort_values("holm_p")[
        ["schedule_a","schedule_b","difference_b_minus_a",
         "mcnemar_exact_p","holm_p","significant_holm_0.05"]
    ].to_string(index=False))

    print("\n=== Each single-V role vs TTTT (Holm corrected) ===")
    show3 = baseline_pairs.copy()
    show3["difference_b_minus_a"] *= 100
    print(show3.sort_values("schedule_b")[
        ["schedule_a","schedule_b","difference_b_minus_a",
         "mcnemar_exact_p","holm_p","significant_holm_0.05"]
    ].to_string(index=False))

    print("\nSaved:", args.out_dir)


if __name__ == "__main__":
    main()
