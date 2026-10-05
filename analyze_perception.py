#!/usr/bin/env python3
"""
Analyze the single-fact perception control.

Question:
Can the model correctly read the same spatial relation when it is presented
alone as Text vs Vision?

If visual perception is high but visual evidence fails mainly in late multi-hop
roles, that points away from basic perception and toward integration/reasoning.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def mcnemar_exact(a, b):
    a = pd.Series(a).astype(bool)
    b = pd.Series(b).astype(bool)
    n01 = int(((a == False) & (b == True)).sum())
    n10 = int(((a == True) & (b == False)).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    p = binomtest(
        min(n01, n10),
        n=n,
        p=0.5,
        alternative="two-sided",
    ).pvalue
    return n01, n10, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results_role_position/qwen3vl_perception.jsonl"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("analysis_perception"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(args.results)
    df["correct"] = df["correct"].astype(bool)

    by_modality = (
        df.groupby("source", as_index=False)
        .agg(n=("correct", "size"), accuracy=("correct", "mean"))
    )
    by_step_modality = (
        df.groupby(["logical_step", "source"], as_index=False)
        .agg(n=("correct", "size"), accuracy=("correct", "mean"))
        .sort_values(["logical_step", "source"])
    )

    by_modality.to_csv(args.out_dir / "accuracy_by_modality.csv", index=False)
    by_step_modality.to_csv(
        args.out_dir / "accuracy_by_step_and_modality.csv",
        index=False,
    )

    print("\n=== Single-fact perception accuracy by modality ===")
    print(by_modality.to_string(index=False))
    print("\n=== Single-fact perception accuracy by logical step ===")
    print(by_step_modality.to_string(index=False))

    pivot = df.pivot_table(
        index=["problem_id", "logical_step"],
        columns="source",
        values="correct",
        aggfunc="first",
    )

    rows = []
    if "T" in pivot.columns and "V" in pivot.columns:
        matched = pivot[["T", "V"]].dropna()
        n01, n10, p = mcnemar_exact(matched["T"], matched["V"])
        rows.append({
            "logical_step": "ALL",
            "text_accuracy": matched["T"].mean(),
            "vision_accuracy": matched["V"].mean(),
            "vision_minus_text": matched["V"].mean() - matched["T"].mean(),
            "text_wrong_vision_right": n01,
            "text_right_vision_wrong": n10,
            "mcnemar_p": p,
            "n": len(matched),
        })

        for step in sorted(df["logical_step"].unique()):
            step_matched = matched.loc[
                matched.index.get_level_values("logical_step") == step
            ]
            n01, n10, p = mcnemar_exact(
                step_matched["T"],
                step_matched["V"],
            )
            rows.append({
                "logical_step": int(step),
                "text_accuracy": step_matched["T"].mean(),
                "vision_accuracy": step_matched["V"].mean(),
                "vision_minus_text": (
                    step_matched["V"].mean() - step_matched["T"].mean()
                ),
                "text_wrong_vision_right": n01,
                "text_right_vision_wrong": n10,
                "mcnemar_p": p,
                "n": len(step_matched),
            })

    paired = pd.DataFrame(rows)
    paired.to_csv(args.out_dir / "paired_text_vs_vision.csv", index=False)

    print("\n=== Paired Text vs Vision perception ===")
    print(paired.to_string(index=False))
    print("\nSaved analysis to:", args.out_dir)


if __name__ == "__main__":
    main()
