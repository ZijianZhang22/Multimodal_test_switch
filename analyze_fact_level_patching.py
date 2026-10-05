#!/usr/bin/env python3
"""
Analyze fact-level causal patching.

The main quantity is rescue_rate(control, role, layer).

Strong evidence for fact-specific causal integration would look like:
  matched_fact rescue >> same_example_other_fact
  matched_fact rescue >> unrelated_same_answer
  matched_fact rescue >> unrelated_other_answer
especially for late logical roles in the mid/late causal window.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binomtest


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def paired_exact_binary(a, b):
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
        default=Path("results_fact_patching/fact_level_patching.jsonl"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("analysis_fact_patching"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(args.results)
    if df.empty:
        raise RuntimeError("No patching results found.")

    df["patched_correct"] = df["patched_correct"].astype(bool)

    summary = (
        df.groupby(
            ["logical_role", "layer", "control"],
            as_index=False,
        )
        .agg(
            n=("patched_correct", "size"),
            rescue_rate=("patched_correct", "mean"),
        )
        .sort_values(["logical_role", "layer", "control"])
    )
    summary.to_csv(
        args.out_dir / "rescue_by_role_layer_control.csv",
        index=False,
    )

    best = (
        summary.sort_values(
            ["logical_role", "control", "rescue_rate", "layer"],
            ascending=[True, True, False, True],
        )
        .groupby(["logical_role", "control"], as_index=False)
        .first()
    )
    best.to_csv(
        args.out_dir / "best_layer_by_role_control.csv",
        index=False,
    )

    print("\n=== Rescue rates ===")
    print(summary.to_string(index=False))
    print("\n=== Best layer by role/control ===")
    print(best.to_string(index=False))

    # Paired matched-fact vs each control at each role/layer.
    comparisons = []
    key_cols = ["problem_id", "order_id", "logical_role", "layer"]
    pivot = df.pivot_table(
        index=key_cols,
        columns="control",
        values="patched_correct",
        aggfunc="first",
    )

    if "matched_fact" in pivot.columns:
        for control in [
            "same_example_other_fact",
            "unrelated_same_answer",
            "unrelated_other_answer",
        ]:
            if control not in pivot.columns:
                continue

            matched = pivot[["matched_fact", control]].dropna()
            for (role, layer), group in matched.groupby(
                level=["logical_role", "layer"]
            ):
                n01, n10, p = paired_exact_binary(
                    group["matched_fact"],
                    group[control],
                )
                comparisons.append({
                    "logical_role": int(role),
                    "layer": int(layer),
                    "control": control,
                    "n": len(group),
                    "matched_rescue": group["matched_fact"].mean(),
                    "control_rescue": group[control].mean(),
                    "delta": (
                        group["matched_fact"].mean()
                        - group[control].mean()
                    ),
                    "matched_wrong_control_right": n01,
                    "matched_right_control_wrong": n10,
                    "paired_exact_p": p,
                })

    comp = pd.DataFrame(comparisons)
    comp.to_csv(
        args.out_dir / "matched_vs_controls_paired_tests.csv",
        index=False,
    )
    if len(comp):
        print("\n=== Matched fact vs controls ===")
        print(comp.to_string(index=False))

    # One rescue plot per logical role.
    for role in sorted(summary["logical_role"].unique()):
        sub = summary[summary["logical_role"] == role]
        plt.figure(figsize=(8, 5))
        for control in sorted(sub["control"].unique()):
            g = sub[sub["control"] == control].sort_values("layer")
            plt.plot(
                g["layer"],
                g["rescue_rate"],
                marker="o",
                label=control,
            )
        plt.xlabel("Patched decoder layer")
        plt.ylabel("Rescue rate")
        plt.title(f"Fact-level causal patching — logical role {role}")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            args.out_dir / f"fact_patch_rescue_role{role}.png",
            dpi=180,
        )
        plt.close()

    print("\nSaved analysis to:", args.out_dir)


if __name__ == "__main__":
    main()
