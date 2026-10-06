#!/usr/bin/env python3
"""
Analyze variable-hop single-Visual depth-scaling experiments.

Primary estimand:
  matched_delta(k,H,model)
    = single-Visual correctness - matched all-Text correctness

Each single-Visual example is matched to the exact same latent problem and
presentation order.

Outputs include:
- all-Text calibration by hop/model
- matched delta by logical role and normalized depth
- early/middle/late depth buckets
- clustered regression of matched delta on normalized depth, hop count, model
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_many(paths):
    rows = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return pd.DataFrame(rows)


def depth_bucket(x):
    if pd.isna(x):
        return None
    if x <= 1 / 3:
        return "early"
    if x <= 2 / 3:
        return "middle"
    return "late"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="One or more result JSONL files (8B and/or 32B).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("analysis_depth_scaling"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_many(args.results)
    if df.empty:
        raise RuntimeError("No results found.")

    df["correct"] = df["correct"].astype(int)
    if "model_label" not in df.columns:
        df["model_label"] = df.get("model", "unknown")

    baseline = df[df["condition"] == "text_baseline"].copy()
    single = df[df["condition"] == "single_visual"].copy()

    required = {
        "problem_id", "order_id", "hop_count",
        "visual_logical_step", "normalized_logical_depth",
    }
    missing = required - set(single.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    calibration = (
        baseline.groupby(["model_label", "hop_count"], as_index=False)
        .agg(n=("correct", "size"), all_text_accuracy=("correct", "mean"))
        .sort_values(["model_label", "hop_count"])
    )
    calibration.to_csv(args.out_dir / "all_text_calibration.csv", index=False)

    base = baseline[
        ["model_label", "problem_id", "order_id", "hop_count", "correct"]
    ].rename(columns={"correct": "baseline_correct"})

    matched = single.merge(
        base,
        on=["model_label", "problem_id", "order_id", "hop_count"],
        how="inner",
        validate="many_to_one",
    )
    matched["delta_vs_text_baseline"] = (
        matched["correct"] - matched["baseline_correct"]
    )
    matched["depth_bucket"] = matched["normalized_logical_depth"].map(depth_bucket)
    matched["relative_physical_position"] = (
        (matched["visual_presentation_position"] - 1)
        / (matched["hop_count"] - 1)
    )

    matched.to_csv(args.out_dir / "matched_examples.csv", index=False)

    by_role = (
        matched.groupby(
            [
                "model_label", "hop_count", "visual_logical_step",
                "normalized_logical_depth",
            ],
            as_index=False,
        )
        .agg(
            n=("correct", "size"),
            single_visual_accuracy=("correct", "mean"),
            matched_text_accuracy=("baseline_correct", "mean"),
            mean_delta=("delta_vs_text_baseline", "mean"),
        )
        .sort_values(
            ["model_label", "hop_count", "visual_logical_step"]
        )
    )
    by_role.to_csv(
        args.out_dir / "matched_delta_by_role_depth.csv",
        index=False,
    )

    by_bucket = (
        matched.groupby(
            ["model_label", "hop_count", "depth_bucket"],
            as_index=False,
            observed=True,
        )
        .agg(
            n=("correct", "size"),
            mean_delta=("delta_vs_text_baseline", "mean"),
            single_visual_accuracy=("correct", "mean"),
            matched_text_accuracy=("baseline_correct", "mean"),
        )
    )
    by_bucket.to_csv(
        args.out_dir / "matched_delta_by_depth_bucket.csv",
        index=False,
    )

    # Clustered OLS on matched difference. The estimand is already paired, so
    # this is a compact trend test rather than a replacement for the raw curves.
    try:
        formula = (
            "delta_vs_text_baseline ~ normalized_logical_depth * hop_count"
        )
        if matched["model_label"].nunique() > 1:
            formula += " * C(model_label)"

        result = smf.ols(formula=formula, data=matched).fit(
            cov_type="cluster",
            cov_kwds={"groups": matched["problem_id"]},
        )
        reg = pd.DataFrame({
            "coef": result.params,
            "std_err_clustered": result.bse,
            "t": result.tvalues,
            "p_value": result.pvalues,
            "ci_low": result.conf_int()[0],
            "ci_high": result.conf_int()[1],
        })
        reg.to_csv(
            args.out_dir / "clustered_depth_trend_regression.csv"
        )
        print("\n=== Clustered matched-delta depth trend ===")
        print(reg.to_string())
    except Exception as exc:
        print("WARNING: regression failed:", exc)

    print("\n=== All-Text calibration ===")
    print(calibration.to_string(index=False))
    print("\n=== Matched delta by role/depth ===")
    print(by_role.to_string(index=False))

    # One plot per model. Each line is one hop count.
    for model_label in sorted(by_role["model_label"].unique()):
        sub = by_role[by_role["model_label"] == model_label]
        plt.figure(figsize=(8, 5))
        for hop_count in sorted(sub["hop_count"].unique()):
            g = sub[sub["hop_count"] == hop_count].sort_values(
                "normalized_logical_depth"
            )
            plt.plot(
                g["normalized_logical_depth"],
                g["mean_delta"],
                marker="o",
                label=f"{hop_count}-hop",
            )
        plt.axhline(0, linewidth=1)
        plt.xlabel("Normalized logical depth")
        plt.ylabel("Accuracy delta vs matched all-Text")
        plt.title(f"Role-dependent visual penalty — {model_label}")
        plt.legend()
        plt.tight_layout()
        safe = str(model_label).replace("/", "_")
        plt.savefig(
            args.out_dir / f"depth_scaling_{safe}.png",
            dpi=180,
        )
        plt.close()

    print("\nSaved analysis to:", args.out_dir)


if __name__ == "__main__":
    main()
