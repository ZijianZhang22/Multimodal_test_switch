#!/usr/bin/env python3
"""
Analyze Round 3: logical reasoning role x physical presentation position.

Primary estimand:
  matched_delta = single-visual accuracy - all-text accuracy
for the same problem and the exact same presentation order.

If the visual penalty follows logical role after controlling for physical
position, that supports a reasoning-role-dependent modality effect.
If it follows physical position instead, the phenomenon is closer to ordinary
prompt/order bias.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results_role_position/qwen3vl_role_position.jsonl"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("analysis_role_position"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(args.results)
    df["correct"] = df["correct"].astype(int)

    single = df[df["condition"] == "single_visual"].copy()
    baseline = df[df["condition"] == "text_baseline"].copy()

    required = {
        "visual_logical_step",
        "visual_presentation_position",
        "order_id",
        "problem_id",
    }
    missing = required - set(single.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Raw 4x4 cell accuracy.
    cell_accuracy = (
        single.groupby(
            ["visual_logical_step", "visual_presentation_position"],
            as_index=False,
        )
        .agg(
            n=("correct", "size"),
            accuracy=("correct", "mean"),
        )
        .sort_values(["visual_logical_step", "visual_presentation_position"])
    )
    cell_accuracy.to_csv(args.out_dir / "cell_accuracy.csv", index=False)

    role_summary = (
        single.groupby("visual_logical_step", as_index=False)
        .agg(n=("correct", "size"), accuracy=("correct", "mean"))
    )
    position_summary = (
        single.groupby("visual_presentation_position", as_index=False)
        .agg(n=("correct", "size"), accuracy=("correct", "mean"))
    )
    role_summary.to_csv(args.out_dir / "accuracy_by_logical_role.csv", index=False)
    position_summary.to_csv(
        args.out_dir / "accuracy_by_presentation_position.csv",
        index=False,
    )

    print("\n=== Single-visual accuracy by logical role ===")
    print(role_summary.to_string(index=False))
    print("\n=== Single-visual accuracy by physical presentation position ===")
    print(position_summary.to_string(index=False))
    print("\n=== 4x4 role x physical-position cell accuracy ===")
    print(cell_accuracy.to_string(index=False))

    # Matched baseline adjustment removes difficulty caused by presentation order.
    if len(baseline):
        base = baseline[
            ["problem_id", "order_id", "correct"]
        ].rename(columns={"correct": "baseline_correct"})

        matched = single.merge(
            base,
            on=["problem_id", "order_id"],
            how="inner",
            validate="many_to_one",
        )
        matched["delta_vs_text_baseline"] = (
            matched["correct"] - matched["baseline_correct"]
        )

        matched.to_csv(
            args.out_dir / "matched_single_visual_vs_text_baseline.csv",
            index=False,
        )

        delta_cells = (
            matched.groupby(
                ["visual_logical_step", "visual_presentation_position"],
                as_index=False,
            )
            .agg(
                n=("delta_vs_text_baseline", "size"),
                mean_delta=("delta_vs_text_baseline", "mean"),
                single_visual_accuracy=("correct", "mean"),
                matched_text_accuracy=("baseline_correct", "mean"),
            )
        )
        delta_cells.to_csv(
            args.out_dir / "matched_delta_cells.csv",
            index=False,
        )

        delta_role = (
            matched.groupby("visual_logical_step", as_index=False)
            .agg(
                n=("delta_vs_text_baseline", "size"),
                mean_delta=("delta_vs_text_baseline", "mean"),
            )
        )
        delta_position = (
            matched.groupby("visual_presentation_position", as_index=False)
            .agg(
                n=("delta_vs_text_baseline", "size"),
                mean_delta=("delta_vs_text_baseline", "mean"),
            )
        )
        delta_role.to_csv(
            args.out_dir / "matched_delta_by_logical_role.csv",
            index=False,
        )
        delta_position.to_csv(
            args.out_dir / "matched_delta_by_presentation_position.csv",
            index=False,
        )

        print("\n=== Matched delta (single visual - all text) by logical role ===")
        print(delta_role.to_string(index=False))
        print("\n=== Matched delta by physical presentation position ===")
        print(delta_position.to_string(index=False))
    else:
        matched = None
        print("\nWARNING: no text_baseline rows found; matched deltas unavailable.")

    # Two-way clustered logistic regression on the single-visual conditions.
    # Reference levels are logical role 1 and physical position 1.
    try:
        model = smf.glm(
            formula=(
                "correct ~ C(visual_logical_step) "
                "+ C(visual_presentation_position)"
            ),
            data=single,
            family=sm.families.Binomial(),
        )
        result = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": single["problem_id"]},
        )

        table = pd.DataFrame({
            "coef_log_odds": result.params,
            "std_err_clustered": result.bse,
            "z": result.tvalues,
            "p_value": result.pvalues,
            "odds_ratio": np.exp(result.params),
            "ci_low_odds_ratio": np.exp(result.conf_int()[0]),
            "ci_high_odds_ratio": np.exp(result.conf_int()[1]),
        })
        table.to_csv(
            args.out_dir / "clustered_logit_role_plus_physical_position.csv"
        )
        print("\n=== Clustered logistic: logical role + physical position ===")
        print(table.to_string())
    except Exception as exc:
        print("\nWARNING: clustered logistic regression failed:", exc)

    # Optional interaction model: useful diagnostically, not the primary claim.
    try:
        interaction_model = smf.glm(
            formula=(
                "correct ~ C(visual_logical_step) "
                "* C(visual_presentation_position)"
            ),
            data=single,
            family=sm.families.Binomial(),
        )
        interaction_result = interaction_model.fit(
            cov_type="cluster",
            cov_kwds={"groups": single["problem_id"]},
        )
        interaction_table = pd.DataFrame({
            "coef_log_odds": interaction_result.params,
            "std_err_clustered": interaction_result.bse,
            "z": interaction_result.tvalues,
            "p_value": interaction_result.pvalues,
            "odds_ratio": np.exp(interaction_result.params),
        })
        interaction_table.to_csv(
            args.out_dir / "clustered_logit_role_x_physical_position.csv"
        )
    except Exception as exc:
        print("\nWARNING: interaction regression failed:", exc)

    # Heatmaps with matplotlib only.
    acc_matrix = cell_accuracy.pivot(
        index="visual_logical_step",
        columns="visual_presentation_position",
        values="accuracy",
    ).sort_index().sort_index(axis=1)

    plt.figure(figsize=(6, 5))
    plt.imshow(acc_matrix.values, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(label="Accuracy")
    plt.xticks(
        range(len(acc_matrix.columns)),
        [str(x) for x in acc_matrix.columns],
    )
    plt.yticks(
        range(len(acc_matrix.index)),
        [str(x) for x in acc_matrix.index],
    )
    plt.xlabel("Physical presentation position")
    plt.ylabel("Logical reasoning role")
    plt.title("Single-visual accuracy: role x presentation position")
    plt.tight_layout()
    plt.savefig(args.out_dir / "role_position_accuracy_heatmap.png", dpi=180)
    plt.close()

    if matched is not None:
        delta_matrix = (
            matched.groupby(
                ["visual_logical_step", "visual_presentation_position"]
            )["delta_vs_text_baseline"]
            .mean()
            .unstack()
            .sort_index()
            .sort_index(axis=1)
        )

        lim = max(0.01, float(np.abs(delta_matrix.values).max()))
        plt.figure(figsize=(6, 5))
        plt.imshow(
            delta_matrix.values,
            aspect="auto",
            vmin=-lim,
            vmax=lim,
        )
        plt.colorbar(label="Accuracy delta vs matched all-text baseline")
        plt.xticks(
            range(len(delta_matrix.columns)),
            [str(x) for x in delta_matrix.columns],
        )
        plt.yticks(
            range(len(delta_matrix.index)),
            [str(x) for x in delta_matrix.index],
        )
        plt.xlabel("Physical presentation position")
        plt.ylabel("Logical reasoning role")
        plt.title("Visual replacement effect after order matching")
        plt.tight_layout()
        plt.savefig(
            args.out_dir / "matched_delta_role_position_heatmap.png",
            dpi=180,
        )
        plt.close()

    print("\nSaved analysis to:", args.out_dir)


if __name__ == "__main__":
    main()
