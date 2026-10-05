#!/usr/bin/env python3
"""
Round-2 analysis for the full 16-schedule factorial experiment.

This script separates:
1) schedule-level performance
2) switch-count effects
3) vision-count effects
4) position-specific modality effects (V1..V4)
5) transition-direction effects (T->V vs V->T)
6) paired within-problem contrasts
7) a clustered logistic regression with problem-level clustered SEs

Important note:
With four binary modality positions, some derived variables are collinear.
For the main regression we use position indicators V1..V4 plus switch_count.
Direction-specific effects are reported separately with paired summaries.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binomtest


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def add_schedule_features(df):
    df = df.copy()

    def features(schedule):
        return {
            "V1": int(schedule[0] == "V"),
            "V2": int(schedule[1] == "V"),
            "V3": int(schedule[2] == "V"),
            "V4": int(schedule[3] == "V"),
            "vision_count": schedule.count("V"),
            "text_count": schedule.count("T"),
            "vision_fraction": schedule.count("V") / len(schedule),
            "switch_count_calc": sum(
                schedule[i] != schedule[i + 1]
                for i in range(len(schedule) - 1)
            ),
            "tv_transitions": sum(
                schedule[i] == "T" and schedule[i + 1] == "V"
                for i in range(len(schedule) - 1)
            ),
            "vt_transitions": sum(
                schedule[i] == "V" and schedule[i + 1] == "T"
                for i in range(len(schedule) - 1)
            ),
            "starts_with": schedule[0],
            "ends_with": schedule[-1],
        }

    feat = pd.DataFrame([features(s) for s in df["schedule"]], index=df.index)

    for col in feat.columns:
        if col not in df.columns:
            df[col] = feat[col]

    if "switch_count" not in df.columns:
        df["switch_count"] = df["switch_count_calc"]

    return df


def mcnemar_exact(a, b):
    a = pd.Series(a).astype(bool)
    b = pd.Series(b).astype(bool)

    n01 = int(((a == False) & (b == True)).sum())
    n10 = int(((a == True) & (b == False)).sum())
    n = n01 + n10

    if n == 0:
        return n01, n10, 1.0

    p_value = binomtest(
        min(n01, n10),
        n=n,
        p=0.5,
        alternative="two-sided",
    ).pvalue

    return n01, n10, p_value


def paired_position_effect(df, position):
    """
    Average within-problem effect of making one position Vision while averaging
    over every setting of the other three positions.
    """
    pos_col = f"V{position}"

    per_problem = (
        df.groupby(["problem_id", pos_col])["correct"]
        .mean()
        .unstack()
        .dropna()
    )

    per_problem["vision_minus_text"] = per_problem[1] - per_problem[0]
    return per_problem


def paired_group_effect(df, mask_a, mask_b, name_a, name_b):
    a = df[mask_a].groupby("problem_id")["correct"].mean().rename(name_a)
    b = df[mask_b].groupby("problem_id")["correct"].mean().rename(name_b)

    out = pd.concat([a, b], axis=1).dropna()
    out[f"{name_a}_minus_{name_b}"] = out[name_a] - out[name_b]
    return out


def fit_clustered_logit(df):
    """
    Primary decomposition:
      correct ~ V1 + V2 + V3 + V4 + switch_count

    This asks whether switch count adds explanatory power after controlling for
    which reasoning positions are visual.
    """
    X = df[["V1", "V2", "V3", "V4", "switch_count"]].astype(float)
    X = sm.add_constant(X)
    y = df["correct"].astype(int)

    model = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(),
    )

    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["problem_id"]},
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

    return result, table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/qwen3vl_results.jsonl"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("analysis"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(args.results)
    df["correct"] = df["correct"].astype(bool)
    df = add_schedule_features(df)

    # 1) Schedule-level results
    by_schedule = (
        df.groupby(
            [
                "schedule",
                "switch_count",
                "vision_count",
                "starts_with",
                "ends_with",
                "tv_transitions",
                "vt_transitions",
            ],
            as_index=False,
        )
        .agg(
            n=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_latency=("latency_sec", "mean"),
            mean_input_tokens=("input_tokens", "mean"),
            mean_output_tokens=("output_tokens", "mean"),
        )
        .sort_values(["vision_count", "switch_count", "schedule"])
    )

    # 2) Aggregate summaries
    by_switch = (
        df.groupby("switch_count", as_index=False)
        .agg(n=("correct", "size"), accuracy=("correct", "mean"))
        .sort_values("switch_count")
    )

    by_vision_count = (
        df.groupby("vision_count", as_index=False)
        .agg(n=("correct", "size"), accuracy=("correct", "mean"))
        .sort_values("vision_count")
    )

    by_start = (
        df.groupby("starts_with", as_index=False)
        .agg(n=("correct", "size"), accuracy=("correct", "mean"))
    )

    by_end = (
        df.groupby("ends_with", as_index=False)
        .agg(n=("correct", "size"), accuracy=("correct", "mean"))
    )

    print("\n=== Accuracy by schedule ===")
    print(by_schedule.to_string(index=False))

    print("\n=== Accuracy by switch count ===")
    print(by_switch.to_string(index=False))

    print("\n=== Accuracy by number of visual facts ===")
    print(by_vision_count.to_string(index=False))

    print("\n=== Accuracy by starting modality ===")
    print(by_start.to_string(index=False))

    print("\n=== Accuracy by ending modality ===")
    print(by_end.to_string(index=False))

    by_schedule.to_csv(args.out_dir / "accuracy_by_schedule.csv", index=False)
    by_switch.to_csv(args.out_dir / "accuracy_by_switch_count.csv", index=False)
    by_vision_count.to_csv(args.out_dir / "accuracy_by_vision_count.csv", index=False)
    by_start.to_csv(args.out_dir / "accuracy_by_start_modality.csv", index=False)
    by_end.to_csv(args.out_dir / "accuracy_by_end_modality.csv", index=False)

    # 3) Position-specific paired effects
    position_rows = []
    for position in range(1, 5):
        paired = paired_position_effect(df, position)
        paired.to_csv(args.out_dir / f"paired_position_V{position}.csv")

        effect = paired["vision_minus_text"].mean()
        position_rows.append({
            "position": position,
            "mean_accuracy_if_vision": paired[1].mean(),
            "mean_accuracy_if_text": paired[0].mean(),
            "paired_vision_minus_text": effect,
            "n_problems": len(paired),
        })

    position_effects = pd.DataFrame(position_rows)
    position_effects.to_csv(args.out_dir / "position_effects.csv", index=False)

    print("\n=== Paired position effects: Vision minus Text ===")
    print(position_effects.to_string(index=False))

    # 4) Start/end paired marginal effects
    start_effect = paired_group_effect(
        df,
        df["starts_with"] == "V",
        df["starts_with"] == "T",
        "vision_start",
        "text_start",
    )
    end_effect = paired_group_effect(
        df,
        df["ends_with"] == "V",
        df["ends_with"] == "T",
        "vision_end",
        "text_end",
    )

    start_effect.to_csv(args.out_dir / "paired_start_effect.csv")
    end_effect.to_csv(args.out_dir / "paired_end_effect.csv")

    print(
        "\nPaired start effect (Vision-start - Text-start):",
        start_effect["vision_start_minus_text_start"].mean(),
    )
    print(
        "Paired end effect (Vision-end - Text-end):",
        end_effect["vision_end_minus_text_end"].mean(),
    )

    # 5) Direction-matched reversal pairs.
    # These are especially interpretable because each pair has the same count
    # of visual facts and same switch count, but opposite modality direction/order.
    pivot = df.pivot_table(
        index="problem_id",
        columns="schedule",
        values="correct",
        aggfunc="first",
    )

    reversal_pairs = [
        ("TTVV", "VVTT"),
        ("TVVT", "VTTV"),
        ("TVTV", "VTVT"),
        ("TTTV", "VTTT"),
        ("TVVV", "VVVT"),
        ("TTVT", "VTVT"),
        ("TVTT", "VTTV"),
    ]

    reversal_rows = []
    for a, b in reversal_pairs:
        if a not in pivot.columns or b not in pivot.columns:
            continue

        matched = pivot[[a, b]].dropna()
        n01, n10, p_value = mcnemar_exact(
            matched[a],
            matched[b],
        )

        reversal_rows.append({
            "schedule_a": a,
            "schedule_b": b,
            "accuracy_a": matched[a].mean(),
            "accuracy_b": matched[b].mean(),
            "a_minus_b": matched[a].mean() - matched[b].mean(),
            "a_wrong_b_right": n01,
            "a_right_b_wrong": n10,
            "mcnemar_p": p_value,
            "n": len(matched),
        })

    reversal_df = pd.DataFrame(reversal_rows)
    reversal_df.to_csv(args.out_dir / "reversal_pair_tests.csv", index=False)

    print("\n=== Reversal-pair tests ===")
    if len(reversal_df):
        print(reversal_df.to_string(index=False))

    # 6) Main clustered logistic model
    try:
        _, regression_table = fit_clustered_logit(df)
        regression_table.to_csv(
            args.out_dir / "clustered_logit_position_plus_switch.csv"
        )
        print("\n=== Clustered logistic regression ===")
        print(regression_table.to_string())
        print(
            "\nInterpretation: V1..V4 estimate position-specific visual effects; "
            "switch_count asks whether switching still matters after controlling "
            "for which positions are visual."
        )
    except Exception as exc:
        print("\nWARNING: clustered logistic regression failed:", exc)

    # 7) Plots
    plt.figure(figsize=(7, 4))
    plt.plot(
        by_vision_count["vision_count"],
        by_vision_count["accuracy"],
        marker="o",
    )
    plt.xlabel("Number of visual facts")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. visual evidence count")
    plt.xticks(sorted(by_vision_count["vision_count"].unique()))
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(args.out_dir / "accuracy_vs_vision_count.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(
        by_switch["switch_count"],
        by_switch["accuracy"],
        marker="o",
    )
    plt.xlabel("Number of V↔T switches")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. cross-modal switch count")
    plt.xticks(sorted(by_switch["switch_count"].unique()))
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(args.out_dir / "accuracy_vs_switches.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(
        position_effects["position"].astype(str),
        position_effects["paired_vision_minus_text"],
    )
    plt.axhline(0)
    plt.xlabel("Reasoning step")
    plt.ylabel("Paired accuracy effect: Vision - Text")
    plt.title("Where does visual evidence help or hurt?")
    plt.tight_layout()
    plt.savefig(args.out_dir / "position_visual_effects.png", dpi=180)
    plt.close()

    print("\nSaved analysis to:", args.out_dir)


if __name__ == "__main__":
    main()
