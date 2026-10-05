#!/usr/bin/env python3
"""
Analyze Experiment 1:
- accuracy by schedule
- accuracy by switch count
- paired high-switch vs low-switch differences
- exact McNemar tests
- accuracy-vs-switch plot
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binomtest


def load_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return pd.DataFrame(rows)


def mcnemar_exact(a, b):
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

    by_schedule = (
        df.groupby(["schedule", "switch_count"], as_index=False)
        .agg(
            n=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_latency=("latency_sec", "mean"),
            mean_input_tokens=("input_tokens", "mean"),
            mean_output_tokens=("output_tokens", "mean"),
        )
        .sort_values(["switch_count", "schedule"])
    )

    by_switch = (
        df.groupby("switch_count", as_index=False)
        .agg(
            n=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_latency=("latency_sec", "mean"),
        )
        .sort_values("switch_count")
    )

    print("\n=== Accuracy by schedule ===")
    print(by_schedule.to_string(index=False))

    print("\n=== Accuracy by switch count ===")
    print(by_switch.to_string(index=False))

    by_schedule.to_csv(
        args.out_dir / "accuracy_by_schedule.csv",
        index=False,
    )
    by_switch.to_csv(
        args.out_dir / "accuracy_by_switch_count.csv",
        index=False,
    )

    low = (
        df[df["switch_count"] == 1]
        .groupby("problem_id")["correct"]
        .mean()
    )
    high = (
        df[df["switch_count"] == 3]
        .groupby("problem_id")["correct"]
        .mean()
    )

    paired = pd.concat(
        [
            low.rename("low_1switch"),
            high.rename("high_3switch"),
        ],
        axis=1,
    ).dropna()

    paired["delta_low_minus_high"] = (
        paired["low_1switch"] - paired["high_3switch"]
    )

    paired.to_csv(args.out_dir / "paired_low_vs_high.csv")

    print(
        "\nMean paired accuracy difference (1-switch - 3-switch):",
        paired["delta_low_minus_high"].mean(),
    )

    pivot = df.pivot_table(
        index="problem_id",
        columns="schedule",
        values="correct",
        aggfunc="first",
    )

    for low_schedule, high_schedule in [
        ("TTVV", "TVTV"),
        ("VVTT", "VTVT"),
    ]:
        if (
            low_schedule in pivot.columns
            and high_schedule in pivot.columns
        ):
            matched = pivot[
                [low_schedule, high_schedule]
            ].dropna()

            n01, n10, p_value = mcnemar_exact(
                matched[low_schedule].astype(bool),
                matched[high_schedule].astype(bool),
            )

            print(
                f"McNemar {low_schedule} vs {high_schedule}: "
                f"low_wrong/high_right={n01}, "
                f"low_right/high_wrong={n10}, "
                f"p={p_value:.6g}"
            )

    plt.figure(figsize=(6, 4))
    plt.plot(
        by_switch["switch_count"],
        by_switch["accuracy"],
        marker="o",
    )
    plt.xlabel("Number of V↔T switches")
    plt.ylabel("Accuracy")
    plt.title("Cross-modal evidence switching pilot")
    plt.xticks(sorted(by_switch["switch_count"].unique()))
    plt.ylim(0, 1)
    plt.tight_layout()

    figure_path = args.out_dir / "accuracy_vs_switches.png"
    plt.savefig(figure_path, dpi=180)

    print("Saved plot:", figure_path)


if __name__ == "__main__":
    main()
