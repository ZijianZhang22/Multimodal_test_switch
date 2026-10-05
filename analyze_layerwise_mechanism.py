#!/usr/bin/env python3
"""
Analyze Round-4 layer-wise mechanism pilot.

Outputs:
- representation similarity by logical role / outcome / layer
- patch rescue rate by logical role / layer
- best rescue layer per logical role
- plots for representation divergence and causal rescue
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
        default=Path("results_mechanism/layerwise_mechanism.jsonl"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("analysis_mechanism"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(args.results)
    if df.empty:
        raise RuntimeError("No mechanism results found.")

    representation = (
        df.groupby(
            ["logical_role", "group", "layer"],
            as_index=False,
        )
        .agg(
            n=("cosine_similarity", "size"),
            mean_cosine=("cosine_similarity", "mean"),
            mean_normalized_l2=("normalized_l2", "mean"),
        )
        .sort_values(["logical_role", "group", "layer"])
    )
    representation.to_csv(
        args.out_dir / "representation_by_role_group_layer.csv",
        index=False,
    )

    patched = df[df["patched_tested"] == True].copy()
    if len(patched):
        patched["patched_correct"] = patched["patched_correct"].astype(bool)
        rescue = (
            patched.groupby(
                ["logical_role", "layer"],
                as_index=False,
            )
            .agg(
                n=("patched_correct", "size"),
                rescue_rate=("patched_correct", "mean"),
            )
            .sort_values(["logical_role", "layer"])
        )
        rescue.to_csv(
            args.out_dir / "patch_rescue_by_role_layer.csv",
            index=False,
        )

        best_rows = []
        for role, group in rescue.groupby("logical_role"):
            best = group.sort_values(
                ["rescue_rate", "n", "layer"],
                ascending=[False, False, True],
            ).iloc[0]
            best_rows.append({
                "logical_role": int(role),
                "best_layer": int(best["layer"]),
                "rescue_rate": float(best["rescue_rate"]),
                "n": int(best["n"]),
            })

        best = pd.DataFrame(best_rows)
        best.to_csv(
            args.out_dir / "best_patch_layer_by_role.csv",
            index=False,
        )

        print("\n=== Patch rescue rate by role/layer ===")
        print(rescue.to_string(index=False))
        print("\n=== Best patch layer by role ===")
        print(best.to_string(index=False))
    else:
        rescue = pd.DataFrame()
        print("\nNo patched rows found.")

    print("\n=== Representation similarity (head) ===")
    print(representation.head(30).to_string(index=False))

    # One plot per logical role for representation similarity.
    for role in sorted(df["logical_role"].unique()):
        sub = representation[
            representation["logical_role"] == role
        ]

        plt.figure(figsize=(7, 4))
        for group_name in ["success", "failure"]:
            g = sub[sub["group"] == group_name]
            if len(g):
                plt.plot(
                    g["layer"],
                    g["mean_cosine"],
                    marker="o",
                    label=group_name,
                )
        plt.xlabel("Decoder layer")
        plt.ylabel("Cosine similarity: Text baseline vs Vision")
        plt.title(f"Representation similarity — logical role {role}")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            args.out_dir
            / f"representation_similarity_role{role}.png",
            dpi=180,
        )
        plt.close()

    if len(rescue):
        for role in sorted(rescue["logical_role"].unique()):
            g = rescue[rescue["logical_role"] == role]
            plt.figure(figsize=(7, 4))
            plt.plot(
                g["layer"],
                g["rescue_rate"],
                marker="o",
            )
            plt.xlabel("Patched decoder layer")
            plt.ylabel("Rescue rate")
            plt.title(
                f"Matched Text-state patch rescue — logical role {role}"
            )
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(
                args.out_dir / f"patch_rescue_role{role}.png",
                dpi=180,
            )
            plt.close()

    # Compact pair counts so we know the pilot support.
    pair_counts = (
        df[[
            "logical_role",
            "group",
            "problem_id",
            "order_id",
            "visual_example_id",
        ]]
        .drop_duplicates()
        .groupby(["logical_role", "group"], as_index=False)
        .size()
        .rename(columns={"size": "n_pairs"})
    )
    pair_counts.to_csv(
        args.out_dir / "pair_counts.csv",
        index=False,
    )

    print("\n=== Pair counts ===")
    print(pair_counts.to_string(index=False))
    print("\nSaved analysis to:", args.out_dir)


if __name__ == "__main__":
    main()
