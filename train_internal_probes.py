#!/usr/bin/env python3
"""
Round 5B: train grouped linear probes over extracted hidden states.

Tasks:
  target_boundary / fact_direction:
      Has the local semantic relation been encoded?
  target_boundary / variant:
      Is source modality still linearly decodable?
  target_boundary / outcome:
      Can the local state predict Visual success vs failure?
  final_prompt / fact_direction:
      Does the target fact survive into the global reasoning state?
  final_prompt / answer:
      Is the final answer linearly decodable?
  final_prompt / variant:
      Does modality identity persist globally?
  final_prompt / outcome:
      When does the global state begin to predict failure?

All train/test splits are GROUPED BY problem_id to prevent paired examples from
the same latent problem leaking across the split.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def tensor_matrix(records):
    return np.stack([
        r["feature"].float().numpy()
        if torch.is_tensor(r["feature"])
        else np.asarray(r["feature"], dtype=np.float32)
        for r in records
    ])


def fit_probe(records, label_key, seed, v_only=False):
    filtered = []
    for r in records:
        if v_only and r["variant"] != "V":
            continue
        label = (
            r["outcome_group"]
            if label_key == "outcome"
            else r[label_key]
        )
        if label is None:
            continue
        filtered.append((r, label))

    if len(filtered) < 20:
        return None

    labels = np.array([x[1] for x in filtered])
    if len(np.unique(labels)) < 2:
        return None

    X = tensor_matrix([x[0] for x in filtered])
    groups = np.array([x[0]["problem_id"] for x in filtered])

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=seed,
    )
    train_idx, test_idx = next(splitter.split(X, labels, groups=groups))

    y_train = labels[train_idx]
    y_test = labels[test_idx]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
        ),
    )
    clf.fit(X[train_idx], y_train)
    pred = clf.predict(X[test_idx])

    majority = pd.Series(y_train).value_counts().idxmax()
    majority_acc = float(np.mean(y_test == majority))

    return {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_classes": int(len(np.unique(labels))),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, pred)
        ),
        "majority_baseline": majority_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--states",
        type=Path,
        default=Path("probe_data/internal_probe_states.pt"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("analysis_probes"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.states, map_location="cpu")
    records = payload["records"]

    tasks = [
        ("target_boundary", "fact_direction", False),
        ("target_boundary", "variant", False),
        ("target_boundary", "outcome", True),
        ("final_prompt", "fact_direction", False),
        ("final_prompt", "answer", False),
        ("final_prompt", "variant", False),
        ("final_prompt", "outcome", True),
    ]

    rows = []
    roles = sorted({int(r["logical_role"]) for r in records})
    layers = sorted({int(r["layer"]) for r in records})

    for role in roles:
        for layer in layers:
            layer_role = [
                r for r in records
                if int(r["logical_role"]) == role
                and int(r["layer"]) == layer
            ]
            for site, label_key, v_only in tasks:
                subset = [r for r in layer_role if r["site"] == site]
                result = fit_probe(
                    subset,
                    label_key=label_key,
                    seed=args.seed,
                    v_only=v_only,
                )
                if result is None:
                    continue
                rows.append({
                    "logical_role": role,
                    "layer": layer,
                    "site": site,
                    "probe_target": label_key,
                    "v_only": v_only,
                    **result,
                })

    df = pd.DataFrame(rows)
    out_csv = args.out_dir / "probe_results.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== Probe results ===")
    print(df.to_string(index=False))
    print("\nSaved:", out_csv)

    for (site, target), sub in df.groupby(["site", "probe_target"]):
        plt.figure(figsize=(8, 5))
        for role in sorted(sub["logical_role"].unique()):
            g = sub[sub["logical_role"] == role].sort_values("layer")
            plt.plot(
                g["layer"],
                g["balanced_accuracy"],
                marker="o",
                label=f"role {role}",
            )
        plt.xlabel("Decoder layer")
        plt.ylabel("Balanced probe accuracy")
        plt.title(f"{site}: decode {target}")
        plt.ylim(0, 1.02)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            args.out_dir / f"probe_{site}_{target}.png",
            dpi=180,
        )
        plt.close()

    # Compact mechanistic summary for easy terminal inspection.
    summary = (
        df.sort_values(
            ["site", "probe_target", "logical_role", "balanced_accuracy"],
            ascending=[True, True, True, False],
        )
        .groupby(["site", "probe_target", "logical_role"], as_index=False)
        .first()
    )
    summary.to_csv(args.out_dir / "best_probe_layer_summary.csv", index=False)

    print("\n=== Best probe layer by task / role ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
