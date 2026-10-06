#!/usr/bin/env python3
"""
Grouped linear probes for arbitrary-hop internal-state datasets.

Supports:
- fact-direction decoding
- modality-source decoding
- raw Visual outcome prediction
- paired V-T outcome prediction
- exact accumulated-state decoding for state1 ... stateH

All splits are grouped by problem_id.
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
            else r.get(label_key)
        )
        if label is not None:
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


def paired_delta_records(records, site):
    keyed = {}
    for r in records:
        if r["site"] != site:
            continue
        key = (
            r["problem_id"],
            r.get("order_id", "o00"),
            int(r["logical_role"]),
            int(r["layer"]),
        )
        keyed.setdefault(key, {})[r["variant"]] = r

    out = []
    for _, pair in keyed.items():
        if "T" not in pair or "V" not in pair:
            continue
        t = pair["T"]
        v = pair["V"]
        vt = v["feature"].float()
        tt = t["feature"].float()

        out.append({
            "problem_id": v["problem_id"],
            "order_id": v.get("order_id", "o00"),
            "hop_count": int(v.get("hop_count", 4)),
            "logical_role": int(v["logical_role"]),
            "normalized_logical_depth": v.get("normalized_logical_depth"),
            "layer": int(v["layer"]),
            "relative_layer_depth": v.get("relative_layer_depth"),
            "site": site,
            "variant": "DELTA",
            "outcome_group": v["outcome_group"],
            "feature": vt - tt,
        })
    return out


def infer_max_hop(records):
    values = [int(r.get("hop_count", 0)) for r in records]
    if max(values, default=0) > 0:
        return max(values)

    state_indices = []
    for key in records[0].keys():
        if key.startswith("state") and key.endswith("_x"):
            try:
                state_indices.append(int(key[5:-2]))
            except Exception:
                pass
    return max(state_indices, default=4)


def make_tasks(max_hop):
    tasks = [
        ("target_boundary", "fact_direction", False),
        ("target_boundary", "variant", False),
        ("target_boundary", "outcome", True),
        ("final_prompt", "fact_direction", False),
        ("final_prompt", "answer", False),
        ("final_prompt", "variant", False),
        ("final_prompt", "outcome", True),
    ]
    for k in range(1, max_hop + 1):
        for coord in ["x", "y"]:
            tasks.append(("final_prompt", f"state{k}_{coord}", False))
            tasks.append(("final_prompt", f"state{k}_{coord}", True))
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, required=True)
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
    max_hop = infer_max_hop(records)
    tasks = make_tasks(max_hop)

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
                subset = [
                    r for r in layer_role
                    if r["site"] == site and r.get(label_key) is not None
                ]
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
                    "feature_type": "hidden_state",
                    **result,
                })

    delta_records = paired_delta_records(records, site="final_prompt")
    for role in roles:
        for layer in layers:
            subset = [
                r for r in delta_records
                if int(r["logical_role"]) == role
                and int(r["layer"]) == layer
            ]
            result = fit_probe(
                subset,
                label_key="outcome",
                seed=args.seed,
                v_only=False,
            )
            if result is None:
                continue
            rows.append({
                "logical_role": role,
                "layer": layer,
                "site": "final_prompt_delta_V_minus_T",
                "probe_target": "outcome",
                "v_only": True,
                "feature_type": "paired_delta",
                **result,
            })

    df = pd.DataFrame(rows)
    out_csv = args.out_dir / "probe_results.csv"
    df.to_csv(out_csv, index=False)

    for (site, target, v_only), sub in df.groupby(
        ["site", "probe_target", "v_only"]
    ):
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
        subset_name = "V-only" if bool(v_only) else "all matched variants"
        plt.title(f"{site}: decode {target} ({subset_name})")
        plt.ylim(0, 1.02)
        plt.legend()
        plt.tight_layout()

        safe_site = site.replace("/", "_")
        safe_target = target.replace("/", "_")
        suffix = "Vonly" if bool(v_only) else "all"
        plt.savefig(
            args.out_dir / f"probe_{safe_site}_{safe_target}_{suffix}.png",
            dpi=180,
        )
        plt.close()

    summary = (
        df.sort_values(
            [
                "site", "probe_target", "v_only",
                "logical_role", "balanced_accuracy",
            ],
            ascending=[True, True, True, True, False],
        )
        .groupby(
            ["site", "probe_target", "v_only", "logical_role"],
            as_index=False,
        )
        .first()
    )
    summary.to_csv(
        args.out_dir / "best_probe_layer_summary.csv",
        index=False,
    )

    focus_rows = []
    for role in roles:
        for coord in ["x", "y"]:
            target = f"state{role}_{coord}"
            sub = df[
                (df["site"] == "final_prompt")
                & (df["probe_target"] == target)
                & (df["v_only"] == True)
                & (df["logical_role"] == role)
            ].copy()
            focus_rows.append(sub)

    focus = (
        pd.concat(focus_rows, ignore_index=True)
        if focus_rows else pd.DataFrame()
    )
    focus.to_csv(
        args.out_dir / "target_role_state_probe_curves.csv",
        index=False,
    )

    outcome_compare = df[
        (df["probe_target"] == "outcome")
        & (
            (df["site"] == "final_prompt")
            | (df["site"] == "final_prompt_delta_V_minus_T")
        )
    ].copy()
    outcome_compare.to_csv(
        args.out_dir / "outcome_probe_raw_vs_delta.csv",
        index=False,
    )

    print("Saved:", out_csv)
    print("Max hop inferred:", max_hop)
    print("Mechanism-focused outputs:")
    print(" ", args.out_dir / "target_role_state_probe_curves.csv")
    print(" ", args.out_dir / "outcome_probe_raw_vs_delta.csv")


if __name__ == "__main__":
    main()
