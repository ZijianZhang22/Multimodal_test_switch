#!/usr/bin/env python3
"""
Round 6: grouped linear probes for exact intermediate reasoning states.

NEW in Round 6
--------------
The earlier probes showed that the target fact itself remained highly
decodable, even on late-Visual failures. That makes "fact forgotten" unlikely.

This version therefore probes the EXACT accumulated reasoning states

    s_k = f_1 + ... + f_k = (x_k, y_k)

instead of only probing the final answer direction.  We decode x_k and y_k
separately because the exact coordinate is richer than the answer class:
(1, 0) and (3, 0) are both EAST, but they are different reasoning states.

We also add a paired-difference outcome probe:

    delta_h = h_visual - h_text

for matched examples.  This reduces the risk that an outcome probe is merely
learning static problem difficulty rather than a Vision-specific failure state.

All train/test splits are GROUPED BY problem_id to avoid leakage from matched
variants of the same latent problem.
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


def paired_delta_records(records, site):
    """
    Build one V-T residual feature per matched pair.

    The label is the Visual run outcome (success/failure).  Because the Text and
    Visual examples share the same latent problem/order, delta_h emphasizes the
    modality-induced representational change rather than static difficulty.
    """
    keyed = {}
    for r in records:
        if r["site"] != site:
            continue
        key = (
            r["problem_id"],
            r["order_id"],
            int(r["logical_role"]),
            int(r["layer"]),
        )
        keyed.setdefault(key, {})[r["variant"]] = r

    out = []
    for key, pair in keyed.items():
        if "T" not in pair or "V" not in pair:
            continue
        t = pair["T"]
        v = pair["V"]
        vt = (
            v["feature"].float()
            if torch.is_tensor(v["feature"])
            else torch.tensor(v["feature"], dtype=torch.float32)
        )
        tt = (
            t["feature"].float()
            if torch.is_tensor(t["feature"])
            else torch.tensor(t["feature"], dtype=torch.float32)
        )

        out.append({
            "problem_id": v["problem_id"],
            "order_id": v["order_id"],
            "logical_role": int(v["logical_role"]),
            "layer": int(v["layer"]),
            "site": site,
            "variant": "DELTA",
            "outcome_group": v["outcome_group"],
            "feature": vt - tt,
        })
    return out


def make_tasks():
    tasks = [
        ("target_boundary", "fact_direction", False),
        ("target_boundary", "variant", False),
        ("target_boundary", "outcome", True),
        ("final_prompt", "fact_direction", False),
        ("final_prompt", "answer", False),
        ("final_prompt", "variant", False),
        ("final_prompt", "outcome", True),
    ]

    # Exact intermediate reasoning states. Probe x and y separately to avoid
    # sparse coordinate classes and to preserve displacement magnitude.
    for k in range(1, 5):
        tasks.append(("final_prompt", f"state{k}_x", False))
        tasks.append(("final_prompt", f"state{k}_y", False))
        tasks.append(("final_prompt", f"state{k}_x", True))
        tasks.append(("final_prompt", f"state{k}_y", True))

    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--states",
        type=Path,
        default=Path("probe_data/internal_probe_states_round6.pt"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("analysis_probes_round6"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.states, map_location="cpu")
    records = payload["records"]

    tasks = make_tasks()
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
                    "feature_type": "hidden_state",
                    **result,
                })

    # NEW control: can the V-T representational CHANGE predict failure?
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
    out_csv = args.out_dir / "probe_results_round6.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== Round-6 probe results ===")
    print(df.to_string(index=False))
    print("\nSaved:", out_csv)

    # Curves. Keep V-only and all-example state probes separate.
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
            args.out_dir
            / f"probe_{safe_site}_{safe_target}_{suffix}.png",
            dpi=180,
        )
        plt.close()

    # Best layer summary for quick inspection.
    summary = (
        df.sort_values(
            [
                "site",
                "probe_target",
                "v_only",
                "logical_role",
                "balanced_accuracy",
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
        args.out_dir / "best_probe_layer_summary_round6.csv",
        index=False,
    )

    # Mechanism-focused subset:
    # For each logical role r, inspect state_r_x / state_r_y in V-only runs.
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
        if focus_rows
        else pd.DataFrame()
    )
    focus.to_csv(
        args.out_dir / "target_role_state_probe_curves.csv",
        index=False,
    )

    # Compare success/failure predictability from raw V state vs paired V-T delta.
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

    print("\n=== Best probe layer by task / role ===")
    print(summary.to_string(index=False))
    print(
        "\nMechanism-focused files:\n"
        f"  {args.out_dir / 'target_role_state_probe_curves.csv'}\n"
        f"  {args.out_dir / 'outcome_probe_raw_vs_delta.csv'}"
    )


if __name__ == "__main__":
    main()
