#!/usr/bin/env python3
"""
Matched Text-vs-Vision intermediate-state decoding.

This is the key control for distinguishing state-composition degradation from
downstream readout failure.

For each logical role k and decoder layer:
  - use the SAME matched latent problems in Text and Vision conditions
  - use the SAME grouped train/test split for both variants
  - decode the exact accumulated state s_k=(x_k,y_k)

Two probe families:
  1) classification of integer x_k / y_k
  2) Ridge regression of continuous x_k / y_k

Primary quantity:
  Vision metric - Text metric

Negative classification/R² gap or positive MAE gap indicates that the same
reasoning state is less linearly accessible after replacing the target fact
with Vision.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def to_numpy(x):
    if torch.is_tensor(x):
        return x.float().numpy()
    return np.asarray(x, dtype=np.float32)


def matched_variant_records(records, role, layer, site="final_prompt"):
    keyed = {}
    for r in records:
        if r["site"] != site:
            continue
        if int(r["logical_role"]) != int(role):
            continue
        if int(r["layer"]) != int(layer):
            continue
        key = (r["problem_id"], r.get("order_id", "o00"))
        keyed.setdefault(key, {})[r["variant"]] = r

    out = []
    for key, pair in keyed.items():
        if "T" in pair and "V" in pair:
            out.append((key, pair["T"], pair["V"]))
    return out


def split_problem_ids(problem_ids, seed, test_fraction):
    unique = np.array(sorted(set(problem_ids)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique)
    n_test = max(1, int(round(len(unique) * test_fraction)))
    test = set(perm[:n_test].tolist())
    train = set(perm[n_test:].tolist())
    return train, test


def prepare_xy(pairs, variant, label_key, train_ids, test_ids):
    train_records = []
    test_records = []

    for _, t, v in pairs:
        r = t if variant == "T" else v
        if r.get(label_key) is None:
            continue
        if r["problem_id"] in train_ids:
            train_records.append(r)
        elif r["problem_id"] in test_ids:
            test_records.append(r)

    if not train_records or not test_records:
        return None

    X_train = np.stack([to_numpy(r["feature"]) for r in train_records])
    y_train = np.array([r[label_key] for r in train_records])
    X_test = np.stack([to_numpy(r["feature"]) for r in test_records])
    y_test = np.array([r[label_key] for r in test_records])

    return X_train, y_train, X_test, y_test


def classification_probe(data):
    X_train, y_train, X_test, y_test = data
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
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, pred)
        ),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_classes_train": len(np.unique(y_train)),
        "n_classes_test": len(np.unique(y_test)),
    }


def regression_probe(data, alpha):
    X_train, y_train, X_test, y_test = data
    reg = make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha),
    )
    reg.fit(X_train, y_train.astype(float))
    pred = reg.predict(X_test)

    return {
        "r2": float(r2_score(y_test.astype(float), pred)),
        "mae": float(
            mean_absolute_error(y_test.astype(float), pred)
        ),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


def summarize_seed_rows(seed_df, metric, higher_is_better=True):
    pivot = seed_df.pivot_table(
        index=[
            "logical_role", "layer", "relative_layer_depth",
            "coord", "seed",
        ],
        columns="variant",
        values=metric,
        aggfunc="first",
    ).reset_index()

    if "T" not in pivot.columns or "V" not in pivot.columns:
        return pd.DataFrame()

    pivot["vision_minus_text"] = pivot["V"] - pivot["T"]
    if not higher_is_better:
        # For MAE, positive V-T means Vision is worse; keep raw V-T but add
        # an explicitly oriented degradation score.
        pivot["vision_degradation"] = pivot["V"] - pivot["T"]
    else:
        pivot["vision_degradation"] = pivot["T"] - pivot["V"]

    summary = (
        pivot.groupby(
            [
                "logical_role", "layer",
                "relative_layer_depth", "coord",
            ],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "size"),
            text_mean=("T", "mean"),
            text_std=("T", "std"),
            vision_mean=("V", "mean"),
            vision_std=("V", "std"),
            vision_minus_text_mean=("vision_minus_text", "mean"),
            vision_minus_text_std=("vision_minus_text", "std"),
            vision_degradation_mean=("vision_degradation", "mean"),
        )
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("analysis_matched_state"),
    )
    parser.add_argument(
        "--seeds",
        default="11,22,33,44,55",
    )
    parser.add_argument("--test_fraction", type=float, default=0.25)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.states, map_location="cpu")
    records = payload["records"]
    n_layers = int(payload.get("n_layers", 0))
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    roles = sorted({int(r["logical_role"]) for r in records})
    layers = sorted({int(r["layer"]) for r in records})

    classification_rows = []
    regression_rows = []

    for role in roles:
        for layer in layers:
            pairs = matched_variant_records(records, role, layer)
            if len(pairs) < 20:
                continue

            relative_layer_depth = (
                layer / (n_layers - 1)
                if n_layers > 1
                else 0.0
            )
            problem_ids = [key[0] for key, _, _ in pairs]

            for seed in seeds:
                train_ids, test_ids = split_problem_ids(
                    problem_ids,
                    seed,
                    args.test_fraction,
                )

                for coord in ["x", "y"]:
                    label_key = f"state{role}_{coord}"

                    for variant in ["T", "V"]:
                        data = prepare_xy(
                            pairs,
                            variant,
                            label_key,
                            train_ids,
                            test_ids,
                        )
                        if data is None:
                            continue

                        cls = classification_probe(data)
                        if cls is not None:
                            classification_rows.append({
                                "logical_role": role,
                                "layer": layer,
                                "relative_layer_depth": relative_layer_depth,
                                "coord": coord,
                                "variant": variant,
                                "seed": seed,
                                **cls,
                            })

                        reg = regression_probe(
                            data,
                            alpha=args.ridge_alpha,
                        )
                        regression_rows.append({
                            "logical_role": role,
                            "layer": layer,
                            "relative_layer_depth": relative_layer_depth,
                            "coord": coord,
                            "variant": variant,
                            "seed": seed,
                            **reg,
                        })

    cls_df = pd.DataFrame(classification_rows)
    reg_df = pd.DataFrame(regression_rows)

    cls_df.to_csv(
        args.out_dir / "matched_state_classification_by_seed.csv",
        index=False,
    )
    reg_df.to_csv(
        args.out_dir / "matched_state_regression_by_seed.csv",
        index=False,
    )

    cls_summary = summarize_seed_rows(
        cls_df,
        metric="balanced_accuracy",
        higher_is_better=True,
    )
    reg_r2_summary = summarize_seed_rows(
        reg_df,
        metric="r2",
        higher_is_better=True,
    )
    reg_mae_summary = summarize_seed_rows(
        reg_df,
        metric="mae",
        higher_is_better=False,
    )

    cls_summary.to_csv(
        args.out_dir / "matched_state_classification_gap.csv",
        index=False,
    )
    reg_r2_summary.to_csv(
        args.out_dir / "matched_state_r2_gap.csv",
        index=False,
    )
    reg_mae_summary.to_csv(
        args.out_dir / "matched_state_mae_gap.csv",
        index=False,
    )

    # Average x/y into one compact role/layer table for the main paper plot.
    if len(reg_r2_summary):
        compact_r2 = (
            reg_r2_summary.groupby(
                [
                    "logical_role", "layer",
                    "relative_layer_depth",
                ],
                as_index=False,
            )
            .agg(
                text_r2=("text_mean", "mean"),
                vision_r2=("vision_mean", "mean"),
                vision_minus_text_r2=(
                    "vision_minus_text_mean", "mean"
                ),
                vision_degradation_r2=(
                    "vision_degradation_mean", "mean"
                ),
            )
        )
        compact_r2.to_csv(
            args.out_dir / "matched_state_compact_r2.csv",
            index=False,
        )
        print("\n=== Compact matched state R² ===")
        print(compact_r2.to_string(index=False))

    if len(cls_summary):
        compact_cls = (
            cls_summary.groupby(
                [
                    "logical_role", "layer",
                    "relative_layer_depth",
                ],
                as_index=False,
            )
            .agg(
                text_bal_acc=("text_mean", "mean"),
                vision_bal_acc=("vision_mean", "mean"),
                vision_minus_text_bal_acc=(
                    "vision_minus_text_mean", "mean"
                ),
                vision_degradation_bal_acc=(
                    "vision_degradation_mean", "mean"
                ),
            )
        )
        compact_cls.to_csv(
            args.out_dir / "matched_state_compact_classification.csv",
            index=False,
        )
        print("\n=== Compact matched state classification ===")
        print(compact_cls.to_string(index=False))

    print("\nSaved matched Text-vs-Vision state decoding to:", args.out_dir)


if __name__ == "__main__":
    main()
