"""Official Rosetta text-only baseline on the same Flickr8K-CN frozen embeddings.

Runs zero-pair and few-pair WassersteinProcrustes.fit() from the upstream
repository, NOT our own reimplementation. The image bridge is evaluation-only.
Run from visual_anchor; see ROSETTA_BASELINE.md for installation and commands.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from core import PCA, procrustes, retrieval
from run import validate


def official_imports():
    try:
        import torch
        from unpaired_rosetta import WassersteinProcrustes
    except ImportError as exc:
        raise RuntimeError(
            "Official Rosetta is missing: pip install "
            "'git+https://github.com/dominik-schnaus/unpaired-rosetta.git'"
        ) from exc
    return torch, WassersteinProcrustes


def ros_transform(model, arr, torch):
    """Apply original Rosetta source->target transform, including centering."""
    return model.transform(torch.as_tensor(np.asarray(arr), dtype=torch.float32)).cpu().numpy()


def ros_target(model, arr, torch):
    """Target-side preprocessing from official center_and_normalize()."""
    from unpaired_rosetta.linalg import center_and_normalize
    y = torch.as_tensor(np.asarray(arr), dtype=torch.float32)
    return center_and_normalize(y, model.mean_y).cpu().numpy()


def evaluate(model, z, data, image_to_english, torch):
    l_test = ros_transform(model, z["l_test"], torch)
    l_img_test = ros_transform(model, z["l_img_test"], torch)
    e_test = ros_target(model, z["e_test"], torch)
    i_test = ros_target(model, z["i_test"] @ image_to_english, torch)
    cases = (
        ("L-E", l_test, e_test, "l_test", "e_test"),
        ("E-L", e_test, l_test, "e_test", "l_test"),
        ("L-I", l_img_test, i_test, "l_img_test", "i_test"),
        ("I-L", i_test, l_img_test, "i_test", "l_img_test"),
    )
    scores = {}
    for direction, x, y, key_x, key_y in cases:
        scores[direction], _ = retrieval(
            x, y, data[key_x + "_ids"], data[key_y + "_ids"]
        )
    return scores


def train_official(x, y, paired_x, paired_y, seed, args, torch, cls):
    # Official method first does geometric initialization; then its own
    # readout and Hungarian/Procrustes refinement, with or without anchors.
    clusters = min(args.clusters, len(x) // 2, len(y) // 2)
    if clusters < 2:
        raise ValueError("Need at least 4 unpaired rows per space")
    model = cls(
        num_clusters=clusters,
        num_restarts=args.restarts,
        batch_size=min(args.batch, len(x), len(y)),
        num_iterations=args.iterations,
        num_workers=1,
        num_threads=1,
        verbose=False,
    )
    def as_tensor(a):
        if a is None:
            return None
        return torch.as_tensor(np.asarray(a), dtype=torch.float32).contiguous()
    with torch.inference_mode():
        model.fit(as_tensor(x), as_tensor(y),
                  as_tensor(paired_x), as_tensor(paired_y),
                  randomness=seed)
    return model


def summarize(rows):
    names = ("rosetta_zero", "rosetta_few")
    directions = ("L-E", "E-L", "L-I", "I-L")
    result = {}
    for method in names:
        matched = [r for r in rows if r["method"] == method]
        if not matched:
            continue
        result[method] = {
            direction: {
                metric: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1))
                           if len(values) > 1 else None,
                }
                for metric in ("R@1", "R@5", "R@10")
                for values in ([r["metrics"][direction][metric]
                                for r in matched],)
            }
            for direction in directions
        }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="real_work/features_stage1.npz")
    p.add_argument("--out", default="real_work/official_rosetta")
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--budgets", type=int, nargs="+", default=[50, 200, 1000])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--clusters", type=int, default=20)
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--iterations", type=int, default=20)
    args = p.parse_args()
    if not args.budgets or not args.seeds:
        p.error("At least one budget and one seed are required")
    if min(args.dim, args.clusters, args.restarts, args.batch,
           args.iterations, *args.budgets) < 1:
        p.error("Dimensions, clusters, restarts, batch, iterations, budgets must be > 0")

    torch, cls = official_imports()
    data = dict(np.load(args.data, allow_pickle=False))
    validate(data)
    if max(args.budgets) > len(data["l_pair"]):
        p.error("Budget exceeds available parallel sentence pairs")
    if set(data["e_pool_ids"]) & set(data["e_bank_ids"]):
        p.error("e_pool overlaps e_bank; run stage1_repool.py first")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pca = {s: PCA(args.dim).fit(data[s + "_fit"]) for s in "lei"}
    z = {k: pca[k[0]].transform(v) for k, v in data.items()
         if not k.endswith("_ids")}
    # Exactly the frozen bridge used in visual_anchor/run.py
    v_ie = procrustes(z["i_bridge"], z["e_bridge"])
    by_budget = {budget: [] for budget in args.budgets}

    for seed in args.seeds:
        # Zero-pair is independent of the supervised budget: run once/seed.
        zero = train_official(z["l_pool"], z["e_pool"], None, None,
                              seed, args, torch, cls)
        zero_metrics = evaluate(zero, z, data, v_ie, torch)
        zero_payload = dict(
            W=zero.weight.detach().cpu().numpy(),
            rosetta_mean_l=zero.mean_x.cpu().numpy(),
            rosetta_mean_e=zero.mean_y.cpu().numpy(),
            V_IE=v_ie,
        )
        for s, model in pca.items():
            zero_payload[s + "_pca_mean"] = model.mean
            zero_payload[s + "_pca_proj"] = model.proj
        np.savez(out / f"rosetta_zero_{seed}.npz", **zero_payload)
        print(f"seed={seed} rosetta_zero {zero_metrics}", flush=True)

        for budget in args.budgets:
            by_budget[budget].append(
                {"seed": seed, "method": "rosetta_zero",
                 "metrics": zero_metrics})
            # Identical draw to visual_anchor/run.py for each (seed,budget).
            idx = np.random.default_rng(seed).choice(
                len(z["l_pair"]), budget, replace=False)
            few = train_official(z["l_pool"], z["e_pool"],
                                 z["l_pair"][idx], z["e_pair"][idx],
                                 seed, args, torch, cls)
            metrics = evaluate(few, z, data, v_ie, torch)
            by_budget[budget].append(
                {"seed": seed, "method": "rosetta_few",
                 "metrics": metrics})
            payload = dict(
                W=few.weight.detach().cpu().numpy(),
                rosetta_mean_l=few.mean_x.cpu().numpy(),
                rosetta_mean_e=few.mean_y.cpu().numpy(),
                anchor_indices=idx,
                V_IE=v_ie,
            )
            for s, model in pca.items():
                payload[s + "_pca_mean"] = model.mean
                payload[s + "_pca_proj"] = model.proj
            np.savez(out / f"rosetta_few_budget_{budget}_{seed}.npz", **payload)
            print(f"budget={budget} seed={seed} rosetta_few {metrics}",
                  flush=True)

    for budget, rows in by_budget.items():
        folder = out / f"results_budget_{budget}"
        folder.mkdir(exist_ok=True)
        report = {
            "config": {**vars(args), "budget": budget},
            "summary": summarize(rows),
            "runs": rows,
            "method_source": "official dominik-schnaus/unpaired-rosetta WassersteinProcrustes.fit",
            "note": (
                "Same training-fit 32D PCA and frozen image->English bridge as "
                "run.py. Official Rosetta fit adds centering/normalization to "
                "both spaces. Zero-pair uses no parallel anchors; few-pair "
                "uses run.py's exact per-budget/seed anchor indices."
            ),
        }
        (folder / "report.json").write_text(json.dumps(report, indent=2))
        print("Report:", folder / "report.json", flush=True)


if __name__ == "__main__":
    main()
