"""Official two-space Rosetta vs. Rosetta with a third target pool, on the same features.npz as run.py.

Methods (all share Rosetta's geometric initialization, readout, batch size and iteration count):
  rosetta     official WassersteinProcrustes, X = low-resource pool, Y = English pool (lambda_z = 0)
  visual      third pool = bank images mapped into English by the frozen V_IE
  bank_text   third pool = the English captions of the SAME bank images (content-matched, modality differs)
  extra_text  third pool = equally many captions of OTHER images (new content, text modality)
  gaussian    third pool = random Gaussian points (geometry-only control)

The key comparison for H1 is visual vs bank_text. Anchor indices per seed are drawn exactly like run.py,
so results pair row-by-row with the earlier NumPy prototype.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from core import PCA, paired_bootstrap, procrustes, retrieval
from rosetta3 import WassersteinProcrustes3
from run import validate

ROSETTA_COMMIT = "fdfad84ea488f29702a61a455be71622c4ff5d49"
METHODS = ["rosetta", "visual", "bank_text", "extra_text", "gaussian"]
DIRECTIONS = {"L-E": ("l_test", "e_test"), "E-L": ("e_test", "l_test"),
              "L-I": ("l_img_test", "i_test"), "I-L": ("i_test", "l_img_test")}


def tensor(x):
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def third_pool(method, z, v, seed):
    if method == "rosetta":
        return None
    if method == "visual":
        return z["i_bank"] @ v
    if method == "bank_text":
        return z["e_bank"]
    rng = np.random.default_rng(seed + 100000)  # same control stream as core.fit
    n = len(z["i_bank"])
    if method == "extra_text":
        if len(z["e_extra"]) < n:
            raise ValueError("e_extra must have at least as many rows as i_bank")
        return z["e_extra"][rng.choice(len(z["e_extra"]), n, replace=False)]
    if method == "gaussian":
        return rng.normal(size=(n, z["e_bank"].shape[1]))
    raise ValueError(method)


def check_overlap(d, methods, allow):
    shared = set(d["e_bank_ids"]) & set(d["e_pool_ids"])
    if shared and not allow and {"visual", "bank_text"} & set(methods):
        raise ValueError(
            f"{len(shared)} bank image IDs also appear in e_pool. The bank's content is then already in the English "
            "pool and visual/bank_text are not a fair test. Rebuild features with e_pool = e_bridge only "
            "(prepare_real.py), or pass --allow-overlap to reproduce the old setting on purpose.")
    return len(shared)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="results_rosetta3")
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--budget", type=int, default=50, help="known pairs; 0 = zero-pair Rosetta")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--lambda-z", type=float, default=0.25, help="weight of third target (0 = official Rosetta)")
    p.add_argument("--clusters", type=int, default=20)
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--whiten", type=float, default=0)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    p.add_argument("--allow-overlap", action="store_true")
    a = p.parse_args()
    if (a.budget < 0 or a.dim < 1 or a.lambda_z < 0
            or min(a.clusters, a.restarts, a.batch, a.iterations, a.workers) < 1):
        p.error("budget >= 0, lambda-z >= 0; dim/clusters/restarts/batch/iterations/workers > 0")

    d = dict(np.load(a.data, allow_pickle=False))
    validate(d)
    overlap = check_overlap(d, a.methods, a.allow_overlap)
    if a.budget > len(d["l_pair"]):
        p.error("budget exceeds available anchors")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # Same preprocessing and frozen image->English map as run.py.
    pcas = {s: PCA(a.dim, a.whiten).fit(d[s + "_fit"]) for s in "lei"}
    z = {k: pcas[k[0]].transform(v) for k, v in d.items() if not k.endswith("_ids")}
    v = procrustes(z["i_bridge"], z["e_bridge"])
    # One fixed image centering for EVERY method, so L-I differences never come from the evaluation preprocessing.
    image_mean = tensor(z["i_bank"] @ v).mean(dim=0, keepdim=True)
    clusters = min(a.clusters, len(z["l_pool"]) // 2, len(z["e_pool"]) // 2)

    print(f"Starting Rosetta3: budget={a.budget} pairs, lambda_z={a.lambda_z}, "
          f"clusters={clusters}, restarts={a.restarts}, batch={a.batch}, "
          f"iterations={a.iterations}, seeds={a.seeds}", flush=True)
    results, hits = [], {}
    init_cache = {}  # same seed + pairs -> same geometric initialization for every method
    for seed in a.seeds:
        idx = (np.random.default_rng(seed).choice(len(z["l_pair"]), a.budget, replace=False)
               if a.budget else np.array([], dtype=int))
        px = tensor(z["l_pair"][idx]) if a.budget else None
        py = tensor(z["e_pair"][idx]) if a.budget else None
        for method in a.methods:
            pool = third_pool(method, z, v, seed)
            model = WassersteinProcrustes3(
                lambda_z=0.0 if pool is None else a.lambda_z, init_cache=init_cache, num_clusters=clusters, num_restarts=a.restarts,
                batch_size=a.batch, num_iterations=a.iterations, num_workers=a.workers, num_threads=1)
            model.fit(tensor(z["l_pool"]), tensor(z["e_pool"]), px, py, randomness=seed,
                      samples_z=None if pool is None else tensor(pool))
            emb = {
                "l_test": model.transform(tensor(z["l_test"])).numpy(),
                "l_img_test": model.transform(tensor(z["l_img_test"])).numpy(),
                "e_test": model.transform_target(tensor(z["e_test"]), model.mean_y).numpy(),
                "i_test": model.transform_target(tensor(z["i_test"] @ v), image_mean).numpy(),
            }
            metrics, per_query = {}, {}
            for name, (xk, yk) in DIRECTIONS.items():
                m, h = retrieval(emb[xk], emb[yk], d[xk + "_ids"], d[yk + "_ids"])
                metrics[name], per_query[name] = m, h["R@1"]
            hits[(seed, method)] = per_query
            z_mean = float(np.mean(model.z_counts)) if model.z_counts else 0.0
            results.append({"seed": seed, "method": method, "metrics": metrics, "mean_z_matches": z_mean})
            np.savez(out / f"{method}_{seed}.npz", W=model.weight.numpy(), mean_x=model.mean_x.numpy(),
                     mean_y=model.mean_y.numpy(), V_IE=v, image_mean=image_mean.numpy(), anchor_indices=idx,
                     **({"mean_z": model.mean_z.numpy()} if model.mean_z is not None else {}))
            print(f"budget={a.budget} seed={seed} method={method} lambda_z={0.0 if pool is None else a.lambda_z} metrics={metrics}", flush=True)

    pairs = [(m, "rosetta") for m in a.methods if m != "rosetta"] + [("visual", "bank_text"), ("visual", "extra_text")]
    comparisons = []
    for seed in a.seeds:
        for left, right in pairs:
            if (seed, left) not in hits or (seed, right) not in hits:
                continue
            for direction, (xk, _) in DIRECTIONS.items():
                ids = d[xk + "_ids"]
                groups = np.unique(ids)
                ha, hb = hits[(seed, left)][direction], hits[(seed, right)][direction]
                aa = np.array([ha[ids == g].mean() for g in groups])
                bb = np.array([hb[ids == g].mean() for g in groups])
                comparisons.append({"seed": seed, "left": left, "right": right, "direction": direction,
                                    **paired_bootstrap(aa, bb, seed)})

    summary = {}
    for method in a.methods:
        rows = [r for r in results if r["method"] == method]
        summary[method] = {direction: {k: {
            "mean": float(np.mean([r["metrics"][direction][k] for r in rows])),
            "std": float(np.std([r["metrics"][direction][k] for r in rows], ddof=1)) if len(rows) > 1 else None}
            for k in ["R@1", "R@5", "R@10"]} for direction in DIRECTIONS}
    report = {"config": vars(a), "rosetta_commit": ROSETTA_COMMIT, "clusters_used": clusters,
              "bank_pool_overlap_ids": overlap, "summary": summary, "runs": results, "paired_bootstrap": comparisons}
    (out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
