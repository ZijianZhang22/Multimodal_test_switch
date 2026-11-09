"""Official two-space Rosetta vs. Rosetta with a third target pool, on the same features.npz as run.py.

Methods (all share Rosetta's geometric initialization, readout, batch size and iteration count):
  supervised  pairs-only orthogonal Procrustes (no unpaired data at all)
  rosetta     official WassersteinProcrustes, X = low-resource pool, Y = English pool (lambda_z = 0)
  visual      third pool = bank images mapped into English by the frozen V_IE
  bank_text   third pool = the English captions of the SAME bank images (content-matched, modality differs)
  extra_text  third pool = equally many captions of OTHER images (new content, text modality)
  gaussian    third pool = random Gaussian points (geometry-only control)
  english_pool third pool = English Y itself, but independently resampled (matching/weight control)
  anchor_downweight  no Z; scale paired term by 1/(1+lambda_z) during refinement

The key comparison for H1 is visual vs bank_text. Anchor indices per seed are drawn exactly like run.py,
so results pair row-by-row with the earlier NumPy prototype.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from core import fit_space_pcas, paired_bootstrap, procrustes, retrieval
from rosetta3 import WassersteinProcrustes3
from run import validate

ROSETTA_COMMIT = "fdfad84ea488f29702a61a455be71622c4ff5d49"
METHODS = ["supervised", "rosetta", "visual", "bank_text", "extra_text", "gaussian",
           "english_pool", "anchor_downweight", "pairs_only", "image_only", "text_only",
           "noise_only", "caption_only", "joint"]
# Five main ablations use IDENTICAL anchor-only initialization; official
# geometric Rosetta is reported separately, not treated as a matched init.
ABLATIONS = {
    "pairs_only": ("supervised", 0.0, 0.0, None),
    "image_only": ("supervised", 0.0, 1.0, "visual"),
    "text_only": ("supervised", 1.0, 0.0, None),
    "noise_only": ("supervised", 0.0, 1.0, "gaussian"),
    "caption_only": ("supervised", 0.0, 1.0, "bank_text"),
    "joint": ("supervised", 1.0, 1.0, "visual"),
}
DIRECTIONS = {"L-E": ("l_test", "e_test"), "E-L": ("e_test", "l_test"),
              "L-I": ("l_img_test", "i_test"), "I-L": ("i_test", "l_img_test")}


def tensor(x):
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def third_pool(method, z, v, seed):
    if method in {"supervised", "rosetta", "anchor_downweight"}:
        return None
    if method == "english_pool":
        return z["e_pool"]
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
    if shared and not allow and {"visual", "bank_text", "image_only", "caption_only", "joint"} & set(methods):
        raise ValueError(
            f"{len(shared)} bank image IDs also appear in e_pool. The bank's content is then already in the English "
            "pool and visual/bank_text are not a fair test. Rebuild features with e_pool = e_bridge only "
            "(prepare_real.py), or pass --allow-overlap to reproduce the old setting on purpose.")
    return len(shared)


def check_bridge_pca(bridge, shared_ei_pca, joint_ei_pca):
    """Validate pre-processing: an identity bridge requires a shared CLIP basis.

    Shared PCA permits distinct E/I means but keeps the projection identical.
    Joint PCA uses a single mean as well as a single projection.
    """
    if shared_ei_pca and joint_ei_pca:
        raise ValueError("Choose only one of --shared-ei-pca or --joint-ei-pca")
    if bridge == "identity" and not (shared_ei_pca or joint_ei_pca):
        raise ValueError("--bridge identity requires --shared-ei-pca or --joint-ei-pca")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="results_rosetta3")
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--budget", type=int, default=50, help="known pairs; 0 = zero-pair Rosetta")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--lambda-z", type=float, default=0.25, help="third-term weight for legacy methods")
    p.add_argument("--lambda-y", type=float, default=1.0, help="Y weight for legacy methods (default 1)")
    p.add_argument("--ablation-lambda-z", type=float, default=1.0,
                   help="Z weight for image_only/noise_only/caption_only/joint, default 1")
    p.add_argument("--bridge", choices=("procrustes", "identity"), default="procrustes",
                   help="identity uses native CLIP coordinates; procrustes is legacy learned bridge")
    p.add_argument("--joint-ei-pca", action="store_true",
                   help="one pooled mean and common projection for E/I (legacy CLIP ablation)")
    p.add_argument("--clusters", type=int, default=20)
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--whiten", type=float, default=0)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    p.add_argument("--allow-overlap", action="store_true")
    p.add_argument("--anchor-centering", choices=("pool", "source_pool", "paired"),
                   default="pool", help="A-E and pairs-only centering: pool=both train pool means; "
                   "source_pool=source pool and target paired mean; paired=both paired means")
    p.add_argument("--image-centering", choices=("bank", "english"), default="bank",
                   help="fixed image evaluation mean, identical across all methods")
    p.add_argument("--shared-ei-pca", action="store_true", help="common projection but separate E and I training means; recommended for CLIP")
    a = p.parse_args()
    if (a.budget < 0 or a.dim < 1 or a.lambda_z < 0 or a.lambda_y < 0
            or a.ablation_lambda_z <= 0
            or min(a.clusters, a.restarts, a.batch, a.iterations, a.workers) < 1):
        p.error("budget >= 0, lambda-z >= 0; dim/clusters/restarts/batch/iterations/workers > 0")

    d = dict(np.load(a.data, allow_pickle=False))
    validate(d)
    overlap = check_overlap(d, a.methods, a.allow_overlap)
    if a.budget > len(d["l_pair"]):
        p.error("budget exceeds available anchors")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # With CLIP, preserve a COMMON projection while allowing either shared
    # modality-specific means or the old joint mean. Identity means no V_IE fit.
    try:
        check_bridge_pca(a.bridge, a.shared_ei_pca, a.joint_ei_pca)
    except ValueError as exc:
        p.error(str(exc))
    pcas = fit_space_pcas(d, a.dim, a.whiten, a.shared_ei_pca, a.joint_ei_pca)
    z = {k: pcas[k[0]].transform(v) for k, v in d.items() if not k.endswith("_ids")}
    v = (np.eye(a.dim) if a.bridge == "identity"
         else procrustes(z["i_bridge"], z["e_bridge"]))
    # Image evaluation must use one FIXED mean for all methods. The image
    # bank is train-only and is also the pool used to center Visual's Z.
    # Retain English-centering as a backwards-compatibility ablation.
    image_mean = tensor(z["i_bank"] @ v).mean(dim=0, keepdim=True)
    clusters = min(a.clusters, len(z["l_pool"]) // 2, len(z["e_pool"]) // 2)

    print(f"Starting Rosetta3: budget={a.budget} pairs, lambda_y={a.lambda_y}, lambda_z={a.lambda_z}, "
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
            if method in ABLATIONS:
                init_mode, lambda_y, lambda_z, z_type = ABLATIONS[method]
                lambda_z = (a.ablation_lambda_z if z_type is not None else 0.0)
                pool = third_pool(z_type, z, v, seed) if z_type is not None else None
            else:
                init_mode, lambda_y = "rosetta", a.lambda_y
                pool = third_pool(method, z, v, seed)
                lambda_z = a.lambda_z if pool is not None else 0.0
                if method == "rosetta":
                    lambda_y = 1.0  # immutable official reference
            if method == "supervised" or method == "pairs_only":
                if not a.budget:
                    print(f"budget=0 seed={seed}: skipping {method} (needs known pairs)", flush=True)
                    continue
            elif init_mode == "supervised" and not a.budget:
                print(f"budget=0 seed={seed}: skipping {method} (anchor init)", flush=True)
                continue
            anchor_scale = 1.0 / (1.0 + a.lambda_z) if method == "anchor_downweight" else 1.0
            model = WassersteinProcrustes3(
                lambda_z=lambda_z, lambda_y=lambda_y, init_mode=init_mode,
                anchor_scale=anchor_scale, anchor_centering=a.anchor_centering,
                init_cache=init_cache, num_clusters=clusters,
                num_restarts=a.restarts, batch_size=a.batch, num_iterations=a.iterations,
                num_workers=a.workers, num_threads=1)
            if method in ("supervised", "pairs_only"):
                model.fit_supervised(tensor(z["l_pool"]), tensor(z["e_pool"]), px, py)
            else:
                model.fit(tensor(z["l_pool"]), tensor(z["e_pool"]), px, py,
                          randomness=seed, samples_z=None if pool is None else tensor(pool))
            emb = {
                "l_test": model.transform(tensor(z["l_test"])).numpy(),
                "l_img_test": model.transform(tensor(z["l_img_test"])).numpy(),
                "e_test": model.transform_target(tensor(z["e_test"]), model.mean_y).numpy(),
                "i_test": model.transform_target(
                    tensor(z["i_test"] @ v), image_mean if a.image_centering == "bank" else model.mean_y).numpy(),
            }
            metrics, per_query = {}, {}
            for name, (xk, yk) in DIRECTIONS.items():
                m, h = retrieval(emb[xk], emb[yk], d[xk + "_ids"], d[yk + "_ids"])
                metrics[name], per_query[name] = m, h["R@1"]
            hits[(seed, method)] = per_query
            z_mean = float(np.mean(model.z_counts)) if model.z_counts else 0.0
            results.append({"seed": seed, "method": method, "metrics": metrics,
                            "mean_z_matches": z_mean, "anchor_scale": anchor_scale,
                            "lambda_y": lambda_y, "lambda_z": lambda_z,
                            "init_mode": init_mode, "anchor_centering": a.anchor_centering,
                            "image_centering": a.image_centering})
            np.savez(out / f"{method}_{seed}.npz", W=model.weight.numpy(), mean_x=model.mean_x.numpy(),
                     mean_y=model.mean_y.numpy(), V_IE=v,
                     image_mean=(image_mean if a.image_centering == "bank" else model.mean_y).numpy(),
                     anchor_indices=idx, lambda_y=np.array(lambda_y), lambda_z=np.array(lambda_z),
                     **({"mean_z": model.mean_z.numpy()} if model.mean_z is not None else {}))
            print(f"budget={a.budget} seed={seed} method={method} "
                  f"lambda_y={lambda_y} lambda_z={lambda_z} init={init_mode} "
                  f"anchor_scale={anchor_scale:.4f} metrics={metrics}", flush=True)

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
        if not rows:
            continue
        summary[method] = {direction: {k: {
            "mean": float(np.mean([r["metrics"][direction][k] for r in rows])),
            "std": float(np.std([r["metrics"][direction][k] for r in rows], ddof=1)) if len(rows) > 1 else None}
            for k in ["R@1", "R@5", "R@10"]} for direction in DIRECTIONS}
    report = {"config": vars(a), "rosetta_commit": ROSETTA_COMMIT, "clusters_used": clusters,
              "bank_pool_overlap_ids": overlap,
              "pca_protocol": ("shared_separate_means" if a.shared_ei_pca else
                               "joint_single_mean" if a.joint_ei_pca else "separate_pca"),
              "summary": summary, "runs": results, "paired_bootstrap": comparisons}
    (out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()