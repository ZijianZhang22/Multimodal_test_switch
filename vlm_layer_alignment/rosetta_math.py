"""Train-only, few-pair orthogonal Procrustes + unpaired OT-like pseudo coupling.

This is a Rosetta-inspired VLM hidden-state variant, NOT a byte-for-byte
reproduction of official Rosetta/WassersteinProcrustes. Image-conditioned VLM
activations already share a hidden dimension; we learn a low-rank *correction*.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-10)


def polar(x):
    u, _, v = np.linalg.svd(x, full_matrices=False)
    return u @ v


def fit_mapping(paired_src, paired_tgt, pool_src, pool_tgt, *, dim=128,
                method="rosetta", iterations=20, pool_weight=0.3,
                seed=0, max_unpaired=128):
    """Inputs are arrays of last-prompt-token decoder outputs in SAME VLM.

    pool_src/pool_tgt must be *different image IDs* and disjoint from paired.
    Means and PCA are fitted on training samples only. W is orthogonal.
    """
    pairs = np.asarray(paired_src, np.float64)
    targets = np.asarray(paired_tgt, np.float64)
    sx, sy = np.asarray(pool_src, np.float64), np.asarray(pool_tgt, np.float64)
    if method not in ("supervised", "rosetta"):
        raise ValueError("Unknown fit method")
    if any(a.ndim != 2 for a in (pairs, targets, sx, sy)) or len(pairs) != len(targets):
        raise ValueError("Invalid anchor/pool shape")
    if min(len(pairs), len(sx), len(sy)) < 2 or len({a.shape[1] for a in (pairs, targets, sx, sy)}) != 1:
        raise ValueError("Insufficient samples or inconsistent hidden size")
    if not all(np.isfinite(a).all() for a in (pairs, targets, sx, sy)):
        raise ValueError("Nonfinite hidden states")
    if not 0 <= pool_weight <= 10 or iterations < 1:
        raise ValueError("Invalid hyperparameters")
    rng = np.random.default_rng(seed)
    means = (np.concatenate((pairs, sx)).mean(0), np.concatenate((targets, sy)).mean(0))
    msrc, mtgt = means
    joint = np.concatenate((pairs - msrc, sx - msrc, targets - mtgt, sy - mtgt))
    # Thin SVD: number of training examples (not hidden width) limits PCA rank.
    rank = min(joint.shape[0] - 1, joint.shape[1])
    if dim > rank:
        raise ValueError(f"dim={dim} exceeds training PCA rank={rank}")
    _, _, v = np.linalg.svd(joint, full_matrices=False)
    basis = v[:dim].T
    a = unit((pairs - msrc) @ basis)
    b = unit((targets - mtgt) @ basis)
    x = unit((sx - msrc) @ basis)
    y = unit((sy - mtgt) @ basis)
    anchor_cov = a.T @ b / len(a)
    w = polar(anchor_cov)
    history = []
    if method == "rosetta":
        for _ in range(iterations):
            count = min(len(x), len(y), max_unpaired)
            ix = rng.choice(len(x), count, replace=False)
            iy = rng.choice(len(y), count, replace=False)
            sim = x[ix] @ w @ y[iy].T
            rr, cc = linear_sum_assignment(-sim)
            # No explicit pair information from unpaired pools is used.
            pseudo = x[ix[rr]].T @ y[iy[cc]] / count
            updated = polar(anchor_cov + pool_weight * pseudo)
            history.append(float(np.linalg.norm(updated - w)))
            w = updated
    return {"basis": basis.astype(np.float32),
            "mean_src": msrc.astype(np.float32), "mean_tgt": mtgt.astype(np.float32),
            "W": w.astype(np.float32), "history": history,
            "method": method, "dim": dim, "pairs": len(pairs),
            "unpaired_src": len(sx), "unpaired_tgt": len(sy)}


def correction(h, mapping, strength=0.2, max_relative_norm=0.15):
    """Low-rank residual correction in VLM's original hidden coordinates.

    Only PCA subspace is modified; orthogonal complement is preserved. Per-token
    norm clamp prevents accidentally destroying VLM image/question content.
    """
    import torch
    if not 0 <= strength <= 2 or not 0 < max_relative_norm <= 1:
        raise ValueError("Invalid strength / norm clamp")
    p = torch.as_tensor(mapping["basis"], dtype=torch.float32, device=h.device)
    w = torch.as_tensor(mapping["W"], dtype=torch.float32, device=h.device)
    ms = torch.as_tensor(mapping["mean_src"], dtype=torch.float32, device=h.device)
    mt = torch.as_tensor(mapping["mean_tgt"], dtype=torch.float32, device=h.device)
    hf = h.float()
    z = (hf - ms) @ p
    desired = (z @ w + (mt - ms) @ p) @ p.T
    original = z @ p.T
    delta = desired - original
    # Clamp correction to a specified fraction of the original token norm.
    dn = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cap = hf.norm(dim=-1, keepdim=True).clamp_min(1e-8) * max_relative_norm
    delta = delta * torch.minimum(torch.ones_like(dn), cap / dn)
    return (hf + strength * delta).to(h.dtype)


def save_mapping(path, mapping, metadata):
    dest = Path(path); dest.parent.mkdir(parents=True, exist_ok=True)
    array_fields = {k: mapping[k] for k in ("basis", "mean_src", "mean_tgt", "W")}
    np.savez_compressed(dest, **array_fields, metadata=np.array(json.dumps({**metadata,
                  "method": mapping["method"], "dim": mapping["dim"], "pairs": mapping["pairs"],
                  "unpaired_src": mapping["unpaired_src"], "unpaired_tgt": mapping["unpaired_tgt"]})))


def load_mapping(path, *, model_id, data_sha, language, layer):
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z["metadata"].item()))
        for name, expected in (("model_id", model_id), ("data_sha", data_sha),
                               ("language", language), ("layer", layer)):
            if metadata.get(name) != expected:
                raise ValueError(f"Checkpoint {name} mismatch: {metadata.get(name)} != {expected}")
        mapping = {k: z[k] for k in ("basis", "mean_src", "mean_tgt", "W")}
    return mapping, metadata