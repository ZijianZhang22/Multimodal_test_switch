"""Reuse existing frozen embeddings for the stage-1 matched-bank comparison.

Only the ordinary English OT pool changes: e_pool = e_bridge.
All other arrays and PCA fitting data remain byte-identical.
"""
import argparse
from pathlib import Path
import numpy as np
from run import validate

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("real_work/features.npz"))
    p.add_argument("--output", type=Path, default=Path("real_work/features_stage1.npz"))
    a = p.parse_args()
    if a.source.resolve() == a.output.resolve():
        p.error("Use a separate output to preserve the original baseline.")
    d = dict(np.load(a.source, allow_pickle=False))
    expected = ("e_bridge", "e_bank")
    # Exact matching of the bank IDs is a necessary, though not sufficient,
    # check that we're migrating the old data protocol.
    if not set(d["e_bank_ids"]).issubset(set(d["e_pool_ids"])):
        raise ValueError("Expected original e_pool to contain all bank IDs")
    d["e_pool"] = d["e_bridge"].copy()
    d["e_pool_ids"] = d["e_bridge_ids"].copy()
    validate(d)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.output, **d)
    print(f"Stage-1 features: {a.output}")
    print("e_pool is now e_bridge only; all other features unchanged.")

if __name__ == "__main__":
    main()
