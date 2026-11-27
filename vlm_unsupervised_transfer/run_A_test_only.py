#!/usr/bin/env python3
"""Resume the previously validated A-only experiment on its fixed TEST split.

This runner reads the original settings.json and invokes ONLY experiment.py
--stage test followed by summarize.py. It never downloads datasets, refits
maps, re-runs a validation search, or executes experiment B.

Usage (inside the package venv):
    python -u run_A_test_only.py --experiment /workspace/vlm_A_zh_seed42/experiment
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

# The flags used to create experiment/settings.json, in its own order.
# Do not allow new CLI defaults to silently override the original protocol.
REQUIRED = (
    "pools", "val", "test", "out", "language", "model", "layers", "arm",
    "dim", "iterations", "batch", "visual_weight", "strengths",
    "oracle_strengths", "max_relative_norm", "max_tokens", "seed",
)
METHODS = ("mean_shift", "text_ot", "text_visual_ot", "text_noise_ot")


def make_commands(experiment: Path, python: str = sys.executable):
    root = experiment.resolve()
    settings_file = root / "settings.json"
    selection_file = root / "selection_val.json"
    if not settings_file.is_file():
        raise FileNotFoundError(f"Missing validation settings: {settings_file}")
    if not selection_file.is_file():
        raise FileNotFoundError(f"Missing completed validation: {selection_file}")
    settings = json.loads(settings_file.read_text(encoding="utf-8"))
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    if settings.get("arm") != "a":
        raise ValueError("This runner is A-only: settings.json must have arm='a'")
    unexpected = set(settings) - set(REQUIRED) - {"file_integrity"}
    missing = set(REQUIRED) - set(settings)
    if unexpected or missing:
        raise ValueError(f"Unexpected/missing settings keys: extra={unexpected}, missing={missing}")
    if Path(settings["out"]).resolve() != root:
        raise ValueError("Settings output directory differs from requested experiment directory")
    chosen = selection.get("selected", {})
    expected = {f"A/{m}" for m in METHODS}
    if set(chosen) != expected:
        raise ValueError(f"Expected four A-only selected methods; got {sorted(chosen)}")
    for m in METHODS:
        c = chosen[f"A/{m}"]
        if c.get("method") != m or c.get("mode") != "direct":
            raise ValueError(f"Unexpected selected method/mode for {m}")
        layer = c["layer"]
        if layer not in settings["layers"]:
            raise ValueError(f"Selected layer {layer} not in initial validation grid")
        map_path = root / "maps" / f"layer{layer}_{m}.npz"
        if not map_path.is_file():
            raise FileNotFoundError(f"Missing learned frozen map: {map_path}")
    for field in ("pools", "val", "test"):
        if not Path(settings[field]).exists():
            raise FileNotFoundError(f"Missing original {field} data at {settings[field]}")
    cmd = [python, "-u", "experiment.py", "--stage", "test"]
    for key in REQUIRED:
        v = settings[key]
        if isinstance(v, list):
            v = ",".join(map(str, v))
        cmd.extend(("--" + key.replace("_", "-"), str(v)))
    summ = [python, "-u", "summarize.py", "--out", str(root)]
    return cmd, summ, selection


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, required=True,
                    help="The EXACT outputs/zh_seed42/experiment directory from validation")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without loading the VLM")
    args = ap.parse_args()
    cmd, summ, selection = make_commands(args.experiment)
    print("Testing 600 CV-Bench items (or the number in saved test.jsonl).", flush=True)
    print("A-only selected validation configurations (FROZEN):", flush=True)
    for k in sorted(selection["selected"]):
        c = selection["selected"][k]
        print(f"  {k}: layer={c['layer']} alpha={c['strength']}", flush=True)
    print("Executing:", shlex.join(cmd), flush=True)
    if args.dry_run:
        print("DRY RUN: no GPU calls made", flush=True)
        return
    package_dir = Path(__file__).resolve().parent
    subprocess.run(cmd, cwd=package_dir, check=True)
    subprocess.run(summ, cwd=package_dir, check=True)
    print("FINISHED:", args.experiment.resolve() / "comparison.csv", flush=True)


if __name__ == "__main__":
    main()