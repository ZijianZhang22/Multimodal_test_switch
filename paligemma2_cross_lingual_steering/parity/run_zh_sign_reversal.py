#!/usr/bin/env python3
"""Frozen-parameter sign reversal of ZH concept steering (+alpha vs -alpha).

All conditions: CV-Bench ZH questions, English task prefix/instruction, same held-out
samples and per-task concept/layer/|alpha| from the earlier ZH Tune split.
Reuses prior positive Both/Image-only, and positive Text-only when available.
Uses original evaluate.py / unmodified upstream mean-shift model for negative alpha.
No hyperparameter retuning, no model training, no vector re-extraction.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
MASKS = ("both", "text_token", "image_token")
LABELS = {"both": "Both", "text_token": "Text-only", "image_token": "Image-only"}
POS_FILES = {"both": "test.csv", "text_token": "test_zh_text_only.csv",
             "image_token": "test_image_only.csv"}
NEG_FILES = {m: f"test_reverse_{m}.csv" for m in MASKS}
TASKS = {"count", "relation", "depth", "distance"}


def csv_rows(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def indexed(rows, variant, n_expected, name):
    subset = [r for r in rows if r["variant"] == variant]
    result = {r["id"]: r for r in subset}
    if len(result) != len(subset) or len(result) != n_expected:
        raise ValueError(f"{name}: expected {n_expected} unique {variant} rows, "
                         f"got {len(subset)} / {len(result)}")
    return result


def check(baseline, rows, jobs, n_expected, mask, sign):
    name = f"{LABELS[mask]} {sign}"
    current = indexed(rows, "none", n_expected, name + " baseline")
    steered = indexed(rows, "zh", n_expected, name)
    if current.keys() != baseline.keys() or steered.keys() != baseline.keys():
        raise ValueError(f"{name}: mismatched test IDs")
    params = {j["task"]: j for j in jobs}
    for i, b in baseline.items():
        c, r = current[i], steered[i]
        for fld in ("task", "prompt_lang", "style", "response", "normalized", "valid", "correct"):
            if c[fld] != b[fld]:
                raise ValueError(f"Baseline drift: {name}, {i}, {fld}")
        if (r["task"], r["prompt_lang"], r["style"], r["concept"], r["variant"]) != (
                b["task"], "zh", "upstream_en", params[b["task"]]["concept"], "zh"):
            raise ValueError(f"{name}: changed prompt/concept at {i}")
        j = params[b["task"]]
        if int(r["layer"]) != int(j["layer"]) or abs(float(r["alpha"]) - sign * float(j["alpha"])) > 1e-6:
            raise ValueError(f"{name}: wrong layer/alpha at {i}")
    return steered


def run_eval(data, jobs_path, target, mask, limit):
    cmd = [sys.executable, str(HERE / "evaluate.py"), "--data", str(data),
           "--split", "test", "--limit", str(limit), "--jobs", str(jobs_path),
           "--out", str(target), "--mask", mask]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def boot_ci(d, rng):
    if not d:
        return (float("nan"), float("nan"))
    a = np.asarray(d, dtype=float)
    take = rng.integers(0, len(a), size=(3000, len(a)))
    lo, hi = np.quantile(a[take].mean(axis=1) * 100, [.025, .975])
    return round(float(lo), 2), round(float(hi), 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=PKG / "outputs/parity_cvbench")
    p.add_argument("--run", default="zh_enprompt_wide20")
    p.add_argument("--limit", type=int, default=20, help="per task, same as previous test")
    a = p.parse_args()
    if a.limit < 1:
        p.error("--limit must be at least 1")
    data = a.data.resolve()
    run = data / "runs" / a.run
    jobs_path = run / "jobs_test.json"
    if not jobs_path.is_file():
        p.error(f"Missing previous tune selection: {jobs_path}")
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    if len(jobs) != 4 or {j["task"] for j in jobs} != TASKS:
        raise ValueError("Expected one frozen ZH winner for each of four CV-Bench tasks")
    if any(j["prompt_lang"] != "zh" or j["style"] != "upstream_en"
           or j["variant"] != "zh" or float(j["alpha"]) <= 0 for j in jobs):
        raise ValueError("Require ZH + upstream_en, original ZH vector, positive selected alpha")

    n = 4 * a.limit
    old = csv_rows(run / POS_FILES["both"])
    image = csv_rows(run / POS_FILES["image_token"])
    baseline = indexed(old, "none", n, "original baseline")
    for fld in ("prompt_lang", "style"):
        expected = {"prompt_lang": "zh", "style": "upstream_en"}[fld]
        if any(r[fld] != expected for r in baseline.values()):
            raise ValueError("Previously collected baseline used the wrong prompt protocol")
    if {r["task"] for r in baseline.values()} != TASKS or any(
            sum(r["task"] == t for r in baseline.values()) != a.limit for t in TASKS):
        raise ValueError("Expected the same per-task held-out sample count")
    check(baseline, old, jobs, n, "both", +1)
    check(baseline, image, jobs, n, "image_token", +1)

    neg_jobs = [{**j, "alpha": -float(j["alpha"])} for j in jobs]
    neg_jobs_path = run / "jobs_reverse_negative.json"
    neg_jobs_path.write_text(json.dumps(neg_jobs, indent=2), encoding="utf-8")
    out_dir = run / "analysis_direction_reversal"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Old script may or may not have run the positive text-only arm yet.
    text_pos = run / POS_FILES["text_token"]
    pos_text_rows = csv_rows(text_pos)
    if len(pos_text_rows) != 2 * n:
        run_eval(data, jobs_path, text_pos, "text_token", a.limit)

    for mask in MASKS:
        neg_file = run / NEG_FILES[mask]
        # evaluate.py resumes unfinished files; skip model loading for finished files.
        if len(csv_rows(neg_file)) != 2 * n:
            run_eval(data, neg_jobs_path, neg_file, mask, a.limit)

    values = {}
    for mask in MASKS:
        values[(mask, 1)] = check(baseline, csv_rows(run / POS_FILES[mask]), jobs, n, mask, 1)
        values[(mask, -1)] = check(baseline, csv_rows(run / NEG_FILES[mask]), jobs, n, mask, -1)

    ids = sorted(baseline)
    rng = np.random.default_rng(42)
    summary = []
    paired = []
    for t in ("overall", "count", "relation", "depth", "distance"):
        subset = ids if t == "overall" else [i for i in ids if baseline[i]["task"] == t]
        bcor = sum(int(baseline[i]["correct"]) for i in subset)
        for mask in MASKS:
            plus, minus = values[(mask, 1)], values[(mask, -1)]
            def cor(m):
                return sum(int(m[i]["correct"]) for i in subset)
            pc, nc = cor(plus), cor(minus)
            pos = [int(plus[i]["correct"]) - int(baseline[i]["correct"]) for i in subset]
            neg = [int(minus[i]["correct"]) - int(baseline[i]["correct"]) for i in subset]
            direction = [int(plus[i]["correct"]) - int(minus[i]["correct"]) for i in subset]
            summary.append({
                "task": t, "mask": mask, "n": len(subset),
                "baseline_acc": round(100 * bcor / len(subset), 2),
                "plus_acc": round(100 * pc / len(subset), 2),
                "minus_acc": round(100 * nc / len(subset), 2),
                "plus_gain_pp": round(100 * sum(pos) / len(subset), 2),
                "minus_gain_pp": round(100 * sum(neg) / len(subset), 2),
                "plus_minus_pp": round(100 * sum(direction) / len(subset), 2),
                "plus_gain_ci": str(boot_ci(pos, rng)),
                "minus_gain_ci": str(boot_ci(neg, rng)),
                "plus_minus_ci": str(boot_ci(direction, rng)),
                "plus_invalid": sum(int(plus[i]["valid"]) == 0 for i in subset),
                "minus_invalid": sum(int(minus[i]["valid"]) == 0 for i in subset),
                "plus_answer_flips": sum(plus[i]["normalized"] != baseline[i]["normalized"] for i in subset),
                "minus_answer_flips": sum(minus[i]["normalized"] != baseline[i]["normalized"] for i in subset),
                "plus_minus_disagree": sum(plus[i]["normalized"] != minus[i]["normalized"] for i in subset),
                "plus_wrong_to_right": sum(pos0 == 1 for pos0 in pos),
                "plus_right_to_wrong": sum(pos0 == -1 for pos0 in pos),
                "minus_wrong_to_right": sum(neg0 == 1 for neg0 in neg),
                "minus_right_to_wrong": sum(neg0 == -1 for neg0 in neg),
            })
    for i in ids:
        b = baseline[i]
        j = next(j for j in jobs if j["task"] == b["task"])
        row = {"id": i, "task": b["task"], "layer": j["layer"], "alpha_abs": j["alpha"],
               "concept": j["concept"], "baseline_response": b["response"],
               "baseline_choice": b["normalized"], "baseline_correct": b["correct"]}
        for mask in MASKS:
            for sign, label in ((1, "plus"), (-1, "minus")):
                r = values[(mask, sign)][i]
                pref = f"{mask}_{label}_"
                row.update({pref + "response": r["response"], pref + "choice": r["normalized"],
                            pref + "correct": r["correct"], pref + "valid": r["valid"]})
        paired.append(row)

    for filename, rows in (("summary.csv", summary), ("paired_predictions.csv", paired)):
        with (out_dir / filename).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    cols = ["task", "mask", "baseline_acc", "plus_acc", "minus_acc", "plus_gain_pp",
            "minus_gain_pp", "plus_minus_pp", "plus_minus_ci", "plus_answer_flips",
            "minus_answer_flips", "plus_minus_disagree"]
    md = ["# ZH Concept Vector — fixed-parameter sign reversal", "",
          "Chinese questions/options, English instruction and task prefix.",
          "Same held-out examples, model, concept, layer, and |alpha| in each pair.",
          "Plus is the original Tune-selected direction; Minus is -alpha.",
          "No extra tuning; accuracy CIs are paired sample bootstrap, not mechanistic proof.",
          "", "| " + " | ".join(cols) + " |",
          "| " + " | ".join("---" for _ in cols) + " |"]
    for r in summary:
        md.append("| " + " | ".join(str(r[k]) for k in cols) + " |")
    md += ["", "Key interpretation: positive gains with negative losses are consistent with "
           "direction-specific effects, but do not alone prove visual understanding. "
           "Inspect per-image responses and compare matched-random controls.",
           "", "No gold-vs-distractor logit margins are measured here.",
           "Result files: summary.csv and paired_predictions.csv."]
    report = out_dir / "SUMMARY.md"
    report.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(report.read_text(), flush=True)
    print("FINISHED:", report, flush=True)


if __name__ == "__main__":
    main()
