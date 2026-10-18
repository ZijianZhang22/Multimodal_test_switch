#!/usr/bin/env python3
"""Full CV-Bench English sign reversal using the paper's fixed BOTH-mask best settings.

Gan et al., ACL 2026, Figure 5(c), PaliGemma2-3B Mean Shift:
  Count=counting L9 alpha .6; Relation=spatial L5 alpha 1;
  Distance=spatial L8 alpha .2; Depth=spatial L11 alpha .6.
The SAME paper-selected settings are frozen across both/text/image masks.
No grid search or retuning; EN questions, EN vectors, official 150/task test.
"""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
PAPER = {
    "count": ("counting", 9, 0.6),
    "relation": ("spatial", 5, 1.0),
    "distance": ("spatial", 8, 0.2),
    "depth": ("spatial", 11, 0.6),
}
MASKS = ("both", "text_token", "image_token")


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def check(rows, ids, task_of, variant, sign, mask):
    selected = [
        r for r in rows
        if r["variant"] == variant and (
            variant == "none" or
            abs(float(r["alpha"]) - sign * PAPER[r["task"]][2]) < 1e-7
        )
    ]
    idx = {r["id"]: r for r in selected}
    if len(idx) != len(selected) or set(idx) != ids:
        raise ValueError(f"{mask} {variant} {sign}: expected {len(ids)} unique rows; got {len(selected)}")
    for i, r in idx.items():
        task = task_of[i]
        if r["task"] != task or r["prompt_lang"] != "en" or r["style"] != "native":
            raise ValueError(f"{mask}: mismatched task or prompt for {i}")
        if variant != "none":
            concept, layer, _ = PAPER[task]
            if r["concept"] != concept or int(r["layer"]) != layer:
                raise ValueError(f"{mask}: mismatched vector or layer for {i}")
    return idx


def ci(changes):
    x = np.array(changes, dtype=float)
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(x), size=(4000, len(x)))
    return [round(float(v), 2) for v in np.quantile(x[draws].mean(axis=1) * 100, [0.025, 0.975])]


def mcnemar_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(a, b) + 1)) / 2**n)


def analyze(out, masks, ids, task_of):
    base = None
    states = {}
    for mask in masks:
        rows = read_csv(out / f"predictions_{mask}.csv")
        b = check(rows, ids, task_of, "none", 0, mask)
        plus = check(rows, ids, task_of, "en", 1, mask)
        minus = check(rows, ids, task_of, "en", -1, mask)
        if len(rows) != 3 * len(ids):
            raise ValueError(f"{mask}: duplicated or unexpected rows")
        if base is None:
            base = b
        elif any((base[i]["response"], base[i]["normalized"], base[i]["correct"], base[i]["valid"])
                 != (b[i]["response"], b[i]["normalized"], b[i]["correct"], b[i]["valid"])
                 for i in ids):
            raise ValueError(f"Baselines differ across token masks at {mask}")
        states[mask] = (plus, minus)

    summary, pairs = [], []
    for mask, (plus, minus) in states.items():
        for task in ("overall", *PAPER):
            subset = sorted(ids) if task == "overall" else sorted(i for i in ids if task_of[i] == task)
            b = [int(base[i]["correct"]) for i in subset]
            p = [int(plus[i]["correct"]) for i in subset]
            m = [int(minus[i]["correct"]) for i in subset]
            dplus = [x - y for x, y in zip(p, b)]
            dminus = [x - y for x, y in zip(m, b)]
            dpm = [x - y for x, y in zip(p, m)]
            n = len(subset)
            better = sum(d > 0 for d in dpm)
            worse = sum(d < 0 for d in dpm)
            summary.append({
                "task": task, "mask": mask, "n": n,
                "baseline_acc": round(100 * sum(b) / n, 2),
                "plus_acc": round(100 * sum(p) / n, 2),
                "minus_acc": round(100 * sum(m) / n, 2),
                "plus_gain_pp": round(100 * sum(dplus) / n, 2),
                "minus_gain_pp": round(100 * sum(dminus) / n, 2),
                "plus_minus_pp": round(100 * sum(dpm) / n, 2),
                "plus_gain_ci95": str(ci(dplus)),
                "minus_gain_ci95": str(ci(dminus)),
                "plus_minus_ci95": str(ci(dpm)),
                "plus_better": better, "minus_better": worse,
                "mcnemar_plus_minus_p": round(mcnemar_p(better, worse), 6),
                "baseline_invalid": sum(int(base[i]["valid"]) == 0 for i in subset),
                "plus_invalid": sum(int(plus[i]["valid"]) == 0 for i in subset),
                "minus_invalid": sum(int(minus[i]["valid"]) == 0 for i in subset),
                "plus_changed": sum(plus[i]["normalized"] != base[i]["normalized"] for i in subset),
                "minus_changed": sum(minus[i]["normalized"] != base[i]["normalized"] for i in subset),
                "plus_minus_disagree": sum(plus[i]["normalized"] != minus[i]["normalized"] for i in subset),
            })
    for i in sorted(ids):
        task = task_of[i]
        concept, layer, alpha = PAPER[task]
        b = base[i]
        row = {"id": i, "task": task, "concept": concept, "layer": layer,
               "alpha_abs": alpha, "baseline_response": b["response"],
               "baseline_choice": b["normalized"], "baseline_correct": b["correct"],
               "baseline_valid": b["valid"]}
        for mask, (plus, minus) in states.items():
            for name, state in (("plus", plus), ("minus", minus)):
                v = state[i]
                prefix = mask + "_" + name + "_"
                row.update({prefix + "response": v["response"],
                            prefix + "choice": v["normalized"],
                            prefix + "correct": v["correct"],
                            prefix + "valid": v["valid"]})
        pairs.append(row)

    analysis = out / "analysis"
    analysis.mkdir(exist_ok=True)
    write_csv(analysis / "summary.csv", summary)
    write_csv(analysis / "paired_predictions.csv", pairs)
    columns = ("task", "mask", "n", "baseline_acc", "plus_acc", "minus_acc",
               "plus_gain_pp", "minus_gain_pp", "plus_minus_pp", "plus_minus_ci95",
               "plus_better", "minus_better", "mcnemar_plus_minus_p", "plus_invalid", "minus_invalid")
    md = ["# English full CV-Bench sign reversal with paper Figure 5(c) parameters", "",
          "EN questions, EN vectors, original paper BOTH-token grid winners.",
          "The same concept/layer/absolute alpha is used across masks (no retuning).",
          "", "| " + " | ".join(columns) + " |",
          "| " + " | ".join("---" for _ in columns) + " |"]
    md += ["| " + " | ".join(str(r[k]) for k in columns) + " |" for r in summary]
    md += ["", "Outputs: summary.csv and paired_predictions.csv.",
           "A change in accuracy alone does not establish enhanced visual understanding.",
           "Logit margins are not measured in this run.", ""]
    (analysis / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md), flush=True)
    print("FINISHED:", analysis / "SUMMARY.md", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=PKG / "outputs/parity_cvbench")
    p.add_argument("--run", default="en_paper_full_sign_reversal")
    p.add_argument("--limit", type=int, default=0, help="0=all 150 test questions/task; >0=smoke size/task")
    p.add_argument("--masks", default="both,text_token,image_token")
    a = p.parse_args()
    if a.limit < 0 or not a.run or "/" in a.run or a.run in (".", ".."):
        p.error("Invalid --limit or --run")
    masks = [x.strip() for x in a.masks.split(",") if x.strip()]
    if not masks or len(masks) != len(set(masks)) or set(masks) - set(MASKS):
        p.error("--masks must be a unique subset of both,text_token,image_token")
    data = a.data.resolve()
    manifest = json.loads((data / "dataset_manifest.json").read_text())
    if manifest.get("protocol_version") != "parity-2026-10-18b":
        raise RuntimeError("Stale dataset protocol: rerun current prepare.py")
    vector_meta = json.loads((data / "vectors/manifest.json").read_text())
    for task, (concept, layer, alpha) in PAPER.items():
        if concept not in vector_meta.get("variants", {}).get("en", []):
            raise RuntimeError(f"Missing original EN vector for {concept}")
        vec = data / "vectors" / "en" / concept / f"layer_{layer}_feature.pt"
        if not vec.exists():
            raise FileNotFoundError(vec)
    samples = [json.loads(line) for line in (data / "visual_eval.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = [s for s in samples if s["split"] == "test"]
    by_task = {t: [s for s in samples if s["task"] == t] for t in PAPER}
    if any(len(v) != 150 for v in by_task.values()):
        raise RuntimeError("Expected original paper test: 150 questions per CV-Bench task")
    if a.limit:
        by_task = {t: v[:a.limit] for t, v in by_task.items()}
    task_of = {s["id"]: t for t, v in by_task.items() for s in v}
    expected = sum(len(v) for v in by_task.values())
    if len(task_of) != expected or (a.limit == 0 and expected != 600):
        raise RuntimeError("Invalid CV-Bench test IDs or sample counts")
    ids = set(task_of)
    out = data / "runs" / a.run
    out.mkdir(parents=True, exist_ok=True)

    jobs = [dict(task=t, prompt_lang="en", style="native", variant="en",
                 concept=c, layer=l, alpha=sign * alpha)
            for t, (c, l, alpha) in PAPER.items() for sign in (1, -1)]
    path = out / "jobs_paper_en.json"
    contents = json.dumps(jobs, indent=2) + "\n"
    if path.exists() and path.read_text() != contents:
        raise RuntimeError("Existing jobs differ; choose another --run")
    path.write_text(contents)
    metadata = {"source": "Gan et al., ACL 2026, Figure 5(c) PaliGemma2-3B Mean Shift",
                "source_url": "https://aclanthology.org/2026.acl-long.1861.pdf",
                "question_lang": "en", "vector_lang": "en", "masks": masks,
                "paper_best_both": {t: list(v) for t, v in PAPER.items()}, "test_limit": a.limit, "test_size": expected,
                "retuned": False}
    meta_path = out / "experiment_manifest.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise RuntimeError("Existing experiment metadata differs; use a new --run")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print("Source: Paper Figure 5(c). Params:", PAPER, flush=True)
    print("Samples:", expected, "masks:", masks, "steering generations:", 2 * expected * len(masks), flush=True)

    expected_hash = hashlib.sha256(json.dumps(jobs, sort_keys=True).encode()).hexdigest()
    for mask in masks:
        output = out / f"predictions_{mask}.csv"
        rows = read_csv(output)
        if len(rows) == 3 * expected:
            settings = json.loads(output.with_suffix(".settings.json").read_text())
            if (settings.get("mask"), settings.get("limit"), settings.get("jobs_sha256")) != (mask, a.limit, expected_hash):
                raise RuntimeError(f"Existing prediction file has different settings: {output}")
            print("Reusing complete:", output, flush=True)
        else:
            cmd = [sys.executable, str(HERE / "evaluate.py"), "--data", str(data),
                   "--split", "test", "--limit", str(a.limit), "--jobs", str(path),
                   "--mask", mask, "--out", str(output)]
            print("RUN:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
    analyze(out, masks, ids, task_of)


if __name__ == "__main__":
    main()
