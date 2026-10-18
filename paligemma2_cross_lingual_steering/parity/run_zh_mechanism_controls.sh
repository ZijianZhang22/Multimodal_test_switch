#!/usr/bin/env bash
# Paired ZH-question / English-task-prompt mechanism controls.
# ZH text-only, EN both (raw and ZH-norm matched), random both (ZH-norm matched).
# Freezes per-task concept/layer/alpha from previous ZH tuning.
set -euo pipefail

REPO="${REPO:-/workspace/Multimodal_test_switch}"
PKG="$REPO/paligemma2_cross_lingual_steering"
DATA="${DATA:-$PKG/outputs/parity_cvbench}"
RUN="${RUN:-zh_enprompt_wide20}"
LIMIT="${LIMIT:-20}"
SEEDS="${SEEDS:-0,1,2}"
OUT="$DATA/runs/$RUN"
PY="$PKG/.venv-parity/bin/python"
cd "$PKG/parity"

[[ -x "$PY" ]] || { echo "Missing Python env: $PY"; exit 1; }
for f in "$OUT/jobs_test.json" "$OUT/test.csv" "$OUT/test_image_only.csv" \
         "$DATA/vectors/manifest.json" "$DATA/visual_eval.jsonl"; do
  [[ -f "$f" ]] || { echo "Missing prior-run file: $f"; exit 1; }
done

# Validate the previous paired run and generate direction controls on CPU.
"$PY" - "$DATA" "$OUT" "$LIMIT" "$SEEDS" <<'PY'
import csv
import json
import sys
import zlib
from pathlib import Path
import torch

data, out = map(Path, sys.argv[1:3])
limit = int(sys.argv[3])
seeds = [int(x) for x in sys.argv[4].split(',') if x.strip()]
assert limit > 0 and seeds and len(seeds) == len(set(seeds)) and min(seeds) >= 0
jobs = json.loads((out / "jobs_test.json").read_text())
assert len(jobs) == 4
assert {j["task"] for j in jobs} == {"count", "relation", "depth", "distance"}
assert all(j["prompt_lang"] == "zh" and j["style"] == "upstream_en"
           and j["variant"] == "zh" for j in jobs)

def load_rows(name, variant):
    with (out / name).open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["variant"] == variant]
    result = {r["id"]: r for r in rows}
    assert len(rows) == len(result) == 4 * limit, (name, variant, len(rows))
    return result

old_baseline = load_rows("test.csv", "none")
new_baseline = load_rows("test_image_only.csv", "none")
assert old_baseline.keys() == new_baseline.keys()
assert all((old_baseline[i]["response"], old_baseline[i]["correct"]) ==
           (new_baseline[i]["response"], new_baseline[i]["correct"])
           for i in old_baseline), "Existing baselines differ: do not compare!"
for name in ("test.csv", "test_image_only.csv"):
    assert load_rows(name, "zh").keys() == old_baseline.keys()

all_jobs, metadata = [], []
vec = data / "vectors"
for j in jobs:
    concept, layer = j["concept"], int(j["layer"])
    z = torch.load(vec / "zh" / concept / f"layer_{layer}_feature.pt",
                   map_location="cpu", weights_only=True).float()
    e = torch.load(vec / "en" / concept / f"layer_{layer}_feature.pt",
                   map_location="cpu", weights_only=True).float()
    assert z.shape == e.shape and z.norm() > 0 and e.norm() > 0
    zn = z.norm().item()

    en_matched = e * (zn / e.norm().item())
    d = vec / "en_normzh" / concept
    d.mkdir(parents=True, exist_ok=True)
    torch.save(en_matched.contiguous(), d / f"layer_{layer}_feature.pt")
    metadata.append({"task": j["task"], "variant": "en_normzh", "layer": layer,
                     "concept": concept, "target_norm": zn,
                     "actual_norm": en_matched.norm().item()})
    for variant in ("en", "en_normzh"):
        all_jobs.append({**j, "variant": variant})

    for seed in seeds:
        variant = f"random_zh_s{seed}"
        generator = torch.Generator().manual_seed(
            99173 + 1000003 * seed + 7919 * layer + zlib.crc32(concept.encode()))
        r = torch.randn(z.shape, generator=generator)
        r *= zn / r.norm().item()
        d = vec / variant / concept
        d.mkdir(parents=True, exist_ok=True)
        torch.save(r.contiguous(), d / f"layer_{layer}_feature.pt")
        assert abs(r.norm().item() / zn - 1) < 1e-5
        metadata.append({"task": j["task"], "variant": variant, "layer": layer,
                         "concept": concept, "target_norm": zn,
                         "actual_norm": r.norm().item(),
                         "cosine_with_zh": torch.nn.functional.cosine_similarity(
                             r[None], z[None]).item()})
        all_jobs.append({**j, "variant": variant})

(out / "jobs_zh_text_only.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")
(out / "jobs_mechanism_controls_both.json").write_text(json.dumps(all_jobs, indent=2), encoding="utf-8")
(out / "mechanism_vector_controls.json").write_text(
    json.dumps({"seeds": seeds, "entries": metadata}, indent=2), encoding="utf-8")
print(f"Verified {4*limit} paired prior Test samples.")
print(f"New runs: {4*limit} ZH text-only steering + "
      f"{4*limit*(2+len(seeds))} EN/random both steering generations.")
print("Layer/Alpha fixed; no new tuning or text-model loading.", flush=True)
PY

"$PY" evaluate.py --data "$DATA" --split test --limit "$LIMIT" \
    --jobs "$OUT/jobs_zh_text_only.json" \
    --out "$OUT/test_zh_text_only.csv" --mask text_token

"$PY" evaluate.py --data "$DATA" --split test --limit "$LIMIT" \
    --jobs "$OUT/jobs_mechanism_controls_both.json" \
    --out "$OUT/test_mechanism_controls_both.csv" --mask both

"$PY" analyze.py \
    --pred "$OUT/test.csv" "$OUT/test_image_only.csv" \
           "$OUT/test_zh_text_only.csv" "$OUT/test_mechanism_controls_both.csv" \
    --out "$OUT/analysis_mechanism_controls" > "$OUT/analysis_mechanism_controls.log"

# Strict paired summary across all old/new conditions.
"$PY" - "$OUT" <<'PY'
import csv
from pathlib import Path

out = Path(__import__("sys").argv[1])
def read(path, variant):
    with (out / path).open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["variant"] == variant]
    result = {r["id"]: r for r in rows}
    assert result and len(rows) == len(result), (path, variant)
    return result

base = read("test.csv", "none")
controls_file = "test_mechanism_controls_both.csv"
with (out / controls_file).open(newline="", encoding="utf-8") as f:
    random_variants = sorted({r["variant"] for r in csv.DictReader(f)
                              if r["variant"].startswith("random_zh_s")})
conditions = {
    "Baseline": base,
    "ZH Both": read("test.csv", "zh"),
    "ZH Image-only": read("test_image_only.csv", "zh"),
    "ZH Text-only": read("test_zh_text_only.csv", "zh"),
    "EN Both": read(controls_file, "en"),
    "EN matched-to-ZH norm Both": read(controls_file, "en_normzh"),
}
conditions.update({v + " Both": read(controls_file, v) for v in random_variants})
for file in ("test_image_only.csv", "test_zh_text_only.csv", controls_file):
    b = read(file, "none")
    assert b.keys() == base.keys(), f"Different sample IDs: {file}"
    assert all((b[i]["response"], b[i]["correct"], b[i]["valid"]) ==
               (base[i]["response"], base[i]["correct"], base[i]["valid"]) for i in base), \
               f"Baseline mismatch: {file}"
for name, items in conditions.items():
    assert items.keys() == base.keys(), f"Different test examples in {name}"
    assert all(items[i]["task"] == base[i]["task"] and
               items[i]["prompt_lang"] == "zh" and items[i]["style"] == "upstream_en"
               for i in base), f"Prompt mismatch: {name}"

def summarize(items, ids):
    n = len(ids)
    correct = sum(int(items[i]["correct"]) for i in ids)
    previous = sum(int(base[i]["correct"]) for i in ids)
    invalid = sum(int(items[i]["valid"]) == 0 for i in ids)
    flips = sum(items[i]["normalized"] != base[i]["normalized"] for i in ids)
    changed = sum(items[i]["response"] != base[i]["response"] for i in ids)
    improved = sum(int(items[i]["correct"]) > int(base[i]["correct"]) for i in ids)
    worsened = sum(int(items[i]["correct"]) < int(base[i]["correct"]) for i in ids)
    return (n, correct, 100 * correct / n, 100 * (correct - previous) / n,
            invalid, flips, changed, improved, worsened)

ids = sorted(base)
header = "| Condition | N | Correct | Accuracy | Gain pp | Invalid | Choice changes | Raw changes | Wrong→right | Right→wrong |"
lines = ["# ZH question / EN instruction: fixed-parameter mechanism controls", "",
         "All conditions share the same held-out images and prompts, with layers and alphas "
         "frozen from the previous ZH Tune run. Three random seeds match the ZH vector norm "
         "at the same layer/concept. EN also has an optional norm-matched variant.", "",
         "## Overall results", "", header, "|" + "---|" * 10]
for label, items in conditions.items():
    n, c, acc, gain, invalid, flips, changed, good, bad = summarize(items, ids)
    lines.append(f"| {label} | {n} | {c} | {acc:.2f}% | {gain:+.2f} | "
                 f"{invalid} | {flips} | {changed} | {good} | {bad} |")
lines += ["", "## Per-task results", "",
          "| Task | Condition | Correct | Accuracy | Gain pp | Choice changes |",
          "|" + "---|" * 6]
for task in ("count", "relation", "depth", "distance"):
    task_ids = [i for i in ids if base[i]["task"] == task]
    for label, items in conditions.items():
        _, c, acc, gain, _, flips, _, _, _ = summarize(items, task_ids)
        lines.append(f"| {task} | {label} | {c}/{len(task_ids)} | {acc:.2f}% | "
                     f"{gain:+.2f} | {flips} |")
lines += ["", "Note: discrete answer changes alone cannot establish improved visual reasoning. "
          "Small test sets also have high statistical uncertainty.", ""]
report = out / "analysis_mechanism_controls" / "MECHANISM_COMPARISON.md"
report.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines), flush=True)
print("Saved:", report, flush=True)
PY

echo "DONE: $OUT/analysis_mechanism_controls/MECHANISM_COMPARISON.md"
echo "Archive all previous+new results:"
echo "tar -czf /workspace/zh_mechanism_controls_results.tar.gz -C \"$DATA\" \"runs/$RUN\""
