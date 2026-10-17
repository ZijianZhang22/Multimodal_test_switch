#!/usr/bin/env bash
# Run ONLY missing visual_first on the EXACT saved 50 MathVerse problems.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${DATA:-idis_math50_new}"
test -s "$DATA/data/manifest.jsonl" || { echo "Missing existing 50-question manifest" >&2; exit 1; }
test -s "$DATA/results/predictions.jsonl" || { echo "Missing existing A/C predictions" >&2; exit 1; }
python idis_math10/test_offline.py
python idis_math10/run.py \
  --manifest "$DATA/data/manifest.jsonl" \
  --out "$DATA/results/visual_first_predictions.jsonl" \
  --model "${MODEL:-Qwen/Qwen3-VL-8B-Thinking}" \
  --conditions visual_first --skip-score \
  --samples "${SAMPLES:-1}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-12288}" \
  --max-image-side "${MAX_IMAGE_SIDE:-1536}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --top-p "${TOP_P:-0.95}" --seed "${SEED:-1043}"
python idis_math10/score_three.py \
  --existing "$DATA/results/predictions.jsonl" \
  --visual-first "$DATA/results/visual_first_predictions.jsonl" \
  --out-dir "$DATA/results/abc_comparison"
