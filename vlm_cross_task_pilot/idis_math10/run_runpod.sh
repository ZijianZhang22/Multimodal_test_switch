#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -r requirements_pilot.txt
python -m py_compile idis_math10/prepare.py idis_math10/run.py idis_math10/score.py idis_math10/test_offline.py
python idis_math10/test_offline.py
OUT="${OUT:-idis_math10}"
VARIANT="${VARIANT:-irrelevant}"
N_DISTRACTORS="${N_DISTRACTORS:-4}"
COUNT="${COUNT:-10}"
SEED="${SEED:-42}"
python idis_math10/prepare.py --out "$OUT/data" --count "$COUNT" \
  --variant "$VARIANT" --n-distractors "$N_DISTRACTORS" --seed "$SEED"
python idis_math10/run.py \
  --manifest "$OUT/data/manifest.jsonl" \
  --out "$OUT/results/predictions.jsonl" \
  --model "${MODEL:-Qwen/Qwen3-VL-8B-Thinking}" \
  --samples "${SAMPLES:-1}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-8192}" \
  --max-image-side "${MAX_IMAGE_SIDE:-1536}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --top-p "${TOP_P:-0.95}" --seed "$SEED"
python idis_math10/score.py --predictions "$OUT/results/predictions.jsonl"
echo "Done. Check $OUT/results/pairs.csv and $OUT/results/summary.csv"
