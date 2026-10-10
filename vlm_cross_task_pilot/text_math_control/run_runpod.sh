#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${DATA:-text_math50}"
python -m pip install -r requirements_pilot.txt
python -m py_compile text_math_control/*.py
python text_math_control/test_offline.py
python text_math_control/prepare.py --out "$DATA/data/manifest.jsonl" \
  --count "${COUNT:-50}" --seed "${SEED:-1043}"
python text_math_control/run.py --manifest "$DATA/data/manifest.jsonl" \
  --out "$DATA/results/predictions.jsonl" \
  --model "${MODEL:-Qwen/Qwen3-VL-8B-Thinking}" \
  --samples "${SAMPLES:-1}" --max-new-tokens "${MAX_NEW_TOKENS:-12288}" \
  --temperature "${TEMPERATURE:-0.7}" --top-p "${TOP_P:-0.95}" --seed "${SEED:-1043}"
python text_math_control/score.py --predictions "$DATA/results/predictions.jsonl" \
  --out-dir "$DATA/results/analysis"
echo "DONE: $DATA/results/analysis/summary.csv"
