#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Reuses existing pilot_math20/data/pairs.jsonl and the original Idis files.
# This does not regenerate questions or change images.
python run_dual_image_q1_control.py \
  --pairs-file "${PAIRS_FILE:-pilot_math20/data/pairs.jsonl}" \
  --out "${OUT:-pilot_math20_dual_image_q1/results/predictions.jsonl}" \
  --variant "${VARIANT:-conflicting}" \
  --conditions "${CONDITIONS:-joint_text,joint_image_q1}" \
  --samples "${SAMPLES:-1}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-4096}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --top-p "${TOP_P:-0.95}" \
  --model "${MODEL:-Qwen/Qwen3-VL-8B-Thinking}" \
  "$@"
