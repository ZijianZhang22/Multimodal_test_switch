#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -r requirements_pilot.txt
BENCHMARK="${BENCHMARK:-gsm8k}"
OUT="${OUT:-pilot_image_text_${BENCHMARK}}"
python run_image_text_q1_control.py \
  --benchmark "$BENCHMARK" \
  --output-dir "$OUT" \
  --pairs "${PAIRS:-8}" \
  --samples "${SAMPLES:-1}" \
  --seed "${SEED:-42}" \
  --math-min-level "${MATH_MIN_LEVEL:-3}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-4096}" \
  --conditions "${CONDITIONS:-q2_only,text_q1_q2,image_q2_only,image_text_q1_q2,image_q1_q2,image_q1_only,text_q1_only}" \
  --model "${MODEL:-Qwen/Qwen3-VL-8B-Thinking}"
