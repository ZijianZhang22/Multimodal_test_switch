#!/usr/bin/env bash
# Screen with baseline only, then rerun selected high-token questions paired at a larger cap.
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -r requirements.txt
python smoke_test.py

OUT_DIR="${OUT_DIR:-long_reasoning_64k}"
SCREEN_EXAMPLES="${SCREEN_EXAMPLES:-80}"
SCREEN_PER_CATEGORY="${SCREEN_PER_CATEGORY:-20}"
SCREEN_MAX_NEW_TOKENS="${SCREEN_MAX_NEW_TOKENS:-8192}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-65536}"
MIN_THINKING="${MIN_THINKING:-3000}"
MAX_LONG="${MAX_LONG:-24}"
SHORT_CONTROLS="${SHORT_CONTROLS:-8}"
SEED="${SEED:-42}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Thinking}"
CATEGORIES="${CATEGORIES:-General Reasoning,Logic,Math,Science}"

echo "===== PREPARE ====="
if [[ ! -f "$OUT_DIR/screen/data/pairs.jsonl" ]]; then
  python prepare.py --out "$OUT_DIR/screen/data" --max-examples "$SCREEN_EXAMPLES" \
    --per-category "$SCREEN_PER_CATEGORY" --seed "$SEED" --categories "$CATEGORIES"
fi

echo "===== SCREEN BASELINES (short cap) ====="
python run.py --data "$OUT_DIR/screen/data/pairs.jsonl" \
  --out "$OUT_DIR/screen/results/predictions.jsonl" --conditions baseline \
  --temperature "${TEMPERATURE:-0}" --seed "$SEED" \
  --max-new-tokens "$SCREEN_MAX_NEW_TOKENS" --model "$MODEL"

echo "===== SELECT LONG AND SHORT CONTROL QUESTIONS ====="
python select_long_cases.py --data "$OUT_DIR/screen/data/pairs.jsonl" \
  --predictions "$OUT_DIR/screen/results/predictions.jsonl" \
  --out "$OUT_DIR/selected/pairs.jsonl" \
  --manifest "$OUT_DIR/selected/selection.csv" \
  --min-thinking "$MIN_THINKING" --max-long "$MAX_LONG" \
  --short-controls "$SHORT_CONTROLS"

echo "===== FULL PAIRED INFERENCE ====="
python run.py --data "$OUT_DIR/selected/pairs.jsonl" \
  --out "$OUT_DIR/results/predictions.jsonl" \
  --temperature "${TEMPERATURE:-0}" --seed "$SEED" \
  --max-new-tokens "$MAX_NEW_TOKENS" --model "$MODEL"

echo "===== SUMMARIZE ====="
python score.py --input "$OUT_DIR/results/predictions.jsonl" --out "$OUT_DIR/results"
echo "===== DONE ====="
