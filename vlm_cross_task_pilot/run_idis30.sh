#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -r requirements_pilot.txt
OUT="${OUT:-idis30}"
COUNT="${COUNT:-30}"
ORIGINAL_ARGS=()
if [[ -n "${CLEAN_ROOT:-}" ]]; then ORIGINAL_ARGS=(--clean-root "$CLEAN_ROOT"); fi
python idis30_prepare.py --out "$OUT/data" --count "$COUNT" --n-distractors "${N_DISTRACTORS:-4}" "${ORIGINAL_ARGS[@]}"
python idis30_run.py --manifest "$OUT/data/manifest.jsonl" --out "$OUT/results/predictions.jsonl" \
  --model "${MODEL:-Qwen/Qwen3-VL-8B-Thinking}" --samples "${SAMPLES:-1}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-4096}"
python idis30_score.py --predictions "$OUT/results/predictions.jsonl"
