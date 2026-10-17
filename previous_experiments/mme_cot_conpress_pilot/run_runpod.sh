#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Use the RunPod image with CUDA PyTorch already installed.
python -m pip install -r requirements.txt
python smoke_test.py

EXAMPLES="${EXAMPLES:-12}"
PER_CATEGORY="${PER_CATEGORY:-2}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Thinking}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
OUT_DIR="${OUT_DIR:-pilot}"

python prepare.py --out "${OUT_DIR}/data" --max-examples "$EXAMPLES" \
  --per-category "$PER_CATEGORY" --seed "${SEED:-42}" \
  --categories "${CATEGORIES:-all}"

# One generation per sample, with generous configurable max output length.
python run.py --data "${OUT_DIR}/data/pairs.jsonl" \
  --out "${OUT_DIR}/results/predictions.jsonl" \
  --temperature "${TEMPERATURE:-0}" --seed "${SEED:-42}" \
  --max-new-tokens "$MAX_NEW_TOKENS" --model "$MODEL"

python score.py --input "${OUT_DIR}/results/predictions.jsonl" \
  --out "${OUT_DIR}/results"
