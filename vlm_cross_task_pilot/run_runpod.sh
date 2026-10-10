#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# MODE=idis preserves the original Idis pilot.
if [[ "${MODE:-idis}" == "image_text" ]]; then
  exec bash run_image_text_control.sh
fi
# Precondition: RunPod has CUDA PyTorch installed (avoid overwriting it)
python -m pip install -r requirements_pilot.txt
BENCHMARK="${BENCHMARK:-gsm8k}"
PAIRS="${PAIRS:-6}"
SAMPLES="${SAMPLES:-1}"
N_CLASSES="${N_CLASSES:-2}"
OUT="pilot_${BENCHMARK}"
EXTRA_ORIGINAL=()
if [[ -n "${ORIGINAL_ROOT:-}" ]]; then
  EXTRA_ORIGINAL=(--original-root "$ORIGINAL_ROOT")
fi
python prepare.py --out "${OUT}/data" --benchmark "$BENCHMARK" --pairs "$PAIRS" --classes "$N_CLASSES" \
  "${EXTRA_ORIGINAL[@]}"
python run.py --pairs "${OUT}/data/pairs.jsonl" --out "${OUT}/results/predictions.jsonl" \
  --conditions "${CONDITIONS:-q2_only,q1_only,image_q2,joint_vt}" \
  --samples "$SAMPLES" --max-new-tokens "${MAX_NEW_TOKENS:-1536}" \
  --model "${MODEL:-Qwen/Qwen3-VL-8B-Thinking}"
python score.py --input "${OUT}/results/predictions.jsonl" --out "${OUT}/results"
