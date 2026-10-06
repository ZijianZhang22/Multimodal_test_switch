#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   MODEL=Qwen/Qwen3-VL-8B-Instruct LABEL=qwen3vl8b ./run_depth_scaling_model.sh
#   MODEL=Qwen/Qwen3-VL-32B-Instruct LABEL=qwen3vl32b ./run_depth_scaling_model.sh

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
LABEL="${LABEL:-qwen3vl8b}"
DATA="${DATA:-data_depth_scaling/depth_scaling_examples.jsonl}"
OUT="${OUT:-results_depth_scaling/${LABEL}.jsonl}"
DTYPE="${DTYPE:-auto}"
ATTN="${ATTN:-sdpa}"
CONDITIONS="${CONDITIONS:-}"
HOPS="${HOPS:-}"

cd "$(dirname "$0")"

ARGS=(
  --data "$DATA"
  --out "$OUT"
  --model "$MODEL"
  --model_label "$LABEL"
  --dtype "$DTYPE"
  --attn_implementation "$ATTN"
  --resume
)

if [[ -n "$CONDITIONS" ]]; then
  ARGS+=(--conditions "$CONDITIONS")
fi
if [[ -n "$HOPS" ]]; then
  ARGS+=(--hops "$HOPS")
fi

python run_qwen3vl.py "${ARGS[@]}"

echo "Saved: $OUT"
