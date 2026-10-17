#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
DATA_DIR="${DATA_DIR:-toy_cot_100/data}"
RESULT_DIR="${RESULT_DIR:-toy_cot_100/results}"
mkdir -p "$RESULT_DIR"
if [[ ! -f "$DATA_DIR/examples.jsonl" ]]; then
  python toy_cot_100/prepare.py --out_dir "$DATA_DIR" --n_problems 100 --seed 42
fi
python toy_cot_100/run.py --data "$DATA_DIR/examples.jsonl" --out "$RESULT_DIR/predictions.jsonl" --model "$MODEL" --reasoning_mode cot --max_new_tokens "${MAX_NEW_TOKENS:-4096}" --dtype "${DTYPE:-auto}" --attn_implementation "${ATTN:-sdpa}" --schedules TTTT,TVVV,VTVV,VVTV,VVVT --resume
python toy_cot_100/analyze.py --input "$RESULT_DIR/predictions.jsonl" --out_dir "$RESULT_DIR"
