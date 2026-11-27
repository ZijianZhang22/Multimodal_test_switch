#!/usr/bin/env bash
# Resume frozen A: only Chinese baseline vs fixed L8 visual OT.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
PY="${PY:-$PWD/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then echo "Missing package .venv. Run the original install first." >&2; exit 1; fi
EXP="${EXP:-/workspace/vlm_A_zh_seed42/experiment}"
DATA="${EXPANDED_DATA:-/workspace/vlm_A_zh_seed42/cvbench_extended1500}"
RESULTS="${RESULTS:-/workspace/vlm_A_zh_seed42/visual_only_1500}"
TOTAL="${TARGET_TOTAL:-1500}"
mkdir -p "$DATA" "$RESULTS"
"$PY" -u prepare_extended_cvbench.py \
  --original-val "$(dirname "$EXP")/cvbench/val.jsonl" \
  --original-test "$(dirname "$EXP")/cvbench/test.jsonl" \
  --out "$DATA" --total "$TOTAL" --mt-batch "${MT_BATCH:-8}"
"$PY" -u run_visual_large.py --experiment "$EXP" \
  --extended "$DATA" --out "$RESULTS"
