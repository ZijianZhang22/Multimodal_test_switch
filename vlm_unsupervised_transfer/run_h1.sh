#!/usr/bin/env bash
# Evaluate H1 directly; reuse PaliGemma2 and existing pretrained unpaired maps.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-$PWD/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then echo "Missing existing .venv at $PY" >&2; exit 1; fi
EXP="${EXP:-/workspace/vlm_A_zh_seed42/experiment}"
MMBENCH="${MMBENCH:-/workspace/vlm_A_zh_seed42/mmbench_dev_bilingual}"
OUT="${H1_OUT:-/workspace/vlm_A_zh_seed42/h1_geometry_n200}"
"$PY" -u h1_geometry.py \
  --experiment "$EXP" --mmbench "$MMBENCH" --out "$OUT" \
  --n "${H1_N:-200}" --layers "${LAYERS:-8,13,18}" \
  --seed "${H1_SEED:-42}" --stage "${H1_STAGE:-all}"