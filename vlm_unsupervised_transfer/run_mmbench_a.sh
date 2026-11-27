#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-$PWD/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then echo "Missing previous package .venv: $PY" >&2; exit 1; fi
EXP="${EXP:-/workspace/vlm_A_zh_seed42/experiment}"
DATA="${MMBENCH_DATA:-/workspace/vlm_A_zh_seed42/mmbench_dev_bilingual}"
OUT="${MMBENCH_OUT:-/workspace/vlm_A_zh_seed42/mmbench_A_search}"
mkdir -p "$DATA" "$OUT"
PREP_EXTRA=()
if [[ -n "${MMBENCH_EN_TSV:-}" ]]; then PREP_EXTRA+=(--en-tsv "$MMBENCH_EN_TSV"); fi
if [[ -n "${MMBENCH_CN_TSV:-}" ]]; then PREP_EXTRA+=(--cn-tsv "$MMBENCH_CN_TSV"); fi
if [[ "${ALLOW_UNPINNED:-0}" == "1" ]]; then PREP_EXTRA+=(--allow-unpinned); fi
"$PY" -u prepare_mmbench.py --out "$DATA" --tune-size "${TUNE_SIZE:-200}" --seed "${SEED:-42}" "${PREP_EXTRA[@]}"
"$PY" -u run_mmbench_a.py --experiment "$EXP" --data "$DATA" --out "$OUT" \
  --layers "${LAYERS:-8,13,18}" --alphas "${ALPHAS:-0.15,0.3,0.6}" --stage "${STAGE:-all}"
