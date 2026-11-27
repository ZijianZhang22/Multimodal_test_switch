#!/usr/bin/env bash
# D1-D4: CPU scoring from frozen H1/Flickr caches; D5 optional new GPU extraction.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-$PWD/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then echo "Missing existing environment: $PY" >&2; exit 1; fi
ARGS=(--experiment "${EXP:-/workspace/vlm_A_zh_seed42/experiment}"
      --h1 "${H1_ROOT:-/workspace/vlm_A_zh_seed42/h1_geometry_n200}"
      --mmbench "${MMBENCH:-/workspace/vlm_A_zh_seed42/mmbench_dev_bilingual}"
      --out "${H1_FOLLOWUP_OUT:-/workspace/vlm_A_zh_seed42/h1_followups}"
      --stage "${H1_FOLLOWUP_STAGE:-diagnostics}")
if [[ -n "${H1_EXTERNAL_PAIRS:-}" ]]; then
  ARGS+=(--external-pairs "$H1_EXTERNAL_PAIRS" --external-n "${H1_EXTERNAL_N:-200}")
fi
"$PY" -u h1_followups.py "${ARGS[@]}"