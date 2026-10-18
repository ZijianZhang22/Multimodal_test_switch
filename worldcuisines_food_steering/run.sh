#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
DATA="${DATA:-outputs/smoke}"
LAYERS="${LAYERS:-8,16}"
ALPHAS="${ALPHAS:-0.1,0.3}"
SPLIT="${SPLIT:-val}"
if [[ ! -x .venv/bin/python ]]; then python -m venv .venv; fi
PY="$PWD/.venv/bin/python"
if [[ "${INSTALL:-1}" == 1 ]]; then "$PY" -m pip install -r requirements.txt; fi
"$PY" -m unittest discover -s tests -v
"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU required"; print(torch.cuda.get_device_name(0))'
if [[ ! -f "$DATA/samples.jsonl" ]]; then
  "$PY" prepare.py --data "$DATA" --source-split "${SOURCE_SPLIT:-test_small}" \
    --n-val "${N_VAL:-50}" --n-test "${N_TEST:-50}" --seed "${SEED:-42}"
fi
if [[ ! -f "$DATA/vectors/manifest.json" ]]; then
  "$PY" extract.py --data "$DATA" --layers "$LAYERS" --concepts "${CONCEPTS:-concepts.json}"
fi
args=(--data "$DATA" --split "$SPLIT" --layers "$LAYERS" --alphas "$ALPHAS"
      --mask "${MASK:-both}" --limit "${LIMIT:-0}" --max-tokens "${MAX_TOKENS:-16}")
if [[ "$SPLIT" == test ]]; then args+=(--selection "$DATA/selection.json"); fi
"$PY" evaluate.py "${args[@]}"
"$PY" report.py --data "$DATA" --split "$SPLIT"
