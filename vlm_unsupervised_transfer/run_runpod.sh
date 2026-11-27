#!/usr/bin/env bash
# Full validation and one fixed test, no LoRA or backbone training.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"
ROOT="$(cd .. && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_DIR="${ENV_DIR:-$HERE/.venv}"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then "$PYTHON_BIN" -m venv --system-site-packages "$ENV_DIR"; fi
PY="$ENV_DIR/bin/python"
if [[ "${INSTALL:-1}" == 1 ]]; then "$PY" -m pip install -r requirements.txt; fi
"$PY" - <<'PY'
import torch, transformers
assert transformers.__version__=='4.51.3',transformers.__version__
assert torch.cuda.is_available(), 'CUDA GPU required to run PaliGemma2'
print('CUDA:',torch.cuda.get_device_name(0),'transformers',transformers.__version__,flush=True)
PY
ARM="${ARM:-all}"
if [[ "$ARM" != a && "$ARM" != all ]]; then echo "ARM must be a or all" >&2; exit 2; fi
LANGUAGE="${LANGUAGE:-zh}"
SEED="${SEED:-42}"
OUT="${OUT:-$HERE/outputs/${LANGUAGE}_seed${SEED}}"
POOLS="${POOLS:-$OUT/flickr}"
EVAL="${EVAL:-$OUT/cvbench}"
EXPERIMENT="${EXPERIMENT:-$OUT/experiment}"
RAW="${RAW:-$ROOT/visual_anchor/real_work_clip_native/raw}"
LAYERS="${LAYERS:-8,13,18}"
DIM="${DIM:-16}"
N_SOURCE="${N_SOURCE:-256}"
N_ENGLISH="${N_ENGLISH:-256}"
N_VISUAL="${N_VISUAL:-256}"
STRENGTHS="${STRENGTHS:-0.15,0.3,0.6}"
ORACLE_STRENGTHS="${ORACLE_STRENGTHS:-0.3,0.6,1.0}"
MODEL="${MODEL:-google/paligemma2-3b-mix-448}"
mkdir -p "$OUT"
if [[ ! -s "$POOLS/manifest.json" ]]; then
    prep=(--out "$POOLS" --raw "$RAW" --language "$LANGUAGE" --seed "$SEED" \
          --n-source "$N_SOURCE" --n-english "$N_ENGLISH" --n-visual "$N_VISUAL")
    if [[ "$LANGUAGE" == sw ]]; then
        : "${SWAHILI_CSV:?For LANGUAGE=sw supply SWAHILI_CSV pointing to SwaFlickr8K CSV}"
        prep+=(--swahili-csv "$SWAHILI_CSV")
    fi
    if [[ -n "${IMAGE_ARCHIVE:-}" ]]; then prep+=(--image-archive "$IMAGE_ARCHIVE"); fi
    "$PY" prepare_data.py "${prep[@]}"
fi
if [[ ! -s "$EVAL/manifest.json" ]]; then
    if [[ -n "${CVBENCH_SOURCE:-}" ]]; then
        "$PY" prepare_eval.py --out "$EVAL" --source "$CVBENCH_SOURCE"
    else
        "$PY" prepare_eval.py --out "$EVAL"
    fi
fi
base=(--pools "$POOLS" --val "$EVAL/val.jsonl" --test "$EVAL/test.jsonl" \
      --out "$EXPERIMENT" --language "$LANGUAGE" --layers "$LAYERS" --arm "$ARM" \
      --model "$MODEL" --dim "$DIM" --seed "$SEED" --strengths "$STRENGTHS" \
      --oracle-strengths "$ORACLE_STRENGTHS" --iterations "${ITERATIONS:-30}" \
      --batch "${BATCH:-128}" --visual-weight "${VISUAL_WEIGHT:-0.25}" \
      --max-relative-norm "${NORM_CAP:-0.15}" --max-tokens "${MAX_TOKENS:-8}")
"$PY" -u experiment.py --stage extract "${base[@]}"
"$PY" -u experiment.py --stage fit "${base[@]}"
if [[ "${RUN_ORACLE:-1}" == 1 ]]; then
    "$PY" -u experiment.py --stage oracle "${base[@]}"
fi
"$PY" -u experiment.py --stage validate "${base[@]}"
if [[ "${RUN_TEST:-1}" == 1 ]]; then
    "$PY" -u experiment.py --stage test "${base[@]}"
fi
if [[ "${RUN_STRESS:-1}" == 1 ]]; then
    "$PY" -u stress_visual.py --out "$EXPERIMENT" --pools "$POOLS" \
      --val "$EVAL/val.jsonl" --language "$LANGUAGE" --model "$MODEL" --arm "$ARM" \
      --seed "$SEED" --max-relative-norm "${NORM_CAP:-0.15}" \
      --max-tokens "${MAX_TOKENS:-8}"
fi
"$PY" summarize.py --out "$EXPERIMENT"
echo "Done: $EXPERIMENT/test_results.json"
