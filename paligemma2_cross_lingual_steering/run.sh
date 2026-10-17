#!/usr/bin/env bash
# Independent Gemma2 2B text-only -> PaliGemma2 3B multilingual Mean Shift.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"
SHARED="$HERE/../gemma3_cross_lingual_steering"
CV="$SHARED/cvbench_extension"
REV="ce1c0f0cffed9b859901ce347f9d2b869a512178"
DATA="${DATA:-outputs/paligemma2_cvbench}"
if [[ "$DATA" != /* ]]; then DATA="$HERE/$DATA"; fi
LAYERS="${LAYERS:-8,16}"
ALPHAS="${ALPHAS:-0.1}"
MASK="${MASK:-both}"
NORMALIZED="${NORMALIZED:-0}"
SPLIT="${SPLIT:-val}"
LIMIT="${LIMIT:-3}"
ALL_FEATURES="${ALL_FEATURES:-0}"
N_VAL="${N_VAL:-60}"
N_TEST="${N_TEST:-150}"
SEED="${SEED:-42}"
INSTALL="${INSTALL:-1}"
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"
MAX_TOKENS="${MAX_TOKENS:-20}"

if [[ ! -d .upstream/.git ]]; then
  git clone --quiet https://github.com/Woodygan/TextSteersVision.git .upstream
fi
git -C .upstream checkout --quiet "$REV"

# Dataset preparation borrows the tested shared translation/anchor construction.
# That routine expects the upstream taxonomy file in its original parent.
if [[ ! -d "$SHARED/.upstream/.git" ]]; then
  git clone --quiet https://github.com/Woodygan/TextSteersVision.git "$SHARED/.upstream"
fi
git -C "$SHARED/.upstream" checkout --quiet "$REV"

if [[ ! -x "$HERE/.venv/bin/python" ]]; then
  python -m venv --system-site-packages "$HERE/.venv"
fi
PY="$HERE/.venv/bin/python"
if [[ "$INSTALL" == "1" ]]; then
  "$PY" -m pip install -r "$SHARED/requirements.txt" 'datasets>=3.3,<5' 'pyarrow>=17' 'sacremoses>=0.1'
fi
"$PY" - <<'PY'
import torch,transformers
assert transformers.__version__=="4.51.3",transformers.__version__
assert torch.cuda.is_available(),"RunPod CUDA GPU required"
print("Transformers",transformers.__version__,"CUDA",torch.cuda.get_device_name(0),flush=True)
PY

if [[ ! -f "$DATA/visual_eval.jsonl" ]]; then
  # Preseed the complete curated 840-item cohort; no NLLB translation needed.
  "$PY" "$HERE/apply_curated.py" --data "$DATA" --seed-cache
  "$PY" "$CV/prepare.py" --data "$DATA" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED"
fi
if [[ "${CURATED_TRANSLATION:-1}" == "1" ]]; then
  "$PY" "$HERE/apply_curated.py" --data "$DATA" --apply
fi
if [[ "$FORCE_EXTRACT" == "1" || ! -f "$DATA/vectors/manifest.json" ]]; then
  "$PY" "$HERE/vectors.py" --data "$DATA" --layers "$LAYERS"
fi
args=(--data "$DATA" --split "$SPLIT" --layers "$LAYERS" --alphas "$ALPHAS" --mask "$MASK"
      --limit "$LIMIT" --normalized "$NORMALIZED" --max-tokens "$MAX_TOKENS")
if [[ "$ALL_FEATURES" == "1" ]]; then args+=(--all-features); fi
"$PY" "$HERE/evaluate.py" "${args[@]}"
"$PY" "$CV/analyze.py" --data "$DATA" --split "$SPLIT"
echo "FINISHED $SPLIT: $DATA/analysis_$SPLIT/task_accuracy.csv"
