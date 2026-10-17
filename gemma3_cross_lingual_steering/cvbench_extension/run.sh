#!/usr/bin/env bash
# Run multilingual real CV-Bench with original unmodified Gemma3 mean-shift wrapper.
set -euo pipefail
cd "$(dirname "$0")"
PARENT="$(cd .. && pwd)"
DATA="${DATA:-outputs/gemma3_cvbench}"
LAYERS="${LAYERS:-8,16}"
ALPHAS="${ALPHAS:-0.1}"
SPLIT="${SPLIT:-val}"
LIMIT="${LIMIT:-0}"
MASK="${MASK:-both}"
NORMALIZED="${NORMALIZED:-0}"
ALL_FEATURES="${ALL_FEATURES:-0}"
N_VAL="${N_VAL:-60}"
N_TEST="${N_TEST:-150}"
SEED="${SEED:-42}"
INSTALL="${INSTALL:-1}"
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"
MAX_TOKENS="${MAX_TOKENS:-20}"

# Keep local artifacts inside this isolated extension (never overwrite toy run).
if [[ "$DATA" != /* ]]; then DATA="$PWD/$DATA"; fi
if [[ ! -d "$PARENT/.upstream/.git" ]]; then
  git clone --quiet https://github.com/Woodygan/TextSteersVision.git "$PARENT/.upstream"
fi
git -C "$PARENT/.upstream" checkout --quiet ce1c0f0cffed9b859901ce347f9d2b869a512178
if [[ ! -x "$PARENT/.venv/bin/python" ]]; then
  python -m venv --system-site-packages "$PARENT/.venv"
fi
PY="$PARENT/.venv/bin/python"
if [[ "$INSTALL" == "1" ]]; then
  "$PY" -m pip install -r "$PARENT/requirements.txt" 'datasets>=3.3,<5' 'pyarrow>=17' 'sacremoses>=0.1' 
fi
"$PY" - <<'PY'
import transformers,torch
assert transformers.__version__=='4.51.3',transformers.__version__
assert torch.cuda.is_available(),'CUDA required for Gemma3'
print('Transformers:',transformers.__version__,'CUDA:',torch.cuda.is_available(),flush=True)
PY
if [[ ! -f "$DATA/visual_eval.jsonl" ]]; then
  "$PY" prepare.py --data "$DATA" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED"
fi
if [[ "$FORCE_EXTRACT" == "1" || ! -f "$DATA/vectors/manifest.json" ]]; then
  "$PY" vectors.py --data "$DATA" --layers "$LAYERS"
fi
args=(--data "$DATA" --split "$SPLIT" --layers "$LAYERS" --alphas "$ALPHAS" \
      --mask "$MASK" --limit "$LIMIT" --normalized "$NORMALIZED" --max-tokens "$MAX_TOKENS")
if [[ "$ALL_FEATURES" == "1" ]]; then args+=(--all-features); fi
"$PY" evaluate.py "${args[@]}"
"$PY" analyze.py --data "$DATA" --split "$SPLIT"
echo "COMPLETE: $DATA/analysis_$SPLIT/task_accuracy.csv"
