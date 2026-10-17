#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

REV=ce1c0f0cffed9b859901ce347f9d2b869a512178
DATA="${DATA:-outputs/gemma3}"
LAYERS="${LAYERS:-8,16}"
ALPHAS="${ALPHAS:-5}"
MASK="${MASK:-both}"
LIMIT="${LIMIT:-4}"
SPLIT="${SPLIT:-val}"
MAX_TOKENS="${MAX_TOKENS:-24}"
INSTALL="${INSTALL:-1}"
REBUILD_DATA="${REBUILD_DATA:-0}"
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"

if [[ ! -d .upstream/.git ]]; then
  git clone --quiet https://github.com/Woodygan/TextSteersVision.git .upstream
fi
git -C .upstream checkout --quiet "$REV"
actual="$(git -C .upstream rev-parse HEAD)"
if [[ "$actual" != "$REV" ]]; then
  echo "Unexpected upstream revision: $actual" >&2
  exit 2
fi

if [[ ! -x .venv/bin/python ]]; then
  python -m venv --system-site-packages .venv
fi
PYTHON=.venv/bin/python
if [[ "$INSTALL" == "1" ]]; then
  "$PYTHON" -m pip install -r requirements.txt
fi
"$PYTHON" - <<'PY'
import torch, transformers
assert transformers.__version__ == "4.51.3", (
    f"Expected upstream Transformers 4.51.3, got {transformers.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA-enabled PyTorch required in venv")
print("Upstream runtime:", transformers.__version__, "CUDA:", torch.cuda.is_available())
PY

if [[ "$REBUILD_DATA" == "1" || ! -f "$DATA/corpus/counting_en.jsonl" || ! -f "$DATA/visual_eval.jsonl" ]]; then
  "$PYTHON" make_data.py --out "$DATA"
  FORCE_EXTRACT=1
fi

if [[ "$FORCE_EXTRACT" == "1" || ! -f "$DATA/vectors/manifest.json" ]]; then
  "$PYTHON" extract.py --data "$DATA" --layers "$LAYERS" --overwrite
fi
"$PYTHON" evaluate.py --data "$DATA" --split "$SPLIT" --layers "$LAYERS" \
  --alphas "$ALPHAS" --mask "$MASK" --limit "$LIMIT" --max-tokens "$MAX_TOKENS"
"$PYTHON" analysis.py --data "$DATA" --split "$SPLIT"
echo "DONE: $DATA/analysis_$SPLIT/paired_accuracy.csv"
