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

if [[ "$INSTALL" == "1" ]]; then
  python -m pip install -r requirements.txt
fi
python - <<'PY'
import transformers
assert transformers.__version__ == "4.51.3", (
    f"Expected transformers==4.51.3, got {transformers.__version__}"
)
print("Using paper's Transformers version",transformers.__version__)
PY

if [[ "$REBUILD_DATA" == "1" || ! -f "$DATA/corpus/counting_en.jsonl" || ! -f "$DATA/visual_eval.jsonl" ]]; then
  python make_data.py --out "$DATA"
  FORCE_EXTRACT=1
fi

if [[ "$FORCE_EXTRACT" == "1" || ! -f "$DATA/vectors/manifest.json" ]]; then
  python extract.py --data "$DATA" --layers "$LAYERS" --overwrite
fi
python evaluate.py --data "$DATA" --split "$SPLIT" --layers "$LAYERS" \
  --alphas "$ALPHAS" --mask "$MASK" --limit "$LIMIT" --max-tokens "$MAX_TOKENS"
python analysis.py --data "$DATA" --split "$SPLIT"
echo "DONE: $DATA/analysis_$SPLIT/paired_accuracy.csv"
