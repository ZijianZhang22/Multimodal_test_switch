#!/usr/bin/env bash
# Independent frozen VLM-native Rosetta and LoRA routes. Full run, no smoke stage.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"
ROOT="$(cd .. && pwd)"
STEERING="$ROOT/paligemma2_cross_lingual_steering"
SHARED="$ROOT/gemma3_cross_lingual_steering"
OUT="${OUT:-$HERE/outputs}"
SOURCE="${SOURCE:-$STEERING/outputs/paligemma2_cvbench_full}"
DATA="${DATA:-$OUT/data}"
LANGUAGE="${LANGUAGE:-zh}"
SEED="${SEED:-42}"
LAYERS="${LAYERS:-8,13,18}"
ALIGN_LAYER="${ALIGN_LAYER:-13}"
PAIR_BUDGET="${PAIR_BUDGET:-25}"
DIM="${DIM:-16}"
STRENGTHS="${STRENGTHS:-0,0.15,0.3,0.6}"
EPOCHS="${EPOCHS:-1}"
INSTALL="${INSTALL:-1}"
RUN_FROZEN="${RUN_FROZEN:-1}"
RUN_LORA="${RUN_LORA:-1}"
RUN_LORA_CONTROL="${RUN_LORA_CONTROL:-1}"
mkdir -p "$OUT"
if [[ ! -x "$HERE/.venv/bin/python" ]]; then python3 -m venv --system-site-packages "$HERE/.venv"; fi
PY="$HERE/.venv/bin/python"
if [[ "$INSTALL" == 1 ]]; then "$PY" -m pip install -r "$HERE/requirements.txt"; fi
"$PY" - <<'PY'
import torch, transformers, peft
assert transformers.__version__=='4.51.3',transformers.__version__
assert torch.cuda.is_available(),'RunPod CUDA is required'
print('GPU:',torch.cuda.get_device_name(0),'Transformers:',transformers.__version__,'PEFT:',peft.__version__,flush=True)
PY
# Prepare an independent WORKING COPY; NEVER change the existing steering
# dataset or archive its predictions by applying curated translations in place.
STAGING="$OUT/source_cvbench840"
mkdir -p "$STAGING"
if [[ ! -s "$STAGING/visual_eval.jsonl" ]]; then
  if [[ -s "$SOURCE/visual_eval.jsonl" ]]; then
    echo "Copying existing steering dataset into independent alignment staging directory."
    cp "$SOURCE/visual_eval.jsonl" "$STAGING/visual_eval.jsonl"
    cp "$SOURCE/dataset_manifest.json" "$STAGING/dataset_manifest.json"
    if [[ ! -e "$STAGING/images" ]]; then ln -s "$(cd "$SOURCE/images" && pwd)" "$STAGING/images"; fi
  else
    echo 'No existing multilingual CV-Bench dataset; preparing original curated cohort in staging.'
    if [[ ! -s "$SHARED/.upstream/get_features/taxonomy_sentences.py" ]]; then
      mkdir -p "$SHARED"
      git clone --quiet https://github.com/Woodygan/TextSteersVision.git "$SHARED/.upstream"
      git -C "$SHARED/.upstream" checkout --quiet ce1c0f0cffed9b859901ce347f9d2b869a512178
    fi
    "$PY" "$STEERING/apply_curated.py" --data "$STAGING" --seed-cache
    "$PY" "$SHARED/cvbench_extension/prepare.py" --data "$STAGING" --n-val 60 --n-test 150 --seed 42 --no-translate
  fi
fi
# Applying curated translations modifies only STAGING, not original files.
"$PY" "$STEERING/apply_curated.py" --data "$STAGING" --apply
if [[ ! -s "$DATA/manifest.json" ]]; then
  "$PY" protocol.py --source "$STAGING/visual_eval.jsonl" --out "$DATA" --seed "$SEED"
else
  echo "Reusing fixed partition: $DATA"
  "$PY" - "$DATA" "$STAGING/visual_eval.jsonl" <<'PYVERIFY'
import json,sys
from pathlib import Path
from protocol import file_hash,load_splits
m=json.loads((Path(sys.argv[1])/'manifest.json').read_text())
assert m['sha256']==file_hash(sys.argv[2]),'Dataset changed; choose a NEW DATA directory'
load_splits(sys.argv[1]); print('Reusing verified train/val/test split',flush=True)
PYVERIFY
fi
if [[ "$RUN_FROZEN" == 1 ]]; then
  "$PY" -u frozen.py --data "$DATA" --out "$OUT/frozen_${LANGUAGE}" --language "$LANGUAGE" \
    --layers "$LAYERS" --pair-budget "$PAIR_BUDGET" --dim "$DIM" \
    --strengths "$STRENGTHS" --seed "$SEED" --stage all
fi
if [[ "$RUN_LORA_CONTROL" == 1 ]]; then
  "$PY" -u lora.py --data "$DATA" --out "$OUT/lora_ce_${LANGUAGE}" --language "$LANGUAGE" \
    --layers "$LAYERS" --align-layer "$ALIGN_LAYER" --pair-budget "$PAIR_BUDGET" --align-weight 0 \
    --epochs "$EPOCHS" --seed "$SEED"
fi
if [[ "$RUN_LORA" == 1 ]]; then
  "$PY" -u lora.py --data "$DATA" --out "$OUT/lora_alignment_${LANGUAGE}" --language "$LANGUAGE" \
    --layers "$LAYERS" --align-layer "$ALIGN_LAYER" --pair-budget "$PAIR_BUDGET" --align-weight "${ALIGN_WEIGHT:-0.3}" \
    --epochs "$EPOCHS" --seed "$SEED"
fi
"$PY" summarize.py --root "$OUT" --language "$LANGUAGE"
echo "FINISHED: $OUT/vlm_alignment_comparison_${LANGUAGE}.csv"