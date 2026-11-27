#!/usr/bin/env bash
# Run from anywhere; reuses the existing visual_anchor CLIP-native caches/maps.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${PY:-$ROOT/visual_anchor/.venv-real/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "Creating visual_anchor/.venv-real (existing RunPod CUDA torch may be reused)"
  python3 -m venv --system-site-packages "$ROOT/visual_anchor/.venv-real"
  PY="$ROOT/visual_anchor/.venv-real/bin/python"
  "$PY" -m pip install -r "$ROOT/visual_anchor/requirements_real.txt"
fi
"$PY" -m unittest discover -s "$HERE/tests" -v
WORK="${CLIP_WORK:-$ROOT/visual_anchor/real_work_clip_native}"
BUDGET="${BUDGET:-50}"
DIM="${DIM:-128}"
PCA_MODE="${PCA_MODE:-shared}"
ANCHOR_CENTERING="${ANCHOR_CENTERING:-source_pool}"
IMAGE_CENTERING="${IMAGE_CENTERING:-bank}"
MAP_DIR="${MAP_DIR:-$WORK/results_clip_${PCA_MODE}_d${DIM}_${ANCHOR_CENTERING}_${IMAGE_CENTERING}_budget_${BUDGET}}"
DATA="${MMBENCH_DATA:-/workspace/vlm_A_zh_seed42/mmbench_dev_bilingual}"
SPLIT="${SPLIT:-tune}"
LIMIT="${LIMIT:-30}"
TOP_K="${TOP_K:-2}"
MAP_SEED="${MAP_SEED:-0}"
OUT="${OUT:-$ROOT/vlm_semantic_bridge/outputs/${SPLIT}_${LIMIT}_b${BUDGET}_seed${MAP_SEED}}"
STAGE="${STAGE:-all}"
if [[ ! -f "$DATA/manifest.json" ]]; then
  mkdir -p "$DATA"
  echo "Preparing pinned bilingual MMBench Dev (public; NOT official blind test)"
  "$PY" "$ROOT/vlm_unsupervised_transfer/prepare_mmbench.py" --out "$DATA" --tune-size "${TUNE_SIZE:-200}" --seed "${DATA_SEED:-42}"
fi
read -r -a ARMS <<< "${ARMS:-baseline_zh baseline_en pairs_only text_only joint random}"
echo "Bridge $STAGE: split=$SPLIT limit=$LIMIT map=$MAP_DIR -> $OUT"
"$PY" -u "$HERE/run.py" --stage "$STAGE" --data-dir "$DATA" \
  --clip-work "$WORK" --map-dir "$MAP_DIR" --out "$OUT" \
  --split "$SPLIT" --limit "$LIMIT" --top-k "$TOP_K" \
  --map-seed "$MAP_SEED" --random-seed "${RANDOM_SEED:-42}" \
  --encoder-device "${ENCODER_DEVICE:-cpu}" --encoder-batch "${ENCODER_BATCH:-32}" \
  --model "${MODEL:-google/paligemma2-3b-mix-448}" --max-new-tokens "${MAX_NEW_TOKENS:-8}" \
  --arms "${ARMS[@]}"