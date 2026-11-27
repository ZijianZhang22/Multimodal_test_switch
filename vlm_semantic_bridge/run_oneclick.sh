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
AUTO_BOOTSTRAP="${AUTO_BOOTSTRAP:-1}"
read -r -a ARMS <<< "${ARMS:-baseline_zh baseline_en pairs_only text_only joint random}"
MAP_ARMS=()
for arm in "${ARMS[@]}"; do
  case "$arm" in
    pairs_only|text_only|joint|joint_caption) MAP_ARMS+=("$arm") ;;
  esac
done

# A new RunPod contains source code after git pull, but NOT old training assets.
# When required CLIP/Rosetta checkpoints are absent, train precisely the maps
# requested by this pilot, using matching PCA/budget/seed settings.
required=(
  "$WORK/features.npz"
  "$WORK/model_revisions.json"
  "$WORK/raw/flickr8kenc.caption.txt"
)
if (( ${#MAP_ARMS[@]} > 0 )); then
  required+=("$MAP_DIR/report.json")
  for arm in "${MAP_ARMS[@]}"; do
    required+=("$MAP_DIR/${arm}_${MAP_SEED}.npz")
  done
fi
missing=()
for item in "${required[@]}"; do
  [[ -s "$item" ]] || missing+=("$item")
done
if (( ${#missing[@]} > 0 )); then
  echo "Missing previously trained CLIP/Rosetta assets on this Pod:"
  printf '  %s\n' "${missing[@]}"
  if [[ "$AUTO_BOOTSTRAP" != "1" ]]; then
    echo "Set AUTO_BOOTSTRAP=1 (default), or provide the existing CLIP_WORK / MAP_DIR." >&2
    exit 2
  fi
  if [[ -n "${MAP_DIR_OVERRIDE:-}" || ( -n "${MAP_DIR:-}" && "$MAP_DIR" != "$WORK/results_clip_${PCA_MODE}_d${DIM}_${ANCHOR_CENTERING}_${IMAGE_CENTERING}_budget_${BUDGET}" ) ]]; then
    echo "Custom MAP_DIR cannot be automatically reconstructed. Point to a complete map directory or unset MAP_DIR." >&2
    exit 2
  fi
  if (( ${#MAP_ARMS[@]} == 0 )); then
    echo "This pilot has no Rosetta mapping arms, but source CLIP features are missing. Run visual_anchor/run_clip_native.sh separately." >&2
    exit 2
  fi
  echo "Bootstrapping CLIP features + ${BUDGET}-pair Rosetta3 maps (seed=${MAP_SEED}, dim=${DIM}, PCA=${PCA_MODE})."
  echo "This downloads Flickr8K/encoders if absent and FITS maps. Source code alone does not contain those files."
  env WORK_DIR="$WORK" BUDGETS="$BUDGET" SEEDS="$MAP_SEED" \\
      METHODS="${MAP_ARMS[*]}" DIM="$DIM" PCA_MODE="$PCA_MODE" \\
      ANCHOR_CENTERING="$ANCHOR_CENTERING" IMAGE_CENTERING="$IMAGE_CENTERING" \\
      DEVICE="${CLIP_DEVICE:-cuda}" \\
      bash "$ROOT/visual_anchor/run_clip_native.sh"
  for item in "${required[@]}"; do
    if [[ ! -s "$item" ]]; then
      echo "Bootstrapping did not produce required file: $item" >&2
      exit 2
    fi
  done
fi
if [[ ! -f "$DATA/manifest.json" ]]; then
  mkdir -p "$DATA"
  echo "Preparing pinned bilingual MMBench Dev (public; NOT official blind test)"
  "$PY" "$ROOT/vlm_unsupervised_transfer/prepare_mmbench.py" --out "$DATA" --tune-size "${TUNE_SIZE:-200}" --seed "${DATA_SEED:-42}"
fi
echo "Bridge $STAGE: split=$SPLIT limit=$LIMIT map=$MAP_DIR -> $OUT"
"$PY" -u "$HERE/run.py" --stage "$STAGE" --data-dir "$DATA" \
  --clip-work "$WORK" --map-dir "$MAP_DIR" --out "$OUT" \
  --split "$SPLIT" --limit "$LIMIT" --top-k "$TOP_K" \
  --map-seed "$MAP_SEED" --random-seed "${RANDOM_SEED:-42}" \
  --encoder-device "${ENCODER_DEVICE:-cpu}" --encoder-batch "${ENCODER_BATCH:-32}" \
  --model "${MODEL:-google/paligemma2-3b-mix-448}" --max-new-tokens "${MAX_NEW_TOKENS:-8}" \
  --arms "${ARMS[@]}"