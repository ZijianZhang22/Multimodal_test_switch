#!/usr/bin/env bash
# Frozen Rosetta3 -> Chinese zero-shot CIFAR-10 CLIP classification.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${PY:-$ROOT/visual_anchor/.venv-real/bin/python}"
WORK="${CLIP_WORK:-$ROOT/visual_anchor/real_work_clip_native}"
BUDGET="${BUDGET:-50}"
MAP_SEED="${MAP_SEED:-0}"
DIM="${DIM:-128}"
PCA_MODE="${PCA_MODE:-shared}"
ANCHOR_CENTERING="${ANCHOR_CENTERING:-source_pool}"
IMAGE_CENTERING="${IMAGE_CENTERING:-bank}"
MAP_DIR="${MAP_DIR:-$WORK/results_clip_${PCA_MODE}_d${DIM}_${ANCHOR_CENTERING}_${IMAGE_CENTERING}_budget_${BUDGET}}"
NUM_IMAGES="${NUM_IMAGES:-500}"
SEED="${SEED:-42}"
DATA_DIR="${CIFAR_DIR:-$HERE/data}"
OUT="${OUT:-$HERE/outputs/cifar10_n${NUM_IMAGES}_b${BUDGET}_mapseed${MAP_SEED}_seed${SEED}}"
read -r -a METHODS <<< "${METHODS:-pairs_only text_only joint}"

# Existing Pod already has these frozen maps from the Semantic Bridge experiment.
# On a new Pod, produce precisely the required maps using the ORIGINAL Rosetta3
# training pipeline; never substitute randomly initialized or new model maps.
required=("$WORK/features.npz" "$WORK/model_revisions.json" "$MAP_DIR/report.json")
for m in "${METHODS[@]}"; do required+=("$MAP_DIR/${m}_${MAP_SEED}.npz"); done
missing=()
for f in "${required[@]}"; do [[ -s "$f" ]] || missing+=("$f"); done
if (( ${#missing[@]} > 0 )); then
  echo 'Missing Rosetta3 training files:'
  printf '  %s\n' "${missing[@]}"
  if [[ "${AUTO_BOOTSTRAP:-1}" != "1" ]]; then
    echo 'Re-enable AUTO_BOOTSTRAP=1 or point MAP_DIR/CLIP_WORK to valid saved assets.' >&2
    exit 2
  fi
  if [[ -n "${MAP_DIR:-}" && "$MAP_DIR" != "$WORK/results_clip_${PCA_MODE}_d${DIM}_${ANCHOR_CENTERING}_${IMAGE_CENTERING}_budget_${BUDGET}" ]]; then
    echo 'Custom MAP_DIR cannot be reconstructed automatically.' >&2
    exit 2
  fi
  if [[ -s "$MAP_DIR/report.json" ]]; then
    echo 'Existing Rosetta report is incomplete for requested methods/seed.' >&2
    echo 'Do not overwrite existing fitted maps. Use a complete MAP_DIR or a new WORK_DIR.' >&2
    exit 2
  fi
  echo 'Training frozen CLIP/Rosetta3 maps using ORIGINAL visual_anchor/run_clip_native.sh ...'
  env WORK_DIR="$WORK" BUDGETS="$BUDGET" SEEDS="$MAP_SEED" \
    METHODS="${METHODS[*]}" DIM="$DIM" PCA_MODE="$PCA_MODE" \
    ANCHOR_CENTERING="$ANCHOR_CENTERING" IMAGE_CENTERING="$IMAGE_CENTERING" \
    DEVICE="${DEVICE:-cuda}" \
    bash "$ROOT/visual_anchor/run_clip_native.sh"
fi

if [[ ! -x "$PY" ]]; then
  echo "Missing Python environment: $PY; creating from original requirements."
  python3 -m venv --system-site-packages "$ROOT/visual_anchor/.venv-real"
  PY="$ROOT/visual_anchor/.venv-real/bin/python"
  "$PY" -m pip install -r "$ROOT/visual_anchor/requirements_real.txt"
fi
"$PY" -m unittest discover -s "$HERE/tests" -v

# Uses official CIFAR-10 *test* split only, and reuses training-only PCA/means.
echo "CLIP transfer: CIFAR10 n=$NUM_IMAGES map=$MAP_DIR methods=${METHODS[*]}"
"$PY" -u "$HERE/run.py" \
  --clip-work "$WORK" --map-dir "$MAP_DIR" --data-dir "$DATA_DIR" --out "$OUT" \
  --budget "$BUDGET" --map-seed "$MAP_SEED" --seed "$SEED" \
  --num-images "$NUM_IMAGES" --batch-size "${BATCH_SIZE:-64}" \
  --bootstraps "${BOOTSTRAPS:-2000}" --device "${DEVICE:-cuda}" \
  --methods "${METHODS[@]}"