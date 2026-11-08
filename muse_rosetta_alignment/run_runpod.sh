#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" - <<'PY'
import sys,torch
assert sys.version_info >= (3,11), "Python 3.11+ required"
print("[env]",sys.version.split()[0],"torch",torch.__version__,"CUDA:",torch.cuda.is_available())
PY
mkdir -p external data outputs
if [[ ! -f external/MUSE/src/trainer.py ]]; then
  git clone https://github.com/facebookresearch/MUSE.git external/MUSE
fi
if [[ ! -f external/unpaired-rosetta/unpaired_rosetta/wasserstein_procrustes.py ]]; then
  git clone --recursive https://github.com/dominik-schnaus/unpaired-rosetta.git external/unpaired-rosetta
fi
"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install -r requirements.txt
"$PYTHON_BIN" -m pip install -e external/unpaired-rosetta
if [[ "${REBUILD:-0}" == "1" ]]; then
  rm -f data/manifest.csv data/train.npz data/test.npz outputs/muse_mapping.npz
fi
if [[ ! -f data/train.npz || ! -f data/test.npz ]]; then
  if [[ ! -f data/manifest.csv ]]; then
    "$PYTHON_BIN" scripts/build_flickr8k_zh.py --out data --train-limit "${TRAIN_LIMIT:-3000}" --test-limit "${TEST_LIMIT:-500}" --batch-size "${BATCH_SIZE:-32}"
  fi
  "$PYTHON_BIN" scripts/extract_embeddings.py --csv data/manifest.csv --out data --batch-size "${BATCH_SIZE:-32}"
fi
if [[ ! -f outputs/muse_mapping.npz ]]; then
  "$PYTHON_BIN" scripts/fit_muse.py --muse-root external/MUSE --train data/train.npz --out outputs/muse_mapping.npz
fi
EXTRA=(--save-maps)
if [[ "${SKIP_SEQUENTIAL:-0}" == "1" ]]; then EXTRA+=(--skip-sequential); fi
if [[ "${MIX_CONTROL:-1}" == "1" ]]; then EXTRA+=(--include-mixture-control); fi
"$PYTHON_BIN" scripts/factorial.py \
  --train data/train.npz --test data/test.npz --muse-mapping outputs/muse_mapping.npz \
  --out-dir outputs --clusters "${ROSETTA_CLUSTERS:-12}" --restarts "${ROSETTA_RESTARTS:-5}" \
  --batch-size "${ROSETTA_BATCH_SIZE:-512}" --iterations "${ROSETTA_ITERATIONS:-20}" "${EXTRA[@]}"
echo "[DONE] Outputs: $(pwd)/outputs/results.csv and results.json"
