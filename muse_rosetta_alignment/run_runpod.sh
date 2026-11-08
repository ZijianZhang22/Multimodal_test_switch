#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" - <<'PY'
import sys
assert sys.version_info >= (3, 11), 'Python >= 3.11 required; choose a recent RunPod PyTorch image'
import torch
print('[env] Python:', sys.version.split()[0], 'PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
PY
mkdir -p external outputs data
if [[ ! -f external/MUSE/src/trainer.py ]]; then git clone https://github.com/facebookresearch/MUSE.git external/MUSE; fi
if [[ ! -f external/unpaired-rosetta/unpaired_rosetta/wasserstein_procrustes.py ]]; then
  git clone --recursive https://github.com/dominik-schnaus/unpaired-rosetta.git external/unpaired-rosetta
fi
"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install -r requirements.txt
"$PYTHON_BIN" -m pip install -e external/unpaired-rosetta
ARGS=(--train-limit "${TRAIN_LIMIT:-3000}" --test-limit "${TEST_LIMIT:-500}" \
      --batch-size "${BATCH_SIZE:-32}" --clusters "${ROSETTA_CLUSTERS:-12}" \
      --restarts "${ROSETTA_RESTARTS:-5}" --iterations "${ROSETTA_ITERATIONS:-20}" \
      --rosetta-batch "${ROSETTA_BATCH_SIZE:-512}")
if [[ "${REBUILD:-0}" == 1 ]]; then ARGS+=(--rebuild); fi
"$PYTHON_BIN" experiment.py "${ARGS[@]}"
echo '[DONE] outputs/results.csv + outputs/results.json'
