#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
MODE="${MODE:-demo}"
ANCHORS="${ANCHORS:-50}"
if [[ "$MODE" == "real" ]]; then
  if [[ -z "${TRAIN_NPZ:-}" || -z "${TEST_NPZ:-}" ]]; then echo "Set TRAIN_NPZ and TEST_NPZ"; exit 2; fi
  mkdir -p external
  if [[ ! -d external/unpaired-rosetta ]]; then
    git clone --recursive https://github.com/dominik-schnaus/unpaired-rosetta.git external/unpaired-rosetta
  fi
  python3 -m pip install -e external/unpaired-rosetta
  python3 run.py --train "$TRAIN_NPZ" --test "$TEST_NPZ" --out outputs/real --anchors "$ANCHORS"
else
  python3 -m unittest discover -s tests -v
  python3 run.py --skip-rosetta --out outputs/demo --anchors "$ANCHORS"
fi
