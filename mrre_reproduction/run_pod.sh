#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${IMAGE_ROOT:?Set IMAGE_ROOT to the directory containing MathVerse images_version_1-4}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
python -m pip install -r requirements.txt
if [ ! -d upstream ]; then git clone https://github.com/QimingLi-scir/MRRE.git upstream; fi
git -C upstream checkout 24fa054932efb5af0e5bd5b3ebb63e4ff97bff0d
python mrre.py prepare --source upstream/evaluation/MathVerse/data/zh.json --out data
python -m unittest -v test_core.py
mkdir -p results
python mrre.py extract --model "$MODEL" --data data/calibration.jsonl --image-root "$IMAGE_ROOT" --out results/vectors.pt
python mrre.py evaluate --model "$MODEL" --data data/validation.jsonl --image-root "$IMAGE_ROOT" --vectors results/vectors.pt --out results/validation.jsonl
# Inspect validation first. Do not tune on test. Then run the test command in README.
