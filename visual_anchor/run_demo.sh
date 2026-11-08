#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
python demo.py
python run.py --data demo_features.npz --dim 16 --budget 30 --iterations 3 --batch 64 --no-cycle --out demo_results
