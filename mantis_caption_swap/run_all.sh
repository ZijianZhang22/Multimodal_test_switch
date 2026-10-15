#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p logs
echo "===== ENVIRONMENT ====="
bash setup.sh
echo "===== SELF TEST ====="
python -u experiment.py selftest
echo "===== PREPARE 50 RANDOM MANTIS-EVAL ITEMS ====="
python -u experiment.py prepare --n 50 --seed 42
echo "===== GENERATE QUESTION-BLIND CAPTIONS ====="
python -u experiment.py caption
echo "===== EVALUATE ALL DISTINCT CONDITIONS ====="
python -u experiment.py evaluate --max-new-tokens 8192
echo "===== SUMMARIZE ====="
python -u experiment.py summarize
python -u token_analysis.py
echo "===== ALL COMPLETED ====="
