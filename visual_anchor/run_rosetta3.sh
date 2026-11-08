#!/usr/bin/env bash
# Official Rosetta vs Rosetta + third target, on features built by prepare_real.py (run run_real.sh first).
# IMPORTANT: rebuild features after the e_pool fix in prepare_real.py; run_rosetta3.py refuses the old overlap.
set -euo pipefail
cd "$(dirname "$0")"
ROSETTA_COMMIT=fdfad84ea488f29702a61a455be71622c4ff5d49
if [[ ! -x .venv-real/bin/python ]]; then python3 -m venv --system-site-packages .venv-real; fi
. .venv-real/bin/activate
python -m pip install -r requirements_rosetta.txt
python -m pip install "unpaired-rosetta @ git+https://github.com/dominik-schnaus/unpaired-rosetta@${ROSETTA_COMMIT}"
python -m unittest -v test_core test_rosetta3
WORK="${WORK_DIR:-real_work}"
read -r -a budgets <<< "${BUDGETS:-0 50 200 1000}"
read -r -a seeds <<< "${SEEDS:-0 1 2}"
for budget in "${budgets[@]}"; do
  python -u run_rosetta3.py --data "$WORK/features.npz" --budget "$budget" --dim 32 --seeds "${seeds[@]}" \
    --lambda-z "${LAMBDA_Z:-1.0}" --iterations "${ITERATIONS:-100}" --workers "${WORKERS:-4}" \
    --out "$WORK/results_rosetta3_budget_${budget}"
done
python summarize_real.py --work "$WORK" --pattern 'results_rosetta3_budget_*' --name summary_rosetta3.csv
