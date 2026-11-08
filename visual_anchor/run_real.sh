#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
command -v curl >/dev/null || { echo 'curl is required'; exit 1; }
if [[ ! -x .venv-real/bin/python ]]; then python3 -m venv --system-site-packages .venv-real; fi
. .venv-real/bin/activate
python -m pip install -r requirements_real.txt
python -m unittest -v test_core test_real
python -u prepare_real.py --work "${WORK_DIR:-real_work}" --device "${DEVICE:-cuda}" --batch-size "${ENCODE_BATCH:-32}" --test-limit "${TEST_LIMIT:-1000}"
read -r -a budgets <<< "${BUDGETS:-50 200 1000}"
read -r -a seeds <<< "${SEEDS:-0 1 2}"
for budget in "${budgets[@]}"; do
  python -u run.py --data "${WORK_DIR:-real_work}/features.npz" --budget "$budget" --dim 32 --seeds "${seeds[@]}" --iterations "${ITERATIONS:-20}" --batch "${OT_BATCH:-256}" --no-cycle --out "${WORK_DIR:-real_work}/results_budget_${budget}"
done
python summarize_real.py --work "${WORK_DIR:-real_work}"
echo "Finished. Summary: ${WORK_DIR:-real_work}/summary.csv"
