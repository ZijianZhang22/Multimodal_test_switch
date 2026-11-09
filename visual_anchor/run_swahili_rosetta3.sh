#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
WORK="${WORK_DIR:-real_work}"
if [[ ! -x .venv-real/bin/python ]]; then python3 -m venv --system-site-packages .venv-real; fi
. .venv-real/bin/activate
python -m pip install -r requirements_real.txt -r requirements_rosetta.txt
python -m pip install "unpaired-rosetta @ git+https://github.com/dominik-schnaus/unpaired-rosetta@fdfad84ea488f29702a61a455be71622c4ff5d49"
python -m unittest -v test_swahili test_rosetta3
if [[ ! -f "$WORK/features_sw.npz" ]]; then
  python -u prepare_swahili.py --base-data "$WORK/features.npz" --out "$WORK/features_sw.npz" --device "${DEVICE:-cuda}" --batch-size "${ENCODE_BATCH:-32}"
fi
read -r -a budgets <<< "${BUDGETS:-10 25 50 200 1000}"
read -r -a seeds <<< "${SEEDS:-0 1 2}"
read -r -a methods <<< "${METHODS:-rosetta visual bank_text extra_text gaussian english_pool anchor_downweight}"
for budget in "${budgets[@]}"; do
  echo "Swahili Rosetta3 budget=$budget seeds=${seeds[*]}"
  python -u run_rosetta3.py --data "$WORK/features_sw.npz" --budget "$budget" --seeds "${seeds[@]}" --methods "${methods[@]}" \
    --lambda-z "${LAMBDA_Z:-0.25}" --dim 32 --clusters "${CLUSTERS:-20}" --restarts "${RESTARTS:-5}" \
    --batch "${BATCH:-256}" --iterations "${ITERATIONS:-20}" --workers "${WORKERS:-1}" \
    --out "$WORK/results_swahili_rosetta3_budget_${budget}"
done
python summarize_real.py --work "$WORK" --pattern "results_swahili_rosetta3_budget_*" --name summary_swahili_rosetta3.csv
echo "Saved: $WORK/summary_swahili_rosetta3.csv"
