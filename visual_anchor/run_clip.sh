#!/usr/bin/env bash
# CLIP setting: E and I = one CLIP checkpoint (a strong, pre-trained English<->image bridge); L = monolingual Chinese.
# Builds features in a SEPARATE work dir, prints the bridge diagnostics, then runs Rosetta3 at low pair budgets.
set -euo pipefail
cd "$(dirname "$0")"
ROSETTA_COMMIT=fdfad84ea488f29702a61a455be71622c4ff5d49
WORK="${WORK_DIR:-real_work_clip}"
if [[ ! -x .venv-real/bin/python ]]; then python3 -m venv --system-site-packages .venv-real; fi
. .venv-real/bin/activate
python -m pip install -r requirements_real.txt -r requirements_rosetta.txt
python -m pip install "unpaired-rosetta @ git+https://github.com/dominik-schnaus/unpaired-rosetta@${ROSETTA_COMMIT}"
python -m unittest -v test_core test_real test_clip test_rosetta3

python -u prepare_real.py --work "$WORK" --encoders clip --clip-model "${CLIP_MODEL:-openai/clip-vit-base-patch32}" \
  --device "${DEVICE:-cuda}" --batch-size "${ENCODE_BATCH:-32}" --test-limit "${TEST_LIMIT:-1000}"
if [[ "${BRIDGE_ONLY:-0}" == "1" ]]; then echo "Bridge diagnostics: $WORK/bridge_test.json"; exit 0; fi

read -r -a budgets <<< "${BUDGETS:-10 25 50 100 200}"
read -r -a seeds <<< "${SEEDS:-0 1 2 3 4}"
for budget in "${budgets[@]}"; do
  python -u run_rosetta3.py --data "$WORK/features.npz" --shared-ei-pca --budget "$budget" --dim "${DIM:-32}" \
    --seeds "${seeds[@]}" --lambda-z "${LAMBDA_Z:-0.25}" --clusters "${CLUSTERS:-20}" --restarts "${RESTARTS:-5}" \
    --batch "${BATCH:-256}" --iterations "${ITERATIONS:-20}" --workers "${WORKERS:-1}" \
    --methods supervised rosetta visual bank_text extra_text gaussian english_pool anchor_downweight \
    --out "$WORK/results_rosetta3_budget_${budget}"
done
python summarize_real.py --work "$WORK" --pattern 'results_rosetta3_budget_*' --name summary_rosetta3_clip.csv
