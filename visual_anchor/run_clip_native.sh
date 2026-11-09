#!/usr/bin/env bash
# Native CLIP coordinates + anchor-only A-E ablations (no learned visual bridge).
set -euo pipefail
cd "$(dirname "$0")"
ROSETTA_COMMIT=fdfad84ea488f29702a61a455be71622c4ff5d49
WORK="${WORK_DIR:-real_work_clip_native}"
if [[ ! -x .venv-real/bin/python ]]; then python3 -m venv --system-site-packages .venv-real; fi
. .venv-real/bin/activate
python -m pip install -r requirements_real.txt -r requirements_rosetta.txt
python -m pip install "unpaired-rosetta @ git+https://github.com/dominik-schnaus/unpaired-rosetta@${ROSETTA_COMMIT}"
python -m unittest -v test_core test_real test_clip test_rosetta3 test_lambda_y test_corrections
# Reuse downloaded raw Flickr assets, if present, to avoid downloading images twice.
if [[ "$WORK" != "real_work" && ! -e "$WORK/raw" && -d real_work/raw ]]; then
  mkdir -p "$WORK"
  ln -s "$(realpath real_work/raw)" "$WORK/raw"
fi
python -u prepare_real.py --work "$WORK" --encoders clip \
  --clip-model "${CLIP_MODEL:-openai/clip-vit-base-patch32}" --diagnostic-dim "${DIM:-64}" \
  --device "${DEVICE:-cuda}" --batch-size "${ENCODE_BATCH:-32}" \
  --test-limit "${TEST_LIMIT:-1000}"
echo "Native CLIP diagnostics: $WORK/bridge_test.json"
if [[ "${BRIDGE_ONLY:-0}" == "1" ]]; then exit 0; fi
read -r -a budgets <<< "${BUDGETS:-10 25 50 100 200}"
read -r -a seeds <<< "${SEEDS:-0 1 2}"
read -r -a methods <<< "${METHODS:-pairs_only image_only text_only noise_only caption_only joint rosetta visual bank_text gaussian english_pool anchor_downweight}"
for budget in "${budgets[@]}"; do
  echo "CLIP-native Rosetta3 | budget=$budget | dim=${DIM:-64} | seeds=${seeds[*]}"
  python -u run_rosetta3.py --data "$WORK/features.npz" \
    --joint-ei-pca --bridge identity --dim "${DIM:-64}" \
    --budget "$budget" --seeds "${seeds[@]}" --methods "${methods[@]}" \
    --lambda-y "${LAMBDA_Y:-1.0}" --lambda-z "${LAMBDA_Z:-0.25}" \
    --anchor-centering "${ANCHOR_CENTERING:-pool}" --image-centering "${IMAGE_CENTERING:-bank}" \
    --ablation-lambda-z "${ABLATION_LAMBDA_Z:-1.0}" \
    --clusters "${CLUSTERS:-20}" --restarts "${RESTARTS:-5}" \
    --batch "${BATCH:-256}" --iterations "${ITERATIONS:-20}" --workers "${WORKERS:-1}" \
    --out "$WORK/results_clip_native_budget_${budget}"
done
python summarize_real.py --work "$WORK" \
  --pattern 'results_clip_native_budget_*' --name summary_clip_native.csv
echo "Saved $WORK/summary_clip_native.csv"
