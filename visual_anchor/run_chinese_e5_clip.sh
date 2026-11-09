#!/usr/bin/env bash
# Chinese/SwaFlickr8k matched-ID language control: SAME multilingual-E5 encoder.
set -euo pipefail
cd "$(dirname "$0")"
WORK="${WORK_DIR:-real_work_clip_native}"
SW_FEATURES="${SW_FEATURES:-$WORK/features_sw_clip.npz}"
ZH_FEATURES="${ZH_FEATURES:-$WORK/features_zh_e5_swmatched.npz}"
[[ -s "$SW_FEATURES" ]] || { echo "Missing $SW_FEATURES. Run bash run_swahili_clip.sh first." >&2; exit 2; }
[[ -x .venv-real/bin/python ]] || { echo "Missing .venv-real. Run BRIDGE_ONLY=1 bash run_clip_native.sh first." >&2; exit 2; }
. .venv-real/bin/activate
python -m unittest -q test_chinese_e5 test_followup test_resume
if [[ ! -s "$ZH_FEATURES" || "${ZH_REBUILD:-0}" == "1" ]]; then
  python -u prepare_chinese_e5.py --sw-data "$SW_FEATURES" \
    --raw-dir "$WORK/raw" --out "$ZH_FEATURES" \
    --device "${DEVICE:-cuda}" --batch-size "${ENCODE_BATCH:-32}"
else
  python prepare_chinese_e5.py --sw-data "$SW_FEATURES" --out "$ZH_FEATURES" --verify-only
fi
DIM="${DIM:-128}"
PCA_MODE="${PCA_MODE:-shared}"
case "$PCA_MODE" in
  shared) PCA_FLAGS=(--shared-ei-pca) ;;
  joint) PCA_FLAGS=(--joint-ei-pca) ;;
  *) echo 'PCA_MODE must be shared or joint' >&2; exit 2 ;;
esac
ANCHOR_MODE="${ANCHOR_CENTERING:-source_pool}"
IMAGE_MODE="${IMAGE_CENTERING:-bank}"
LABEL="${RUN_LABEL:-zh_e5_swmatched_strong_h1}"
RESULT_TAG="${LABEL}_${PCA_MODE}_d${DIM}_${ANCHOR_MODE}_${IMAGE_MODE}"
read -r -a budgets <<< "${BUDGETS:-10 25 50}"
read -r -a seeds <<< "${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
read -r -a methods <<< "${METHODS:-pairs_only image_only noise_only caption_only text_only joint joint_caption joint_noise joint_extra_english}"
RESUME_FLAGS=()
if [[ "${RESUME:-0}" == "1" ]]; then RESUME_FLAGS=(--resume); fi
for budget in "${budgets[@]}"; do
  dest="$WORK/results_${RESULT_TAG}_budget_${budget}"
  python -u run_rosetta3.py --data "$ZH_FEATURES" \
    "${PCA_FLAGS[@]}" --bridge identity --dim "$DIM" \
    --budget "$budget" --seeds "${seeds[@]}" --methods "${methods[@]}" \
    --anchor-centering "$ANCHOR_MODE" --image-centering "$IMAGE_MODE" \
    --ablation-lambda-z "${ABLATION_LAMBDA_Z:-1.0}" \
    --clusters "${CLUSTERS:-20}" --restarts "${RESTARTS:-5}" \
    --batch "${BATCH:-256}" --iterations "${ITERATIONS:-20}" --workers "${WORKERS:-1}" \
    "${RESUME_FLAGS[@]}" --out "$dest"
done
python summarize_real.py --work "$WORK" --pattern "results_${RESULT_TAG}_budget_*" --name "summary_${RESULT_TAG}.csv"
python analyze_ablation.py --work "$WORK" --pattern "results_${RESULT_TAG}_budget_*" \
  --output "paired_${RESULT_TAG}.csv" --repetitions "${BOOTSTRAP_REPS:-2000}"
echo "Chinese E5 matched result: $WORK/summary_${RESULT_TAG}.csv"
echo "Paired contrasts: $WORK/paired_${RESULT_TAG}.csv"
# Only run across-language difference-in-differences when both complete configs exist.
SW_PREFIX="${SW_RESULT_PREFIX:-sw_strong_h1_shared_d128_source_pool_bank}"
if [[ "$DIM" == "128" && "$PCA_MODE" == "shared" && "$ANCHOR_MODE" == "source_pool" && "$IMAGE_MODE" == "bank" &&
      "${BUDGETS:-10 25 50}" == "10 25 50" &&
      "${SEEDS:-0 1 2 3 4 5 6 7 8 9}" == "0 1 2 3 4 5 6 7 8 9" &&
      "${METHODS:-pairs_only image_only noise_only caption_only text_only joint joint_caption joint_noise joint_extra_english}" == "pairs_only image_only noise_only caption_only text_only joint joint_caption joint_noise joint_extra_english" ]]; then
  if [[ -s "$WORK/results_${SW_PREFIX}_budget_10/report.json" &&
        -s "$WORK/results_${SW_PREFIX}_budget_25/report.json" &&
        -s "$WORK/results_${SW_PREFIX}_budget_50/report.json" ]]; then
    python compare_e5_languages.py --work "$WORK" \
      --zh-prefix "$RESULT_TAG" --sw-prefix "$SW_PREFIX" \
      --output "language_diff_${RESULT_TAG}.csv" \
      --repetitions "${BOOTSTRAP_REPS:-2000}"
  else
    echo "Swahili result reports incomplete; skip cross-language differences for now."
  fi
fi