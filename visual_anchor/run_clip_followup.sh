#!/usr/bin/env bash
# Strong-H1 control experiment, 10 seeds, crossed paired bootstrap.
set -euo pipefail
cd "$(dirname "$0")"
WORK="${WORK_DIR:-real_work_clip_native}"
DIM="${DIM:-128}"
PCA_MODE="${PCA_MODE:-shared}"
case "$PCA_MODE" in
  shared) PCA_FLAGS=(--shared-ei-pca) ;;
  joint) PCA_FLAGS=(--joint-ei-pca) ;;
  *) echo "PCA_MODE must be 'shared' or 'joint'" >&2; exit 2 ;;
esac
[[ -s "$WORK/features.npz" ]] || { echo "Missing $WORK/features.npz. First run BRIDGE_ONLY=1 bash run_clip_native.sh" >&2; exit 2; }
[[ -x .venv-real/bin/python ]] || { echo "Missing .venv-real. First run BRIDGE_ONLY=1 bash run_clip_native.sh" >&2; exit 2; }
. .venv-real/bin/activate
python -m unittest -q test_followup test_corrections test_lambda_y
# Keep outputs separate from earlier 3-seed experiments.
ANCHOR_MODE="${ANCHOR_CENTERING:-source_pool}"
IMAGE_MODE="${IMAGE_CENTERING:-bank}"
LABEL="${RUN_LABEL:-strong_h1}"
RESULT_TAG="${LABEL}_${PCA_MODE}_d${DIM}_${ANCHOR_MODE}_${IMAGE_MODE}"
read -r -a budgets <<< "${BUDGETS:-10 25 50}"
read -r -a seeds <<< "${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
read -r -a methods <<< "${METHODS:-pairs_only image_only noise_only caption_only text_only joint joint_caption joint_noise joint_extra_english}"
RESUME_FLAGS=()
if [[ "${RESUME:-0}" == "1" ]]; then RESUME_FLAGS=(--resume); fi
for budget in "${budgets[@]}"; do
  dest="$WORK/results_${RESULT_TAG}_budget_${budget}"
  echo "Running CLIP follow-up: budget=$budget, seeds=${seeds[*]}, methods=${methods[*]}"
  python -u run_rosetta3.py --data "$WORK/features.npz" \
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
echo "Results: $WORK/summary_${RESULT_TAG}.csv and $WORK/paired_${RESULT_TAG}.csv"