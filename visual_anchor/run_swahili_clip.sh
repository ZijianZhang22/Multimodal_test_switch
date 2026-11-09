#!/usr/bin/env bash
# Reuse the SAME frozen CLIP E and I representations, changing only the L encoder/captions.
# The Swahili source uses multilingual E5, and the Chinese source text2vec:
# do NOT treat absolute cross-language scores as a matched-encoder comparison.
set -euo pipefail
cd "$(dirname "$0")"
WORK="${WORK_DIR:-real_work_clip_native}"
[[ -s "$WORK/features.npz" ]] || { echo "Missing $WORK/features.npz. First run BRIDGE_ONLY=1 bash run_clip_native.sh" >&2; exit 2; }
[[ -x .venv-real/bin/python ]] || { echo "Missing .venv-real. First run BRIDGE_ONLY=1 bash run_clip_native.sh" >&2; exit 2; }
. .venv-real/bin/activate
python -m unittest -q test_swahili test_followup
python - "$WORK/model_revisions.json" <<'PYLOCK'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); m=json.loads(p.read_text())["models"]
if not (m['e']==m['i'] and 'clip' in m['e'].lower()):
    raise SystemExit(f"Expected CLIP-based E/I features. Got {m}")
PYLOCK
SW_FEATURES="${SW_FEATURES:-$WORK/features_sw_clip.npz}"
if [[ ! -s "$SW_FEATURES" || "${SW_REBUILD:-0}" == "1" ]]; then
  csv_args=()
  [[ -z "${SW_CSV:-}" ]] || csv_args=(--csv "$SW_CSV")
  python -u prepare_swahili.py --base-data "$WORK/features.npz" \
    --english-captions "$WORK/raw/flickr8kenc.caption.txt" \
    --out "$SW_FEATURES" "${csv_args[@]}" \
    --model "${SW_MODEL:-intfloat/multilingual-e5-base}" \
    --device "${DEVICE:-cuda}" --batch-size "${ENCODE_BATCH:-32}"
else
  echo "Reusing existing Swahili embeddings: $SW_FEATURES"
fi
DIM="${DIM:-128}"
PCA_MODE="${PCA_MODE:-shared}"
case "$PCA_MODE" in
  shared) PCA_FLAGS=(--shared-ei-pca) ;;
  joint) PCA_FLAGS=(--joint-ei-pca) ;;
  *) echo "PCA_MODE must be shared or joint" >&2; exit 2 ;;
esac
ANCHOR_MODE="${ANCHOR_CENTERING:-source_pool}"
IMAGE_MODE="${IMAGE_CENTERING:-bank}"
LABEL="${RUN_LABEL:-sw_strong_h1}"
RESULT_TAG="${LABEL}_${PCA_MODE}_d${DIM}_${ANCHOR_MODE}_${IMAGE_MODE}"
read -r -a budgets <<< "${BUDGETS:-10 25 50}"
read -r -a seeds <<< "${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
read -r -a methods <<< "${METHODS:-pairs_only image_only noise_only caption_only text_only joint joint_caption joint_noise joint_extra_english}"
RESUME_FLAGS=()
if [[ "${RESUME:-0}" == "1" ]]; then RESUME_FLAGS=(--resume); fi
for budget in "${budgets[@]}"; do
  dest="$WORK/results_${RESULT_TAG}_budget_${budget}"
  echo "Swahili CLIP: budget=$budget, seeds=${seeds[*]}, methods=${methods[*]}"
  python -u run_rosetta3.py --data "$SW_FEATURES" \
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