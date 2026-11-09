#!/usr/bin/env bash
# Native CLIP coordinates + anchor-only A-E ablations (no learned visual bridge).
set -euo pipefail
cd "$(dirname "$0")"
# Default 128D shared-PCA protocol (E and I share axes, but not the mean).
# Keep the existing feature location so encoded CLIP vectors are reused.
ROSETTA_COMMIT=fdfad84ea488f29702a61a455be71622c4ff5d49
WORK="${WORK_DIR:-real_work_clip_native}"
DIM="${DIM:-128}"
PCA_MODE="${PCA_MODE:-shared}"
case "$PCA_MODE" in
  shared) PCA_FLAGS=(--shared-ei-pca) ;;
  joint)  PCA_FLAGS=(--joint-ei-pca) ;;
  *) echo "PCA_MODE must be shared or joint (got: $PCA_MODE)" >&2; exit 2 ;;
esac
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
if [[ ! -s "$WORK/features.npz" || "${REBUILD_FEATURES:-0}" == "1" ]]; then
  python -u prepare_real.py --work "$WORK" --encoders clip \
    --clip-model "${CLIP_MODEL:-openai/clip-vit-base-patch32}" --diagnostic-dim "$DIM" \
    --device "${DEVICE:-cuda}" --batch-size "${ENCODE_BATCH:-32}" \
    --test-limit "${TEST_LIMIT:-1000}"
else
  echo "Reusing existing CLIP features: $WORK/features.npz (set REBUILD_FEATURES=1 to regenerate)"
fi
# Detect accidentally mixing non-CLIP features into identity-bridge experiments.
python - "$WORK/model_revisions.json" <<'PYLOCK'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(f"Missing {p}; cannot confirm CLIP provenance. Set REBUILD_FEATURES=1")
models = json.loads(p.read_text()).get('models', {})
if not (models.get('e') == models.get('i') and 'clip' in models.get('e', '').lower()):
    raise SystemExit(f"Expected identical E/I CLIP encoders in {p}, got: {models}")
PYLOCK
# Optional standalone diagnostic using already encoded features (no new model inference).
if [[ "${BRIDGE_ONLY:-0}" == "1" || "${RECHECK_BRIDGE:-0}" == "1" ]]; then
  python - "$WORK/features.npz" "$DIM" "$WORK/bridge_test_dim_${DIM}.json" <<'PYDIAG'
import json, sys
import numpy as np
from pathlib import Path
from prepare_real import bridge_diagnostics
d = dict(np.load(sys.argv[1], allow_pickle=False))
res = bridge_diagnostics(d, 'clip', dim=int(sys.argv[2]))
res['diagnostic_dim'] = int(sys.argv[2])
Path(sys.argv[3]).write_text(json.dumps(res, indent=2))
for method, scores in res['variants'].items():
    print(f'English->image bridge [{method}]: {scores}', flush=True)
print('Diagnostics saved:', sys.argv[3], flush=True)
PYDIAG
fi
if [[ "${BRIDGE_ONLY:-0}" == "1" ]]; then exit 0; fi
ANCHOR_MODE="${ANCHOR_CENTERING:-source_pool}"
IMAGE_MODE="${IMAGE_CENTERING:-bank}"
RESULT_TAG="clip_${PCA_MODE}_d${DIM}_${ANCHOR_MODE}_${IMAGE_MODE}"
read -r -a budgets <<< "${BUDGETS:-10 25 50}"
read -r -a seeds <<< "${SEEDS:-0 1 2}"
read -r -a methods <<< "${METHODS:-pairs_only image_only caption_only noise_only text_only joint}"
for budget in "${budgets[@]}"; do
  echo "CLIP-native Rosetta3 | budget=$budget | PCA=$PCA_MODE | dim=$DIM | seeds=${seeds[*]}"
  python -u run_rosetta3.py --data "$WORK/features.npz" \
    "${PCA_FLAGS[@]}" --bridge identity --dim "$DIM" \
    --budget "$budget" --seeds "${seeds[@]}" --methods "${methods[@]}" \
    --lambda-y "${LAMBDA_Y:-1.0}" --lambda-z "${LAMBDA_Z:-0.25}" \
    --anchor-centering "$ANCHOR_MODE" --image-centering "$IMAGE_MODE" \
    --ablation-lambda-z "${ABLATION_LAMBDA_Z:-1.0}" \
    --clusters "${CLUSTERS:-20}" --restarts "${RESTARTS:-5}" \
    --batch "${BATCH:-256}" --iterations "${ITERATIONS:-20}" --workers "${WORKERS:-1}" \
    --out "$WORK/results_${RESULT_TAG}_budget_${budget}"
done
python summarize_real.py --work "$WORK" \
  --pattern "results_${RESULT_TAG}_budget_*" --name "summary_${RESULT_TAG}.csv"
echo "Saved $WORK/summary_${RESULT_TAG}.csv"