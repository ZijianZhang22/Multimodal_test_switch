#!/usr/bin/env bash
# RunPod entry point: bash cross_lingual_steering/run.sh
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
DATA="${DATA:-outputs/pilot}"
LAYERS="${LAYERS:-6,12}"
ALPHAS="${ALPHAS:-5}"
LIMIT="${LIMIT:-8}"
SPLIT="${SPLIT:-val}"
MASK_TYPE="${MASK_TYPE:-both}"
INSTALL="${INSTALL:-1}"
RECREATE_DATA="${RECREATE_DATA:-0}"
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"
if [[ "$INSTALL" == "1" ]]; then
 python -m pip install -r requirements.txt
fi
if [[ "$RECREATE_DATA" == "1" || ! -f "$DATA/visual_eval.jsonl" ]]; then
 python make_pilot_data.py --out "$DATA"
 FORCE_EXTRACT=1
fi
# Extract every execution unless the caller explicitly opts to reuse vectors.
# Prevents silent model/corpus mismatch; extraction is the inexpensive stage.
if [[ "$FORCE_EXTRACT" == "1" || ! -f "$DATA/vectors.pt" ]]; then
 python run_experiment.py --mode extract --model "$MODEL" --data "$DATA" \
   --layers "$LAYERS" --vectors "$DATA/vectors.pt"
else
 # Fail fast if a requested layer is absent from a cached vector file.
 python - "$DATA/vectors.pt" "$LAYERS" "$MODEL" <<'PY'
import sys, torch
obj=torch.load(sys.argv[1],map_location="cpu",weights_only=True)
requested=[int(v) for v in sys.argv[2].split(",")]
if obj.get("model") != sys.argv[3] or any(
    str(layer) not in obj["vectors"][concept]
    for layer in requested for concept in ("counting","spatial")
):
    sys.exit("Cached vectors do not match model/layers. Run FORCE_EXTRACT=1 bash run.sh")
PY
fi
python run_experiment.py --mode evaluate --model "$MODEL" --data "$DATA" \
 --vectors "$DATA/vectors.pt" --layers "$LAYERS" --alphas "$ALPHAS" \
 --limit "$LIMIT" --split "$SPLIT" --mask-type "$MASK_TYPE"
python analyze_results.py --vectors "$DATA/vectors.pt" \
 --results "$DATA/results_$SPLIT.csv" --out "$DATA/analysis_$SPLIT"
echo "DONE: $DATA/analysis_$SPLIT/paired_accuracy.csv"
