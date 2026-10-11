#!/usr/bin/env bash
set -euo pipefail

# Full 4-hop logical-role curve for InternVL3.5-14B.
# Reuses the exact same 100 latent problems as the Qwen direct-answer control:
#   TTTT, V@1, V@2, V@3, V@4
# If V@1/V@4 were already run in the earlier replication, --resume skips them
# and only the missing V@2/V@3 examples are evaluated.

MODEL="${MODEL:-OpenGVLab/InternVL3_5-14B}"
LABEL="${LABEL:-internvl35_14b_4hop_direct}"
DTYPE="${DTYPE:-bf16}"
N_PROBLEMS="${N_PROBLEMS:-100}"
SEED="${SEED:-42}"
DATA_DIR="${DATA_DIR:-data_4hop_trace}"
RESULT_DIR="${RESULT_DIR:-results_internvl14b}"
ANALYSIS_DIR="${ANALYSIS_DIR:-analysis_internvl14b_full_roles}"
DATA="${DATA_DIR}/depth_scaling_examples.jsonl"
RESULTS="${RESULT_DIR}/${LABEL}.jsonl"

cd "$(dirname "$0")"
mkdir -p "$RESULT_DIR" "$ANALYSIS_DIR"

if [ ! -f "$DATA" ]; then
  echo "Dataset not found; generating the same 4-hop matched dataset..."
  python generate_depth_scaling_dataset.py \
    --out_dir "$DATA_DIR" \
    --hops 4 \
    --n_problems_per_hop "$N_PROBLEMS" \
    --seed "$SEED" \
    --order_mode identity
fi

echo "=== [1/2] Run full logical-role curve: TTTT + V@1..V@4 ==="
python run_internvl35.py \
  --data "$DATA" \
  --out "$RESULTS" \
  --model "$MODEL" \
  --model_label "$LABEL" \
  --dtype "$DTYPE" \
  --conditions text_baseline,single_visual \
  --visual_roles 1,2,3,4 \
  --hops 4 \
  --max_new_tokens 16 \
  --resume

echo
echo "=== [2/2] Analyze matched logical-depth curve ==="
python analyze_depth_scaling.py \
  --results "$RESULTS" \
  --out_dir "$ANALYSIS_DIR"

echo
echo "=== Exact matched role tests ==="
python analyze_role_curve_exact.py \
  --results "$RESULTS" \
  --out_dir "$ANALYSIS_DIR"

echo
echo "Key outputs:"
echo "  $ANALYSIS_DIR/matched_delta_by_role_depth.csv"
echo "  $ANALYSIS_DIR/exact_mcnemar_by_role.csv"
echo "  $ANALYSIS_DIR/depth_scaling_${LABEL}.png"
