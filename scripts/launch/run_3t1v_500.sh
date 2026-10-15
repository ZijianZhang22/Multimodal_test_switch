#!/usr/bin/env bash
set -euo pipefail

# 4-hop single-visual logical-role experiment.
# 500 matched latent problems per condition.
#
# Conditions:
#   TTTT baseline
#   VTTT = Vision at role 1
#   TVTT = Vision at role 2
#   TTVT = Vision at role 3
#   TTTV = Vision at role 4
#
# Total: 5 conditions x 500 matched problems = 2,500 evaluations.
#
# Usage:
#   MODEL_FAMILY=qwen scripts/launch/run_3t1v_500.sh
#   MODEL_FAMILY=internvl scripts/launch/run_3t1v_500.sh

MODEL_FAMILY="${MODEL_FAMILY:-qwen}"
N_PROBLEMS="${N_PROBLEMS:-500}"
SEED="${SEED:-20261014}"
SCHEDULES="TTTT,VTTT,TVTT,TTVT,TTTV"

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DATA_DIR="data_3t1v_500"
DATA="${DATA_DIR}/examples.jsonl"
mkdir -p results_3t1v_500 analysis_3t1v_500

if [ ! -f "$DATA" ]; then
  echo "=== Generate 500 new matched 4-hop problems ==="
  python -m scripts.data.generate_factorial_schedule_dataset \
    --out_dir "$DATA_DIR" \
    --hops 4 \
    --n_problems "$N_PROBLEMS" \
    --seed "$SEED"
fi

if [ "$MODEL_FAMILY" = "qwen" ]; then
  MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
  LABEL="${LABEL:-qwen3vl8b_3t1v_500}"
  RESULTS="results_3t1v_500/${LABEL}.jsonl"

  echo "=== Qwen: TTTT + four 3T/1V schedules x 500 matched problems ==="
  python -m scripts.inference.run_qwen3vl \
    --data "$DATA" \
    --out "$RESULTS" \
    --model "$MODEL" \
    --model_label "$LABEL" \
    --schedules "$SCHEDULES" \
    --hops 4 \
    --reasoning_mode direct \
    --max_new_tokens 16 \
    --resume

elif [ "$MODEL_FAMILY" = "internvl" ]; then
  MODEL="${MODEL:-OpenGVLab/InternVL3_5-14B}"
  LABEL="${LABEL:-internvl35_14b_3t1v_500}"
  RESULTS="results_3t1v_500/${LABEL}.jsonl"

  echo "=== InternVL: TTTT + four 3T/1V schedules x 500 matched problems ==="
  python -m scripts.inference.run_internvl35 \
    --data "$DATA" \
    --out "$RESULTS" \
    --model "$MODEL" \
    --model_label "$LABEL" \
    --dtype bf16 \
    --schedules "$SCHEDULES" \
    --hops 4 \
    --max_new_tokens 16 \
    --resume
else
  echo "MODEL_FAMILY must be 'qwen' or 'internvl'."
  exit 2
fi

echo
echo "=== Matched statistical analysis ==="
python -m scripts.analysis.analyze_3t1v_500 \
  --results "$RESULTS" \
  --out_dir "analysis_3t1v_500/${LABEL}"

echo
echo "Done."
echo "Results:   $RESULTS"
echo "Accuracy:  analysis_3t1v_500/${LABEL}/accuracy_by_schedule.csv"
echo "Overall:   analysis_3t1v_500/${LABEL}/cochrans_q_single_visual.csv"
echo "Primary:   analysis_3t1v_500/${LABEL}/prespecified_primary_contrasts.csv"
echo "Role pairs:analysis_3t1v_500/${LABEL}/pairwise_single_visual_mcnemar_holm.csv"
echo "Vs TTTT:   analysis_3t1v_500/${LABEL}/single_visual_vs_text_mcnemar_holm.csv"

