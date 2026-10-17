#!/usr/bin/env bash
set -euo pipefail

# Six 4-hop schedules with exactly 2 Text + 2 Vision relations.
# 200 matched latent problems per schedule => 1,200 evaluations total.
#
# Usage:
#   MODEL_FAMILY=qwen scripts/launch/run_2v2t_200.sh
#   MODEL_FAMILY=internvl scripts/launch/run_2v2t_200.sh
#
# Defaults to Qwen3-VL-8B.

MODEL_FAMILY="${MODEL_FAMILY:-qwen}"
N_PROBLEMS="${N_PROBLEMS:-200}"
SEED="${SEED:-20261012}"
SCHEDULES="TTVV,TVTV,TVVT,VTTV,VTVT,VVTT"

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DATA_DIR="data_2v2t_200"
DATA="${DATA_DIR}/examples.jsonl"
mkdir -p results_2v2t_200 analysis_2v2t_200

if [ ! -f "$DATA" ]; then
  echo "=== Generate 200 matched 4-hop problems ==="
  python -m scripts.data.generate_factorial_schedule_dataset \
    --out_dir "$DATA_DIR" \
    --hops 4 \
    --n_problems "$N_PROBLEMS" \
    --seed "$SEED"
fi

if [ "$MODEL_FAMILY" = "qwen" ]; then
  MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
  LABEL="${LABEL:-qwen3vl8b_2v2t_200}"
  RESULTS="results_2v2t_200/${LABEL}.jsonl"

  echo "=== Qwen: six 2T/2V schedules × 200 matched problems ==="
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
  LABEL="${LABEL:-internvl35_14b_2v2t_200}"
  RESULTS="results_2v2t_200/${LABEL}.jsonl"

  echo "=== InternVL: six 2T/2V schedules × 200 matched problems ==="
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
python -m scripts.analysis.analyze_2v2t_200 \
  --results "$RESULTS" \
  --out_dir "analysis_2v2t_200/${LABEL}"

echo
echo "Done."
echo "Results: $RESULTS"
echo "Stats:   analysis_2v2t_200/${LABEL}/pairwise_mcnemar_holm.csv"

