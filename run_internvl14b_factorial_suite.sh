#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-OpenGVLab/InternVL3_5-14B}"
LABEL4="${LABEL4:-internvl35_14b_h4_all16}"
LABEL5="${LABEL5:-internvl35_14b_h5_all32}"
DTYPE="${DTYPE:-bf16}"
N4="${N4:-100}"
N5="${N5:-100}"
SEED="${SEED:-42}"
RUN_H4="${RUN_H4:-1}"
RUN_H5="${RUN_H5:-1}"

cd "$(dirname "$0")"

if [ "$RUN_H4" = "1" ]; then
  echo "=== H=4 full factorial: all 16 modality schedules ==="
  python generate_factorial_schedule_dataset.py \
    --out_dir data_internvl_h4_all16 \
    --hops 4 \
    --n_problems "$N4" \
    --seed "$SEED"

  python run_internvl35.py \
    --data data_internvl_h4_all16/examples.jsonl \
    --out "results_internvl14b/${LABEL4}.jsonl" \
    --model "$MODEL" \
    --model_label "$LABEL4" \
    --dtype "$DTYPE" \
    --hops 4 \
    --max_new_tokens 16 \
    --resume

  python analyze_factorial_schedules.py \
    --results "results_internvl14b/${LABEL4}.jsonl" \
    --out_dir analysis_internvl14b_h4_all16
fi

if [ "$RUN_H5" = "1" ]; then
  echo
  echo "=== H=5 full factorial: all 32 modality schedules ==="
  echo "NOTE: SAME is unreachable for odd H under unit cardinal steps, so H=5 balances the 8 reachable direction classes."
  python generate_factorial_schedule_dataset.py \
    --out_dir data_internvl_h5_all32 \
    --hops 5 \
    --n_problems "$N5" \
    --seed "$SEED"

  python run_internvl35.py \
    --data data_internvl_h5_all32/examples.jsonl \
    --out "results_internvl14b/${LABEL5}.jsonl" \
    --model "$MODEL" \
    --model_label "$LABEL5" \
    --dtype "$DTYPE" \
    --hops 5 \
    --max_new_tokens 16 \
    --resume

  python analyze_factorial_schedules.py \
    --results "results_internvl14b/${LABEL5}.jsonl" \
    --out_dir analysis_internvl14b_h5_all32
fi

echo
echo "Done."
echo "H4 outputs: analysis_internvl14b_h4_all16/"
echo "H5 outputs: analysis_internvl14b_h5_all32/"
