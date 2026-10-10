#!/usr/bin/env bash
set -euo pipefail

# One-click 6-hop pilot for Qwen3-VL.
#
# Default behavior:
#   1) generate 100 six-hop latent problems
#   2) run all-Text calibration only
#   3) print baseline accuracy
#   4) if RUN_FULL=1, continue with V@1 ... V@6 and analyze
#
# Examples:
#   ./run_6hop_pilot.sh
#   RUN_FULL=1 ./run_6hop_pilot.sh
#   N_PROBLEMS=200 RUN_FULL=1 ./run_6hop_pilot.sh
#   MODEL=Qwen/Qwen3-VL-32B-Instruct LABEL=qwen3vl32b_6hop DTYPE=bf16 RUN_FULL=1 ./run_6hop_pilot.sh

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
LABEL="${LABEL:-qwen3vl8b_6hop}"
DTYPE="${DTYPE:-auto}"
ATTN="${ATTN:-sdpa}"
N_PROBLEMS="${N_PROBLEMS:-100}"
SEED="${SEED:-42}"
RUN_FULL="${RUN_FULL:-0}"

DATA_DIR="${DATA_DIR:-data_6hop_pilot}"
RESULT_DIR="${RESULT_DIR:-results_6hop_pilot}"
ANALYSIS_DIR="${ANALYSIS_DIR:-analysis_6hop_pilot}"

DATA="${DATA_DIR}/depth_scaling_examples.jsonl"
RESULTS="${RESULT_DIR}/${LABEL}.jsonl"
ANALYSIS="${ANALYSIS_DIR}/${LABEL}"

cd "$(dirname "$0")"

mkdir -p "$DATA_DIR" "$RESULT_DIR" "$ANALYSIS"

echo "=== [1/4] Generate 6-hop dataset ==="
python generate_depth_scaling_dataset.py \
  --out_dir "$DATA_DIR" \
  --hops 6 \
  --n_problems_per_hop "$N_PROBLEMS" \
  --seed "$SEED" \
  --order_mode identity

echo
echo "=== [2/4] Run all-Text calibration ==="
python run_qwen3vl.py \
  --data "$DATA" \
  --out "$RESULTS" \
  --model "$MODEL" \
  --model_label "$LABEL" \
  --dtype "$DTYPE" \
  --attn_implementation "$ATTN" \
  --conditions text_baseline \
  --hops 6 \
  --resume

echo
echo "=== 6-hop all-Text accuracy ==="
python - "$RESULTS" <<'PY'
import json
import sys

path = sys.argv[1]
rows = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        x = json.loads(line)
        if x.get("condition") == "text_baseline" and int(x.get("hop_count", 0)) == 6:
            rows.append(x)

if not rows:
    raise SystemExit("No 6-hop text_baseline results found.")

correct = sum(bool(x["correct"]) for x in rows)
acc = correct / len(rows)
print(f"{correct}/{len(rows)} = {acc:.1%}")
print()
if acc < 0.50:
    print("Baseline is below 50%; this may be close to a floor regime.")
elif acc < 0.65:
    print("Baseline is usable for a pilot, but still fairly difficult.")
else:
    print("Baseline looks suitable for the full single-Visual role sweep.")
PY

if [[ "$RUN_FULL" != "1" ]]; then
  echo
  echo "Calibration finished."
  echo "To continue with V@1 ... V@6, run:"
  echo "  RUN_FULL=1 ./run_6hop_pilot.sh"
  exit 0
fi

echo
echo "=== [3/4] Run full 6-hop single-Visual sweep ==="
# --resume skips the all-Text examples already completed above.
python run_qwen3vl.py \
  --data "$DATA" \
  --out "$RESULTS" \
  --model "$MODEL" \
  --model_label "$LABEL" \
  --dtype "$DTYPE" \
  --attn_implementation "$ATTN" \
  --hops 6 \
  --resume

echo
echo "=== [4/4] Analyze matched role-depth effect ==="
python analyze_depth_scaling.py \
  --results "$RESULTS" \
  --out_dir "$ANALYSIS"

echo
echo "Done."
echo "Results:  $RESULTS"
echo "Analysis: $ANALYSIS"
echo "Main table: $ANALYSIS/matched_delta_by_role_depth.csv"
