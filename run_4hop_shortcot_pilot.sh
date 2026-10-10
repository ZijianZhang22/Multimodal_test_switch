#!/usr/bin/env bash
set -euo pipefail

# One-click 4-hop short-CoT sanity check.
# Generates 100 four-hop problems, runs all-Text only with bounded short CoT,
# and reports accuracy. The runner limits visible reasoning to <=2 short lines
# and defaults to 48 new tokens.

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
LABEL="${LABEL:-qwen3vl8b_4hop_shortcot}"
DTYPE="${DTYPE:-auto}"
ATTN="${ATTN:-sdpa}"
N_PROBLEMS="${N_PROBLEMS:-100}"
SEED="${SEED:-42}"

DATA_DIR="${DATA_DIR:-data_4hop_shortcot}"
RESULT_DIR="${RESULT_DIR:-results_4hop_shortcot}"
DATA="${DATA_DIR}/depth_scaling_examples.jsonl"
RESULTS="${RESULT_DIR}/${LABEL}.jsonl"

cd "$(dirname "$0")"
mkdir -p "$DATA_DIR" "$RESULT_DIR"

echo "=== [1/3] Generate 4-hop dataset ==="
python generate_depth_scaling_dataset.py \
  --out_dir "$DATA_DIR" \
  --hops 4 \
  --n_problems_per_hop "$N_PROBLEMS" \
  --seed "$SEED" \
  --order_mode identity

echo
echo "=== [2/3] Run all-Text with bounded short CoT ==="
python run_qwen3vl.py \
  --data "$DATA" \
  --out "$RESULTS" \
  --model "$MODEL" \
  --model_label "$LABEL" \
  --dtype "$DTYPE" \
  --attn_implementation "$ATTN" \
  --conditions text_baseline \
  --hops 4 \
  --reasoning_mode short_cot \
  --max_new_tokens 48 \
  --resume

echo
echo "=== [3/3] Accuracy + output sanity ==="
python - "$RESULTS" <<'PY'
import json, sys
path = sys.argv[1]
rows=[]
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        x=json.loads(line)
        if x.get("condition")=="text_baseline" and int(x.get("hop_count",0))==4:
            rows.append(x)

correct=sum(bool(x["correct"]) for x in rows)
invalid=sum(x.get("prediction") is None for x in rows)
mean_tok=sum(x.get("output_tokens",0) for x in rows)/len(rows)
print(f"4-hop short-CoT all-Text: {correct}/{len(rows)} = {correct/len(rows):.1%}")
print(f"Invalid prediction: {invalid}")
print(f"Mean output tokens: {mean_tok:.2f}")
print("\nFirst 10 outputs:")
for x in rows[:10]:
    print("\nID:", x["example_id"])
    print("GT:", x["answer"], "Pred:", x["prediction"])
    print("Raw:", repr(x["raw_output"]))
PY
