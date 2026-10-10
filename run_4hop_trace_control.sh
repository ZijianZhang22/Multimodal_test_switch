#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
LABEL="${LABEL:-qwen3vl8b_4hop_trace}"
DTYPE="${DTYPE:-auto}"
ATTN="${ATTN:-sdpa}"
N_PROBLEMS="${N_PROBLEMS:-100}"
SEED="${SEED:-42}"

DATA_DIR="${DATA_DIR:-data_4hop_trace}"
RESULT_DIR="${RESULT_DIR:-results_4hop_trace}"
ANALYSIS_DIR="${ANALYSIS_DIR:-analysis_4hop_trace}"
DATA="${DATA_DIR}/depth_scaling_examples.jsonl"
RESULTS="${RESULT_DIR}/${LABEL}.jsonl"

cd "$(dirname "$0")"
mkdir -p "$DATA_DIR" "$RESULT_DIR" "$ANALYSIS_DIR"

echo "=== [1/3] Generate 4-hop matched dataset ==="
python generate_depth_scaling_dataset.py \
  --out_dir "$DATA_DIR" \
  --hops 4 \
  --n_problems_per_hop "$N_PROBLEMS" \
  --seed "$SEED" \
  --order_mode identity

echo
echo "=== [2/3] Run TTTT, V@1, V@4 with coordinate trace ==="
python run_qwen3vl.py \
  --data "$DATA" \
  --out "$RESULTS" \
  --model "$MODEL" \
  --model_label "$LABEL" \
  --dtype "$DTYPE" \
  --attn_implementation "$ATTN" \
  --conditions text_baseline,single_visual \
  --visual_roles 1,4 \
  --hops 4 \
  --reasoning_mode coordinate_trace \
  --max_new_tokens 64 \
  --resume

echo
echo "=== [3/3] Matched summary ==="
python - "$RESULTS" <<'PY'
import json, sys
from collections import defaultdict

path = sys.argv[1]
rows=[]
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

def key(r):
    if r.get("condition") == "text_baseline":
        return "TTTT"
    if r.get("condition") == "single_visual":
        return f"V@{r.get('visual_logical_step')}"
    return r.get("condition","other")

groups=defaultdict(list)
for r in rows:
    groups[key(r)].append(r)

for name in ["TTTT","V@1","V@4"]:
    g=groups.get(name,[])
    if not g:
        print(name, "N=0")
        continue
    correct=sum(bool(x.get("correct")) for x in g)
    valid=sum(x.get("prediction") is not None for x in g)
    toks=sum(x.get("output_tokens",0) for x in g)/len(g)
    print(f"{name}: {correct}/{len(g)} = {correct/len(g):.1%} | valid={valid}/{len(g)} | mean_tokens={toks:.1f}")

base={r["problem_id"]:r for r in groups.get("TTTT",[])}
for name in ["V@1","V@4"]:
    deltas=[]
    flips01=flips10=0
    for r in groups.get(name,[]):
        b=base.get(r["problem_id"])
        if b is None:
            continue
        bv=int(bool(b["correct"]))
        rv=int(bool(r["correct"]))
        deltas.append(rv-bv)
        if bv==0 and rv==1: flips01+=1
        if bv==1 and rv==0: flips10+=1
    if deltas:
        print(f"{name} - TTTT matched delta: {sum(deltas)/len(deltas):+.1%} | T wrong->V right={flips01} | T right->V wrong={flips10}")

print("\nFirst 6 outputs:")
for r in rows[:6]:
    print("\n", key(r), r["example_id"], "GT=",r["answer"], "Pred=",r["prediction"], "Correct=",r["correct"], "tokens=",r["output_tokens"])
    print(repr(r["raw_output"]))
PY
