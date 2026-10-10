#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-OpenGVLab/InternVL3_5-14B}"
LABEL="${LABEL:-internvl35_14b_4hop_direct}"
DTYPE="${DTYPE:-bf16}"
N_PROBLEMS="${N_PROBLEMS:-100}"
SEED="${SEED:-42}"
DATA_DIR="${DATA_DIR:-data_4hop_trace}"
RESULT_DIR="${RESULT_DIR:-results_internvl14b}"
DATA="${DATA_DIR}/depth_scaling_examples.jsonl"
RESULTS="${RESULT_DIR}/${LABEL}.jsonl"

cd "$(dirname "$0")"
mkdir -p "$RESULT_DIR"

if [ ! -f "$DATA" ]; then
  echo "Dataset not found; generating the same 4-hop matched dataset..."
  python generate_depth_scaling_dataset.py \
    --out_dir "$DATA_DIR" \
    --hops 4 \
    --n_problems_per_hop "$N_PROBLEMS" \
    --seed "$SEED" \
    --order_mode identity
fi

echo "=== Smoke test: 6 examples ==="
python run_internvl35.py \
  --data "$DATA" \
  --out "${RESULT_DIR}/smoke_${LABEL}.jsonl" \
  --model "$MODEL" \
  --model_label "$LABEL" \
  --dtype "$DTYPE" \
  --conditions text_baseline,single_visual \
  --visual_roles 1,4 \
  --hops 4 \
  --limit 6 \
  --max_new_tokens 16

echo
echo "=== Full TTTT / V@1 / V@4 replication ==="
python run_internvl35.py \
  --data "$DATA" \
  --out "$RESULTS" \
  --model "$MODEL" \
  --model_label "$LABEL" \
  --dtype "$DTYPE" \
  --conditions text_baseline,single_visual \
  --visual_roles 1,4 \
  --hops 4 \
  --max_new_tokens 16 \
  --resume

echo
echo "=== Summary ==="
python - "$RESULTS" <<'PY'
import json, sys
from collections import defaultdict

path=sys.argv[1]
rows=[json.loads(x) for x in open(path,encoding="utf-8") if x.strip()]
g=defaultdict(list)
for r in rows:
    k="TTTT" if r["condition"]=="text_baseline" else f"V@{r['visual_logical_step']}"
    g[k].append(r)

for k in ["TTTT","V@1","V@4"]:
    a=g[k]
    c=sum(bool(x["correct"]) for x in a)
    v=sum(x.get("prediction") is not None for x in a)
    print(f"{k}: {c}/{len(a)} = {c/len(a):.1%} | valid={v}/{len(a)}")

base={x["problem_id"]:x for x in g["TTTT"]}
for k in ["V@1","V@4"]:
    d=[]; wr=rw=0
    for r in g[k]:
        b=base[r["problem_id"]]
        bv=int(bool(b["correct"])); rv=int(bool(r["correct"]))
        d.append(rv-bv)
        if bv==0 and rv==1: wr+=1
        if bv==1 and rv==0: rw+=1
    print(f"{k}-TTTT matched delta: {sum(d)/len(d):+.1%} | T wrong->V right={wr} | T right->V wrong={rw}")
PY
