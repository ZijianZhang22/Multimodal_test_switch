#!/usr/bin/env bash
# ZH question + EN task prefix/instruction; tune ZH steering vector on ZH tune,
# then evaluate once on held-out ZH test. Does not fine-tune model weights.
set -euo pipefail

REPO="\${REPO:-/workspace/Multimodal_test_switch}"
HERE="$REPO/paligemma2_cross_lingual_steering/parity"
DATA="\${DATA:-$REPO/paligemma2_cross_lingual_steering/outputs/parity_cvbench}"
RUN="\${RUN:-zh_only_enprompt20}"
LIMIT="\${LIMIT:-20}"          # per task, independently for tune/test
OUT="$DATA/runs/$RUN"
PY="$REPO/paligemma2_cross_lingual_steering/.venv-parity/bin/python"

cd "$HERE"
[[ -x "$PY" ]] || { echo "Missing environment: $PY"; exit 1; }
[[ -f "$DATA/visual_eval.jsonl" ]] || { echo "Missing prepared data: $DATA/visual_eval.jsonl"; exit 1; }
[[ -f "$DATA/vectors/manifest.json" ]] || { echo "Missing cached concept vectors. Run original parity pipeline first."; exit 1; }
mkdir -p "$OUT"

# Concept fixed per task; only alpha/layer tuned on the ZH tune split.
"$PY" - "$OUT/jobs_tune.json" <<'PY'
import json, sys
from pathlib import Path
concepts={'count':'counting','relation':'spatial','depth':'spatial','distance':'spatial'}
jobs=[dict(task=task, prompt_lang='zh', style='upstream_en', concept=concept,
           variant='zh', layer=layer, alpha=alpha)
      for task, concept in concepts.items() for layer in (5,8,12) for alpha in (0.3,1.0)]
Path(sys.argv[1]).write_text(json.dumps(jobs, indent=2), encoding='utf-8')
print(f'Tune: {len(jobs)} settings/sample × {4} tasks × limit', flush=True)
PY

"$PY" evaluate.py --data "$DATA" --split tune --limit "$LIMIT" \
    --jobs "$OUT/jobs_tune.json" --out "$OUT/tune.csv"
"$PY" select_best.py --pred "$OUT/tune.csv" --out "$OUT/best_zh.json"

"$PY" - "$OUT/best_zh.json" "$OUT/jobs_test.json" <<'PY'
import json, sys
from pathlib import Path
best=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
groups=[g for g in best['groups'] if g['prompt_lang']=='zh' and g['style']=='upstream_en' and g['variant']=='zh']
assert len(groups)==4, f'Expected 4 task settings; got {groups}'
jobs=[{k:g[k] for k in ('task','prompt_lang','style','concept','variant','layer','alpha')} for g in groups]
Path(sys.argv[2]).write_text(json.dumps(jobs, indent=2), encoding='utf-8')
print('Chosen test settings:')
for j in jobs: print(j)
PY

"$PY" evaluate.py --data "$DATA" --split test --limit "$LIMIT" \
    --jobs "$OUT/jobs_test.json" --out "$OUT/test.csv"
"$PY" analyze.py --pred "$OUT/test.csv" --out "$OUT/analysis"

echo ""
echo "DONE. Results: $OUT/analysis/SUMMARY.md"
echo "Raw paired outputs: $OUT/test.csv"
echo "Upload to ChatGPT by running:"
echo "  tar -czf /workspace/zh_enprompt20_results.tar.gz -C \"$DATA\" \"runs/$RUN\""
