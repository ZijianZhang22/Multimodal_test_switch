#!/usr/bin/env bash
# ZH questions + EN prefix/instruction. Tune ZH steering on ZH tune split only,
# then evaluate selected settings on held-out ZH test. No model weight updates.
set -euo pipefail

REPO="${REPO:-/workspace/Multimodal_test_switch}"
HERE="$REPO/paligemma2_cross_lingual_steering/parity"
DATA="${DATA:-$REPO/paligemma2_cross_lingual_steering/outputs/parity_cvbench}"
RUN="${RUN:-zh_only_enprompt20_wide}"
LIMIT="${LIMIT:-20}"                 # per task in both tune and test
LAYERS="${LAYERS:-5,8,10,12,14,16,18,20}"
ALPHAS="${ALPHAS:-0.1,0.3,0.6,1.0,1.5}"
OUT="$DATA/runs/$RUN"
PY="$REPO/paligemma2_cross_lingual_steering/.venv-parity/bin/python"

cd "$HERE"
[[ -x "$PY" ]] || { echo "Missing environment: $PY"; exit 1; }
[[ -f "$DATA/visual_eval.jsonl" ]] || { echo "Missing prepared data: $DATA/visual_eval.jsonl"; exit 1; }
[[ -f "$DATA/vectors/manifest.json" ]] || { echo "Missing cached concept vectors. Run original parity pipeline first."; exit 1; }
mkdir -p "$OUT"
echo "ZH steering + EN prompt | RUN=$RUN | LIMIT=$LIMIT | LAYERS=$LAYERS | ALPHAS=$ALPHAS"

# Aligned concept per task; search alpha/layer using only ZH tuning examples.
"$PY" - "$OUT/jobs_tune.json" "$LAYERS" "$ALPHAS" "$LIMIT" <<'PY'
import json
import sys
from pathlib import Path

def read_layers(text):
    result = []
    for item in text.split(','):
        item = item.strip()
        if not item:
            continue
        if '-' in item:
            start, end = (int(x) for x in item.split('-', 1))
            if end < start:
                raise ValueError(f'Invalid layer range: {item}')
            result.extend(range(start, end + 1))
        else:
            result.append(int(item))
    result = sorted(set(result))
    if not result or any(x < 0 or x >= 26 for x in result):
        raise ValueError(f'Layers must be in 0..25; got {result}')
    return result

layers = read_layers(sys.argv[2])
alphas = sorted(set(float(x.strip()) for x in sys.argv[3].split(',') if x.strip()))
if not alphas or any(a <= 0 or a > 3 or round(a, 4) != a for a in alphas):
    raise ValueError(f'Choose positive alphas <=3, max 4 decimal places; got {alphas}')
limit = int(sys.argv[4])
if limit < 1:
    raise ValueError('LIMIT must be a positive number per task for this pilot.')
concepts = {'count': 'counting', 'relation': 'spatial', 'depth': 'spatial', 'distance': 'spatial'}
jobs = [dict(task=task, prompt_lang='zh', style='upstream_en', concept=concept,
             variant='zh', layer=layer, alpha=alpha)
        for task, concept in concepts.items() for layer in layers for alpha in alphas]
Path(sys.argv[1]).write_text(json.dumps(jobs, indent=2), encoding='utf-8')
print(f'Tune: {len(layers)} layers × {len(alphas)} alphas × 4 tasks × {limit} examples = '
      f'{len(jobs) * limit} steering generations (+ {4 * limit} baseline).', flush=True)
PY

"$PY" evaluate.py --data "$DATA" --split tune --limit "$LIMIT" \
    --jobs "$OUT/jobs_tune.json" --out "$OUT/tune.csv"
"$PY" select_best.py --pred "$OUT/tune.csv" --out "$OUT/best_zh.json"

"$PY" - "$OUT/best_zh.json" "$OUT/jobs_test.json" <<'PY'
import json
import sys
from pathlib import Path

best = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
groups = [g for g in best['groups'] if g['prompt_lang'] == 'zh'
          and g['style'] == 'upstream_en' and g['variant'] == 'zh']
assert len(groups) == 4, f'Expected four task settings; got {groups}'
jobs = [{k: g[k] for k in ('task', 'prompt_lang', 'style', 'concept', 'variant', 'layer', 'alpha')}
        for g in groups]
Path(sys.argv[2]).write_text(json.dumps(jobs, indent=2), encoding='utf-8')
print('Chosen held-out test settings:')
for job in jobs:
    print(job)
PY

"$PY" evaluate.py --data "$DATA" --split test --limit "$LIMIT" \
    --jobs "$OUT/jobs_test.json" --out "$OUT/test.csv"
"$PY" analyze.py --pred "$OUT/test.csv" --out "$OUT/analysis"

echo
echo "DONE. Summary: $OUT/analysis/SUMMARY.md"
echo "Raw paired results: $OUT/test.csv"
echo "Archive results: tar -czf /workspace/zh_enprompt_wide_results.tar.gz -C \"$DATA\" \"runs/$RUN\""
