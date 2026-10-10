# Idis-math50: 50 genuinely new visual mathematics problems

Excludes all versions of the initial 10 source MathVerse problems (problem indices: 153, 332, 202, 272, 238, 686, 147, 261, 556, 573) as well as their original sample indices.

RunPod:
```bash
cd /workspace/Multimodal_test_switch
git pull origin main
cd vlm_cross_task_pilot
bash idis_math10/run_50new.sh
```

Default: 50 distinct previously unseen source problems x 2 conditions = 100 generations; official Idis-math irrelevant/n4 visuals, Qwen3-VL-8B-Thinking, temp=0.7, 8192->12288 max new tokens to reduce truncation, seed=1043.

Outputs separate from first run: `idis_math50_new/results/predictions.jsonl`, `pairs.csv`, `summary.csv`, `manual_labels.csv`, `needs_review.csv`. Resumable. Correctness for nontrivial expressions still requires manual validation.

Disjointness check:
```bash
python - <<'PY'
import json
rows=[json.loads(s) for s in open('idis_math50_new/data/manifest.jsonl')]
old={'153','332','202','272','238','686','147','261','556','573'}
ids=[str(r['problem_index']) for r in rows]
assert len(ids)==50 and len(set(ids))==50 and old.isdisjoint(ids)
print('50 NEW distinct MathVerse problems, 0 overlap with prior 10')
PY
```

Package result:
```bash
tar czf idis_math50_new_results.tar.gz idis_math50_new/results/ idis_math50_new/data/manifest.jsonl
```

Official images are downloaded by `prepare.py` only for chosen questions; no image generation.
