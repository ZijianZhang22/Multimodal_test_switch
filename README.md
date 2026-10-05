# Multimodal Test Switch

Controlled pilot for studying **where modality matters inside a multi-hop reasoning chain**.

## Round 1 result

The first 100-problem pilot did **not** show a monotonic switching-cost effect.
However, it showed a large order asymmetry, especially:

- \`TTVV\`: 22%
- \`VVTT\`: 40%

Because both conditions use exactly 2 text facts, 2 visual facts, and 1 switch, the
next question is whether the difference is caused by:

- evidence position,
- transition direction,
- visual/text difficulty,
- start/end modality,
- or switching itself.

## Round 2: full 16-schedule factorial design

For each 4-hop latent problem, each fact can independently be carried by Text (T)
or Vision (V). This gives all \(2^4 = 16\) schedules:

\`\`\`text
TTTT TTTV TTVT TTVV
TVTT TVTV TVVT TVVV
VTTT VTTV VTVT VTVV
VVTT VVTV VVVT VVVV
\`\`\`

The latent facts, order, question, answer, and task difficulty are unchanged.

## Install / update

On RunPod:

\`\`\`bash
cd /workspace/Multimodal_test_switch
git pull
pip install -r requirements.txt
\`\`\`

## Generate Round-2 data

This now defaults to all 16 schedules:

\`\`\`bash
python generate_switching_dataset.py \
  --out_dir data16 \
  --n_problems 100 \
  --seed 42
\`\`\`

Expected output:

\`\`\`text
Generated 100 latent problems.
Generated 1600 examples.
\`\`\`

To reproduce the old pilot instead:

\`\`\`bash
python generate_switching_dataset.py \
  --out_dir data_old \
  --n_problems 100 \
  --seed 42 \
  --schedule_set core8
\`\`\`

## Smoke test

\`\`\`bash
python run_qwen3vl.py \
  --data data16/examples.jsonl \
  --out results16/smoke.jsonl \
  --limit 16
\`\`\`

## Full Round-2 run

\`\`\`bash
python run_qwen3vl.py \
  --data data16/examples.jsonl \
  --out results16/qwen3vl_results.jsonl
\`\`\`

## Analyze

\`\`\`bash
python analyze_results.py \
  --results results16/qwen3vl_results.jsonl \
  --out_dir analysis16
\`\`\`

The analysis now reports:

- accuracy for all 16 schedules
- accuracy by switch count
- accuracy by number of visual facts
- start-modality and end-modality effects
- paired visual-vs-text effects at reasoning steps 1, 2, 3, and 4
- reversal-pair McNemar tests
- clustered logistic regression:
  \`correct ~ V1 + V2 + V3 + V4 + switch_count\`

The key scientific question is now:

> After controlling for which reasoning positions are visual, does switch count
> still matter? Or is performance mainly determined by *where* visual evidence
> appears in the reasoning chain?

## Recommended interpretation

Do **not** treat a simple Image-first/Text-first difference as the final finding.
The useful result would be something more specific, for example:

- visual evidence helps at early steps but hurts at late steps,
- T→V transitions are harder than V→T,
- or switch count becomes negligible after controlling for step position.

Those findings are more informative than a generic "order matters" result.
