# Multimodal Test Switch

Pilot code for a controlled **cross-modal evidence switching** experiment.

## Question

Holding the latent 4-hop reasoning problem and the **2 Text + 2 Vision evidence budget**
fixed, does increasing the number of V↔T evidence-source transitions reduce reasoning accuracy?

Core schedules:

- `TTVV`, `VVTT`: 1 switch
- `TVVT`, `VTTV`: 2 switches
- `TVTV`, `VTVT`: 3 switches

The same latent facts are reused across all six variants of each problem.

## Install

```bash
pip install -r requirements.txt
```

## Generate data

```bash
python generate_switching_dataset.py \
  --out_dir data \
  --n_problems 100 \
  --seed 42 \
  --include_unimodal
```

## Smoke-test Qwen3-VL

```bash
python run_qwen3vl.py \
  --data data/examples.jsonl \
  --out results/smoke.jsonl \
  --limit 30
```

## Full run

```bash
python run_qwen3vl.py \
  --data data/examples.jsonl \
  --out results/qwen3vl_results.jsonl
```

## Analyze

```bash
python analyze_results.py \
  --results results/qwen3vl_results.jsonl \
  --out_dir analysis
```

Main outputs:
- accuracy by schedule
- accuracy by switch count
- paired 1-switch vs 3-switch difference
- exact McNemar comparisons
- accuracy-vs-switches plot
