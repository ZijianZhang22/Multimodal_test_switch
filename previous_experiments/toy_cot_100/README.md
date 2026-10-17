# Four-hop toy: TTTT versus 1T3V, with CoT

100 latent spatial-chain problems, each evaluated under **TTTT, TVVV, VTVV, VVTV, VVVT**: 500 generations total. The single text fact occupies logical step 1, 2, 3 or 4. Evidence order, latent relations, diagrams, questions and gold answers are matched across conditions. Uses the existing `scripts/data/generate_switching_dataset.py`; this is the original toy, not a new benchmark. Seed 42 and the original approximately answer-balanced sampling are preserved (latent chains may repeat).

## Run from repository root

```bash
pip install -r requirements.txt
bash toy_cot_100/run_runpod.sh
```

Default model: `Qwen/Qwen3-VL-8B-Instruct`. Uses unrestricted visible step-by-step CoT and requires `FINAL: <LABEL>`, not the old 48-token short-CoT prompt. Default cap is 4096 generated tokens, greedy decoding, batch size 1.

```bash
MODEL=/workspace/models/Qwen3-VL-8B-Instruct MAX_NEW_TOKENS=8192 bash toy_cot_100/run_runpod.sh
```

Interrupt and rerun the same command to resume. Do not change model, prompt or budget while reusing an output directory. Use a fresh `RESULT_DIR` for a different setting. The existing data is reused; delete it or select another `DATA_DIR` to regenerate. To rerun capped cases with a different cap, start a fresh result directory rather than resuming old outputs.

## Outputs

- `data/latent_problems.jsonl`, `examples.jsonl`, diagrams and config.
- `results/predictions.jsonl`: raw CoT, gold/predicted answer, correctness, input/output token counts, generated token IDs, generation latency and conservative `hit_token_cap` flag.
- `results/summary.csv`: per-condition accuracy, mean/median output tokens, cap and invalid counts.
- `results/paired.csv`: matched accuracy/token deltas versus TTTT, correct-to-wrong and wrong-to-correct counts; all-pair and uncapped-pair analyses reported separately.

The Instruct model's **output tokens are not hidden thinking tokens**; `thinking_tokens` is null. Output includes CoT, final answer and any terminal special token. Missing FINAL is invalid. Reaching the cap is conservatively flagged even if EOS occurs exactly at that boundary. Do not interpret a capped output as efficient reasoning. Accuracy differences mix modality allocation with position and transition pattern; this five-condition design does not identify pure switching cost. No full GPU inference was run during script validation.
