# Idis30 re-tokenization of saved Thinking

Run on the previous RunPod (GPU **not required**):

```bash
cd /workspace/Multimodal_test_switch
git pull origin main
cd vlm_cross_task_pilot
bash run_idis30_thinking_tokens.sh
```

Reads the **existing 240** `idis30/results/predictions.jsonl` outputs.
Requires `transformers` and Hugging Face access for the tokenizer on the first run;
the existing experiment environment already has this dependency.

Outputs `idis30/results/thinking_token_analysis/` with:
- `retokenized_summary.csv`: comparison across clean/aligned/conflicting/irrelevant + all
- `retokenized_pairs.csv`: 120 pairwise thinking/final token comparisons
- `retokenized_outputs.csv`: 240 individually reencoded outputs and diagnostic reconstruction gaps
- `top_compressions.csv` / `top_expansions.csv`

Outputs use the same model tokenizer as the original Qwen3-VL-8B-Thinking run, but
re-tokenization of **decoded text** is only an **estimate** of original token count
because original token IDs were not archived. Original total output token counts
remain exact from `num_new_tokens`. Multi-task thinking includes reasoning for both
visual Q1 and arithmetic Q2 and must not be called Q1-specific compression.

Statistics: ID-cluster bootstrap intervals (30 underlying images, 4 image variants)
rather than treating 120 augmented variants as independent.

Compress the results:
```bash
tar czf idis30_thinking_token_results.tar.gz idis30/results/thinking_token_analysis/
```
