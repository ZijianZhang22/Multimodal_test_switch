# MME-CoT × ConPress: Original Task vs Original Task + 1+1

Independent, quick pilot for **Project 1 (Reasoning Compression)**. This folder is isolated from all other experiments in Multimodal_test_switch.

**Official benchmark:** [CaraJ/MME-CoT](https://huggingface.co/datasets/CaraJ/MME-CoT), ICML 2025. `prepare.py` downloads the official test split and filters to `question_type == Reasoning` with at least one image. It stratifies across the original `category` labels, rather than selecting one toy math domain.

**Default model:** [Qwen/Qwen3-VL-8B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking).

## Quick start on RunPod

Use an NVIDIA GPU with CUDA PyTorch; A100 80GB recommended for long reasoning. Model weights and the ~279 MB benchmark dataset download from Hugging Face.

```bash
cd /workspace
git clone https://github.com/ZijianZhang22/Multimodal_test_switch.git
cd Multimodal_test_switch/mme_cot_conpress_pilot
bash run_runpod.sh
```

This starts a 12-problem stratified reasoning-only smoke pilot. It executes **two inference conditions per selected example**: baseline and plus_arithmetic, using the same original images, original question, answer options, model, and seed.

To increase the number of questions:

```bash
EXAMPLES=30 PER_CATEGORY=5 bash run_runpod.sh
```

To inspect just one category, set its exact label after reviewing the printed eligible-category counts:

```bash
CATEGORIES=Science EXAMPLES=10 PER_CATEGORY=10 bash run_runpod.sh
```

Resume unfinished experiments simply by running the same command again; completed `(index,condition)` pairs are skipped.

## Two conditions

**Baseline**: Original MME-CoT image(s) and question, plus the shared prompt asking to reason and output `<target_answer>...</target_answer>`.

**Plus arithmetic**: Exactly the same prompt, with only an extra independent task appended: `What is 1 + 1?` and a request to output `<arithmetic_answer>...</arithmetic_answer>`.

The model is NEVER instructed to compress. The 1+1 prompt is deliberately easy. There is no artificial guidance forcing shorter reasoning in either group.

See `prompts.py` for the fully specified paired prompts.

## No arbitrary output-length truncation

`run.py` deliberately has **no global max_new_tokens / max output count**. A generation call emits up to `--chunk-tokens` tokens (default 1024), and if the model has not emitted its real EOS, the script passes the complete accumulated token IDs back into generation and **continues**, repeatedly, until a natural model turn-end token. It does not count each full chunk as a completed response.

This is slower than one-shot generation because the prompt/images are processed again for each chunk. Finite context limits and GPU memory still apply to any transformer; if the model exhausts those resources, the case is explicitly marked `generation_error`, NEVER silently scored as a completed truncated response. Use Ctrl+C to interrupt infinite/very lengthy reasoning.

Change the internal chunk allocation size without imposing a total cap:

```bash
CHUNK_TOKENS=2048 bash run_runpod.sh
```

`TEMPERATURE=0` is deterministic by default to limit stochastic noise in a paired comparison; use nonzero values and multiple independent runs for a future statistical study.

## Files and measurements

Output under `pilot/` (gitignored):

- `data/pairs.jsonl` and `data/images/`: official questions/images, answers, categories and key-step annotations.
- `results/predictions.jsonl`: **full generated outputs**, model thinking if present, exact raw output tokens, decoded readable response, finished EOS status, prompt, latency; append-only and resumable.
- `results/scored.csv`: parsed target final answer, simple arithmetic correctness, generation and thinking tokens.
- `results/paired.csv`: **matched** baseline-vs-plus_arithmetic CoT token changes and outcome flips for the same item.
- `results/summary.csv`: all-domain and per-category mean/median compression, token changes, multiple-choice correctness.

`thinking_tokens` is measured from text between `<think>...</think>` by re-tokenizing that span (an approximation); `generation_tokens` is the **exact** total emitted token count. The joint answer may include some reasoning devoted to 1+1, so interpret thinking-token differences conservatively.

For final answer grading, **multiple-choice** is parsed strictly from `<target_answer>`. Non-MC free-form answers receive a strict normalized-string equality proxy, **not the official MME-CoT LLM-judge evaluation**. For meaningful accuracy claims, manually inspect free-form answers or use the authors’ judge framework: https://github.com/MME-Benchmarks/MME-CoT .

## Interpretation limits

- The benchmark includes reasoning-focused questions, but some may still elicit short traces; inspect the baseline CoT length before declaring these long-reasoning tasks.
- Positive `compression_pct` = fewer thinking tokens in original+1+1; negative = expansion.
- A paired token difference is not yet evidence of better or worse reasoning. Compare accuracy and preserved key inference/visual evidence steps before making a research claim.
- The two prompts differ in total length and output requirements. This is a minimal replication-style pilot, not a full mechanism-controlled causal study.
- Runtime is not tested with GPU in this chat; the included smoke test checks prompt/scoring logic offline.

## Underlying references

- [MME-CoT official code](https://github.com/MME-Benchmarks/MME-CoT)
- [MME-CoT dataset](https://huggingface.co/datasets/CaraJ/MME-CoT)
- [Qwen3-VL Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking)
