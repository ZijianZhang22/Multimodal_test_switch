# Mantis-Eval: 50 random questions, replace every image position

Uses the official [Mantis-Eval](https://huggingface.co/datasets/TIGER-Lab/Mantis-Eval) test split (217 questions). HF schema: `images` (list of 2–5 images), `question` with `<image>` markers, `options`, `answer`, `question_type`, `category`.

**Sampling:** 50 items with random seed 42 from the complete test split, with **no filtering by image count**. Each question keeps all original images and their order.

**Interventions:** For N images, run the distinct schedules corresponding to:
- All-V = N original images.
- All-T = N independent captions (no V tokens).
- T_at_i = replace image i with caption, keeping the other N-1 originals.

Each question runs **N+2** schedules: N=2 → 4, N=3 → 5, N=4 → 6, N=5 → 7. There are no keep-one-V experiments. **Total evaluation count depends on sampled image counts**.

Captions use `llava-hf/llava-1.5-7b-hf`, fixed detailed description prompt from *Bridging the Missing-Modality Gap*, greedy decoding, max 200 tokens. QA default model `Qwen/Qwen3-VL-8B-Thinking` with greedy decoding and 8192 max new tokens. Caption generation and QA inference are different Python processes so the models are not loaded simultaneously.

### New GPU Pod

```bash
cd /workspace
git clone -b main --single-branch \
  https://github.com/ZijianZhang22/Multimodal_test_switch.git mantis_project
cd /workspace/mantis_project/mantis_caption_swap
bash setup.sh
python experiment.py selftest
python -u experiment.py prepare --n 50 --seed 42
```

Only if prepare reports SELECTED: 50, start caption+QA in the background:

```bash
mkdir -p logs
nohup bash -c 'set -e; python -u experiment.py caption; python -u experiment.py evaluate --max-new-tokens 8192; python -u experiment.py summarize' > logs/run.log 2>&1 < /dev/null &
echo $! > logs/run.pid
tail -f logs/run.log
```

Or to run setup+prepare+caption+evaluate+summarize together:

```bash
mkdir -p logs
nohup bash run_all.sh > logs/run.log 2>&1 < /dev/null &
echo $! > logs/run.pid
```

Logs/results: `logs/run.log`, `data/pilot/manifest.jsonl`, `results/captions.jsonl`, `results/predictions.jsonl`, `results/summary.csv`, and `results/flips.csv`. Resume caption/evaluate by rerunning after interruption. The summary compares fully paired questions only and stratifies by original image count.

**Research caution:** This experiment measures modality substitution for physical image slots, not necessarily the step in a reasoning chain. A caption can drop task-critical information. Review paired examples/caption quality before making causal claims.

**Status:** Code tested on synthetic examples; Mantis-Eval end-to-end GPU run is not yet verified.

## Extend an ongoing 50-item pilot to ALL 217 Mantis-Eval questions

**Wait until any previous `experiment.py evaluate` / caption process has stopped or finished.** Do not write the same results files with two inference processes at once. From the Pod:

```bash
cd /workspace/mantis_project
git pull  # Updates your current branch; new clones above use main
cd mantis_caption_swap
python experiment.py selftest
cp data/pilot/manifest.jsonl data/pilot/manifest_before_all.jsonl
python -u experiment.py prepare --all --extend --seed 42
wc -l data/pilot/manifest.jsonl   # Should be 217
mkdir -p logs
nohup bash -c 'set -e; python -u experiment.py caption; python -u experiment.py evaluate --max-new-tokens 8192; python -u experiment.py summarize; echo ALL_217_COMPLETED' > logs/all_217.log 2>&1 < /dev/null &
echo $! > logs/all_217.pid
tail -f logs/all_217.log
```

`--extend` retains the original 50 selected rows in their original order, image paths, captions, and completed QA records, and adds the other 167 records; `--all` selects every question in the dataset. The manifest is only replaced after new images are successfully materialized. Caption and QA generation skip saved nontruncated work; QA retries older truncated records when their cap was below 8192. If the original run is still active, **do not execute prepare or launch another evaluation** until it exits.

## Token–accuracy analysis for the complete 217-item benchmark

After the 217-item experiment finishes, run `python -u token_analysis.py`. Results:

- `results/token_accuracy_summary.csv`: per-condition accuracy, mean/median/min/max generated tokens, differences versus All-V and a paired token-change versus correctness-change correlation, broken out by image count.
- `results/paired_token_deltas.csv`: question-level token changes, flips, truncation status, and token ratios against All-V.
- `results/token_accuracy_flips.csv`: counts and token delta distribution by correct→wrong / wrong→correct / same-correct / same-wrong transition.

The **primary cohort excludes any question with a truncated generation in any condition**, so changing token limits does not silently bias condition comparisons. The all-complete cohort is also retained for diagnostic reference. Output token count includes all generated thinking plus final answer, not just the reasoning trace.
