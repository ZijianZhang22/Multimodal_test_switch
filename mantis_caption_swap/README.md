# Mantis-Eval: 50 random questions, replace every image position

Uses the official [Mantis-Eval](https://huggingface.co/datasets/TIGER-Lab/Mantis-Eval) test split (217 questions). HF schema: `images` (list of 2–5 images), `question` with `<image>` markers, `options`, `answer`, `question_type`, `category`.

**Sampling:** 50 items with random seed 42 from the complete test split, with **no filtering by image count**. Each question keeps all original images and their order.

**Interventions:** For N images, run the distinct schedules corresponding to:
- All-V = N original images.
- All-T = N independent captions (no V tokens).
- T_at_i = replace image i with caption, keeping the other N-1 originals.

Each question runs **N+2** schedules: N=2 → 4, N=3 → 5, N=4 → 6, N=5 → 7. There are no keep-one-V experiments. **Total evaluation count depends on sampled image counts**.

Captions use `llava-hf/llava-1.5-7b-hf`, fixed detailed description prompt from *Bridging the Missing-Modality Gap*, greedy decoding, max 200 tokens. QA default model `Qwen/Qwen3-VL-8B-Thinking` with greedy decoding and 1536 max new tokens. Caption generation and QA inference are different Python processes so the models are not loaded simultaneously.

### New GPU Pod

```bash
cd /workspace
git clone -b experiments/mantis-eval-caption-swap --single-branch \
  https://github.com/ZijianZhang22/Multimodal_test_switch.git mantis_project
cd /workspace/mantis_project/mantis_caption_swap
bash setup.sh
python experiment.py selftest
python -u experiment.py prepare --n 50 --seed 42
```

Only if prepare reports SELECTED: 50, start caption+QA in the background:

```bash
mkdir -p logs
nohup bash -c 'set -e; python -u experiment.py caption; python -u experiment.py evaluate --max-new-tokens 1536; python -u experiment.py summarize' > logs/run.log 2>&1 < /dev/null &
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
