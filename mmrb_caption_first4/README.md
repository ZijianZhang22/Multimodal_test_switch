# MMRB caption substitution: first four image slots

Research experiment on [MMRB](https://github.com/LesterGong/MMRB) with >=4 candidate images. Sample target: **50 questions**. Preserve image slots 5+ as images in **every condition**.

Schedules, applied only to slots 1–4: **VVVV, TTTT, VTTT, TVTT, TTVT, TTTV**. For questions with 6 images, VTTT means V T T T V V. Thus TTTT is **not** all-text when total images >4.

V = original image. T = a caption generated independently using LLaVA-1.5-7B, the fixed detailed-description instruction, greedy decoding and max 200 new tokens as in *Bridging the Missing-Modality Gap*. Final QA model defaults to Qwen3-VL-8B-Thinking. Check original captions before interpreting swaps as semantic-equivalent.

## On a new GPU Pod

```bash
git clone -b experiments/mmrb-caption-first4 --single-branch https://github.com/ZijianZhang22/Multimodal_test_switch.git /workspace/mmrb_project
cd /workspace/mmrb_project/mmrb_caption_first4
bash setup.sh
python experiment.py selftest
python experiment.py prepare --n 50 --max-images 10 --seed 42
python experiment.py caption
python experiment.py evaluate --max-new-tokens 1536
python experiment.py summarize
```

### Launch in background after confirming prepare works

```bash
mkdir -p logs
nohup bash -c 'set -e; python -u experiment.py caption; python -u experiment.py evaluate --max-new-tokens 1536; python -u experiment.py summarize' > logs/run.log 2>&1 < /dev/null &
echo $! > logs/run.pid
tail -f logs/run.log
```

Run prepare *in the foreground first* to see whether the Hugging Face release exposes decoded images through `load_dataset("HarrytheOrange/MMRB","all")`. The official downloader uses this API, but the HF release may be ZIP-backed on your machine. If preparing fails, paste the error and the `list_repo_files` output; do not attempt 300 GPU runs until loading is fixed.

Results: `data/pilot/manifest.jsonl`, `results/captions.jsonl`, `results/predictions.jsonl`, `results/summary.csv`, `results/flips.csv`.

The first four images are **physical slots**, not necessarily annotated logical hops. Record total original image counts and task types; don't infer causal reasoning roles from position alone.

If memory is insufficient, reduce `--max-images` during prepare and/or `--max-new-tokens` during evaluation; don't change the benchmark without recording it.
