# MMRB first-four image caption substitution

The official MMRB release stores **annotated multi-image questions inside `data_MMRB_20250531.zip`**. Do **not** run `load_dataset("HarrytheOrange/MMRB", "all")` or `"default"`: the modern datasets imagefolder loader does not return annotated question rows.

## New GPU Pod

```bash
git clone -b main --single-branch https://github.com/ZijianZhang22/Multimodal_test_switch.git /workspace/mmrb_project
cd /workspace/mmrb_project/mmrb_caption_first4
bash setup.sh
python experiment.py selftest
python -u experiment.py prepare --n 50 --max-images 10 --seed 42
```

Prepare downloads the official archive (about 6 GB), inspects its question JSON, prints image-count distribution, samples 50 questions with >=4 images, and extracts only their needed images. If archive schema or name differs, it fails with useful diagnostics rather than inventing data.

Conditions for the first four slots: VVVV / TTTT / VTTT / TVTT / TTVT / TTTV. **All later images remain V**. Caption model LLaVA-1.5-7B, greedy, max 200 tokens; QA model Qwen3-VL-8B-Thinking. A 6-image task in TTTT still includes 2 original images.

## After prepare succeeds, run in the background

```bash
mkdir -p logs
nohup bash -c 'set -e; python -u experiment.py caption; python -u experiment.py evaluate --max-new-tokens 1536; python -u experiment.py summarize' > logs/run.log 2>&1 < /dev/null &
echo $! > logs/run.pid
tail -f logs/run.log
```

Outputs: data/pilot/manifest.jsonl, results/captions.jsonl, results/predictions.jsonl, results/summary.csv and results/flips.csv.

**Caution:** Only the first four *physical slots* are intervened on, not necessarily four distinct logical hops. Review caption fidelity and ground-truth answer parsing before interpreting effects.
