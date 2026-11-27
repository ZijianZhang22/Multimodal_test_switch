# Larger test: Chinese baseline vs Text + Visual OT only

This extension **does not train a new map**. It reuses the **validation-selected** `A/text_visual_ot` map from the first 237-item Chinese CV-Bench validation (in the initial run: layer **8**, alpha **0.6**). Both conditions use **the same PaliGemma2-3B**, prompt style, answer extraction, and image for each question. The sole difference is the frozen visual-assisted OT hidden-state intervention. No other method, Oracle, English reference, LoRA, or hyperparameter search is run.

## What are the extra questions?

Official CV-Bench contains **2,638 publicly released test questions**. The original local multilingual cohort has **237 validation** and **600 test** rows, with fixed EN/ZH/SW prompts. The larger test, by default, has:

- **600 original curated Chinese test items**, unchanged;
- **900 additional CV-Bench items** not in the 237-item validation nor original 600 test, translated *only for this evaluation* from English to Chinese with `facebook/nllb-200-distilled-600M`, using a resolved, recorded immutable Hugging Face revision. Answer labels and multiple-choice option order are unchanged. Numeric and simple spatial-direction options have deterministic translations.

The extra items are stratified across `count`, `relation`, `depth`, and `distance`. They never share decoded image pixels with validation; the sampler prefers unique test images, but allows different test questions about the same image if necessary. The final confidence interval therefore clusters by image. Train/test disjointness is checked using exact decoded-image content against the original Flickr8K training pools. These checks do **not** detect near-duplicate or model-pretraining leakage.

This is an **exploratory local extension of a public test benchmark**, not a new official blind test split. Additional Chinese translations are **machine-generated and not manually verified**. In particular, the 600 curated and 900 MT questions have different translation provenance, so separate group accuracy and deltas are essential. Inspect `translation_audit.csv` (especially spatial relations) before publication. Automatic checks only verify response options/labels and nonempty translated strings; they cannot verify that NLLB preserved the visual question meaning. Translation errors can make the expanded group misleading.

## Run on the same RunPod as the first completed experiment

Original files expected:

```text
/workspace/vlm_A_zh_seed42/
  flickr/{source,english,visual}.jsonl
  cvbench/{val,test}.jsonl
  experiment/{settings.json,selection_val.json,maps/layer8_text_visual_ot.npz}
```

Update the existing branch:

```bash
cd /workspace/Multimodal_test_switch
git pull origin experiment/vlm-unsupervised-a-b
```

Start only the new two-arm comparison, in background:

```bash
mkdir -p /workspace/vlm_A_zh_seed42/logs
nohup env TARGET_TOTAL=1500 bash vlm_unsupervised_transfer/run_visual_large.sh \
  > /workspace/vlm_A_zh_seed42/logs/visual1500.log 2>&1 < /dev/null &
echo $! > /workspace/vlm_A_zh_seed42/logs/visual1500.pid
```

Watch:

```bash
tail -f /workspace/vlm_A_zh_seed42/logs/visual1500.log
```

The first run downloads NLLB for **translation only** and prepares the extra images. The translation model and the VLM run in **different processes**, so GPU memory is released between them. After preparation it loads the existing PaliGemma2, runs exactly two generations per question, and appends predictions to durable JSONL caches after each example. If interrupted, **rerun the same command** with the same paths, and previously saved predictions will be skipped. The checkpoint includes source-prompt and dataset hashes to refuse stale/mixed results. The latest audit also rehashes the actual image pixels at startup; if the final JSONL line was interrupted mid-write, only that incomplete trailing line is discarded and recomputed.

For more than 1500, use `TARGET_TOTAL=2000`, **along with new independent output directories**, e.g.:

```bash
TARGET_TOTAL=2000 \
EXPANDED_DATA=/workspace/vlm_A_zh_seed42/cvbench_extended2000 \
RESULTS=/workspace/vlm_A_zh_seed42/visual_only_2000 \
  bash vlm_unsupervised_transfer/run_visual_large.sh
```

**Do not increase to 2000 after seeing the 1500-question result merely to obtain significance**. Choose the expanded sample size beforehand if treating the result as confirmatory. The maximum pool is smaller than 2,638, because validation questions (and possible image duplicates) must be excluded. `TARGET_TOTAL=2000` may still require more compute; on RTX 2000 Ada, generation dominates the runtime.

## Outputs

```text
/workspace/vlm_A_zh_seed42/
  cvbench_extended1500/
    test_extended.jsonl
    manifest.json
    translation_audit.csv
    translation_cache_zh.json
    new_images/*.png
  visual_only_1500/
    run_manifest.json
    predictions/{baseline_zh,text_visual_ot}.jsonl
    comparison.csv
    summary.json
```

The JSON report provides both models' accuracy, repaired/harmed questions, invalid-response rate, exact McNemar p-value **only if every question has a unique image**; otherwise an exploratory image-cluster sign-flip permutation p-value, and a 95% paired **image-cluster bootstrap** interval, and counts by task. It explicitly separates `curated_existing`, `new_mt_zh`, and `all`. The interval does not correct for prior selection among 36 validation configurations or all repeated research attempts; interpret conservatively.

`TARGET_TOTAL=1500` means approximately **3,000 PaliGemma2 generations**, much less than the previous 36-arm validation sweep, but on a modest GPU may still take hours. Use the run log and actual throughput to estimate time rather than a generic time promise.

CPU tests (without downloading models):

```bash
cd /workspace/Multimodal_test_switch/vlm_unsupervised_transfer
.venv/bin/python -m unittest discover -s tests -v
```
