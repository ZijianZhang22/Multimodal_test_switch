# WorldCuisines custom food steering pilot

Independent new experiment folder. Fixed **English** questions/answers, culturally diverse food images. This pilot asks whether a generic food concept improves dish recognition; it does not yet measure multilingual transfer, cultural fairness, or reasoning chains. No VFA or RAG is included in this first ablation.

## Quick start (RunPod / CUDA Linux)

From the repository root, after `git pull`:

```bash
# Accept BOTH Google model licenses on Hugging Face first:
# https://huggingface.co/google/gemma-2-2b
# https://huggingface.co/google/paligemma2-3b-mix-448
# Supply a read-access HF_TOKEN in your environment, or log in to Hugging Face.

# First run: 50 validation images; prepare another 50 held-out test images.
# 50 x (2 + 5) = 350 generations at this one-layer, one-alpha setting.
LAYERS=8 ALPHAS=0.1 bash worldcuisines_food_steering/run.sh
```

The script creates its own `.venv`, installs requirements, runs CPU protocol/hook tests, prepares data, extracts text-only vectors, evaluates **validation only**, then writes `summary_val.csv` and `selection.json`. Extraction and VLM evaluation run in separate processes so both models need not fit simultaneously. BF16 CUDA is required; a 24 GB GPU is a reasonable starting point, but peak VRAM has not been measured. Batch size is 1. Internet and gated model access are required.

For a larger experiment, start a **new** output directory:

```bash
DATA=outputs/full SOURCE_SPLIT=test_large N_VAL=100 N_TEST=200 \
  LAYERS=8,16 ALPHAS=0.1,0.3 INSTALL=0 \
  bash worldcuisines_food_steering/run.sh

# After inspecting validation and freezing the selected setting:
DATA=outputs/full SPLIT=test INSTALL=0 \
  bash worldcuisines_food_steering/run.sh
```

Test reads the layer, alpha and mask from validation `selection.json`; it evaluates only that setting plus its matched negative/random controls and the two baselines. Do not repeatedly tune on test. A selected positive vector may still be worse than baseline. Defaults without environment overrides search layers 8,16 and magnitudes 0.1,0.3 (22 generations/image).

`DATA` is relative to **this folder**, not the repository root. Do not change preparation/extraction options in an existing DATA directory: the runner reuses those artifacts. Use a new DATA path for changed concepts, layers, cohort sizes, seed, or source split. Evaluation refuses to mix different settings in existing results; exact reruns resume completed rows. A partially written final JSONL line must be removed manually after checking the file. Failed images stop preparation; they are not silently excluded from the sample.

## Method and controls

* Extraction model: `google/gemma-2-2b`; target VLM: `google/paligemma2-3b-mix-448`.
* `concepts.json`: 20 authored generic food anchors (food, ingredients, meal, cuisine, cooking, etc.). No test answer names are used to construct or select per-image vectors.
* For each zero-based decoder layer, pool **all target-span tokens** across the corpus and subtract the pooled remaining non-special tokens. Tokenizer offsets align spans. Keep the vector's original norm. Hooks collect raw block outputs, including for the final layer, avoiding final-layer-normalization ambiguity.
* Intervention: `h += alpha * vector` at one decoder block, on masked **prompt-prefill tokens only**. Default `MASK=both` includes image and ordinary text tokens. `MASK=image`/`text` are supported. Generated tokens are not directly shifted; the modified prompt influences subsequent decoding through cached states. Greedy generation, no gradient training.
* Controls: baseline; ordinary food-focused prompt; positive food vector; negative food vector; three independent random directions (seeds 0,1,2), each with the **same norm as the food vector** at that layer. Alpha is a raw-vector multiplier, not a hidden-state percentage.

This is an independent adaptation inspired by [TextSteersVision](https://github.com/Woodygan/TextSteersVision), **not an exact replication** of its published results or upstream wrapper. In particular our prefill-only hook and special-token mask differ from its implementation. A broad concept can increase food-related output without improving fine-grained recognition; the controls and paired accuracy are the test of that hypothesis.

## Data and metrics

Uses [WorldCuisines v1.1](https://huggingface.co/datasets/worldcuisines/vqa-v1.1), config `task1`, language `en`, `prompt_type=1` (no context), **zero-based** `mcq_answer_index`. Five original options are rendered as A-E; question and option content/order are preserved. Download only selected source image URLs. For an existing v1.0 image archive, manually run `prepare.py --image-root /path/to/archive --data ...` before the runner; the source `image_path` must resolve underneath that root. Source URLs can become unavailable; a local archive may be needed.

Deduplicate by source image URL, then deterministically partition by `food_id`; the same dish ID or URL cannot occur in both local val and test. This does not detect perceptually duplicate images hosted at different URLs or establish absence from model pretraining. Samples are drawn from the official evaluation split, so these are **local pilot scores, not official benchmark scores**. The dataset is CC-BY-SA 4.0; retain source attribution and image-specific license information when redistributing data. We commit code and generic anchors, not dataset images.

Results include image accuracy, dish-macro accuracy, valid-output rate, paired gains/losses and delta vs baseline. Invalid generations count as incorrect. The strict parser accepts a standalone A-E, `Answer: A`, or a unique exact option name; explanations and ambiguous responses are invalid. Always inspect `raw` outputs if validity is low. Validation selects positive-food settings by image accuracy, with ties resolved by smaller alpha then lower layer. Random seeds are reported separately, not cherry-picked. This pilot has no region annotations or significance claims; add region-level evaluation and dish-cluster bootstrap intervals before claiming cultural gains.

Artifacts under `outputs/<run>/`:

* `samples.jsonl`, `images/`, `dataset_manifest.json`: source IDs, split, options, URLs, image hashes and resolved dataset revision.
* `vectors/layer_*.pt`, `vectors/manifest.json`: vectors, anchors, norms, resolved model revision and package versions.
* `predictions_val.jsonl` / `predictions_test.jsonl`: prompts, raw output, parsed answer, ground truth and correctness per condition.
* `settings_*.json`, `summary_*.csv`, `selection.json`: experiment provenance and frozen choice. Model revisions, data and vector hashes are checked before held-out evaluation.

## Custom concepts and checks

Copy `concepts.json`, write `{ "sentence": "...", "target": "..." }` entries with each target occurring exactly once, then set `CONCEPTS=/absolute/path/my_concept.json DATA=outputs/new_concept`. A `dumplings` vector is possible, but applying the correct dish-specific vector only to known dumpling answers would leak labels. Use the same chosen vector on all images or a routing mechanism that does not see ground truth.

CPU checks, no downloads:

```bash
cd worldcuisines_food_steering
python -m unittest discover -s tests -v
python -m compileall -q .
```

Torch hook tests automatically skip if torch is absent; the GPU runner installs torch and runs them. They cover mask placement, zero strength, first-pass-only behavior, cleanup and equal-norm random controls. Dependency-free tests cover v1.1 answer indexing, grouped splits, strict parsing and incomplete-result rejection.

**Validation status at initial commit:** six dependency-free tests passed and Python/shell syntax checked locally; three torch-dependent hook tests skipped because torch is absent. The live official English `test_small` JSONL was fetched: 300 rows, 100 no-context rows, and a successful 50/50 food-disjoint partition. No pretrained model forward pass, full dataset/image download, or GPU end-to-end run has been completed in this environment. No accuracy improvement is claimed. First run the small setting above and inspect raw predictions before spending compute on the grid.

References: [WorldCuisines code](https://github.com/worldcuisines/worldcuisines), [WorldCuisines paper](https://arxiv.org/abs/2410.12705), [TextSteersVision code](https://github.com/Woodygan/TextSteersVision). Cite the original papers and dataset in course reports.
