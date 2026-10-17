# Cross-lingual Textual Steering (Qwen3-VL)

Standalone experiment based on the **Mean Shift** idea from
[TextSteersVision](https://github.com/Woodygan/TextSteersVision):
extract concept-minus-control hidden-state directions from English and Chinese
text-only examples, then steer the **same** Qwen3-VL on paired visual questions.

This is an **adaptation, not an exact reproduction** of the original paper:
vectors are extracted from text-only passes through the VLM's own language
decoder to avoid misaligned hidden spaces.

## One command on RunPod

From the root of `Multimodal_test_switch`:

```bash
bash cross_lingual_steering/run.sh
```

This installs Python dependencies, creates a small synthetic counting/spatial
pilot, extracts bilingual vectors, evaluates EN→EN, ZH→EN, EN→ZH, ZH→ZH
plus a no-steering baseline and a random-vector control, and writes CSVs.

Defaults: Qwen/Qwen3-VL-2B-Instruct, decoder layers 6 and 12 (zero-based),
unit-normalized strength 5, first 8 validation images **per task**,
both image and text positions. Requires CUDA and access to the model weights.

```bash
# Example: more thorough validation sweep
LAYERS=6,12,18 ALPHAS=3,8 LIMIT=0 bash cross_lingual_steering/run.sh

# Recompute vectors after changing model / concept corpus
FORCE_EXTRACT=1 bash cross_lingual_steering/run.sh

# Recreate pilot images/corpus AND re-extract vectors
RECREATE_DATA=1 bash cross_lingual_steering/run.sh

# Held-out synthetic test. Select layer/strength on val first.
SPLIT=test LAYERS=12 ALPHAS=5 LIMIT=0 bash cross_lingual_steering/run.sh
```

## Outputs

In `cross_lingual_steering/outputs/pilot/`:

- `vectors.pt` — concept/layer → EN, ZH, EN_A, EN_B, ZH_A, ZH_B
- `results_val.csv` / `results_test.csv` — prediction for every image, prompt
  language, vector language, layer, alpha and baseline/control condition
- `analysis_val/vector_geometry.csv` — EN–ZH and within-language split cosine
- `analysis_val/paired_accuracy.csv` — paired accuracy improvements (percentage
  points) with 95% bootstrap intervals

The toy images are a **pipeline sanity check** and not a robust claim about
real-world visual comprehension. A meaningful paper experiment must replace
them with translated, held-out CV-Bench (or comparable) questions and use a
validation/test split. All hyperparameter selection must happen on validation.

## Architecture

- `make_pilot_data.py`: bilingual semantically paired concept corpus and visual PIL images.
- `model_utils.py`: tokenizer offset targets, Qwen3-VL loading, selected-token
  decoder output hooks, generated-answer canonicalization.
- `run_experiment.py`: Mean Shift vector extraction, 2×2 bilingual intervention,
  baseline and random control generation.
- `analyze_results.py`: same-language split stability, cross-language cosine,
  paired accuracy and bootstrap CIs.
- `run.sh`: one-command GPU pipeline.

**Important restrictions:** never compare hidden vectors from different
un-aligned model checkpoints; steering is applied to selected prefill token
states without training model weights. The hook normalizes steering directions
before applying alpha. English/Chinese concept prompts can still differ in
tokenization or lexical statistics, so an observed performance difference
cannot automatically be called a shared-semantic mechanism.

## Input format

`corpus/counting_en.jsonl`, `corpus/counting_zh.jsonl` etc:

```json
{"id":0,"sentence":"There are three red stars.","target":"three"}
```

`visual_eval.jsonl`:

```json
{"id":"q1","split":"test","task":"counting","image":"images/q1.png","answer":"3","questions":{"en":"How many red circles?","zh":"有几个红色圆形？"}}
```

Image paths are relative to the data root. Spatial answers are canonical
`left`, `right`, `above`, `below`.

## Checks and limitations

```bash
cd cross_lingual_steering
python -m pytest -q tests/
python -m py_compile make_pilot_data.py model_utils.py run_experiment.py analyze_results.py
```

End-to-end model loading and GPU evaluation have **not** been verified here;
check RunPod console logs and sampled per-question predictions before relying
on aggregate findings.

Upstream: Woodygan/TextSteersVision. No upstream source files copied verbatim.
