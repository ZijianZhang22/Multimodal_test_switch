# Paper-native Gemma3 cross-lingual steering

This folder is a **new, isolated implementation** based directly on
[Woodygan/TextSteersVision](https://github.com/Woodygan/TextSteersVision),
pinned to `ce1c0f0cffed9b859901ce347f9d2b869a512178`.

**Model:** `google/gemma-3-4b-it` (same Gemma3 checkpoint supported by the
original authors). This is different from the previous Qwen3-VL pilot in
`cross_lingual_steering/`. It does not load or reuse Qwen vectors.

The inference implementation uses the author's **unmodified**
`models/gemma3_meanshift_model.py`, specifically
`Gemma3MeanShiftManipulator` and `MeanShiftConfig`. The original repository
is fetched at runtime into `.upstream/`; no large upstream source is bundled.
The original Gemma3 wrapper's hook, mask handling and generation logic remain
unchanged. The dependencies pin `transformers==4.51.3`, the original version.

**Mean Shift extraction:** same pooled *highlighted-token hidden state minus
other-token hidden state* rule as the upstream
`get_features/get_features_mean_shift.py`, excluding BOS. To make Chinese
target spans reliable, this experiment uses tokenizer **character offsets**
instead of assuming whitespace prefix token matching. It obtains text-only
activations from the **Gemma3 VLM's own language decoder** to keep vectors and
VLM in the same activation space. This is a deliberate adaptation of the
author's extraction script, **not byte-for-byte original extraction**.

## RunPod one-command background run

Run from repository root (GPU required, and Gemma3 may require HF license access):

```bash
git pull
bash gemma3_cross_lingual_steering/run_background.sh
```

To check the log:

```bash
cd gemma3_cross_lingual_steering
tail -f "$(cat logs/latest_log_path.txt)"
```

Or run in the terminal: `bash gemma3_cross_lingual_steering/run.sh`.

Small default smoke check: 20 original English taxonomy examples per concept
with 20 paired Chinese translations, 4 images per concept, both prompt languages,
EN/ZH directions, layers 8/16, normalized strength 5, original `both` mask.
Two tasks: counting and spatial relation. Images are synthetically generated
and intended to verify execution, **not** a paper-quality visual benchmark.

```bash
# More thorough validation (no extra model training)
LAYERS=8,16 ALPHAS=5,15 LIMIT=0 bash gemma3_cross_lingual_steering/run_background.sh

# Force a different layer set to be EXTRACTED and then evaluated
FORCE_EXTRACT=1 LAYERS=5,10,15,20 ALPHAS=5 bash gemma3_cross_lingual_steering/run_background.sh

# After choosing a setting on VAL, only then evaluate on TEST.
SPLIT=test LIMIT=0 LAYERS=8,16 ALPHAS=5 bash gemma3_cross_lingual_steering/run_background.sh

# Recreate samples/images and re-extract
REBUILD_DATA=1 bash gemma3_cross_lingual_steering/run_background.sh
```

Run foreground by replacing `run_background.sh` with `run.sh`.
If using a notebook image without Git, install Git through the system package
manager before running. Use a GPU with enough memory for a BF16 4B model.

## Experiment

For a matched image and pair of semantically parallel English/Chinese questions,
run the author's Gemma3 manipulator for:

| | EN prompt | ZH prompt |
| --- | --- | --- |
| EN vector | EN→EN | EN→ZH |
| ZH vector | ZH→EN | ZH→ZH |

Every image/question-language also has a non-steered baseline, produced by the
same upstream `generate_with_manipulation` method.

We instantiate **separate manipulators sharing the same GPU model**, one per
concept/layer/language/alpha. This avoids the original source's feature caching
behavior in `set_config`, which otherwise may silently use a previously loaded
language vector after switching `feature_paths`.

Layer numbers are zero-based. Vectors are unit normalized through the original
`MeanShiftConfig(normalize_features=True)`; strength is applied by the original
implementation. Use the same layer, strength and mask across EN/ZH comparisons.

## Outputs

Stored only in `outputs/gemma3/`:

- `corpus/counting_{en,zh}.jsonl` and `spatial_{en,zh}.jsonl`
- `vectors/{counting,spatial}/{en,zh}/layer_N_feature.pt`
- `vectors/manifest.json` (model, upstream commit, Transformers version, data hash, layers)
- `predictions_val.csv` / `predictions_test.csv`: original output per image and condition
- `analysis_val/vector_geometry.csv`: EN/ZH cosine and cross-*concept* cosines
- `analysis_val/paired_accuracy.csv`: matched differences with paired bootstrap CIs

The manifest prevents accidental cross-model or cross-corpus vector reuse. If you
change the source corpus, rerun with `FORCE_EXTRACT=1`. If you request new layers,
also rerun extraction.

## Tests

```bash
cd gemma3_cross_lingual_steering
python -m pytest -q tests/
python -m py_compile make_data.py extract.py evaluate.py analysis.py
bash -n run.sh run_background.sh
```

**Caveats:** cannot conclude language-independent semantics from high EN/ZH
cosine alone; check counting-vs-spatial cosine for shared nuisance directions.
The test images are a smoke dataset, not CVBench. The author's wrapper is used
unchanged; its support is tied to the pinned Transformers version, and end-to-end
GPU success still needs to be confirmed on RunPod. If the original wrapper fails,
do not silently substitute a different hook; report that incompatibility.

Upstream source credit: Woodygan/TextSteersVision, revision
`ce1c0f0cffed9b859901ce347f9d2b869a512178`.
