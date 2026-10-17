# Gemma3 multilingual steering on **real CV-Bench** (EN / ZH / SW)

Isolated follow-up to `../` toy validation. Based on the **unmodified** author's
`Woodygan/TextSteersVision/models/gemma3_meanshift_model.py` at upstream commit
`ce1c0f0cffed9b859901ce347f9d2b869a512178` (Transformers 4.51.3).

## What is being evaluated?

Two distinct sets of four — do not confuse them:

1. **CV-Bench's 4 real-world subtasks / reported accuracies:** **Count,
   Relation, Depth, Distance**. Paper-native steering main tables tune on
   **Count & Relation**; our Depth/Distance are explicit extensions. The official
   dataset, `nyu-visionx/CV-Bench`, has 2,638 image-QA examples in a **single
   original test split**; we deterministically allocate local VAL/TEST subsets,
   and do not imply these are official CV-Bench split names.
2. **4 textual feature categories:** Counting, Spatial Relationship, Attribute,
   Entity. Each EN concept uses the author's 20 original anchors. ZH and SW use
   20 paired human-drafted *pilot* anchors each, with explicit target spans.
   SW is an additional relatively low-resource African language; translations
   **are NOT professionally validated** and need independent language review.

Prompt language = EN, ZH, SW; steering-vector language = EN, ZH, SW.
Same image + same answer letter across languages.

CV-Bench questions/options are automatically translated by
`facebook/nllb-200-distilled-600M` using the local GPU before Gemma is loaded.
Output choices **keep original A-E labels**; model answer is scored as a letter,
never by keyword matching. Machine translations are **not** equivalent to a
human-validated multilingual CV-Bench. Always inspect
`translation_audit.csv` before claiming accuracy differences are due to
language itself; errors in low-resource translations are a confounder.

## One-command RunPod VAL run (default larger real-image setting)

From repository root:

```bash
cd /Multimodal_test_switch
git pull
N_VAL=60 N_TEST=150 LAYERS=8,16 ALPHAS=0.1 \
  bash gemma3_cross_lingual_steering/cvbench_extension/run_background.sh
```

The command prepares **60 VAL + 150 held-out TEST examples per task** (840
unique real images, 2,520 prompts across three languages), extracts all
12 concept/language vectors at the selected layers, and **evaluates VAL
only**. No test-set tuning. This can take significant disk, download time and
GPU computation. It does *not* run all 2,638 examples automatically.

Check logs:

```bash
cd /Multimodal_test_switch/gemma3_cross_lingual_steering/cvbench_extension
tail -f "$(cat logs/latest_log_path.txt)"
```

**Cheaper functional test before the full run:**

```bash
N_VAL=60 N_TEST=150 LIMIT=3 LAYERS=8 ALPHAS=0.1 \
 bash gemma3_cross_lingual_steering/cvbench_extension/run_background.sh
```

Here `LIMIT=3` evaluates only 3 images **per task** while preparing full
translated data (which is downloaded/prepared only once). To later run **all**
VAL samples, remove the previous `predictions_val.csv` and its settings file
(or use a separate `DATA` directory), then run `LIMIT=0`; the script
intentionally refuses silently mixing incompatible configurations.

### After validation, freeze one hyperparameter configuration for TEST

```bash
SPLIT=test INSTALL=0 LAYERS=8 ALPHAS=0.1 LIMIT=0 \
 bash gemma3_cross_lingual_steering/cvbench_extension/run_background.sh
```

Use exactly the selected layer, alpha, mask and normalization setting.

## Config options

- `N_VAL`, `N_TEST`, `SEED`: immutable after preparation; use a fresh `DATA`
  directory for a new split. If `N_TEST=0`, use all remaining official samples.
- `LAYERS=8,16`, `ALPHAS=0.1`, `MASK=both`.
- `NORMALIZED=0` (paper-like *unnormalized* Mean Shift). Optional
  `NORMALIZED=1 ALPHAS=5` for comparability with earlier toy run, but
  these are distinct interventions; **do not directly equate alphas**.
- `ALL_FEATURES=1`: cross-concept interventions for **every** CV-Bench task
  (roughly 4x GPU cost compared with task-aligned); otherwise Count uses
  Counting and Relation/Depth/Distance use Spatial. Attribute and Entity
  vectors are still extracted and included in geometry with default setting.
- `LIMIT=0`: all prepared examples in chosen split, not all original 2,638.
- `FORCE_EXTRACT=1`: repeat vector extraction (needed when requesting layers
  absent from the manifest).

In `ALL_FEATURES=1` mode, `task_accuracy.csv` distinguishes all concepts,
while the default task-aligned `four_task_macro_accuracy.csv` is not produced
because mixing concepts would be invalid.

## Outputs

Under `cvbench_extension/outputs/gemma3_cvbench/`:

- `dataset_manifest.json`: provenance, sampling, languages, translation status.
- `visual_eval.jsonl`, `images/`, `translation_audit.csv`.
- `corpus/{counting,spatial,attribute,entity}_{en,zh,sw}.jsonl`.
- `vectors/<concept>/<language>/layer_<layer>_feature.pt` and manifest.
- `predictions_{val,test}.csv`: row-level baseline and intervention answers.
- `analysis_val/` or `analysis_test/`:
  `baseline_accuracy.csv`, `task_accuracy.csv`, `vector_geometry.csv`,
  and (for task-matched configs) `four_task_macro_accuracy.csv`.

## Limitations

- **Only Mean Shift** intervention, not paper's SAE and linear probe methods.
- Annotation translations are **machine-generated and not verified**; pilot
  SW anchor sentences are human-drafted, not proofread by native speakers.
- Our Gemma3 setup is an extension, not a numerical replication of the
  paper's main PaliGemma2/Idefics3 results.
- The benchmark's 2,638 original examples are publicly labeled. Our fixed
  locally held-out test prevents our own grid-search leakage but is not a
  newly created blind official hidden test.
- English/Chinese/Swahili performance gap may reflect translation quality,
  model training exposure, or answer-format adherence, not just concept vectors.
