# CLIP Rosetta3: Strong-H1 Controls, Paired Statistics, and Swahili

This branch extends the **128D, shared-projection CLIP experiment** without changing the frozen English-CLIP text or image encoder.

## Goals

1. **Same total unsupervised coefficient / matched third-pool size.** The stronger hypothesis is that image geometry contributes beyond equally weighted extra English text or Gaussian controls.
2. **Paired evaluation.** Ten training seeds by default; store each query's held-out `R@1` hit and image ID. Report seed- and image-cluster-paired bootstrap CIs, plus two-sided seed-level sign-flip p-values.
3. **Swahili validation.** Reuse CLIP English/image features and the original Flickr8k partition, replacing only the L-language captions and encoder using SwaFlickr8k.

## A–I matched-initialization arms

| Method | `lambda_y` | `lambda_z` | Additional Z pool | Supervised init |
|---|---:|---:|---|---|
| `pairs_only` | 0 | 0 | none | same anchors |
| `image_only` | 0 | 1 | image bank | same anchors |
| `noise_only` | 0 | 1 | Gaussian | same anchors |
| `caption_only` | 0 | 1 | captions of the exact same image bank | same anchors |
| `text_only` | 1 | 0 | none | same anchors |
| **`joint`** | 1 | 1 | image bank | same anchors |
| **`joint_caption`** | 1 | 1 | same-bank English captions | same anchors |
| **`joint_noise`** | 1 | 1 | Gaussian | same anchors |
| **`joint_extra_english`** | 1 | 1 | extra English captions of other images | same anchors |

`joint_caption` is an especially informative **same-content, same-size** control; `joint_extra_english` tests simply adding other English data. All added Z terms use the same nominal coefficient and number of Z rows, with deterministic matched selection if some Swahili IDs are unavailable. Hungarian matches and cross-covariance scales still vary with the content and geometry; coefficient equality is not identical gradient/effective signal strength.

### Run Chinese controls

First make sure your original CLIP features exist:

```bash
cd /Multimodal_test_switch
# Get experiment branch as instructed in the chat, then:
cd visual_anchor
# If necessary, populate real_work_clip_native/features.npz:
BRIDGE_ONLY=1 bash run_clip_native.sh
# Optional fast smoke run first:
BUDGETS='25' SEEDS='0 1' ITERATIONS=2 BOOTSTRAP_REPS=100 bash run_clip_followup.sh
# Main experiment (10 seeds, 10/25/50 pairs, 9 methods):
bash run_clip_followup.sh
```

Default environment: `WORK_DIR=real_work_clip_native`, `DIM=128`, `PCA_MODE=shared`, `ANCHOR_CENTERING=source_pool`, `IMAGE_CENTERING=bank`, `BUDGETS='10 25 50'`, `SEEDS='0 1 2 3 4 5 6 7 8 9'`, `METHODS` = all nine rows above, `ITERATIONS=20`, `BATCH=256`, `ABLATION_LAMBDA_Z=1.0`, no whitening. No new CLIP feature extraction if the cached feature file already exists. The output tag defaults to `strong_h1` and will not overwrite the previous three-seed runs.

Outputs:

```
real_work_clip_native/
  results_strong_h1_shared_d128_source_pool_bank_budget_10/
    report.json
    hits_joint_0.npz     # hits_L_E, ids_L_E, hits_L_I, ids_L_I, ...
    ...
  results_strong_h1_shared_d128_source_pool_bank_budget_25/
  results_strong_h1_shared_d128_source_pool_bank_budget_50/
  summary_strong_h1_shared_d128_source_pool_bank.csv
  paired_strong_h1_shared_d128_source_pool_bank.csv
  paired_strong_h1_shared_d128_source_pool_bank.json
```

Focus on `image_only - pairs_only`, `image_only - noise_only`, `image_only - caption_only`, `joint - text_only`, `joint - joint_caption`, `joint - joint_noise`, and `joint - joint_extra_english`. Report both **L→E** and **L→I**, with E→L / I→L supplementary.

### Paired inference

`analyze_ablation.py` computes per-seed image-ID cluster means from binary `R@1` hits, then applies a **two-way crossed paired bootstrap** by resampling training seeds and test image IDs independently, *using the same draws for both methods*. The 95% percentile CI is reported in percentage points, alongside a two-sided sign-flip p-value on the paired seed means. The latter has low power with few seeds. This is exploratory statistical inference on one dataset; multiple contrasts/directions are tested, so report the full table and consider multiplicity in formal claims. Do **not** treat 1,000 correlated test queries as 1,000 independent training replicates.

### Run Swahili with frozen CLIP E/I

The repo includes `prepare_swahili.py` and `test_swahili.py` from the prior Swahili experiment. These files align English/Swahili captions by Flickr8k image ID and normalized English sentence; missing/unavailable IDs are audited, and train/test ID overlap is rejected. The script may refill missing supervised anchors from the **training bridge split only**; its manifest records any such changes. A wrong caption is not silently substituted by a different image caption. The dataset is [Alfaxad/SwaFlickr8k](https://huggingface.co/datasets/Alfaxad/SwaFlickr8k).

```bash
# Run after the Chinese CLIP work directory has features.npz and the original raw captions.
cd /Multimodal_test_switch/visual_anchor
bash run_swahili_clip.sh
```

By default: download Swahili dataset CSV from Hugging Face; load `intfloat/multilingual-e5-base` for Swahili, and **reuse** the cached frozen CLIP English/image representations, held-out image IDs and data groups. If the dataset cannot be downloaded automatically, use an audited local copy with `SW_CSV=/path/to/captions_sw.csv bash run_swahili_clip.sh`. To force new source representations set `SW_REBUILD=1`.

Swahili outputs use `sw_strong_h1_*` names in the same work directory; `features_sw_clip.npz` and `features_sw_clip.manifest.json` record its preparation. Missing IDs can reduce the eligible test set and change the exact training splits; **do not directly compare Chinese/Swahili absolute R@1 values as if they use identical evaluated examples**. Evaluate each language's *within-language method differences*.

**Encoder caveat:** Chinese L is `text2vec-base-chinese`, while Swahili L is multilingual E5. This is a cross-language replication of the effect, **not** a controlled same-architecture language comparison. A proper controlled language study should re-encode both L languages with the same frozen multilingual model and keep identical eligible image IDs.

## What this does *not* prove

- It is **retrieval/representation alignment**, not generative VQA or Qwen2.5-VL reasoning.
- CLIP image-text training and jointly fitted E/I projection still supply English/text information even for `image_only`. `source_pool` only excludes the English **unpaired OT matching and centering mean** in those arms.
- Candidate metric R@1 can remain low even with meaningful relative improvement.
- These controls match pool size and nominal coefficients, but do not guarantee equal effective gradient, objective magnitude, or data semantics.
- Current code has CPU-only synthetic integration tests; it has **not** run an end-to-end Swahili/CLIP GPU experiment in the author's environment.