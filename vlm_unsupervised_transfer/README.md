# From Rosetta3 retrieval to frozen-VLM cross-lingual interventions

**Status:** Research implementation with CPU/unit tests, **not** a completed GPU experiment or an established accuracy gain. The code has not been run end-to-end on RunPod or checked against actual downloaded PaliGemma2 weights. Verify GPU logs before interpreting results. It is **not** a replication of the original Rosetta paper or of supervised cross-lingual steering.

This folder contains two clearly different research paths, both using the same frozen `google/paligemma2-3b-mix-448` and the same local multilingual CV-Bench evaluation:

- **A — Unsupervised VLM-native latent alignment:** Fit a low-rank, orthogonal map on **unpaired** Chinese/Swahili vs English **decoder hidden states**. Modify the *last text prompt token* at a chosen decoder layer during prefill, not the entire model. Test mean-shift, text-only OT (Hungarian pseudo-matching / Procrustes), text+visual OT, and an equal-third-pool **noise control**.
- **B — Geometry-guided dynamic steering:** Reuse the exact **A** maps, but apply selective corrections based on (1) English-bank nearest-neighbor confidence and/or (2) an input-specific visual-token-compatibility gate. A direct ungated correction is the common A baseline; `h + alpha(T(h)-h)` is **not** reported as a separate B method. Include a constant-half gate to test whether any gains merely reflect weaker intervention.

**Oracle paired activation patching** is a third, explicitly **diagnostic** step: same image and corresponding EN+non-EN CV-Bench questions; patch only the final prompt token hidden state from the EN run into the source-language run, with an interpolation strength/norm cap. It uses the EN paired question **at evaluation**, so it is *not* an unsupervised model, deployment strategy, or fair competitor. The oracle scans decoder layers 8, 13, 18 by default (override to include 4 and 22).

## What exactly is unsupervised?

Training draws **three disjoint image-ID sets from Flickr8K's official training split**:

| Source pool X | English pool Y | Visual pool Z |
|---|---|---|
| Chinese caption (Flickr8K-CN MT) + its image | English caption + a **different** image | **Other** Flickr8K images, no captions; neutral English prompt |
| Last text-token decoder states | Last text-token decoder states | Mean image-prefix **decoder states** at same layer |
| e.g. 256 | e.g. 256 | e.g. 256 |

- No bilingual caption pairs, shared source/English image IDs, pseudo-pair labels, VQA answers, or language-side labels enter the alignment objective. The **source-language identity** is known. Source captions are associated with their own images, so this is *cross-lingually unpaired fitting*, **not** fully self-supervised training from raw pixels.
- The source dataset itself was made using translated captions, and PaliGemma2 is a multilingual pretrained checkpoint. Do **not** claim the overall system has never seen cross-lingual supervision or translations; only this **new fitting stage** uses no bilingual pairs.
- Three pools are checked for disjoint image IDs **and SHA256 file bytes**. A separate preflight checks for exact-byte overlap with CV-Bench val/test. This does not detect perceptual duplicates, translations with repeated content, or pretrained-data overlap.
- `Z` is **not the CLIP image encoder** from old Rosetta3; it is an average of PaliGemma2's **image-prefix decoder tokens** after its own vision/language merging. Comparing this bank with textual representations is a hypothesis and can fail. The visual vs noise comparison tests whether gains follow from an arbitrary third bank.
- We **do not** reuse the old E5→CLIP 128-dimensional W on PaliGemma2's hidden dimensions. A fresh map is fitted for each PaliGemma2 decoder layer.

### Why a geometry effect may not improve reasoning

OT only matches distributions, not semantics. The learned map may match wrong concepts or move activations off the manifold the remaining decoder expects. Therefore we report actual **generated answer accuracy** (with paired confidence intervals), not retrieval R@K or cosine similarity alone. We also track invalid outputs and an image-shuffling sensitivity check; sensitivity is a diagnostic, not proof of correct visual grounding.

## Files

- `prepare_data.py`: Download/reuse pinned Flickr8K-CN caption sources; sample disjoint training image-ID pools, extract necessary JPEGs and audit corpus hashes. `LANGUAGE=sw` uses `SWAHILI_CSV` explicitly.
- `prepare_eval.py`: Build the exact 840 previously curated EN/ZH/SW CV-Bench prompt cohort from the original CV-Bench public `test` split, or copy a given previous `visual_eval.jsonl` with resolved images. It **does not** use CV-Bench for alignment fitting.
- `vlm.py`: HF PaliGemma2 `transformers==4.51.3` decoder hooks, last prompt-token extraction, image-prefix mean, greedy generation, and answer parser.
- `geometry.py`: Shared PCA, unpaired iterative Hungarian/Procrustes, source→target center shift, norm-clamped residual intervention and B gates.
- `experiment.py`: extraction cache, map fitting, oracle diagnostic, validation-only hyperparameter selection, fixed test evaluation. `A` direct has four controls; `B` gates are explored at A-selected layers.
- `stress_visual.py`: validation-only image-swap diagnostic for baseline vs A vs B.
- `summarize.py`: paired bootstrap **by image/question ID** across fixed model predictions.
- `tests/`: CPU tests with synthetic embeddings, no PaliGemma2 download.

## Run direction A only first (no B)

Use `ARM=a` to run **only A's direct alignment maps** (`mean_shift`,
`text_ot`, `text_visual_ot`, `text_noise_ot`) plus the optional paired
Oracle diagnostic. B's confidence/visual/constant gate search and B-specific tests are skipped.
The A-only shuffled-image diagnostic can still run with `RUN_STRESS=1`. Choose a fresh `OUT`
when switching from `ARM=all` to `ARM=a` because the settings and selection
protocol differ. `RUN_TEST=0` keeps the first pass validation-only; after
inspecting validation, rerun with `RUN_TEST=1` and the *exact same configuration*
to evaluate the locked holdout once.

```bash
cd /workspace/Multimodal_test_switch
git fetch origin && git switch experiment/vlm-unsupervised-a-b && git pull
ARM=a LANGUAGE=zh LAYERS=8,13,18 \
  N_SOURCE=256 N_ENGLISH=256 N_VISUAL=256 \
  RUN_ORACLE=1 RUN_STRESS=0 RUN_TEST=0 \
  OUT=/workspace/vlm_A_zh_seed42 \
  bash vlm_unsupervised_transfer/run_runpod.sh
```

To evaluate the fixed test after validating the run, use the same command
with `RUN_TEST=1` (and `INSTALL=0` if dependencies are installed). Re-running
cached extraction and fitting is idempotent but still takes resources to load
activations; do not modify seed, image pools, layer grid, or strength grid.

## GPU run in RunPod (one command)

**Hardware:** An A100 80GB is recommended for all-layer repeated 448px prefill; smaller cards might work but OOM has not been checked. Model may require HF license approval and `hf auth login`. Existing repo branch includes the previously built `paligemma2_cross_lingual_steering`, `visual_anchor`, `vlm_layer_alignment` folders; none is changed by this package.

```bash
cd /workspace/Multimodal_test_switch
git fetch origin
git switch experiment/vlm-unsupervised-a-b
git pull origin experiment/vlm-unsupervised-a-b

# If the checkpoint is gated, accept its license and run: hf auth login
LANGUAGE=zh LAYERS=8,13,18 N_SOURCE=256 N_ENGLISH=256 N_VISUAL=256 \
  RUN_ORACLE=1 RUN_STRESS=1 RUN_TEST=1 \
  bash vlm_unsupervised_transfer/run_runpod.sh
```

The launcher creates a separate `vlm_unsupervised_transfer/.venv`, installs requirements (`INSTALL=0` to skip later), downloads missing Flickr8K training images and official CV-Bench with Hugging Face Datasets, extracts PaliGemma2 hidden states, fits all maps, runs validation oracle and A/B, runs **one** selected test evaluation per named arm, then generates `comparison.csv`. Note full experimental runs involve many multimodal forward passes; use `tmux` or a persistent pod shell.

If previous user-generated curated CV-Bench `visual_eval.jsonl` already exists and you trust its translation/split protocol, you can reuse it **without modifying it**:

```bash
CVBENCH_SOURCE=/workspace/Multimodal_test_switch/paligemma2_cross_lingual_steering/outputs/paligemma2_cvbench_full/visual_eval.jsonl \
INSTALL=0 bash vlm_unsupervised_transfer/run_runpod.sh
```

Other options: `LANGUAGE=sw SWAHILI_CSV=/path/to/SwaFlickr8K.csv` (only if the file exists), `SEED=43`, `LAYERS=4,8,13,18,22`, `ITERATIONS=30`, `BATCH=128`, `VISUAL_WEIGHT=0.25`, `NORM_CAP=0.15`, `STRENGTHS=0.15,0.3,0.6`, `RUN_STRESS=0`, `RUN_ORACLE=0`, `RUN_TEST=0`. **Use an independent `OUT` for each seed/protocol**, do not merge predictions from different configurations. If Flickr8K image archives already exist, the code reuses them from `visual_anchor/real_work_clip_native/raw` or `IMAGE_ARCHIVE`.

To run CPU unit tests without downloads:

```bash
cd vlm_unsupervised_transfer
python -m unittest discover -s tests -v
```

### Stepwise reproducible stages

With an activated virtual environment and existing `outputs/...` data:

```bash
python experiment.py --stage extract --pools outputs/zh_seed42/flickr --val outputs/zh_seed42/cvbench/val.jsonl --test outputs/zh_seed42/cvbench/test.jsonl --out outputs/zh_seed42/experiment
python experiment.py --stage fit --pools outputs/zh_seed42/flickr --val outputs/zh_seed42/cvbench/val.jsonl --test outputs/zh_seed42/cvbench/test.jsonl --out outputs/zh_seed42/experiment
python experiment.py --stage oracle --pools outputs/zh_seed42/flickr --val outputs/zh_seed42/cvbench/val.jsonl --test outputs/zh_seed42/cvbench/test.jsonl --out outputs/zh_seed42/experiment
python experiment.py --stage validate --pools outputs/zh_seed42/flickr --val outputs/zh_seed42/cvbench/val.jsonl --test outputs/zh_seed42/cvbench/test.jsonl --out outputs/zh_seed42/experiment
python experiment.py --stage test --pools outputs/zh_seed42/flickr --val outputs/zh_seed42/cvbench/val.jsonl --test outputs/zh_seed42/cvbench/test.jsonl --out outputs/zh_seed42/experiment
```

Pass the same layer grid, map dim, strengths, seed and model to **every stage** if changing defaults.

## Expected outputs

```text
outputs/zh_seed42/
  flickr/{source,english,visual}.jsonl
  flickr/manifest.json
  cvbench/{val,test}.jsonl
  experiment/
    settings.json
    features/{source,english,visual}/00000.npz
    maps/layer13_{mean_shift,text_ot,text_visual_ot,text_noise_ot}.npz
    oracle_val.json                       # paired diagnostic ONLY
    selection_val.json                    # A layer+alpha; B gate+alpha selected on val
    predictions/{val,test}_*.csv
    test_results.json
    comparison.csv                        # paired bootstrap delta vs unchanged ZH baseline
    visual_sensitivity_val.json           # deliberately swapped images, diagnostic only
```

## Non-negotiable interpretation notes

1. **No claim of an unsupervised improvement until real GPU evaluation confirms it.** Synthetic numerical tests only check implementation sanity.
2. **Patching is not independent-language evidence.** Oracle English states require paired images/questions, are reported separately, and are used for *diagnosis only*. Selection of layers/hyperparameters on CV-Bench validation is supervised model selection, even though OT mapping *fitting* used no cross-language labels. Therefore call the method **unpaired fitting + labeled downstream validation**, not fully unsupervised end-to-end model selection.
3. **PaliGemma2 448px is expensive.** Scanning extra layers and alpha values multiplies autoregressive decoding cost. Validation first searches A's direct maps; B gates are then tested only at A's selected map layer, so we do not create a combinatorial all-method grid.
4. **Proper counterfactual controls.** A compares `mean_shift`, `text_ot`, `text_visual_ot`, and noise-bank OT. B compares the same A maps under geometry confidence, visual gate, joint gate and constant-half gate. English reference is never directly patched in deployed A/B. `direct` is one baseline shared by both.
5. **Limitations:** The 840 CV-Bench prompts are GPT-authored EN/ZH/SW variants that have not been verified by independent bilingual raters, constructed from a publicly available benchmark **test** split. This is a *local experimental benchmark*, not an official independent test. Older project evaluations may already have inspected these IDs. Repeated seeds and an external dataset are needed for credible publication.
6. A gain in accuracy (even if real) is **not proof of English-like internal reasoning**, nor of zero translation pretraining. Report task-wise results and paired repairs/harm, not anthropomorphic internal-language claims.

## Related background (not claimed as replications)

- Dumas et al., *Separating Tongue from Thought*, ACL 2025: https://aclanthology.org/2025.acl-long.1536/ . Text-only activation patching supports testing language-specific vs conceptual layers, but does not establish optimal PaliGemma2 layers.
- Hugging Face transformers v4.51.3 PaliGemma documentation: https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/paligemma
- MUSE (Conneau et al.), *Word Translation Without Parallel Data*: https://arxiv.org/abs/1710.04087 . Inspired unpaired distribution-matching, not byte-identical implementation.