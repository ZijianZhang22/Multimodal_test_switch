# CLIP Native + lambda_Y Rosetta3 (A-E experiment)

This is a self-contained **overlay** on the existing `visual_anchor/` folder, built from the CLIP code bundle supplied in the conversation and updated with native CLIP geometry and lambda_Y ablations. Copy the archive contents into `visual_anchor/`, keeping existing requirements, `summarize_real.py`, `test_core.py`, `test_real.py`, `test_rosetta3.py` intact.

## What is different from v1?

- `L`: frozen Chinese **monolingual** text2vec-base-chinese (no CLIP fine-tuning).
- `Y`: frozen **CLIP English text** tower.
- `Z`: frozen **CLIP image** tower from **the same checkpoint**, no trained bridge.
- `--joint-ei-pca`: E and I use **one pooled mean and the same PCA projection**. Never project them with separate PCA for native CLIP experiments.
- `--bridge identity`: image embeddings stay in the English-CLIP coordinates; no V_IE learning. An identity `V_IE` is saved for compatibility.
- Native CLIP eval uses the same Rosetta target mean for E and I; raw CLIP bridge diagnosis is reported separately, including a **cosine** result before PCA.
- `lambda_y` controls the English-unpaired matching term in refinement. When 0, its Hungarian matching **and the Y batch draw are skipped**.
- **Matched A-E ablations** use **supervised Procrustes initialization** from identical anchors. They do not run Rosetta's initial cross-lingual geometric alignment. Official Rosetta is still included as an additional baseline.

## One-command execution

```bash
cd /Multimodal_test_switch/visual_anchor
# First, only verify the CLIP native English-image diagnostic and cache features.
BRIDGE_ONLY=1 bash run_clip_native.sh
# Then experiments on 10,25,50,100,200 anchors with 3 seeds.
bash run_clip_native.sh
```

Tip: run a short smoke test first after the bridge diagnosis:

```bash
BUDGETS='10 25' SEEDS='0' METHODS='pairs_only image_only text_only noise_only caption_only joint rosetta' \
  CLUSTERS=8 RESTARTS=1 ITERATIONS=2 bash run_clip_native.sh
```

Change `DIM=32` if you want to match the earlier text-only pilot; default is `DIM=64` to limit possible 32-d CLIP information loss. The same DIM is shared across all methods in one experiment.

## Core comparisons (all use the same initial 10-200 parallel pairs)

| Method | Initialization | Y matching lambda_Y | Z matching lambda_Z | Z source |
|---|---|---:|---:|---|
| `pairs_only` (A) | supervised | 0 | 0 | none |
| `image_only` (B) | supervised | 0 | 1 | CLIP image bank |
| `text_only` (C) | supervised | 1 | 0 | none |
| `noise_only` (D) | supervised | 0 | 1 | Gaussian |
| `caption_only` (E) | supervised | 0 | 1 | same bank's CLIP English captions |
| `joint` | supervised | 1 | 1 | CLIP image bank |
| `rosetta` | **official Rosetta** | 1 | 0 | none |
| `visual` | official Rosetta | CLI lambda_Y (default 1) | CLI lambda_Z (default .25) | CLIP image bank |

A-E use strictly identical paired-only initialization, so `image_only - pairs_only`, `image_only - noise_only`, and `image_only - caption_only` have interpretable changes. `rosetta` uses its own geometric initialization and should not be treated as identically initialized with A-E.

**Scope qualification:** Even in A-E, the common joint PCA is fit on both unlabeled E and I feature pools; therefore `image_only` means **no English unpaired matching or English geometric initialization**, not absolutely no English unpaired data at any preprocessing stage. If a rigorous no-English experiment is needed, change the PCA fitting protocol accordingly and make it identical for all compared methods.

**Model scope:** A successful retrieval result gives representation-level image retrieval (and perhaps zero-shot classification after additional validation); this code does **not** fine-tune a VLM or give generative visual question answering.

**Interpretation:** Stronger CLIP can improve the bridge, but a large fraction of performance change might arise from a different pretrained encoder. Compare ablation deltas within the same CLIP run, not across CLIP and original MiniLM/DINOv2 as if only the bridge changed.

**Repository dependencies:** uses existing `requirements_real.txt`, `requirements_rosetta.txt` and `summarize_real.py`. Upstream official Rosetta is pinned to `fdfad84ea488f29702a61a455be71622c4ff5d49`. Data prep downloads Flickr8k-CN captions and original images if absent.

**Result files:** `real_work_clip_native/bridge_test.json`, `real_work_clip_native/features.npz`, `real_work_clip_native/results_clip_native_budget_*/report.json`, `real_work_clip_native/summary_clip_native.csv`.

**Validation status:** Offline syntax checks and NumPy unit tests pass. Official Rosetta, Hugging Face model downloads and end-to-end GPU retrieval need to be run in your connected environment; this local packaging environment does not have the Rosetta package or transformers installed.


## Correction (image eval, anchor centering, diagnostic dimensions)

- `--image-centering bank` (**default**): for every method, center held-out image features by the **same training-bank image mean** in the joint CLIP-PCA coordinates. This matches the Z pool centering used by the `visual` refinement method and avoids treating text/image modality means as identical. `--image-centering english` reproduces the previous evaluation behavior for an explicit sensitivity analysis. The plain original CLIP image/text cosine diagnostic is still reported separately and should not be confused with Rosetta's centered evaluation.
- `--anchor-centering pool` (**default for A-E/pairs-only**): center X with the **unpaired X training pool mean** and Y with the **unpaired English Y training pool mean**. This uses English statistics even for `image_only`, although no English pseudo-pair matching happens.
- `--anchor-centering source_pool`: X pool mean and **paired Y mean**, a stricter control when testing whether `image_only` avoids the English unpaired pool; use this mode for that assertion.
- `--anchor-centering paired`: original anchor-only means on both sides, for reproducing the earlier A-E results.
- **Official Rosetta** still uses its native pool-centering regardless of the A-E flag, and its bit-for-bit baseline path remains unchanged. All A-E methods share identical anchor choices, centering and initialization within any one run.
- `prepare_real.py` now has `--diagnostic-dim` and `run_clip_native.sh` passes `${DIM:-64}` to it. Diagnostic dimension is saved in `bridge_test.json`. Changing DIM requires recomputing diagnostics; the raw cached features do not need re-encoding.
- `test_rosetta3.py` restores the `english_pool`, `anchor_downweight`, and `anchor_scale` regression checks; `test_corrections.py` checks all centering modes.
- Low budget: a 64D orthogonal map estimated from 10/25 pairs remains rank-deficient. Compare e.g. `DIM=32` and `DIM=64`, but don't use the final test set to choose hyperparameters.

Example 32D stricter source-only English dependence:

```bash
DIM=32 ANCHOR_CENTERING=source_pool IMAGE_CENTERING=bank BUDGETS="10 25 50 100" bash run_clip_native.sh
```

Example 64D default pool-mean centering:

```bash
DIM=64 ANCHOR_CENTERING=pool IMAGE_CENTERING=bank BUDGETS="10 25 50 100" bash run_clip_native.sh
```

**Important**: change `WORK_DIR` when switching dimensions, or results directories will be overwritten. E.g. `WORK_DIR=real_work_clip32` and `WORK_DIR=real_work_clip64`. The two setups may reuse downloaded raw assets via symlinks.
