# CLIP Rosetta3 — corrected centering release

Overlay this folder's files onto an existing `visual_anchor/` working checkout (e.g. your CLIP experimental branch). It does NOT include original `test_core.py` and `test_real.py` which already exist on the base repository branch; retain these.

Main changes:
1. Native CLIP image evaluation now uses a **fixed training-image-bank mean** for ALL methods by default. Legacy English-mean evaluation available by `IMAGE_CENTERING=english`.
2. Anchor-only methods default to **training pool means** for both X and Y. `ANCHOR_CENTERING=paired` restores old low-budget behavior, while `ANCHOR_CENTERING=source_pool` uses X pool mean and paired-Y mean (no English unpaired pool dependence during refinement or supervised centering, except for upstream PCA).
3. `prepare_real.py --diagnostic-dim` matches experiment DIM (default 64 in shell).
4. Restores three previous baseline tests and adds new centering regressions.

RunPod, from repository root after overlay:

```bash
cd visual_anchor
DIM=32 WORK_DIR=real_work_clip32 BUDGETS="10 25 50 100" ANCHOR_CENTERING=source_pool bash run_clip_native.sh
DIM=64 WORK_DIR=real_work_clip64 BUDGETS="10 25 50 100" ANCHOR_CENTERING=pool bash run_clip_native.sh
```

`--bridge identity` and `--joint-ei-pca` remain active in `run_clip_native.sh`.

Verification here: Python compileall, bash -n, 8 NumPy tests passed / 2 transformer-dependent skipped; 7 focused algorithm tests passed with a simulated upstream Rosetta API only. True upstream Rosetta + CLIP end-to-end GPU tests must run in your environment.

**Do not mix results across anchor/image-centering protocols**: report all centering settings with each result. Original CLIP cosine retrieval remains a separate diagnostic.
