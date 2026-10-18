# MRRE-inspired language steering × EN concept steering

This new isolated Parity Pipeline adds paired EN-minus-(ZH/SW) final-prompt-token vectors,
norm-preserving language steering on prefill and decode, the original text-derived
Mean Shift concept vectors, and first-option-letter-token margin analysis.

**Not a reproduction of MRRE reasoning-trace steering.** It uses final prompt tokens.
Tune split is divided into alternating extraction/selection halves. Test is untouched
until layer/alpha selection is frozen. Random directions are matched to language-vector
norms. Combo uses default paper EN concept parameters and ZH/SW prompts.

RunPod quick pilot (requires prepared parity_cvbench dataset and vector cache):

```bash
cd /workspace/Multimodal_test_switch
git pull origin parity-pipeline-20261018
RUN=combo_smoke LIMIT=5 LAYERS=5,8,12 ALPHAS=0.25,0.5 \
  bash paligemma2_cross_lingual_steering/parity/run_combo.sh
```

Results: `outputs/parity_cvbench/runs/combo_smoke/analysis/SUMMARY.md`.
For a full run use a fresh DATA directory or explicitly recompute language
vectors if the extraction sample IDs differ; cache validation fails closed.
Use a new RUN when changing hyperparameter grids.

Caution: the first-token bare A-F margin may not represent multi-token
answer probabilities, especially for outputs beginning with parentheses.
GPU execution remains to be validated on RunPod.
