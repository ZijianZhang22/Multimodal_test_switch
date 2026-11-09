# Controlled Chinese vs Swahili: SAME multilingual E5 encoder, SAME CLIP E/I

This is the follow-up to the 10-seed CLIP/Rosetta3 experiments. Its purpose is to remove the source-**encoder** and image-ID confounds from the Chinese/Swahili comparison.

### Exact design

- **Chinese L**: `intfloat/multilingual-e5-base`, *same pinned revision and exact query-prefix/mean-pooling/max-length protocol* as Swahili.
- **Swahili L**: previously encoded `features_sw_clip.npz` with `intfloat/multilingual-e5-base`.
- **English E and image I**: copied *byte-for-byte* from the Swahili `.npz`, so CLIP towers and PCA training inputs match exactly.
- **IDs**: *all* `*_ids`, including paired anchor rows, test images, image bank and unpaired pools are copied from the Swahili `.npz` in the original row order. This includes the Swahili coverage filtering and any bridge→anchor replacements.
- **Chinese captions**: original Flickr8K-CN Google translation of EN caption #0 for `l_pair` **and** `l_pool`; human translation of English caption #0 for L↔E held-out tests; Google translation of English caption #1 for L↔I tests (explicitly audited #0 fallback if missing). Swahili uses its corresponding SwaFlickr8k translations.
- **Training**: same `DIM=128`, `PCA_MODE=shared`, `--bridge identity`, `ANCHOR_CENTERING=source_pool`, `IMAGE_CENTERING=bank`, budgets `10 25 50`, seeds `0..9`, 9 methods, 20 iterations, batch 256.
- **Paired language comparison**: matched seed and image-ID crossed bootstrap on *difference of within-language method contrasts*: `(Joint Image − Joint Caption)_ZH − (Joint Image − Joint Caption)_SW` and six other contrasts. This isolates the effect of the source language more tightly than comparing raw R@1 percentages across different test sets.

### RunPod: on existing completed Swahili machine

```bash
cd /workspace/Multimodal_test_switch
# Switch to the branch advertised in ChatGPT (after git fetch).
cd visual_anchor

# Test only 25 pairs, 2 seeds and 2 iterations; unique RUN_LABEL avoids overwriting main results.
BUDGETS='25' SEEDS='0 1' ITERATIONS=2 BOOTSTRAP_REPS=100 RUN_LABEL=smoke_zh_e5 \
    bash run_chinese_e5_clip.sh

# Main 10-seed run. RESUME=1 can safely reuse saved model files after interruption.
RESUME=1 bash run_chinese_e5_clip.sh
```

Prerequisite: complete `bash run_swahili_clip.sh` once, such that both `real_work_clip_native/features_sw_clip.npz` and `real_work_clip_native/features_sw_clip.manifest.json` exist. The same `real_work_clip_native/raw/` text files and `.venv-real` must be present. **No CLIP re-encoding, image downloads or Swahili re-encoding is required.** Only Chinese E5 vectors are newly encoded. `--verify-only` confirms future cache reuse exactly matches the Swahili source and IDs.

Expected outputs:

```text
real_work_clip_native/features_zh_e5_swmatched.npz
real_work_clip_native/features_zh_e5_swmatched.manifest.json
real_work_clip_native/summary_zh_e5_swmatched_strong_h1_shared_d128_source_pool_bank.csv
real_work_clip_native/paired_zh_e5_swmatched_strong_h1_shared_d128_source_pool_bank.csv
real_work_clip_native/language_diff_zh_e5_swmatched_strong_h1_shared_d128_source_pool_bank.csv
```

The last comparison is generated automatically if the full original Swahili experiment with 10 seeds/3 budgets and reports is present. If not, complete the original Swahili runner first or call `compare_e5_languages.py` manually after both languages finish.

### Caveats

- This controls the **encoder checkpoint, E/I embeddings, image IDs, available training anchors, seed selections and evaluation set**, but not translation quality, tokenizer behavior by language, language distribution in E5 pretraining, or exact caption semantics. A shared multilingual encoder is not the same thing as identical language-specific representations.
- Original Chinese/text2vec results are not a pure *encoder* ablation here: those used a different caption protocol and 1,000 test images vs 984 matched IDs. The cleanest result is new Chinese E5 vs original Swahili E5, compared by *within-language gains and their difference*.
- E/I shared PCA is fitted on English and image training embeddings. Even `image_only` is not strictly English-unexposed in preprocessing.
- All tests are frozen-feature **retrieval**, not generative VLM reasoning.
- The cross-language bootstrap is exploratory, not proof that language alone caused the difference.