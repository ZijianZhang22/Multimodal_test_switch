# H1 follow-up diagnostics (D1–D5)

This **adds** experiments to the frozen PaliGemma2 H1 probe without replacing `h1_geometry.py` or refitting the Flickr8K maps. The previous H1 result already showed high RSA but lower retrieval after full OT. These are mechanism checks, not new evidence of VQA accuracy.

## Five experiments and expected files

| Test | Question | File | Compute |
|---|---|---|---|
| D1: W vs Wᵀ | Is mapping orientation wrong? `raw`, `center`, trained `W`, transpose `Wᵀ` | `direction_check.csv` | Cache-only |
| D2: partial OT | Does full rotation overcorrect? Normalized linear blend `(1−α)z + αzW`, α=0/.1/.25/.5/.75/1 | `partial_ot.csv` | Cache-only |
| D3: native vs PCA | Does PCA destroy correspondence? Original hidden states vs low-rank PCA, same source/target centering; also full-space mapped projection | `pca_vs_full.csv` | Cache-only |
| D4: caption/question shift | Do Flickr8K caption and MMBench question activations differ? Unit-vector RBF MMD², 199 label-permutation null, centroid similarity | `distribution_shift.csv` | Cache-only |
| D5: fixed independent replication | Does the **predeclared** L8 / gray-image / question-stem diagnostic replicate on new, disjoint bilingual image-question pairs? | `external/<dataset SHA>/external_locked.csv` | GPU prefill for new pairs |

D1–D3 additionally write `paired_deltas.csv` with paired-question bootstrap 95% intervals for **average bidirectional R@1 delta** relative to the same method's PCA-centered baseline. D1 verifies the recorded original H1 raw/center/W scores to 1e-6 before producing output. The script checks settings, dataset manifest and map hashes against the original H1 manifest. Original map fitting is untouched.

**Important cautions:** `W` is the correctly oriented mapping fitted by `fit_map` (`source @ W`); `Wᵀ` is a negative diagnostic, not a remedy. D2 linear interpolation is not generally orthogonal and is **not** the same as original norm-clipped VQA steering. D3 full-space mapped representations only change the learned PCA subspace; the complement remains unrotated. D4 is *not* a causal separation of caption-vs-question semantics: the datasets, pictures, question templates and prompts differ. The current 200 MMBench questions were already inspected, so D1–D4 and their CIs are exploratory; don't claim selected α/layer as blind confirmation.

## Run D1–D4 (existing RunPod, no generation or model load)

This needs `/workspace/vlm_A_zh_seed42/experiment` from Experiment A, existing `h1_geometry_n200/features`, and original `mmbench_dev_bilingual` v2. The script **does not** contain or download the saved activations or fitted maps.

```bash
cd /workspace/Multimodal_test_switch
git fetch origin
git switch experiment/vlm-unsupervised-a-b
git pull origin experiment/vlm-unsupervised-a-b

bash vlm_unsupervised_transfer/run_h1_followups.sh

ls /workspace/vlm_A_zh_seed42/h1_followups
```

Results:

```text
/workspace/vlm_A_zh_seed42/h1_followups/
  diagnostics_summary.json
  direction_check.csv
  partial_ot.csv
  pca_vs_full.csv
  distribution_shift.csv
  paired_deltas.csv
```

## D5: independent bilingual dataset, fixed configuration

Supply a **new** JSONL with at least 20 paired question stems and different real images, none identical by decoded RGB pixels to either MMBench DEV or the original Flickr8K fit pools. Each language must also have unique question stems, with no answer-option blocks; the loader rejects exact repeated stems after case/whitespace normalization. Use a held-out external corpus that has not been inspected for H1 and avoid paraphrases/near duplicates (the script detects only exact image duplicates). One record per line:

```json
{"id":"new_001","image":"/workspace/independent_images/001.jpg","questions":{"zh":"一个人拿着什么？","en":"What is the person holding?"}}
```

Images exist on RunPod, and `image` may be absolute or relative to the JSONL. **This record is a format example only, not an included dataset.** Clean/verify translations and question specificity before evaluation. There is no answer label requirement because H1 measures correspondence, not VQA accuracy. Supply at least 200 pairs for default `H1_EXTERNAL_N=200` or explicitly set a smaller locked count (≥20).

```bash
cd /workspace/Multimodal_test_switch
H1_FOLLOWUP_STAGE=external \
H1_EXTERNAL_PAIRS=/workspace/new_bilingual_pairs.jsonl \
H1_EXTERNAL_N=200 \
  bash vlm_unsupervised_transfer/run_h1_followups.sh
```

D5 uses one frozen Layer 8, shared gray 448×448 image and question-stem-only prompts. It reports raw / source-center-only / original W for every saved map; **no tuning or layer/α selection on this new corpus**. The original v2 MMBench splits and Flickr fit pools are excluded by decoded-image hash. A JSON manifest binds each run to the external dataset SHA, mapping SHAs and sample IDs. Intermediate prefill activations are resumable. Do **not** reuse the external dataset to pick a new α before reporting the fixed D5 test.

## Bundle outputs for analysis

```bash
cd /workspace/vlm_A_zh_seed42
tar -czf /workspace/H1_Followups_results.tar.gz \
  h1_followups/diagnostics_summary.json \
  h1_followups/direction_check.csv \
  h1_followups/partial_ot.csv \
  h1_followups/pca_vs_full.csv \
  h1_followups/distribution_shift.csv \
  h1_followups/paired_deltas.csv
```

If D5 has also run, separately share the matching `h1_followups/external/<dataset SHA>/external_locked.csv` and `manifest.json` (not the large activation caches).

## Review status

Seven CPU unit tests pass for direction, blending endpoints, PCA projection, MMD and input validation. These tests do not substitute for a RunPod GPU end-to-end check. D1-D4 are exploratory on previously viewed MMBench samples; D5 requires a separately sourced and reviewed dataset.