# H1: Shared geometry of Chinese/English activations in frozen PaliGemma2

**Hypothesis H1:** Chinese and English question representations in the frozen
PaliGemma2 decoder exhibit shared relational geometry, and a previously learned
**unpaired** orthogonal mapping can recover the matched bilingual question
correspondence. This is **not** a new VQA accuracy experiment and does not
retrain OT or the VLM.

## Design

- Source: **existing** official EN/CN paired MMBench DEV prepared by
  `prepare_mmbench.py`; use only the previous **local test split**. This is an
  exploratory follow-up, because the MMBench test split's accuracy was already
  viewed. No new blind benchmark is claimed.
- From that split, sample `H1_N=200` **different images**, deterministically.
  Pick **one question per image**; remove rotated/repeated-option records for
  retrieval. This prevents same-image repeated questions from creating multiple
  indistinguishable positives. Bilingual question pairing is used for
  **evaluation only**; the existing maps were fitted on disjoint, unpaired
  Flickr8K image/caption pools.
- Frozen layers `8,13,18` from the earlier experiment A. Every forward pass
  captures the **last text prompt-token hidden state** at all layers.
- Conditions:
  - `blank/stem` (**primary**): all rows use the exact same RGB gray image;
    remove options and boilerplate instructions, retain question + hint. Keep
    the same PaliGemma task prefix for both languages. This reduces matching
    by image identity and answer-choice position.
  - `real/stem`: original image, question + hint.
  - `blank/full`: identical blank image, original full MMBench question including
    options and shared task prefix.
  - `real/full`: real image, full prompt, closest to earlier VQA benchmark.
- Methods: `mean_shift`, `text_ot`, `text_visual_ot`, `text_noise_ot` using the
  **exact frozen** layer-specific maps already fitted in `experiment A`.
  - `raw`: Chinese and English are projected into the map's own PCA space,
    both centered on the **English** mean; no correction.
  - `center`: source language gets its learned mean-centering shift, but no
    rotation; controls for translation without meaningful geometric matching.
  - `mapped`: apply the saved orthogonal `W` in the shared PCA space. This is
    a **full geometry test**, not the clipped residual intervention/alpha used
    for generation. We do **not** optimize the alignment using MMBench pairs.
  - PCA bases can differ between map methods, so compare `mapped` versus
    `center` **within each method**, and compare retrieval rates across methods.
    Absolute cosine values are not inherently comparable across different PCAs.

## Metrics

1. **RSA** (Representational Similarity Analysis): Spearman correlation
   between upper-triangular within-Chinese and within-English pairwise cosine
   similarity matrices on matched questions. Orthogonal `W` **cannot change**
   within-language relational structure: RSA tests the shared geometry itself,
   separately from whether `W` identifies the correct cross-language matching.
   For the primary `blank/stem, Layer 8` condition, a permutation of English
   labels gives a null reference (399 shuffles; minimum Monte Carlo p=.0025).
2. **Cross-language paired retrieval**: bidirectional `R@1`, `R@5`, `MRR`,
   paired cosine, and margin against the hardest incorrect item. With 200
   distinct-image candidates, chance `R@1` is ~0.5%.
3. **Paired bootstrap** (4000 repeats): R@1 delta from `mapped` to its own
   `raw` and `center` control, and Visual OT versus Text-only OT on the same
   selected examples. CI is an exploratory interval (do not select the best
   layer and claim confirmatory significance).
4. `blank` versus `real` and `stem` versus `full` isolate whether retrieval is
   driven by the common input picture or answer-option text.

**Predeclared primary probe:** Layer **8**, `blank/stem`, RSA and average
bidirectional R@1. Other layer/prompt/image choices are diagnostic.

## RunPod

Use the **original** RunPod instance containing:

```
/workspace/vlm_A_zh_seed42/experiment/settings.json
/workspace/vlm_A_zh_seed42/experiment/maps/layer8_text_visual_ot.npz
/workspace/vlm_A_zh_seed42/mmbench_dev_bilingual/manifest.json
/workspace/Multimodal_test_switch/vlm_unsupervised_transfer/.venv
```

```bash
cd /workspace/Multimodal_test_switch
git fetch origin
git switch experiment/vlm-unsupervised-a-b
git pull origin experiment/vlm-unsupervised-a-b
mkdir -p /workspace/vlm_A_zh_seed42/logs
nohup env H1_N=200 LAYERS=8,13,18 \
  bash vlm_unsupervised_transfer/run_h1.sh \
  > /workspace/vlm_A_zh_seed42/logs/h1.log 2>&1 < /dev/null &
echo $! > /workspace/vlm_A_zh_seed42/logs/h1.pid

tail -f /workspace/vlm_A_zh_seed42/logs/h1.log
```

This extracts 200 examples × 2 languages × 2 image conditions × 2 prompt
formats = **1600 prefill-only forward passes**, but captures all 3 decoder layers
at once. **No autoregressive answer generation** is required. Feature files are
saved atomically per example and will be reused on a re-run. It may be faster
than the previous large Accuracy experiment but actual GPU runtime is unknown.

For a first GPU smoke test with only 20 image pairs, use **a new output path**:

```bash
H1_N=20 H1_OUT=/workspace/vlm_A_zh_seed42/h1_geometry_smoke20 \
  bash vlm_unsupervised_transfer/run_h1.sh
```

Afterward run the 200-pair command above in its **separate** default output.

## Results

```
/workspace/vlm_A_zh_seed42/h1_geometry_n200/
  h1_manifest.json       # data/model/maps/sample IDs SHA256 tracking
  neutral_gray_448.png   # shared input image control
  features/              # interruption-safe activation caches
  summary.json           # primary and full H1 results + CIs
  results.csv            # each setting's RSA / R@1 / R@5 / MRR / cosine
  per_pair.csv           # per-question ranks, cosine and margin
```

```bash
cat /workspace/vlm_A_zh_seed42/h1_geometry_n200/results.csv
cd /workspace/vlm_A_zh_seed42
# omit huge raw activation caches from sharing
tar -czf /workspace/H1_Geometry_results.tar.gz \
  h1_geometry_n200/summary.json \
  h1_geometry_n200/results.csv \
  h1_geometry_n200/per_pair.csv \
  h1_geometry_n200/h1_manifest.json
```

## Interpretation

| Observation | What it would support |
|---|---|
| RSA high in blank/stem, retrieval poor | Similar relational geometry exists, but unpaired map has not identified cross-language correspondences |
| RSA high; mapped R@1 improves relative to center and noise | Stronger support for H1 and a useful learned mapping |
| Visual OT beats Text OT and noise consistently | Evidence that the visual pool helps recover language correspondences; still replicate elsewhere |
| Real-image R@1 high but blank-image R@1 poor | Matched image identity drives retrieval; **not** evidence of text semantic alignment |
| RSA low and mapped R@1 at chance | Little support for H1 **in these prompt positions and frozen layers**, not a general refutation of shared geometry |

**Limitations:** the two languages are different natural-language prompts and
translation style might affect representations. A blank image does not make
PaliGemma2 a text-only LLM; it controls visual *content*. RSA correlations can
reflect syntactic/template similarity rather than grounded semantics. The
paired dataset has already been used to inspect VQA accuracy; this is a
post-hoc exploratory analysis, not independently blinded confirmation.