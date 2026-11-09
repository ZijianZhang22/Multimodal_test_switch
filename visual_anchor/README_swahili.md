# Swahili Rosetta3 (Flickr8k controlled language transfer)

This branch tests whether the third-target effect persists on **Swahili (sw)** rather than Chinese (zh). It reuses the exact original Flickr8k-CN group IDs, held-out image IDs, English MiniLM vectors, DINOv2 vectors, PCA protocol, and E-I Procrustes training pairs, replacing only the L-language embeddings with `intfloat/multilingual-e5-base` Swahili embeddings. Source: [SwaFlickr8k](https://huggingface.co/datasets/Alfaxad/SwaFlickr8k).

**Limitations:** This is an encoder-change and dataset-change experiment, not a strictly isolated language effect; compare within-language method deltas, not absolute Chinese/Swahili scores. English and Swahili are matched by image ID **and normalized English caption**. By default missing exact matches abort, rather than silently substituting a caption of the same image. SwaFlickr8k's translation provenance and quality must be evaluated separately. Unlike Chinese's original setup, the Swahili text test uses a translated version of the English test caption; the image retrieval test instead uses a different Swahili caption for the same image when available.

## Run

Requires existing `visual_anchor/real_work/features.npz` created by the repaired `prepare_real.py` pipeline, together with `real_work/raw/flickr8kenc.caption.txt`. The first run downloads SwaFlickr8k and multilingual-E5, encodes Swahili, and caches the new vectors. It reuses the English/image features already produced by `prepare_real.py`.

```bash
cd /Multimodal_test_switch
git fetch origin
git switch experiment/swahili-rosetta3
cd visual_anchor
bash run_swahili_rosetta3.sh
```

Default budgets are 10, 25, 50, 200, 1000; seeds 0, 1, 2; seven methods (including the 2 weight/matching controls). To do a quick trial:

```bash
BUDGETS="10 50" SEEDS="0" METHODS="rosetta visual bank_text english_pool anchor_downweight" bash run_swahili_rosetta3.sh
```

If source English captions do not exactly match the SwaFlickr8k caption text, the script will **stop and explain which sample failed**. Inspect the English caption discrepancy before deliberately using `python prepare_swahili.py --allow-image-fallback`; doing so weakens sentence-level anchor alignment.

`real_work/features_sw.npz` preserves existing key names so it can be passed unchanged into `run_rosetta3.py`. Feature metadata and caption audit are written to `real_work/features_sw.manifest.json`. All budget runs and reports go to `real_work/results_swahili_rosetta3_budget_*/`, summarized into `real_work/summary_swahili_rosetta3.csv`.

## Model comparison

- Existing Chinese result remains the within-Chinese reference.
- Swahili: compare original Rosetta vs Visual vs Bank Text vs Extra Text vs Gaussian vs English Pool vs Anchor Downweight with matched seeds and caption banks.
- Focus particularly on Visual minus Bank Text for 10/25/50 anchors.
- No claim of significant improvement without paired per-image confidence intervals and an audit of caption translation quality.

## Missing image ID fix (SwaFlickr8k subset)

Some Flickr8K-CN IDs are absent from `captions_sw.csv`, or have no exact English caption match. The preparation script now performs an audited dataset intersection instead of crashing on the first unavailable image:

- Removes missing/unmatched IDs consistently across each shared train/test group.
- Refills missing supervision anchors **only** from the original bridge group with verified exact English-caption match, and removes each moved ID from both bridge modalities and the English OT pool. This preserves the 1,000-anchor budget if sufficient verified bridge IDs remain, but changes the training/bridge split and should be reported.
- Keeps original test IDs in their original order **when available**, but removes unmatched tests. Hence test size can become smaller than 1,000; comparison against the Chinese test numbers is not a paired apples-to-apples evaluation. Do not claim that it is.
- Writes all excluded IDs and replacements into `features_sw.manifest.json`, as well as reporting new test count at preparation time.
- If the remaining verified bridge pool cannot provide enough anchor replacements, the script raises an explicit error rather than fabricating translated labels.

Rebuild Swahili features after updating this branch (original features untouched):

```bash
rm -f real_work/features_sw.npz
python -m unittest -v test_swahili
bash run_swahili_rosetta3.sh
```

For a strict apples-to-apples Chinese-vs-Swahili evaluation, rerun the **Chinese** experiments on the repaired Swahili intersection IDs too. Within-Swahili ablations remain comparable because all seven methods use the same repaired splits.
