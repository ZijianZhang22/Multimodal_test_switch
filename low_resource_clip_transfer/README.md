# Low-resource CLIP Transfer — Rosetta3 -> Chinese zero-shot visual classification

**Question:** Can the same 50 bilingual Flickr8K-CN anchor pairs and *frozen* CLIP-native Rosetta3 alignment transfer CLIP's image understanding to **Chinese class descriptions**, without any downstream Chinese image-classification training labels?

This is a direct **visual transfer** test, not PaliGemma2, not VQA, and not the English Caption Retrieval Bridge. It does not presume accuracy improvements. We use the independent official **CIFAR-10 test split**, not Flickr8K test image-caption retrieval. Classes are common objects; cross-cultural/low-resource-language reasoning is **not** measured.

## Reuse original Rosetta

- Original training code: `../visual_anchor/rosetta3.py`, `../visual_anchor/run_rosetta3.py`, `../visual_anchor/run_clip_native.sh`.
- Frozen train features: `../visual_anchor/real_work_clip_native/features.npz`.
- Existing 50-pair **128-D shared PCA** checkpoints (seed 0): `results_clip_shared_d128_source_pool_bank_budget_50/{pairs_only,text_only,joint}_0.npz`.
- Source language: `shibing624/text2vec-base-chinese`; target English and image towers: **same** `openai/clip-vit-base-patch32`, both pinned by `model_revisions.json`.
- The exact original `visual_anchor/core.fit_space_pcas()` fits on **Flickr8K training features only**, never CIFAR labels/images. We use the same `W`, `mean_x`, `V_IE=I`, and the **fixed training-image-bank mean** for target images. Different Rosetta methods evaluate against *identical* frozen image features.
- Comparison: `pairs_only` (50 paired anchors); `text_only` (pairs + English unpaired matching); `joint` (pairs + English unpaired matching + CLIP image matching). Optional additional `image_only`, `caption_only`, `joint_caption`, etc. if complete matching checkpoints are already available.

### Direct classification architecture

```text
Chinese class prompt (e.g., "这是一张飞机的照片。")
   -> frozen Chinese text2vec -> training-fitted Chinese PCA
   -> Rosetta3 W (pairs_only / text_only / joint)
   -> image-text shared comparison space
CIFAR-10 test image
   -> frozen CLIP image encoder -> training-fitted image PCA
   -> fixed Flickr8K image-bank centering
   -> nearest of ten Chinese category prototypes = prediction
```

We average two **predeclared** Chinese templates per class (no tuning on CIFAR), then normalize. For reference, we also measure: (1) `english_clip_reference`, direct raw CLIP image vs English category prompt; (2) `chinese_clip_direct`, raw CLIP image vs the **same Chinese label templates** encoded by CLIP's text tower. The English reference uses translations of labels and is **not** a low-resource-language method. The direct Chinese CLIP control is not trained with our 50 anchors.

## One-click RunPod command

On the Pod where the prior `vlm_semantic_bridge` experiment finished (therefore Rosetta 50-pair models already exist):

```bash
cd /workspace/Multimodal_test_switch
git switch experiment/vlm-unsupervised-a-b
git pull origin experiment/vlm-unsupervised-a-b
bash low_resource_clip_transfer/run_oneclick.sh
```

**Defaults:** 500 class-balanced CIFAR-10 test images, 50 anchors, same map seed=0, Chinese templates, 2000 stratified paired bootstrap replicates. Downloads the official **CIFAR-10 binary archive (~162 MiB)** into `low_resource_clip_transfer/data/` the first time, with pinned MD5 verification. Does **not** download PaliGemma2 or run 6×VQA. If Rosetta artifacts are absent, the one-click script invokes the **original** `visual_anchor/run_clip_native.sh` pipeline to prepare/fix them (potentially substantial Flickr8K download and encoding). A partial old result directory is not silently overwritten.

A cheaper **100-image plumbing pilot** (10 images per class):

```bash
NUM_IMAGES=100 bash low_resource_clip_transfer/run_oneclick.sh
```

Full **10,000-image** test after locking design:

```bash
NUM_IMAGES=0 OUT=/workspace/clip_transfer_cifar10_full bash low_resource_clip_transfer/run_oneclick.sh
```

To use other complete checkpoints:

```bash
MAP_DIR=/workspace/your_run/results_clip_joint_d64_source_pool_bank_budget_50 \
  CLIP_WORK=/workspace/your_run BUDGET=50 MAP_SEED=0 \
  DIM=64 PCA_MODE=joint AUTO_BOOTSTRAP=0 \
  OUT=/workspace/clip_transfer_joint_d64 \
  bash low_resource_clip_transfer/run_oneclick.sh
```

To also test `image_only`/`caption_only`, the selected Rosetta report and saved maps must already contain those methods, fitted on the exact same anchors/protocol:

```bash
METHODS='pairs_only text_only image_only caption_only joint' AUTO_BOOTSTRAP=0 \
  OUT=/workspace/clip_transfer_ablation bash low_resource_clip_transfer/run_oneclick.sh
```

Do **not** add missing ablations by retraining over an existing report. Use a separate fresh `CLIP_WORK`/`MAP_DIR` experiment for an expanded method set.

## Outputs

Default: `low_resource_clip_transfer/outputs/cifar10_n500_b50_mapseed0_seed42/`

```text
config.json          # immutable method/model/dataset/hash/label/template protocol
raw_encodings.npz    # frozen raw CLIP image/EN/ZH and Chinese text2vec prompt features; safely resumable
summary.csv          # accuracy, delta vs pairs_only and text_only
report.json          # paired image-level within-class bootstrap CIs and repairs/harms
per_example.csv      # each CIFAR test index, true class, prediction & correctness for every method
per_class.csv        # 10-class accuracy breakdown per method
```

## Interpreting results

Primary question: does `joint` beat **both** `pairs_only` and `text_only` on image classification when using *the same Chinese target prompts and class prototypes*? The report shows paired per-image bootstrap confidence intervals and net correct-image repairs/harms. If `joint` beats neither, there is **no verified transfer gain**, even if prior Flickr8K caption-retrieval metrics improved. Do not compare this 10-class CIFAR accuracy to MMBench VQA accuracy or Flickr R@1. Avoid treating a 100-image pilot as independent evidence after selecting templates or hyperparameters.

Caveats: CIFAR classes are easy and CLIP's native English model may dominate; images are 32×32 upscaled by the CLIP processor; a caption-trained Chinese encoder may generalize poorly to class labels; CLIP pretraining data may have included benchmark images. This measures **Chinese zero-shot object-category transfer**, not full multilingual VQA or cultural understanding. This version runs **one saved Rosetta seed** by default; repeat with independent fit seeds and preferably additional datasets before strong conclusions.

Official dataset: [CIFAR-10, Alex Krizhevsky](https://www.cs.toronto.edu/~kriz/cifar.html); binary archive MD5 `c32a1d4ab5d03f1284b67883e8d87530`, record format 1 label byte + 3072 image bytes.

CPU tests:

```bash
python -m unittest discover -s low_resource_clip_transfer/tests -v
```