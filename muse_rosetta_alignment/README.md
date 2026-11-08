# MUSE × Rosetta: Chinese image-caption retrieval (RunPod A100)

Pilot experiment comparing (i) no extra alignment, (ii) MUSE language alignment, (iii) Rosetta image-text alignment, (iv) fixed MUSE + Rosetta, and (v) **language-first retraining Rosetta**. Also evaluates the vital raw-language-mixture control.

## One command on a fresh RunPod A100 PyTorch terminal

```bash
git clone https://github.com/ZijianZhang22/Multimodal_test_switch.git && bash Multimodal_test_switch/muse_rosetta_alignment/run_runpod.sh
```

Choose a RunPod PyTorch CUDA image with Python >= 3.11, git and >= 20GB free disk.

Small smoke run:

```bash
TRAIN_LIMIT=700 TEST_LIMIT=120 ROSETTA_CLUSTERS=8 ROSETTA_RESTARTS=2 ROSETTA_ITERATIONS=8 bash Multimodal_test_switch/muse_rosetta_alignment/run_runpod.sh
```

Run again in the same directory to reuse precomputed embeddings (use `REBUILD=1` to regenerate).

Outputs under `muse_rosetta_alignment/outputs/`:
- `results.csv`: Recall@1/5/10 and MRR for Chinese image-to-text and text-to-image.
- `results.json`: same plus paired bootstrap intervals of Recall@1 gains.
- `muse_mapping.npz`: the supervised Chinese→English Procrustes matrix.
- `rosetta_english.pt` and `rosetta_mixed_aligned.pt`: learned Rosetta maps.

## Methods

Upstream implementations are cloned into ignored `external/` and called directly:
- [Facebook MUSE](https://github.com/facebookresearch/MUSE): **supervised** `src.trainer.Trainer.procrustes` on training caption embeddings. NOT adversarial unsupervised MUSE.
- [Unpaired Rosetta](https://github.com/dominik-schnaus/unpaired-rosetta): `WassersteinProcrustes.fit` with shuffled, unpaired image/text training embeddings.

All results use one held-out test split, one caption per image. Data: Flickr8k English captions plus *machine-translated* Chinese captions; pretrained multilingual CLIP 512-d shared-space encoders make the unmodified baseline meaningful. Therefore, this is a **retrieval pilot only**, not CV-Bench/CVQA reasoning accuracy, nor a faithful reproduction of Rosetta's independently trained encoders. Baseline is already aligned; additional alignment may hurt.

Conditions: `Baseline`, `A_MUSE_only`, `B_Rosetta_only`, `C_MUSE_plus_Rosetta_fixed`, `D_Language_first_retrain_Rosetta`, and `E_Rosetta_raw_language_mix_control` plus English references. D retrains Rosetta using a 50/50 distribution of English and MUSE-mapped training Chinese; E matches the mixture without MUSE.

No test captions/images are used to fit mappings. Machine-generated Chinese is not a culturally representative benchmark. The original full GitHub/Hugging Face download and A100 workflow have **not** been executed end to end here. If an installation or model step fails, save the complete log.
