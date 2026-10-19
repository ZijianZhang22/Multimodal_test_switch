# Language Steering for Multilingual ICL: VLM modality ablation

Based on Kirtane & Huang (2026): https://github.com/lab-flair/language-vector . This is NOT the Gan et al. semantic concept-vector method.

## Method

Use PaliGemma2-3B mix 448. Build English/Chinese (or Swahili) parallel Q&A blocks from CV-Bench, with 6 Q&A examples per extraction sample. Mean-pool token hidden states in the *same VLM's language decoder*, then average target-minus-English paired differences for language vectors. At inference use English task demonstrations, target-language question, and the same image. Compare baseline (no steering), text decoder token steering, projected image decoder token steering, and both. Injection uses additive h'=h+alpha*v at decoder layer outputs during **prefill** only. It does not modify image pixels or the vision encoder. This is a VLM adaptation of the original pure-text method.

Each task's existing parity tune split is divided into disjoint extraction and validation halves; existing CV-Bench test data is held out. Layers and strengths are selected on validation independently for each mask and frozen before test. The test file is not overwritten on rerun. Compute paired accuracy gains, regressions, disagreement, and invalid outputs.

## RunPod

Prepare CV-Bench images and translations with the sibling paligemma2_cross_lingual_steering/parity pipeline. Accept Google model license, run hf auth login, install parity requirements (transformers==4.51.3), and run:

```bash
cd /workspace/Multimodal_test_switch
git fetch origin parity-pipeline-20261018
git checkout parity-pipeline-20261018
PY=paligemma2_cross_lingual_steering/.venv-parity/bin/python
DATA=paligemma2_cross_lingual_steering/outputs/parity_cvbench
$PY language_icl_vlm_masks/tests.py
$PY language_icl_vlm_masks/run.py --data "$DATA" --out "$DATA/runs/kirtane_mask_smoke" --lang zh --layers 8,12,16 --alphas 0.5,1.0,2.0 --extract-limit 12 --limit-val 3 --limit-test 3
```

After smoke testing, choose a NEW --out path and use --extract-limit 0 --limit-val 0 --limit-test 0 for all samples. Outputs: language_vectors.pt, vector_manifest.json, validation.csv, chosen.json, test.csv, SUMMARY.md.

## Limitations

This is NOT a reproduction of the original Kirtane benchmarks MGSM/XNLI/MSVAMP. Existing translated CV-Bench prompts may have errors. Since image token counts differ from text token counts, a later controlled study should normalize intervention budgets. No GPU test or accuracy improvement is claimed yet. The local CPU environment cannot load gated PaliGemma weights. Recommended further controls: random norm-matched vector, reversal, 3 seeds, and confidence intervals.
