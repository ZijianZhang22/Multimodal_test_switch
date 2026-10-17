# 30 matched Idis samples × 4 variants × 2 conditions

Use OFFICIAL existing Idis-perception `aligned`, `conflicting`, `irrelevant` images, plus OFFICIAL ImageNet-9 `original/val` clean images. No image synthesis. All four images must share the **same original ImageNet image stem**; otherwise fail before inference rather than compare mismatched examples.

The study selects **30 distinct original images**, distributed across 9 object classes. One inference for each variant × each of `visual_only` and `visual_plus_text`: **240 generations** by default. Q2 is fixed simple arithmetic `What is 17 + 23?`, answer 40.

## One-click RunPod

```bash
cd /workspace/Multimodal_test_switch
git pull origin main
cd vlm_cross_task_pilot
bash run_idis30.sh
```

The preparation step downloads the official ImageNet-9 testing release (~280 MB) once unless CLEAN_ROOT is provided, extracts original validation files only, then downloads selected official Idis distractor images from Hugging Face. GitHub dataset release: https://github.com/MadryLab/backgrounds_challenge/releases/tag/data . Official Idis dataset: https://huggingface.co/datasets/Vail-2000/Idis .

If you already have extracted clean originals, use `CLEAN_ROOT=/workspace/backgrounds_challenge/original/val bash run_idis30.sh`. Script searches `<root>/<class>/<stem>.JPEG`, `<root>/val/<class>/...`, `<root>/original/val/<class>/...`.

Optional: `SAMPLES=3 bash run_idis30.sh` gives 720 generations. `COUNT=30` is the number of distinct originals, not total model calls. Existing manifests are reused, model inference resumes by sample/variant/condition/repetition.

Two prompts differ only by addition of an independent easy arithmetic question and its requested answer tag. Visual-only uses the previously stored Idis official perception base prompt.

Results: `idis30/results/predictions.jsonl` and `scored.csv`, `summary.csv`, `paired.csv`, `paired_summary.csv`, `interaction_vs_clean.csv`. Report all three: Q1 accuracy delta, output-token length delta, and whether compression actually occurred. Total output-token reduction does not necessarily mean visual-Q1 reasoning was compressed. Thinking-phase *word* count is a coarse proxy, not exact tokenizer count.

No experimental accuracy claims are made until the model has run. 30 originals is a pilot; repeat sampling for uncertainty, particularly because visual classification errors may be rare.
