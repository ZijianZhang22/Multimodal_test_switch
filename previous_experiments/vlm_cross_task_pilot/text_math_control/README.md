# Text-only reasoning compression control (same Qwen3-VL model)

Use 50 diagram-independent MATH-500 text math problems, stratified 10 per difficulty level, and run **Qwen/Qwen3-VL-8B-Thinking** with no image input. A = only text math; B = text math then 17+23; C = 17+23 then text math.

Run on RunPod:

```bash
cd /workspace/Multimodal_test_switch
git pull origin main
cd vlm_cross_task_pilot
bash text_math_control/run_runpod.sh
```

Requires CUDA and enough VRAM for the 8B model; 150 generations, resumable. Same base settings as Math50: temperature .7, top-p .95, max output 12288, seed 1043. Results: `text_math50/results/analysis/{summary.csv,pairs.csv,scored.csv,needs_review.csv,manual_labels_abc.csv}`. Complete outputs: `text_math50/results/predictions.jsonl`. Re-run score independently after editing the manual labels CSV.

```bash
tar -czf text_math50_results.tar.gz text_math50/data/manifest.jsonl text_math50/results/
```

The exact total Thinking count comes from generated token IDs. Target-only Thinking counts are estimated by retokenizing the target section, and only when task-switch boundaries are clear; ambiguous examples are flagged. Unknown math correctness is NOT automatically incorrect. Track whether B and C answer both tasks and exclude truncated outputs from completed-Thinking comparisons. Differences against MathVerse image tasks can reflect dataset and difficulty, not solely modality.
