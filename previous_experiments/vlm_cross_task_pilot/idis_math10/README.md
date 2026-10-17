# Idis-math10: Add one unrelated arithmetic question FIRST

Pilot: **10 distinct official Idis-math / MathVerse testmini questions × 2 prompt conditions = 20 generations**. **No model training and no image synthesis.** The SAME augmented image (default: official Idis-math `irrelevant/n4`), mathematical question and ground-truth answer are reused in both conditions.

| Condition | Prompt content |
| --- | --- |
| `original` | Original visual math question, with final answer in `<math_answer>` |
| `easy_first` | **Question 1: What is 17 + 23?** and then the exact same original visual math question, answered in `<math_answer>` |

Both conditions share the same mathematical instruction suffix; Q1 is added only as a prefix. The original question is numbered Question 2 in both conditions. This tests multi-task prompt effects; it does not guarantee that thinking compression occurs.

## RunPod: one command

```bash
cd /workspace/Multimodal_test_switch
git pull origin main
cd vlm_cross_task_pilot
bash idis_math10/run_runpod.sh
```

For `nohup` background execution:

```bash
nohup bash idis_math10/run_runpod.sh > idis_math10.log 2>&1 &
echo $!
tail -f idis_math10.log
```

### Settings

- Default model: `Qwen/Qwen3-VL-8B-Thinking`; temperature 0.7, top-p 0.95, output cap 8192, one generation per condition.
- `VARIANT=aligned`, `VARIANT=conflicting`, or `VARIANT=irrelevant`. Default is **irrelevant** because Idis-math focuses on robust visual math; all images are pre-built official Idis images.
- `N_DISTRACTORS=1..4` (default 4).
- `SAMPLES=3` repeats both conditions 3 times (60 model generations total).
- `COUNT=10` distinct source math problems. Change `OUT=idis_math10_conflicting` when switching benchmark variant to avoid mixing manifests.
- `MAX_NEW_TOKENS=8192` is generous for a pilot but still may truncate difficult questions. The script flags truncated samples and prints exact thinking-token counts from special token boundaries.
- `MAX_IMAGE_SIDE=1536` keeps fine MathVerse diagram text more legible than the 512px perception preprocessing.

## Dataset

- Official metadata and **only 10 selected images**: `Vail-2000/Idis`, `Idis-math/visual_distractor/<variant>/meta.jsonl`, `n4/...`.
- Original questions and gold answers: `AI4Math/MathVerse`, `testmini.json`.
- Prefer `Vision Intensive` / `Vision Dominant` MathVerse versions with nonempty questions; choose distinct problem IDs with a fixed seed, without using model correctness.
- The script does NOT download the full 70GB Idis dataset or generate distractors.

## Results

Directory: `idis_math10/results/`

- `predictions.jsonl`: raw model output, prompt, image path, exact generated total and thinking-token counts, truncation.
- `pairs.csv`: 10 paired comparisons of token and thinking lengths, both math answers, easy Q1 answer, provisional grades.
- `summary.csv`: mean output/thinking token deltas, compression rates, scoring coverage.
- `scored.csv`: per-condition Q2 answer extraction.
- `needs_review.csv`: MathVerse answers requiring manual review.
- `manual_labels.csv`: 10-row template; manually fill `original_correct` and `easy_first_correct` with `1` / `0`, then rerun `python idis_math10/score.py --predictions idis_math10/results/predictions.jsonl`.

Only exact, numerical and verified mathematical equivalence is automatically treated as correct. Unknowns are **not** automatically marked wrong. For a 10-question pilot manually review all 20 visual-math predictions; otherwise no reliable accuracy claim.

If no GPU or the official HF layout differs, the script exits with an error rather than silently substituting other images. The preparation phase requires access to Hugging Face from the RunPod container.

Official resources: https://github.com/effl-lab/Idis ; https://huggingface.co/datasets/Vail-2000/Idis ; https://huggingface.co/datasets/AI4Math/MathVerse .
