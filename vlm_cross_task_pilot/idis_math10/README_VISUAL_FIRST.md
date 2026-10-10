# Idis-math50 — Condition B: visual math FIRST, easy arithmetic SECOND

Reuse the **exact same 50 source questions/images** from the previous Idis-math50 pilot.
Old A/C predictions remain untouched, and **only 50 new Condition B generations**
are appended to `idis_math50_new/results/visual_first_predictions.jsonl`.

- **A**: Original visual mathematics question only (saved).
- **B**: Original visual mathematics question first, then `17+23` (new).
- **C**: `17+23` first, then the visual mathematics question (saved).

The B prompt retains the original A visual mathematics prompt as its exact prefix;
the unrelated arithmetic task is added after it. The legacy A prompt labels the
visual question "Question 2"; we deliberately **do not edit it**, to avoid
changing the original comparison. In B, the new arithmetic task is numbered Q3.
This preserves the original math question text and answer tags.

## RunPod

```bash
cd /workspace/Multimodal_test_switch
git pull origin main
cd vlm_cross_task_pilot
bash idis_math10/run_visual_first_50.sh
```

Default settings exactly match previous 50-item generation:
Qwen3-VL-8B-Thinking, 12,288 output-token cap, temperature 0.7,
top-p 0.95, same seed 1043, same image size 1536, one sample per condition.
If your prior run customized these variables, use the SAME settings here.

The script verifies all 50 old A/C and B outputs have identical sample IDs,
images, questions, gold answers, models, model settings and seeds.
It will refuse to compare mismatched settings. The existing A/C outputs
are not altered. Run is resumable.

Output:
- `idis_math50_new/results/abc_comparison/summary_abc.csv`: mean/median thinking deltas and 95% paired-bootstrap intervals.
- `paired_abc.csv`: question-by-question A/B/C thinking, total tokens and final answers.
- `predictions_abc_scored.csv`: provisional automatic mathematical equivalence.
- `manual_labels_abc.csv`: annotate all three conditions as 1/0 for final accuracy.
- `needs_review_abc.csv`: uncertain or truncated answers requiring review.

Incomplete `</think>` or truncated generation is excluded from
all three-condition thinking comparisons. Math judgments that cannot
be verified are NOT automatically counted wrong; manual audit is necessary.
Thinking span includes ALL tasks, not pure visual-math-only thought.

Pack for analysis:
```bash
tar czf idis_math50_abc_results.tar.gz \
  idis_math50_new/results/ idis_math50_new/data/manifest.jsonl
```
