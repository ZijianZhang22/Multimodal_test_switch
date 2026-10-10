# VLM Cross-Task Interference Pilot

An experimental extension of the [Idis benchmark](https://github.com/effl-lab/Idis).

Use the official Idis-perception visually distracted images for a Q1 object-classification task, then pair each input with an unrelated math Q2 from official GSM8K or MATH-500. The default model is Qwen/Qwen3-VL-8B-Thinking.

## Run on RunPod (NVIDIA GPU with CUDA PyTorch)

    git clone https://github.com/ZijianZhang22/Multimodal_test_switch.git
    cd Multimodal_test_switch/vlm_cross_task_pilot
    bash run_runpod.sh

Default: 6 image–text pairs from 2 classes, conflicting/irrelevant Idis distractor variants, 1 sampled output per condition. Only selected Hugging Face images are downloaded.

Use harder math questions:

    BENCHMARK=math500 PAIRS=12 N_CLASSES=3 bash run_runpod.sh

Add reversed ordering:

    CONDITIONS=q2_only,q1_only,image_q2,joint_vt,joint_tv bash run_runpod.sh

Repeat 5 times (matching the paper's sampling count):

    SAMPLES=5 PAIRS=24 N_CLASSES=4 bash run_runpod.sh

## Conditions

- q2_only: independent text-only baseline (no image).
- q1_only: original Idis visual category task prompt plus Idis distracted image.
- image_q2: distracted image and math Q2, no visual Q1.
- joint_vt: visually answer Q1 and mathematically answer Q2.
- joint_tv: reverse question order (optional control).
- original visual images (optional): ORIGINAL_ROOT=/path/to/ImageNet-9/original/val.

## Results

Generated in pilot_gsm8k/results or pilot_math500/results:

- predictions.jsonl: model outputs
- scored.csv: predictions and validity of each answer
- summary.csv: accuracy, generation lengths, and truncation summary
- candidate_interference.csv: Q2 correct with image only but incorrect in Q1+Q2

Primary contrast: paired Q2 accuracy(joint_vt) minus Q2 accuracy(image_q2). Interpret carefully: additional instructions and outputs can influence accuracy separately from actual visual leakage. Review truncated/ill-formatted model outputs, and use larger matched samples and matched text-only task-load controls before claiming causal interference.

Reproduction notes: original Idis Q1 base prompt and 512x512 preprocessing, T=0.7/top-p=0.95, but the default pilot uses 1 generation (paper used 5). Inference uses transformers rather than vLLM. MATH-500 verification may require manual review for symbolic answers.

Datasets: https://huggingface.co/datasets/Vail-2000/Idis ; https://huggingface.co/datasets/openai/gsm8k ; https://huggingface.co/datasets/HuggingFaceH4/MATH-500.

## Image-of-text Q1 control: 17 + 23 = ?

Image asset: [assets/q1_text_17_plus_23.png](assets/q1_text_17_plus_23.png). Its equivalent textual Q1 is **17 + 23 = ?**; gold answer **40**.

This standalone modality-control experiment holds the SAME official GSM8K or MATH-500 reasoning Q2 fixed across every condition. It does not modify the original Idis experiment above.

Run on RunPod (CUDA PyTorch and internet are required):

```bash
git pull origin main
cd vlm_cross_task_pilot
MODE=image_text BENCHMARK=gsm8k PAIRS=8 bash run_runpod.sh
# Harder official questions (MATH-500 level >=3):
MODE=image_text BENCHMARK=math500 PAIRS=12 bash run_runpod.sh
# Increase repetitions:
MODE=image_text BENCHMARK=math500 PAIRS=12 SAMPLES=5 bash run_runpod.sh
```

Alternatively run `bash run_image_text_control.sh` directly. Use `OUT=pilot_new` for a fresh output directory; same directory resumes the existing manifest and predictions.

| Condition | Image included | Tasks |
|---|---|---|
| `q2_only` | No | Q2 only |
| `text_q1_q2` | No | Written equation Q1 + Q2 |
| `image_q2_only` | Yes | Attached image, but Q2 only |
| `image_text_q1_q2` | Yes | Written equation Q1 + Q2, image also included |
| `image_q1_q2` | Yes | Must read equation from image, then solve Q2 |
| `image_q1_only` / `text_q1_only` | Yes / No | Single-task Q1 sanity checks |

Optional order controls: `image_q2_q1` and `text_q2_q1`, enabled via `CONDITIONS`.

Outputs in `pilot_image_text_gsm8k/` or `pilot_image_text_math500/`: `questions.jsonl` (frozen paired Q2 set), `predictions.jsonl` (raw generations), `summary.csv`, `scored.csv`, `paired_comparisons.csv`, `candidate_interference.csv` (control correct -> treatment incorrect). The Q1-only conditions run only once per repeat, not for every Q2.

Primary comparisons: `image_q1_q2` vs `image_q2_only` (active visual Q1); `image_q1_q2` vs `image_text_q1_q2` (visual vs text Q1 with the SAME picture attached); `text_q1_q2` vs `q2_only` (multitasking baseline); `image_q2_only` vs `q2_only` (passive image presence). These measure behavioral changes, not causal neural mechanisms.

Offline test: `python test_image_text_control.py`. To rescore, run `python run_image_text_q1_control.py --score-only --output-dir pilot_image_text_gsm8k`.

Review truncated and badly formatted generations; the main Q1 is deliberately easy, serving as a text-vs-image control rather than a spatial-reasoning stress test.

## Dual-image Q1 control: original Idis image plus arithmetic image

This follow-up reuses the **same** local `pilot_math20/data/pairs.jsonl` from the MATH-500 pilot. No re-sampling or rewriting of Q2 or the Idis image.

| Condition | Input images | Q1 | Q2 |
|---|---|---|---|
| `joint_text` | Original Idis image only | Written `17 + 23?` | Original MATH-500 text |
| `joint_image_q1` | Original Idis AND equation PNG | Solve arithmetic in SECOND image | Identical MATH-500 text |

Run on the SAME RunPod filesystem where `pilot_math20/data/pairs.jsonl` and its image files already exist:

```bash
git pull origin main
cd vlm_cross_task_pilot
python test_dual_image_q1_control.py
bash run_dual_image_q1_control.sh
```

To reuse earlier matching text-Q1 outputs instead of recomputing them:

```bash
bash run_dual_image_q1_control.sh --reuse-text-file pilot_math20/results/text_q1_predictions.jsonl
```

Default: all existing 20 paired questions (or however many are in pairs.jsonl), Idis variant `conflicting`, temperature 0.7, top-p 0.95, 1 sample, 4096 generated tokens. Set `SAMPLES=5` for repeats. Use `--limit 3 --dry-run` to inspect input structure without GPU.

Rescore existing predictions with:

```bash
python score_dual_image_q1_control.py --predictions pilot_math20_dual_image_q1/results/predictions.jsonl
```

Outputs under `pilot_math20_dual_image_q1/results/`: `predictions.jsonl`, `summary_dual_image_q1.csv`, `paired_summary_dual_image_q1.csv`, `paired_details_dual_image_q1.csv`, and `candidate_interference_dual_image_q1.csv`.

Primary metric is Q2 accuracy in `joint_image_q1` minus that in `joint_text`, matched by image ID, exact Q2 ID, variant and repetition. Inspect both correct-to-wrong and wrong-to-correct flips, format failures, and truncation. This is a behavioral control, not proof of causal visual leakage.

Unlike `run_image_text_q1_control.py`, which replaces the original Idis image with an arithmetic-only image, this new script retains Idis as image #1 and appends the arithmetic PNG as image #2.
