# Experiments A & B: Language Steering across multimodal training

This is an **extension** of [Kirtane & Huang (2026), Language Steering for Multilingual In-Context Learning](https://github.com/lab-flair/language-vector), using the pre-existing PaliGemma2 CV-Bench parallel-prompt data. It is NOT the Gan et al. semantic concept-vector method.

## Experimental conditions

- **A, external backbone**: Extract parallel target-language-minus-English pooled vectors from `google/gemma-2-2b` (original text model), inject into `google/paligemma2-3b-mix-448`.
- **B, VLM decoder**: Extract the **same paired Q&A texts** from PaliGemma2's own language decoder without any image, inject into the **same** PaliGemma2 checkpoint.
- For each vector source: steering **text-only**, **image-only**, and **both**, with **identical test images/prompts** and a shared unsteered baseline.
- Image token steering affects **projected image positions inside the language decoder**, not SigLIP weights or pixels. Hooks act during prefill only.
- Both vectors use per-layer token-mean pooling and average `target - English` over paired six-Q&A text blocks. By design, this is an adaptation of the authors' MGSM/XNLI protocol to the existing CV-Bench dataset.
- The original model is unloaded before the VLM is loaded, avoiding simultaneous VRAM residency.
- The output `vector_cosine_norms.csv` reports cosine similarity, both norms, and ratio at matched decoder layers.

## Experimental rigor

The existing parity dataset's local `tune` split is divided into mutually exclusive **compute** and **validation** subsets by task. The existing `test` set remains held out. Layer and alpha are selected separately for each source × modality using validation accuracy, then **frozen** for the held-out test. Raw per-example generations, correctly parsed predictions, accuracy change and per-question corrections/regressions are saved.

Caution: this experiment measures a *language-direction change* between checkpoints. Its cosine does not directly measure generic representation drift or prove a causal mechanism. Differences in norms, per-token intervention counts, and translated prompt quality are potential confounders. For publication-quality claims add random/norm-matched directions, multiple seeds and confidence intervals.

## RunPod quickstart

Prerequisites: Hugging Face access to both gated Google checkpoints and a prepared parity dataset from `paligemma2_cross_lingual_steering/parity/`. The existing parity virtualenv requires `transformers==4.51.3`. First ensure that environment and dataset are prepared using that pipeline; this experiment **does not rerun** its expensive vector grid.

```bash
cd /workspace/Multimodal_test_switch
git fetch origin parity-pipeline-20261018
git checkout parity-pipeline-20261018
hf auth login

PY=paligemma2_cross_lingual_steering/.venv-parity/bin/python
DATA=paligemma2_cross_lingual_steering/outputs/parity_cvbench

# Lightweight CPU unit tests (ordinary python with torch also works):
$PY language_icl_vlm_masks/test_ab.py

# Small GPU end-to-end pilot; creates a NEW isolated output directory.
$PY language_icl_vlm_masks/run_ab.py \
  --data "$DATA" \
  --out "$DATA/runs/kirtane_ab_pilot" \
  --lang zh --sources base,vlm --masks text,image,both \
  --layers 8,12,16 --alphas 0.5,1.0,2.0 \
  --extract-limit 12 --limit-val 3 --limit-test 3
```

After validating the GPU pilot, use a **new** output directory for the full run:

```bash
$PY language_icl_vlm_masks/run_ab.py \
  --data "$DATA" \
  --out "$DATA/runs/kirtane_ab_full" \
  --lang zh --sources base,vlm --masks text,image,both \
  --layers 8,12,16 --alphas 0.5,1.0,2.0 \
  --extract-limit 0 --limit-val 0 --limit-test 0
```

Use `--lang sw` with another `--out` directory to test Swahili. `--skip-test` generates vectors, validation, and frozen selections without reading test; a subsequent run without that switch evaluates test exactly once. Existing test files are never overwritten.

## Outputs

- `config.json`: corpus and model provenance plus all experimental options.
- `vectors_base.pt`, `vectors_vlm.pt`: matched layer-wise language directions.
- `vector_cosine_norms.csv`: per-layer cosine similarity and norm comparisons.
- `validation.csv`, `selected_validation.json`: independent selection.
- `test.csv`: raw predicted text, parsed correctness, intervention masks, edited token counts.
- `summary.csv`, `SUMMARY.md`: paired baseline vs six steering cells.

## Status

Code pushed to the experiment branch. **GPU end-to-end execution has not been performed here** because gated Google weights and an NVIDIA GPU are not available in this environment. A CPU test validates mask behavior, split integrity and vector geometry. The small GPU pilot must pass before results can be interpreted.

## Resumable full sweep (checkpoint update)

The evaluation now saves each generated prediction to `validation.partial.csv` or `test.partial.csv` and prints sample/cell progress. When all cells are present, the checkpoint is atomically renamed to `validation.csv` or `test.csv`. Restart the same command in the same output directory after an interruption; only missing cells run. Existing complete test results are verified and reused. Run `python language_icl_vlm_masks/test_checkpoint.py` for the local CPU unit test.

Do not launch two simultaneous copies of the same experiment writing to one output directory. Keep `--data`, `--layers`, and all other settings identical on a resume. If desired, keep terminal output in `run.log` with `nohup` and inspect progress with `tail -f`.
