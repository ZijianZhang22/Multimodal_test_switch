# MRRE independent reproduction pilot

Implementation from the paper, NOT official core code and NOT a verified reproduction of reported accuracy. Official repository inspected at commit `24fa054932efb5af0e5bd5b3ebb63e4ff97bff0d` only publishes data/evaluation scripts. No large-model GPU experiment has been run here.

Sources:
- https://aclanthology.org/2026.acl-long.1138/
- https://arxiv.org/html/2511.23231v1
- https://github.com/QimingLi-scir/MRRE

## What is implemented

English/Chinese paired reasoning generation; last non-EOS response-token decoder-block activation extraction; mean English-minus-Chinese reasoning vectors; Chinese-minus-English language-forcing prompt vectors; two-layer last-token additive intervention with original-norm restoration; baseline/reason/anchor/both/random controls. Model weights remain unchanged. Full outputs, output token counts, truncations, scores and configuration are retained.

Native VLM support: Qwen2.5-VL. LLM adapter: Hugging Face causal LMs with supported decoder paths. Other VLMs need their own image preprocessing adapter. Different model/checkpoint requires new vectors. Closed chat APIs cannot use this intervention.

## Run on a GPU pod

Unzip, enter this folder, use a clean Python 3.11/3.12 environment. Install an appropriate CUDA PyTorch build on the pod before installing requirements. The CPU torch used for local checks is NOT bundled.

Download MathVerse images from https://huggingface.co/datasets/AI4Math/MathVerse . Set IMAGE_ROOT to the extracted parent of `images_version_1-4` (verify paths against `data/calibration.jsonl`). Images and model weights are not bundled.

```bash
export IMAGE_ROOT=/workspace/datasets/MathVerse
bash run_pod.sh
```

This downloads the pinned upstream metadata, prepares 100 calibration / 30 validation / 100 test question groups, runs implementation checks, extracts vectors, then runs validation. It deliberately does not automatically spend GPU time on a large hyperparameter sweep. The initial layer indices 22 and 24 and strengths 1 and 0.75 are starting choices, not validated optima for your checkpoint. All indices are ZERO-BASED decoder blocks.

After validation, freeze your chosen settings and run:

```bash
python mrre.py evaluate --model Qwen/Qwen2.5-VL-7B-Instruct \
  --data data/test.jsonl --image-root "$IMAGE_ROOT" \
  --vectors results/vectors.pt --reason-layer 22 --anchor-layer 24 \
  --alpha-reason 1 --alpha-anchor 0.75 --out results/test.jsonl
python mrre.py summarize --out results/test.jsonl
```

Use a fresh output name per evaluation; output overwrite is refused. Calibration vectors/traces are overwritten by extract, so archive results or change --out before repeating extraction. There is no automatic resume yet. Errors stop execution rather than silently dropping evaluation questions.

For a quick end-to-end GPU smoke run, prepare 4/2/2 groups into `smoke_data`, then pass its calibration and validation files and separate output names to extract/evaluate. Default 1024 output tokens may truncate difficult questions; inspect logs and increase `--max-new-tokens` consistently before a proper experiment.

## Data and measurement choices

- One first available translated variant per shuffled original problem group; all variants of a problem stay in the same split. 788/3940 records in the pinned Chinese MathVerse JSON lack a nonempty zh_question and are skipped. Thus this pilot is a translated subset, not all paper conditions.
- Default greedy decoding improves comparability but differs from the paper's sampled decoding.
- Explicit bilingual prompt requests `Final: answer`. This differs from the paper's boxed-answer formatting.
- Extract vectors at the final NON-EOS generated token; paper's wording leaves exact handling of terminal tokens/implementation unclear. Entire prompt+response is re-forwarded while retaining the image inputs. Both sides must finish without truncation. No correctness-based filtering.
- Store raw mean-difference vectors. The paper uses a tilde on intervention vectors without a fully explicit separate scaling definition in the inspected equations; this implementation adds raw vectors then restores activation norm. Treat strength calibration as essential.
- Anchor uses one fixed pair of language-forcing instructions, within chat templates and without an image. Template and token-position choices are documented reproduction choices.
- Eager attention, batch size one, no quantization, capped image resolution. This is a compatibility-first implementation, not optimized throughput. The cap must match calibration and evaluation.
- Deterministic final-answer scoring handles simple option letters, numbers and exact strings. Missing/unparseable answers count as failures in reported accuracy and are separately counted. This is NOT official benchmark equivalence scoring or the paper's LLM judge. Review raw outputs before drawing conclusions.
- Language consistency is NOT automatically scored. Annotate output language separately; script does not confuse Chinese-character frequency with language fidelity.
- Output tokens are tokenizer-dependent and are not a clean measure of reasoning efficiency across languages.
- Calibration/validation/test separation prevents same-question leakage; it does not establish broad task or language generalization.

## Validation status

`python -m unittest -v test_core.py` uses a randomly initialized tiny Qwen2 decoder on CPU: verifies norm preservation, last-position-only changes, real decoder intervention, cached generation-step compatibility, hook removal (including exceptions), disjoint data groups and conservative scoring. This establishes implementation behavior, NOT semantic gains or Qwen2.5-VL image-pipeline correctness. Full VLM execution and accuracy reproduction require the GPU run.

## Next experiment

First compare baseline vs both on validation; use reason/anchor/random controls to interpret changes. Then test frozen settings on held-out questions. Keep an English baseline separately (`--language en --conditions baseline`). For visual dependence, add matched image/no-image/caption conditions as a separate extension rather than claiming this pilot already tests them.

## RunPod one-click entry (recommended)

Use a RunPod PyTorch CUDA GPU template. Default target is Qwen2.5-VL-7B; a 48 GB GPU and at least 40 GiB free disk are recommended starting resources, not measured peak requirements. CPU offload can make smaller GPUs slow. CUDA visibility is checked before downloading assets.

```bash
cd /workspace/Multimodal_test_switch/mrre_reproduction
bash runpod_oneclick.sh
```

The entry point creates `/workspace/mrre-venv` with access to the template's installed PyTorch, installs remaining requirements, downloads the pinned upstream question metadata, downloads/extracts MathVerse images and model weights, verifies images, runs CPU implementation checks, then runs a real GPU smoke experiment (4 calibration + 2 validation) and the pilot (100 calibration + 30 validation). It preserves `/workspace/hf-cache` and `/workspace/mrre-runs/qwen25vl7b-zh`. Use a persistent RunPod volume mounted at `/workspace` if results must survive pod deletion.

For only the initial GPU smoke experiment:

```bash
bash runpod_oneclick.sh --smoke-only
```

Then run without `--smoke-only` to reuse its completed stages and continue. Completed-stage reuse requires the same configuration and present outputs; an interrupted stage restarts from its beginning after preserving partial outputs. This is stage-level recovery, not per-example resume. Model downloads use Hugging Face's cache. Do not run two copies against the same run directory concurrently.

Optional existing images: `--image-root /workspace/datasets/MathVerse`. Optional alternate experiment: `--run-dir /workspace/mrre-runs/experiment2`. Changing token budget, layer settings, or model requires a new run directory to avoid mixing results. Model overrides must still be Qwen2.5-VL-compatible and layer indices must fit the checkpoint.

Inspect `smoke_validation.jsonl`, `pilot_validation.jsonl`, `.log` files and `run_config.json` in the run directory. Default generation cap is 2048; all truncated calibration pairs are excluded. Unparseable evaluation answers remain in the accuracy denominator as failures. The script prints unparsed/truncated counts and retains full responses. It does not claim official benchmark scores or measure language fidelity automatically.

After freezing your settings using validation, run the generated `run_test_after_validation.sh` in the run directory for the held-out 100-question test. It is intentionally not auto-executed.

Validation performed during development: shell syntax, CLI, CPU decoder tests, ZIP traversal rejection and image-root discovery. The GPU download/inference workflow remains unexecuted in the development environment (no GPU).
