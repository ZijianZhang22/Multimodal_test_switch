# Multimodal Test Switch

[中文](README.md) | English

This repository studies how the form and position of visual and textual evidence, as well as additional tasks, affect VLM answer accuracy and generation cost. Internal representation analysis is used to investigate possible mechanisms.

The experiments connect three lines of investigation: **controlled toy tasks → real benchmarks → mechanism diagnostics**. Additional-task and compression experiments constitute a separate intervention and should be analyzed separately.

## Where to start

| Goal | Entry point | Experimental design |
|---|---|---|
| Run 100 toy problems with CoT: TTTT vs. 1T3V | [toy_cot_100](toy_cot_100/README.md) | Move the only text fact across the four logical steps of a matched spatial chain |
| Run 500 toy problems: TTTT vs. 3T1V | [run_3t1v_500.sh](scripts/launch/run_3t1v_500.sh) | Move the only visual fact; this is a different design from 1T3V |
| Separate logical role, presentation order, and single-fact perception | [Experiment index](docs/EXPERIMENTS.md) | Factorial, role-position, and perception tools under scripts/ |
| Replace images with captions in real multi-image tasks | [Mantis](mantis_caption_swap/README.md), [MMQA](mmqa_four_image/README.md), [MMRB](mmrb_caption_first4/README.md) | Intervene on physical image slots, which are not necessarily reasoning steps |
| Measure additional-task effects and token usage | [Cross-task pilot](vlm_cross_task_pilot/README.md), [MME-CoT](mme_cot_conpress_pilot/README.md) | Joint tasks, task order, and an appended simple task |
| Analyze layers, probes, and activation patching | [Mechanism analysis](docs/MECHANISM.md) | Current tools mainly examine the language decoder; they do not directly establish encoder information loss |

**Notation:** V denotes an original image. In the toy experiments, T denotes a structured textual fact; in the benchmark experiments, T denotes a generated caption. A four-character schedule specifies four logical evidence steps in the toy, or four image slots in the relevant benchmark design.

Mantis evaluates All-V, All-T, and one-image-at-a-time caption replacements for variable image counts. MMRB intervenes only on the first four images; later images remain visual, so its TTTT condition is not necessarily an entirely textual input.

## Documentation

1. [Experiment index](docs/EXPERIMENTS.md): conditions, datasets, commands, and metrics.
2. [Mechanism analysis](docs/MECHANISM.md): what the tools observe and which interpretations they support.
3. [Result status and comparison guidelines](docs/STATUS.md): validation coverage and evidence still needed.
4. [Branch provenance](docs/BRANCHES.md): experiment snapshots consolidated into main.
5. [Script migration table](docs/SCRIPT_PATHS.md): old and new paths and updated commands.

These overview documents are currently in Chinese; individual experiment READMEs provide English instructions.

[RESEARCH_PLAN.md](RESEARCH_PLAN.md) preserves the historical research plan. The previous round-by-round README is archived [here](docs/legacy/README_rounds.md). Historical numerical claims have not been independently revalidated during repository organization.

## Quick start: 100-problem toy CoT experiment

```bash
git clone https://github.com/ZijianZhang22/Multimodal_test_switch.git
cd Multimodal_test_switch
pip install -r requirements.txt
bash toy_cot_100/run_runpod.sh
```

The default model is Qwen3-VL-8B-Instruct. The experiment runs 100 latent problems under five matched conditions—**TTTT, TVVV, VTVV, VVTV, VVVT**—for 500 generations, with a default limit of 4096 output tokens per generation.

Follow each experiment's README for dependencies and GPU requirements. The root requirements file does not replace benchmark-specific setup.

Use a fresh results directory when changing the model, prompt, token budget, or conditions. Do not run concurrent writers against the same manifest or predictions file.

## Repository layout

| Directory | Contents |
|---|---|
| scripts/data/ | Toy dataset generators |
| scripts/inference/ | Qwen and InternVL inference runners |
| scripts/analysis/ | Accuracy, position effects, token usage, and statistical analysis |
| scripts/mechanism/ | Hidden-state extraction, probes, and activation patching |
| scripts/launch/ | Shell launchers that locate the repository root |
| toy_cot_100/ | Matched 100-problem CoT experiment |
| mantis_caption_swap/ | Mantis-Eval caption substitution |
| mmqa_four_image/ | MultiModalQA four-image experiment |
| mmrb_caption_first4/ | MMRB first-four-image substitution |
| vlm_cross_task_pilot/ | Joint-task and task-order controls |
| mme_cot_conpress_pilot/ | MME-CoT additional-task and compression experiments |
| docs/ | Experiment documentation and historical records |

Run Python tools as modules from the repository root:

```bash
python -m scripts.inference.run_qwen3vl --help
bash scripts/launch/run_3t1v_500.sh
```

Root-level Python and shell files have moved into scripts/. Data and result directory locations are unchanged. Update commands from older notes using the [migration table](docs/SCRIPT_PATHS.md); reorganizing code does not require rerunning existing GPU experiments.

The directory name mme_cot_conpress_pilot retains its original spelling for compatibility.

## Interpreting results

- **Structured text and generated captions are different interventions.** Captions can omit task-critical information; toy and benchmark results should not be pooled without accounting for this difference.
- **Output tokens, parsed thinking tokens, and visual input tokens are distinct metrics.** Visible CoT from an Instruct model is not the same measurement as the thinking segment of a Thinking model.
- **Fewer tokens alone do not establish better efficiency.** Report accuracy, truncation, and matched comparisons together. Total output from a joint task does not isolate the original task's reasoning cost. End-to-end efficiency must also account for caption generation.
- **Probes measure readability, not necessarily causal use.** The current experiments do not establish a single shared mechanism, a pure modality-switching cost, or visual encoder information loss.

## Validation status

Repository reorganization was checked with Python and shell syntax validation, local import-target and documentation-link checks, deterministic toy generation comparisons, benchmark offline tests, and launcher dispatch checks.

No new GPU inference was run as part of that work. The organization environment lacked torch, so model loading was not verified. Code availability should not be interpreted as confirmation that a complete benchmark run has finished.
