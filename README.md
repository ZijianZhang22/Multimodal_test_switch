# Multimodal reasoning experiments

This repository separates the earlier multimodal experiments from the MRRE reproduction pilot.

| Directory | Contents |
| --- | --- |
| [previous_experiments/](previous_experiments/) | All repository contents before this reorganization: toy switching, caption replacement, Mantis, compression, benchmark pilots, scripts and research notes. |
| [mrre_reproduction/](mrre_reproduction/) | Independent MRRE-style vector extraction, decoder intervention, controls and evaluation pilot. |

## Earlier experiments

Run previous commands from the preserved project root so relative paths retain their original meaning:

```bash
cd previous_experiments
```

See the original [English README](previous_experiments/README_EN.md) and [original README](previous_experiments/README.md). Existing command paths assume this working directory. External absolute paths in old configuration files still need to match your machine.

The previous contents are preserved byte-for-byte from commit `c5069df3e0ecd0fcde1875d09674b436dd4cb6e7`; only their enclosing directory changed. Git history remains available.

## MRRE pilot

```bash
cd mrre_reproduction
export IMAGE_ROOT=/workspace/datasets/MathVerse
bash run_pod.sh
```

Download and extract MathVerse images first; IMAGE_ROOT must contain the paths referenced by the dataset, e.g. `images_version_1-4`. See [setup and methodological differences](mrre_reproduction/README.md).

Status: CPU implementation tests passed on a tiny randomly initialized Qwen2 decoder. Full Qwen2.5-VL GPU execution and paper accuracy reproduction have NOT been completed. This is an independent implementation, not the authors' unpublished core code.
