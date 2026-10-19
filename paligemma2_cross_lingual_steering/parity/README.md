# Upstream-aligned ("parity") pipeline: Gemma2-2B → PaliGemma2-3B, EN/ZH/SW

This folder re-implements the CV-Bench evaluation so that the **English condition is
the protocol of Gan et al. (ACL 2026) / `Woodygan/TextSteersVision@ce1c0f0`**, and the
cross-lingual extension differs from it only where a language must differ.
The legacy pipeline in `..` (results in `steering_summary_cvbench`) is kept unchanged
except for one parser fix; use `rescore_legacy.py` to put its CSVs on the upstream metric.

## One click

```bash
cd /workspace/Multimodal_test_switch && git pull
hf auth login          # accept google/gemma-2-2b and google/paligemma2-3b-mix-448 licenses first

# smoke test: 3 samples per task, writes to outputs/parity_cvbench/runs/smoke_3
LIMIT=3 bash paligemma2_cross_lingual_steering/parity/run_parity.sh

# full run, detached
bash paligemma2_cross_lingual_steering/parity/run_parity_background.sh
tail -f "$(cat paligemma2_cross_lingual_steering/parity/logs/latest_log_path.txt)"
```

Result: `outputs/parity_cvbench/runs/main/analysis/SUMMARY.md` (+ CSVs).

## What is now identical to upstream

| Item | Upstream | Legacy pipeline | Parity pipeline |
| --- | --- | --- | --- |
| Split | pandas `sample(frac=1, random_state=42)`, test = last 150/task, tune on the rest | own shuffle, 60 val / 150 test | upstream code path (`common.upstream_split`, tested) |
| EN prompt | `answer en ` + instruction + CV-Bench `prompt` | instruction duplicated, no "Select from the following choices." | upstream, byte-identical |
| Scoring | `normalize_answer` (letter **or exact choice text**) | letter-prefix regex; `"a cat"`→A, `"2"`/`"left"`→wrong | `normalize_answer` verbatim (tested against upstream source) |
| Grid | layers 5–20 × α {0.1,0.2,0.4,0.6,0.8,1.0} | 5 layers × 5 α | same as upstream |
| Selection | best per (task, concept), ties → min α | one global (L8, α=1.0) | upstream rule (`select_best.py`) |
| Concepts | all 4 taxonomies per task | fixed count→counting, rest→spatial | all 4; per task the best on EN tuning (`CONCEPT_MODE=aligned` for legacy mapping) |
| EN extraction | `mean_shift_probing`, fp32 | offset re-implementation, bf16 | the upstream function itself (AST-extracted from the pinned file), fp32 |
| Model load | bf16, default attention | bf16, forced `sdpa` | bf16, default attention |
| Steering | `PaliGemmaMeanShiftManipulator`, mask=both, prefill only | same | same (unmodified author file) |

## Deliberate, documented differences

* **Tuning subset.** `N_TUNE=150` per task (first 150 rows of the upstream train split).
  Upstream tunes on the whole train split; set `N_TUNE=0` for exact parity (≈3.4× the Stage-A cost).
* **ZH/SW prompts.** `native` style: `answer zh|sw` + translated instruction + translated CV-Bench
  prompt (no more English instruction glued in front). `STYLES=native,upstream_en` also runs the
  legacy mixed-language prompt as an ablation. Translation is template + lexicon; it reproduces all
  840 GPT-curated rows exactly (`test_parity.py`). Objects outside the curated lexicon use
  `glossary_extra.tsv`, then NLLB *for that term only*; everything machine-translated is listed in
  `DATA/missing_terms.txt` and `translation_audit.csv`. Nothing is native-speaker verified yet.
* **ZH/SW scoring** applies `normalize_answer` with the prompt-language choices, then the English
  choices; `答案：`/`jibu:` are stripped like upstream strips `answer:`.
* **ZH/SW extraction** uses the same pooling with tokenizer-offset target alignment. EN is extracted
  both ways and the per-layer cosine is written to `vectors/en_extraction_sanity.csv`.

## Stages and cost (generations; baselines extra)

| Stage | What | Default size |
| --- | --- | --- |
| A-tune | EN×EN, 4 tasks × 4 concepts × 16 layers × 6 α on N_TUNE/task | 600 × 384 = 230,400 |
| A-test | EN×EN, best (L, α) per task × concept on the 600 test images | 2,400 |
| B-test | 3 prompt langs × {en, zh, sw, zh_normen, sw_normen, random_s0}, Stage-A config per task | 10,800 |
| C (opt.) | `RETUNE_PER_LANG=1`: tune L/α per prompt × vector language | 600 × 9 × 96 ≈ 518k — narrow `C_LAYERS`/`C_ALPHAS` |

Stage B freezes concept/layer/α from the English tuning, so language comparisons carry no
per-language selection bias; `*_normen` rescales a ZH/SW vector to the EN norm (direction vs.
magnitude), `random_s0` is a random direction with the EN norm. Every run is resumable.

## Knobs

`DATA, N_TUNE, LAYERS, ALPHAS, CONCEPTS, CONCEPT_MODE, VARIANTS, STYLES, RANDOM_SEEDS,
RETUNE_PER_LANG, C_LAYERS, C_ALPHAS, LIMIT, RUN, INSTALL, STRICT_LEXICON` — see `run_parity.sh`.
Changing `N_TUNE` needs a new `DATA`; changing grids/variants needs a new `RUN`.

## Legacy CSVs on the upstream metric (CPU, no download)

```bash
python paligemma2_cross_lingual_steering/parity/rescore_legacy.py --pred /path/to/predictions_test.csv
```

Prints legacy vs upstream-scored accuracy, gain, image-clustered CI and invalid counts per
(prompt, vector) cell.

## Status

CPU tests (`test_parity.py`) and a synthetic end-to-end of plan → select → analyze pass.
The GPU stages have **not** been executed by the author of this folder; run the `LIMIT=3`
smoke test first.
