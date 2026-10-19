# Gemma2-2B (text only) -> PaliGemma2-3B (VLM), cross-lingual Mean Shift

> **2026-10-17 — use [`parity/`](parity/README.md) for new runs.** It reproduces the upstream
> split, prompt, scoring, grid and selection rule for English and runs EN/ZH/SW on top of that
> (one click: `bash parity/run_parity_background.sh`). The pipeline below is the *legacy* protocol
> behind `steering_summary_cvbench`; its prompt, 60/150 split, letter-only scoring and single
> global (layer, alpha) differ from the paper. `parity/rescore_legacy.py` re-scores its CSVs with
> the upstream metric. Its `parse_letter` is now case-sensitive (`"a cat"` is no longer read as A);
> existing prediction CSVs are protected from being resumed with the new parser.

**Original paper parity:** Uses two separate, matched checkpoints:
- Extract: `google/gemma-2-2b` (independent **text-only** language model).
- Intervene: `google/paligemma2-3b-mix-448` (vision-language model).
- Original **unmodified** `models/paligemma_meanshift_model.py` at Woodygan/TextSteersVision commit `ce1c0f0cffed9b859901ce347f9d2b869a512178`.
- 20 upstream English anchor sentences each for counting, spatial_relationship, attribute, entity.
- Mean Shift pooled target token hidden states minus non-target token hidden states, no BOS; zero-based decoder layers.
- Unnormalized features by default (`NORMALIZED=0`), with positive additive strength `ALPHAS=0.1`; intervention defaults to original `MASK=both`.
- Uses official CV-Bench and original multiple-choice scoring target, with deterministic per-task local validation/test partition.

**Explicit multilingual adaptations (not a byte-identical replication):**
- 20 paired draft anchor sentences for EN/ZH/SW per concept, Chinese/Swahili target spans aligned by tokenizer character offsets rather than whitespace token-ID heuristics. The mathematical extraction is the same; target-token alignment differs. Translation anchors require fluent-language quality review.
- Questions and choices machine-translated by `facebook/nllb-200-distilled-600M` while preserving fixed answer choice letters; the same source image is reused for each language.
- CVBench questions are prefixed by the same `answer en` task prefix for PaliGemma2 across all three languages. This is a hypothesis about prompt compatibility; check whether Swahili outputs valid letters.
- Gemma2 and PaliGemma2 components load sequentially (not simultaneously). Separate isolated .venv and outputs.
- Reuses the already checked-in `gemma3_cross_lingual_steering/cvbench_extension/prepare.py` and `analyze.py` for exact matching multilingual test rows/metrics. Doesn't use Gemma3 model/vectors.
- No SAE/linear probe reproduction in this folder.

## RunPod, first test (new pod)

Run from repo root:
```bash
cd /workspace
git clone https://github.com/ZijianZhang22/Multimodal_test_switch.git  # omit if already cloned
cd Multimodal_test_switch
LIMIT=3 LAYERS=8 ALPHAS=0.1 bash paligemma2_cross_lingual_steering/run_background.sh
```

If repo is already at `/Multimodal_test_switch`:
```bash
cd /Multimodal_test_switch && git pull
LIMIT=3 LAYERS=8 ALPHAS=0.1 bash paligemma2_cross_lingual_steering/run_background.sh
```

```bash
cd paligemma2_cross_lingual_steering
tail -f "$(cat logs/latest_log_path.txt)"
```

Gemma2 and PaliGemma2 are license-gated Google model repos: accept BOTH licenses and log in with `hf auth login`. Use GPU with sufficient VRAM (BF16). `transformers==4.51.3` is installed within an isolated virtual environment. This pipeline has been statically reviewed but **has not yet passed a real RunPod GPU end-to-end test**.

## Full validation (leave 150 per task held out)

Use a NEW `DATA` directory so earlier smoke CSV/settings cannot conflict:
```bash
DATA=outputs/paligemma2_cvbench_full LIMIT=0 N_VAL=60 N_TEST=150 \
  LAYERS=8,16 ALPHAS=0.1 MASK=both NORMALIZED=0 \
  bash paligemma2_cross_lingual_steering/run_background.sh
```

For all remaining original CV-Bench examples as the locally held-out test, set `N_TEST=0` with another fresh DATA path. The pipeline only evaluates `SPLIT=val` unless explicitly changed.

Only after freezing the validation choice:
```bash
DATA=outputs/paligemma2_cvbench_full SPLIT=test INSTALL=0 LIMIT=0 \
  LAYERS=8,16 ALPHAS=0.1 MASK=both NORMALIZED=0 \
  bash paligemma2_cross_lingual_steering/run_background.sh
```

`ALL_FEATURES=1` evaluates all four concept steering vectors on every task (four times the task-aligned cost); with default `ALL_FEATURES=0`, count uses counting and relation/depth/distance use spatial, but all four concept vectors are extracted and compared geometrically.

Outputs at `paligemma2_cross_lingual_steering/outputs/<DATA NAME>/`:
- `corpus/`, `vectors/`, `dataset_manifest.json`, `visual_eval.jsonl`, `translation_audit.csv`
- `predictions_val.csv`, `predictions_test.csv`
- `analysis_val/` and `analysis_test/` baseline, task accuracy, vector geometry and four-task macro.

**Main limitations:** NLLB translations and SW anchors are unverified; Swahili 0% in an earlier pilot might be formatting/translation rather than incapacity. This method uses the original Mean Shift calculation, but exact prompt-language changes, per-language tokenization, translated multiple choice tasks, different dataset sizes and hyperparameter grids are intentional deviations from the original publication. Validate on English CVBench before interpreting language effect.

## GPT-authored bilingual prompt dataset (840 items; 2026-10-17)

Curated, deterministic EN/ZH/SW CV-Bench questions are checked into `curated_questions/*.jsonl.gz.b64` as four compressed plain-text shards, with 210 question records per task. Each shard contains the original English question and options, correct answer letter, and rewritten Chinese and Swahili question/option text. `curated_questions/objects.tsv` provides the 304-term object glossary used to build these translations. The translations are GPT-authored, **not** independently verified by a fluent Swahili-speaking reviewer; this is a controlled pilot dataset, not a certified benchmark translation.

* Numbers are **never translated**: Count choices are the exact Arabic numerals in the English original.
* Relation directional choices are fixed pairs: left/right = 左边/右边 = kushoto/kulia, above/below = 上方/下方 = juu/chini.
* English option letters, correct answer, ordering, split, and sample IDs remain unchanged, including F on six-option questions.
* Source text and colored-box referents are preserved in controlled task-specific templates. Terminology can still require native-speaker review.

`run.sh` now automatically seeds the curated bilingual translation cache before dataset preparation (avoids downloading NLLB on a fresh pod), and applies the curated texts to an existing 840-sample cohort. An automatic preflight checks all sample IDs, source texts, answer letters, and choices. When applying to older data, the previous `predictions_*.csv`, settings, and analysis directories are archived under `outputs/.../legacy_runs/<timestamp>/`, not silently mixed with the new prompt protocol. Stored vectors remain reusable; these are *question translations*, not feature-extraction anchor modifications.

```bash
cd /workspace/Multimodal_test_switch
git pull
cd paligemma2_cross_lingual_steering

# Check all curated 840 prompts on CPU; no downloads or GPU required:
./.venv/bin/python test_curated.py

# Smoke test with old data, using the corrected prompts; previous predictions archived
INSTALL=0 FORCE_EXTRACT=0 LIMIT=3 LAYERS=8,12,16,20 \
  ALPHAS=0.1,0.2,0.4,0.6,0.8,1.0 NORMALIZED=0 \
  bash run_background.sh
tail -f "$(cat logs/latest_log_path.txt)"
```

For the 240-row validation dataset, run `LIMIT=0` **only after** verifying that sample questions and answers parse correctly. Holdout Test should not be tuned on. Full 840 translations are *not* equivalent to official CV-Bench source language labels or official translation certification. To switch off translation auto-apply, set `CURATED_TRANSLATION=0`; avoid combining settings or results from the two translation protocols.
