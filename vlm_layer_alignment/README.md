# VLM-native multilingual alignment: Frozen Rosetta vs LoRA

**目的：**独立测试两条能直接改变 PaliGemma2 视觉问答行为的路线，不需要将原来的 E5/CLIP 映射与 Steering 混合：

- **A / Frozen Rosetta-inspired alignment (zero VLM weight updates).** Freeze `google/paligemma2-3b-mix-448`. Extract image-conditioned *last prompt text token* states from the same decoder layer for matched English and Chinese/Swahili CV-Bench prompts. Fit a low-rank shared PCA and orthogonal mapping with **25 paired image/questions**. The Rosetta arm also uses *unpaired*, image-disjoint source and target activation distributions, with batched Hungarian pseudo-coupling and Procrustes updates. At generation time, a forward hook applies a norm-clamped low-rank correction **only to the prompt prefill token**, not generated tokens. Compare both **supervised-only Procrustes** and **supervised + unpaired Rosetta-style** methods.
- **B / LoRA alignment.** Freeze the vision tower, multimodal projector and base decoder. Update LoRA on the **selected middle decoder layers**' `q_proj`/`v_proj` only. Train on the **same 25 training image/question IDs** as A, using answer-label cross-entropy plus a cosine matching loss between the source-language prompt hidden state and the frozen base VLM's English prompt hidden state **for the same image**. Compare to **answer-only LoRA with alignment weight zero**.

Both routes are separately evaluated with **held-out VLM generation accuracy / invalid-answer rate**, not CLIP R@K. These are new VLM-native methods inspired by Rosetta; **they are not the published Rosetta code or direct reuse of `visual_anchor`'s 128D E5→CLIP W**.

## Middle-layer motivation

Clément Dumas et al., *Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers*, ACL 2025: https://aclanthology.org/2025.acl-long.1536/ . In their **text-only translation models**, language and concept information emerge at different layers. They did **not** establish the best layers for PaliGemma2 multilingual VQA. Our candidate decoder layers default to **8, 13, 18 (0-indexed)** and are tested/selected on local validation; they are **not** claimed to reproduce the paper's specific layer transitions.

## Full RunPod setup (single command, no smoke test)

Assumes this repository (new branch) is checked out and a CUDA GPU is available. Some Google checkpoints are gated; accept the model terms for `google/paligemma2-3b-mix-448` on Hugging Face and authenticate via `hf auth login` before running. Use enough GPU VRAM for full multimodal 448-resolution forward/backward passes (A100 80 GB recommended for training). Run:

```bash
cd /workspace/Multimodal_test_switch
# git fetch origin && git switch experiment/vlm-native-rosetta-vs-lora
# git pull origin experiment/vlm-native-rosetta-vs-lora

LANGUAGE=zh PAIR_BUDGET=25 LAYERS=8,13,18 ALIGN_LAYER=13 \
  RUN_FROZEN=1 RUN_LORA_CONTROL=1 RUN_LORA=1 \
  bash vlm_layer_alignment/run_runpod.sh
```

Run under `tmux` to survive terminal disconnects. The launcher creates isolated `vlm_layer_alignment/.venv` and installs `requirements.txt` by default. `INSTALL=0` skips installation on subsequent runs. The full run intentionally includes the standard LoRA control; set `RUN_LORA_CONTROL=0` to omit it. For Swahili use `LANGUAGE=sw` **and an independent `OUT=...` if the data or hyperparameters change**. To explore only A, set `RUN_LORA=0 RUN_LORA_CONTROL=0`. To explore only B, set `RUN_FROZEN=0`. Use `SEED=43` with a **different `OUT` directory** to repeat another seed (no same-result-path mixing).

### Data and leakage protocol

The launcher **copies** the existing prepared `paligemma2_cross_lingual_steering/outputs/paligemma2_cvbench_full/visual_eval.jsonl` into a separate `outputs/source_cvbench840` staging folder if present; it does **not mutate or archive the original steering dataset/predictions**. The images folder is symlinked from the existing dataset when available. If missing, it builds the **same 840-row curated, translated local CV-Bench cohort** from this repository's previously stored translations and the `nyu-visionx/CV-Bench` public test split. This requires access to the public dataset and upstream taxonomy code. No API-based translation is requested. Data preparation refuses a nonmatching source cohort. The `outputs/source_cvbench840` working copy owns any curated-prompt changes.

For these 840 rows, existing local `val` (60/task) is split into **train (48/task) and new validation (12/task)**, while existing local `test` (150/task, 600 total) remains held out. Data selection is deterministic and checked for image-ID overlap. **CV-Bench has one public test split**: this is a **local pilot training/validation/test partition**, *not* an official training split or fresh independent test benchmark. The GPT-authored Chinese/Swahili translations from the steering project are **not native-speaker verified**; report this limitation. If these examples have already been inspected/tuned in prior work, the local holdout is **not globally untouched**; external test validation is required for publication.

The 25 supervised anchor IDs are the same for Frozen and both LoRA arms, using a fixed random seed. Only A gets **additional unlabeled training-side** examples, split into image-disjoint source and English pools without exposing paired correspondences. English language teacher activations for LoRA come from the **unadapted PaliGemma2**, not an external English answer oracle. Validation alone selects the frozen map layer, method, and intervention alpha, and LoRA's best epoch. The fixed test split is used once for reporting; **never change settings after looking at test results**.

### Defaults and outputs

```text
Language             zh (or sw)
Pair budget           25
Candidate layers      8,13,18 (0-indexed decoder layers)
Frozen PCA            16D (few pairs, avoids rank-deficient 128D supervised fit)
Frozen alpha grid     0,0.15,0.3,0.6  [alpha 0 is original VLM]
Frozen norm cap       15% of original prompt token norm per unit strength
Unpaired OT rounds    20; pseudo-coupling weight 0.3
LoRA rank             8; q_proj/v_proj at selected layers
LoRA alignment weight 0.3; CE weight 1.0; epochs 1; grad accum 4
Decode                greedy; max_new_tokens 8; answer letter A–F
```

```text
vlm_layer_alignment/outputs/
  data/{train,val,test}.jsonl
  data/manifest.json
  frozen_zh/activation_cache/*.npz
  frozen_zh/mappings/layer{8,13,18}_{supervised,rosetta}.npz
  frozen_zh/val_grid.json
  frozen_zh/selection_by_method.json
  frozen_zh/test_baseline_{en,zh}.csv
  frozen_zh/test_layer*_*.csv
  lora_ce_zh/{best_adapter/,best_val.json,test_baseline_*.csv,test_lora_*.csv}
  lora_alignment_zh/{best_adapter/,best_val.json,test_baseline_*.csv,test_lora_*.csv}
  vlm_alignment_comparison_zh.csv
```

The `comparison_zh.csv` contains accuracy before/after, accuracy gain (percentage points), paired image bootstrap CIs, exact McNemar p-value, invalid response counts. Report multiple comparisons and seed limitations. Other raw prediction CSVs include task IDs, language, answer letters, parsed outputs and per-question correctness.

## Manual entry points

```bash
cd vlm_layer_alignment
. .venv/bin/activate
python -m unittest -v test_cpu                   # CPU tests, no model downloads
python protocol.py --source /path/visual_eval.jsonl --out outputs/data
python frozen.py --data outputs/data --out outputs/frozen_zh --language zh --pair-budget 25 --stage all
python lora.py --data outputs/data --out outputs/lora_ce_zh --language zh --pair-budget 25 --align-weight 0
python lora.py --data outputs/data --out outputs/lora_alignment_zh --language zh --pair-budget 25 --align-weight 0.3
python summarize.py --root outputs --language zh
```

### Scientific interpretation and caveats

1. **Do not call this official Rosetta replication.** The unpaired step uses Euclidean/cosine Hungarian pseudo-coupling plus polar Procrustes in image-conditioned VLM activations. There is no separately extracted image-target Z pool here, no Sinkhorn regularizer and no assurance of the original paper's objective equivalence.
2. **Do not assume middle layers are language-only.** We probe them. Interventions may alter reasoning, option-format, and image reliance, and can degrade English performance.
3. **Matched labels and capacity matter.** LoRA CE only is essential. The frozen route uses 25 labeled prompts plus unlabeled data, while LoRA gets the same 25 labels and its aligned English equivalents as auxiliary supervision; these objectives are not perfectly supervision-equivalent. Distinguish them.
4. **Small labeled budget + reused public test split:** accuracy fluctuations and overfitting are likely. Rerun multiple independent seeds, correct for multiple testing and add an external vision-language dataset before making research claims.
5. The **model weights are not shipped** in the package. No GPU end-to-end results are claimed here. We verified offline synthetic numerical and protocol tests; GPU dependency/version-specific failures should be checked against actual RunPod logs.