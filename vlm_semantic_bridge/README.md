# English Semantic Retrieval Bridge — CLIP Rosetta3 -> PaliGemma2

**Hypothesis:** Does a Chinese question benefit from English *semantic cues* retrieved using a mapping fitted with unpaired English/Chinese captions and visual data? This pilot tests the answer using frozen `google/paligemma2-3b-mix-448` on an image-disjoint, bilingual subset of public **MMBench Dev**. It does not claim improved VLM accuracy until VQA runs finish.

## What is connected

```
Chinese MMBench question (NO paired English question, NO answer label)
  -> frozen Chinese text2vec encoder [pinned revision]
  -> SAME train-fitted Chinese PCA as CLIP Rosetta3
  -> centered + normalized source -> Rosetta3 saved W (one of four map arms)
  -> cosine search over SAME Flickr8K *training-image bank* English captions
     [frozen English CLIP embeddings, same PCA, same checkpoint target mean]
  -> 1-2 retrieved descriptions from OTHER images, marked as *language hints only*
  -> original input image + unchanged Chinese MCQ + hints -> frozen PaliGemma2
  -> answer letter; paired accuracy, repairs/harms, image-cluster bootstrap
```

This is **not decoder OT steering**, not training a new image-caption system, and not retrieving the gold English MMBench question. The CLIP-native `joint` Rosetta3 map was trained with image features; the `text_only` map is identical except for the visual term. The retriever uses exactly the same English caption bank for all map conditions.

### Controls

| Arm | What enters the PaliGemma2 prompt | Interpretation |
|---|---|---|
| `baseline_zh` | Image + original Chinese question | Main baseline |
| `baseline_en` | Image + OFFICIAL English translated question | **English upper reference only**, not a deployable no-parallel-data method |
| `pairs_only` | English snippets retrieved using only 50 paired anchors | Paired-only baseline |
| `text_only` | English snippets retrieved using 50 paired anchors + unpaired text | Textual Rosetta baseline |
| `joint` | English snippets retrieved using *same anchors* + unpaired text + CLIP-image pool | **Visual Rosetta method** |
| `joint_caption` (optional) | Same as `joint`, but third pool is bank's English captions | Matched semantic-content / different-modality control; requires prior checkpoint |
| `random` | Same number of randomly selected snippets from same bank | Tests added-context effects |

**Important:** CLIP was already pretrained using image-text pairs. `joint`'s vision contribution is an *extra training objective* over that pretrained space, not proof of an independently learned image grounding capability. The bank is Flickr8K caption domain whereas MMBench is question/answer domain, so poor transfer is a meaningful possible outcome. Added snippets are from **other** images and MUST NOT be treated as evidence about the current input image; the prompt explicitly cautions the model.

## One-click, reusing your existing RunPod files

First ensure you have *previously run* native CLIP Rosetta3 at 50 pairs, e.g.:

```bash
cd /workspace/Multimodal_test_switch
# Only when original feature files or map checkpoints do not exist:
cd visual_anchor
BUDGETS='50' SEEDS='0' METHODS='pairs_only text_only joint' bash run_clip_native.sh
cd ..
```

In the repository root:

```bash
# Low-cost local-tune 30-question VQA pilot (requires CUDA for PaliGemma2):
bash vlm_semantic_bridge/run_oneclick.sh
```

Default paths:

```
CLIP_WORK=$ROOT/visual_anchor/real_work_clip_native
MAP_DIR=$CLIP_WORK/results_clip_shared_d128_source_pool_bank_budget_50
MMBENCH_DATA=/workspace/vlm_A_zh_seed42/mmbench_dev_bilingual
OUT=$ROOT/vlm_semantic_bridge/outputs/tune_30_b50_seed0
```

The launcher prepares the *official bilingual MMBench Dev* locally if needed. Use the exact original Rosetta training features and checkpoint; the code rejects incorrect PCA modes, non-CLIP bridges, anchor mismatch, map dimension mismatch, model provenance mismatch, image-bank/evaluation image overlap, mismatched datasets and reusing output with changed settings.

Useful examples:

```bash
# Inspect retrieved English prompts BEFORE running the VLM (uses CPU, and may download the pinned Chinese encoder):
STAGE=prepare bash vlm_semantic_bridge/run_oneclick.sh
sed -n '1,5p' vlm_semantic_bridge/outputs/tune_30_b50_seed0/prompts.jsonl

# Run the frozen VLM on the cached prompts and resume completed items:
STAGE=eval bash vlm_semantic_bridge/run_oneclick.sh

# For matched caption control if saved as joint_caption_0.npz in the same map dir:
ARMS='baseline_zh baseline_en pairs_only text_only joint joint_caption random' \
  OUT=/workspace/bridge_tune_caption bash vlm_semantic_bridge/run_oneclick.sh

# After protocol decisions are frozen using the tune split, evaluate ALL held-out questions.
# Note: MMBench Dev is public; this is a LOCAL held-out comparison, not leaderboard Test.
SPLIT=test LIMIT=0 OUT=/workspace/bridge_mmbench_test bash vlm_semantic_bridge/run_oneclick.sh

# Point to a different complete Rosetta3 run, including one saved with joint PCA/different dimension:
MAP_DIR=/workspace/custom/results_clip_joint_d64_source_pool_bank_budget_50 \
CLIP_WORK=/workspace/custom OUT=/workspace/bridge_custom bash vlm_semantic_bridge/run_oneclick.sh
```

If the original 50-pair runs used a different filename or protocol, **set `MAP_DIR` explicitly** and keep it unchanged across arms. `--stage prepare` requires the sentence encoder; `--stage eval` loads PaliGemma2. The mapping never uses MMBench labels, questions' paired English translations, or test images during fit. The English reference arm intentionally uses its translation, and is not included in the low-pair training comparison.

## Output

```
OUT/config.json                   # immutable dataset/map/checkpoint hashes, exact IDs, settings
OUT/prompts.jsonl                # per-arm exact prompt, retrieved bank IDs/captions/scores
OUT/predictions/{arm}.jsonl      # restartable per-example generated option + correct/valid
OUT/summary.csv / summary.json   # VQA accuracy and paired deltas vs Chinese baseline
```

The default uses 30 **tune** questions for a plumbing check. **Never call the tune result an independent improvement**. Use a separate untouched test split and image-cluster bootstrap intervals. All runs are paired on identical images and question IDs. Different models, map seeds, budgets, top-K, prompt templates, or subsets require a NEW OUT directory.

The script's current retrieval is **cosine**, not the CSLS used in the earlier Rosetta3 retrieval R@1 tables. There is no guarantee the 36.72% original held-out caption retrieval transfers to out-of-domain MMBench questions. The fixed image in PaliGemma2 remains the actual MMBench input image; retrieved English captions are *not* descriptions generated from that image. An effective use of the current image during retrieval would be a separate, image-conditioned reranking extension.

### Scientific limitations

- The current code uses **Chinese** as the target language to reuse your pinned `text2vec-base-chinese`. Swahili requires a separately trained Rosetta map and corresponding frozen language encoder; switching the text string without retraining is invalid.
- PaliGemma2 mix-448 outputs can be sensitive to prompt wording; an incorrect retrieved caption may *reduce* accuracy. Inspect `prompts.jsonl` for answer cues/hallucinations, and compare `random`, `text_only`, and optionally `joint_caption`.
- Flickr8K training captions (objects and activities) are often not close to MMBench relational questions. This is a hard cross-domain transfer pilot, **not** a test of true question translation or high-quality RAG.
- CLIP's pretrained data may include Flickr8K images. We verify only exact pixel duplicates of the bank against the locally evaluated images, not near duplicates or model pretraining overlap.
- `baseline_en` is an English reference; giving the actual gold English MMBench question to an English teacher is a separate supervised-input upper bound, not a valid low-resource bridge control.
- Uses simple L2-normalized cosine retrieval and a deterministic `max_new_tokens=8` multiple-choice letter parser; no model training, no VQA adapter and no claim of SOTA.

## CPU tests

```bash
python -m unittest discover -s vlm_semantic_bridge/tests -v
```

The tests validate mapping conventions, retrieval controls and failure paths. They do not require CLIP/PaliGemma2 weights; full RunPod GPU generation is not executed in the local development environment.