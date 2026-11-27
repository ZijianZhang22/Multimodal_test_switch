# Bilingual MMBench A: tune on one part, test on disjoint images

**Research protocol, not an official MMBench leaderboard result.** Official MMBench v1.0 Dev EN and verified CN translations (1,164 entries each) are downloaded from official VLMEvalKit URLs and checked using the published MD5s. The **public DEV** is randomly split **by decoded image hash**, holding ~200 questions for local parameter selection; all remaining images and associated questions go into a local held-out test. This is not the official secret TEST split or its CircularEval scoring.

## What this experiment answers

Do representations learned **without paired cross-lingual training samples** improve Chinese visual question answering on a new benchmark? How much of the gain is specific to visual assistance?

- Model: frozen `google/paligemma2-3b-mix-448`.
- Frozen original Flickr8K-fitted maps: `text_visual_ot` and `text_ot`; **no OT refitting and no LoRA**.
- Tune parameters **only on MMBench-Dev local tuning subset**: decoder layers `8,13,18`; alphas `0.15,0.3,0.6`, selected independently per method. This grid can be customized but must be frozen before viewing test results.
- Final test: Chinese baseline, English reference, selected Visual OT and selected Text-only OT. Test accuracy is not used to choose parameters.
- Same image and same question prompt for the Chinese baseline and both intervention methods. The official optional `hint` field is included (as in VLMEvalKit). Chinese/English prompts use the corresponding *official* bilingual multiple-choice questions. Correct answer and option letters are verified to agree.
- Parser reports **valid option rate**, accuracy, correct-item repairs/harms, cluster-bootstrap 95% CI, image-cluster sign-flip p-value, and category results. Ordinary McNemar is reported only when each question has a distinct image.
- Per-condition JSONL is append-only and resumable; a truncated final JSONL line can be recovered without repeating completed examples.
- **Important:** Existing Flickr pool, map and experiment settings must still exist under the same saved paths. This test runner does not alter the old experiment.

## One-command RunPod background execution

Run **in the original RunPod instance** where `/workspace/vlm_A_zh_seed42/experiment` and the earlier `.venv` already exist:

```bash
cd /workspace/Multimodal_test_switch
git fetch origin
git switch experiment/vlm-unsupervised-a-b
git pull origin experiment/vlm-unsupervised-a-b
mkdir -p /workspace/vlm_A_zh_seed42/logs
nohup env TUNE_SIZE=200 LAYERS=8,13,18 ALPHAS=0.15,0.3,0.6 \
  bash vlm_unsupervised_transfer/run_mmbench_a.sh \
  > /workspace/vlm_A_zh_seed42/logs/mmbench_A.log 2>&1 < /dev/null &
echo $! > /workspace/vlm_A_zh_seed42/logs/mmbench_A.pid
```

Monitor:

```bash
tail -f /workspace/vlm_A_zh_seed42/logs/mmbench_A.log
```

**Protocol v2 includes optional MMBench hints.** If you previously generated MMBench data with the older script, use a fresh `MMBENCH_DATA` and `MMBENCH_OUT` directory; the updated manifest refuses silently reusing the older prompts.

**First download may be large.** The official MMBench image TSV files embed base64 images; disk/RAM needs may be sizable. If network download fails, download official `MMBench_DEV_EN.tsv`, `MMBench_DEV_CN.tsv` manually, then pass `MMBENCH_EN_TSV=/path/to/en.tsv MMBENCH_CN_TSV=/path/to/cn.tsv` in `nohup env ...`. An incorrect or changed dataset triggers an MD5 error; don't bypass it casually. You may explicitly opt in to a different release with `ALLOW_UNPINNED=1`, but its results must not be combined with the pinned version.

You can split into stages after data prep:

```bash
STAGE=tune bash vlm_unsupervised_transfer/run_mmbench_a.sh
STAGE=test bash vlm_unsupervised_transfer/run_mmbench_a.sh
```

**Warning:** two methods × three layers × three alphas = **18 tune variants**, then four frozen test conditions. At 200 tune + ~964 test questions, this is on the order of 7,600 separate VLM generations. It can take substantial time on a small GPU; no specific runtime guaranteed. For a cheaper exploratory run, use **a fresh output directory** and narrower grid, e.g. `LAYERS=8,13 ALPHAS=0.3,0.6`. Never change grid/dataset in an existing run directory after tuning.

## Results

```text
/workspace/vlm_A_zh_seed42/mmbench_A_search/
  run_config.json
  selection_mmbench.json          # selected layer and alpha from local tune only
  comparison.csv                 # FINAL held-out accuracies
  test_results.json              # paired statistics and category breakdown
  predictions/
    tune_baseline_zh.jsonl
    tune_text_visual_ot_*.jsonl
    tune_text_ot_*.jsonl
    test_baseline_zh.jsonl
    test_baseline_en.jsonl
    test_text_visual_ot_*.jsonl
    test_text_ot_*.jsonl
```

Bundle for review:

```bash
cd /workspace/vlm_A_zh_seed42
tar -czf /workspace/MMBench_A_results.tar.gz \
    mmbench_A_search/ mmbench_dev_bilingual/manifest.json
```

## Data provenance and scientific interpretation

- Dataset: [MMBench official GitHub](https://github.com/open-compass/MMBench) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). URLs and pinned MD5s are in `prepare_mmbench.py`.
- Official MMBench Dev contains repeated-option/circular passes. The local row-wise accuracy here is **not official MMBench CircularEval**. The split is **image-disjoint** so rotated questions and multiple questions on the same picture don't cross tuning/test.
- The original maps were selected on CV-Bench in an earlier experiment. The MMBench tuning search introduces additional downstream supervision for hyperparameter selection, even though fitting the maps themselves was unpaired.
- Images are strictly matched across EN and CN at decoded RGB pixel level. An exact duplicate between the previous Flickr fit pools and MMBench is rejected. This does not detect semantic near-duplicates or pretrained-data overlap.
- **A fixed cross-lingual representation intervention is not itself evidence that a model now 'thinks in English'.** Accuracy gains plus Text OT controls support a more modest representation-utility claim.
- Scripts have CPU unit tests only until the first RunPod end-to-end run; downloads, HF gated model access and GPU runtime can fail in that environment.

CPU tests:

```bash
cd vlm_unsupervised_transfer
python -m unittest discover -s tests -v
```
