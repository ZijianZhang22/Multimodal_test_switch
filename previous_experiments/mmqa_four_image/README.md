# MultiModalQA: four images, 50 questions, six matched conditions

Select exactly 50 real MultiModalQA dev questions, each with exactly four candidate image documents and image + another modality in its annotated supporting context.

Six conditions on identical questions and contexts: **VVVV, TTTT, VTTT, TVTT, TTVT, TTTV**. V is the original picture; T is a caption from LLaVA-1.5-7B (greedy, max_new_tokens=200, fixed image-description prompt, no downstream QA question), following the captioning procedure in *Bridging the Missing-Modality Gap (2026)*.

Run from this directory on your GPU pod:

```bash
bash setup.sh
python experiment.py prepare --n 50 --seed 42
python experiment.py caption
python experiment.py evaluate --max-new-tokens 1536
python experiment.py summarize
```

Outputs are written to `data/pilot/manifest.jsonl`, `results/captions.jsonl`, `results/predictions.jsonl`, `results/flips.jsonl`, and `results/summary.csv`. Expected number of QA evaluations: 50 × 6 = 300.

**Important:** No official dataset count could be verified from this environment. The prepare command fails explicitly if fewer than 50 qualifying examples/images exist; it does not invent images. Four image slots are *physical candidate-image positions*, not necessarily logical reasoning hops. Some may be distractors; the manifest includes supporting-image IDs. Inspect caption fidelity and paired flips before attributing differences to modality alone.
