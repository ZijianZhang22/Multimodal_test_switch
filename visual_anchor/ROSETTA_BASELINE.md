# Official Rosetta text-only baselines (Flickr8K-CN)

This experiment calls **the original** [unpaired-rosetta](https://github.com/dominik-schnaus/unpaired-rosetta) `WassersteinProcrustes.fit()` including its geometric initialization and readout. It is *not* OT->refinement and does not modify our text/visual/bank_text models.

## Install and run

Requires Python >=3.11. In `visual_anchor/`:

```bash
python -m pip install 'git+https://github.com/dominik-schnaus/unpaired-rosetta.git'
python -m unittest -v test_core test_real test_rosetta_baseline

# If features_stage1.npz is missing but you retained the old features.npz:
python stage1_repool.py --source real_work/features.npz --output real_work/features_stage1.npz

python run_rosetta_baseline.py \
  --data real_work/features_stage1.npz \
  --budgets 50 200 1000 --seeds 0 1 2 --dim 32 \
  --clusters 20 --restarts 5 --batch 256 --iterations 20 \
  --out real_work/official_rosetta
```

`pip install` installs the official library from upstream; its `pylibmgm` dependency is used by the quadratic-assignment initializer. If installation fails, use the upstream repository's documented pixi environment. For reproducibility, record and pin the exact upstream commit (a Git URL with `@<commit-sha>`).

## Protocol and outputs

* Two methods: `rosetta_zero` (no paired examples) and `rosetta_few` (same 50/200/1000 anchors, sampled with exactly the same per-seed NumPy selection as `run.py`).
* Both fit only Chinese `l_pool` and English `e_pool`; no images enter Rosetta training. A fixed `V_IE` estimated from `i_bridge`/`e_bridge` is used **only** for L-I and I-L evaluation. Thus `rosetta_zero` is zero-pair for the **Chinese-English mapping**, not zero-pair for the whole evaluation pipeline.
* Reuses the same training-only PCA at d=32 and the same evaluation IDs and CSLS recall as `run.py`. Rosetta itself additionally mean-centers and L2-normalizes the PCA features; its test features, including the mapped image bank, are evaluated in that centered target space.
* `real_work/official_rosetta/results_budget_{50,200,1000}/report.json` each has `summary`, `runs`, `config`. Individual model parameters are written as .npz.
* `num_clusters=20`, restarts=5, batch=256, iterations=20 are **pilot settings** reduced for this dataset; not the upstream paper's default experimental settings. This is a first smoke/pilot baseline, not an optimized state-of-the-art comparison. Use a validation split, not the held-out test set, for hyperparameter selection.
* A full official Rosetta run is more expensive than our original OT updates because its geometric initialization performs QAP matching; for faster plumbing checks, try `--budgets 50 --seeds 0 --clusters 8 --restarts 1 --iterations 2`.
* Supervised and text/visual/bank_text comparisons remain in the existing stage-1 results. This script does not rewrite or overwrite those artifacts.
