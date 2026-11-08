# Stage 1: same-image visual versus bank-text controls

This branch changes only the training data-pool protocol and adds a **bank_text** control. It does not change the English/image bridge checkpoint, the image encoder, the PCA-fitting samples, or the evaluation set.

## Reuse existing embeddings (no re-encoding)

From `visual_anchor/`:

```bash
python -m unittest -v test_core test_real
python stage1_repool.py --source real_work/features.npz --output real_work/features_stage1.npz
for budget in 50 200 1000; do
  python run.py --data real_work/features_stage1.npz --budget "$budget" --dim 32 \
    --seeds 0 1 2 --iterations 20 --batch 256 --no-cycle \
    --methods supervised text visual bank_text extra_text gaussian \
    --out "real_work/stage1_budget_${budget}"
done
```

To create features from scratch instead, `prepare_real.py` uses the corrected disjoint pool and continues to write `features.npz`. Save the old features and results before overwriting.

The key controlled comparison is **visual minus bank_text** at the same budget, seed, image IDs, and optimization budget. In both methods, the third OT target has 500 rows; visual uses image embeddings transformed through the frozen bridge, bank_text uses the paired English-caption embeddings. The caption/image content is matched, but the embeddings and distributions differ by modality. `extra_text` is an additional disjoint-data control.

The `e_fit` and `i_fit` PCA populations remain unchanged on purpose for stage 1, including 500 bank items. The new protocol removes bank captions from **e_pool** (OT training), not from PCA fitting. `broken_pairs` is omitted because with `--no-cycle` it is identical to visual by design. Run it only with cycle enabled if you want to study pseudo-pair voting.

These experiments do not imply any improvement in Qwen-VL reasoning.
