"""Paired accuracy, parse validity, dish macro; validation-only model selection."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from core import read_jsonl, write_json


def summarize(rows, expected_ids):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['condition'], r['layer'], r['alpha'], r['random_seed'])].append(r)
    baseline = {r['id']: r['correct'] for r in groups[('baseline', -1, 0., -1)]}
    if set(baseline) != set(expected_ids):
        raise ValueError('Incomplete baseline')
    result = []
    for (condition, layer, alpha, seed), items in groups.items():
        if len(items) != len(expected_ids) or {r['id'] for r in items} != set(expected_ids):
            raise ValueError('Incomplete/duplicate condition; finish evaluation before reporting')
        dishes = defaultdict(list)
        for r in items:
            dishes[r['food_id']].append(r['correct'])
        n = len(items)
        result.append(dict(condition=condition, layer=layer, alpha=alpha, random_seed=seed, n=n,
            accuracy=sum(r['correct'] for r in items) / n,
            valid_rate=sum(r['valid'] for r in items) / n,
            dish_macro=sum(sum(v) / len(v) for v in dishes.values()) / len(dishes),
            gained=sum(r['correct'] and not baseline[r['id']] for r in items),
            lost=sum(not r['correct'] and baseline[r['id']] for r in items),
            delta=sum(r['correct'] - baseline[r['id']] for r in items) / n))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='outputs/smoke')
    p.add_argument('--split', choices=['val', 'test'], default='val')
    a = p.parse_args()
    root = Path(a.data)
    settings = json.loads((root / f'settings_{a.split}.json').read_text())
    rows = read_jsonl(root / f'predictions_{a.split}.jsonl')
    summary = summarize(rows, settings['sample_ids'])
    expected_groups = 2 + 5 * len(settings['layers']) * len(settings['alphas'])
    if len(summary) != expected_groups:
        raise ValueError('Missing experimental conditions; complete evaluation first')
    with (root / f'summary_{a.split}.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for s in summary:
        print(f"{s['condition']:8s} L={s['layer']:2d} a={s['alpha']:5.2f} seed={s['random_seed']:2d} "
              f"acc={s['accuracy']:.3f} valid={s['valid_rate']:.3f} delta={s['delta']:+.3f}")
    if a.split == 'val':
        # Prespecified: maximize image accuracy, then smaller alpha, then lower layer.
        best = sorted((s for s in summary if s['condition'] == 'food'),
                      key=lambda s: (-s['accuracy'], s['alpha'], s['layer']))[0]
        selection = {k: settings[k] for k in ['dataset_sha256', 'vector_manifest_sha256',
            'vlm_revision', 'mask', 'max_tokens']}
        selection.update(layer=best['layer'], alpha=best['alpha'],
            vector_sha256=settings['vector_hashes'][str(best['layer'])],
            validation_accuracy=best['accuracy'], validation_delta=best['delta'],
            selection_rule='max positive-food val accuracy; ties smaller alpha then layer')
        path = root / 'selection.json'
        if path.exists() and json.loads(path.read_text()) != selection:
            raise ValueError('Frozen selection differs; use a new experiment directory')
        write_json(path, selection)
        print('Frozen selection.json. A selection is not evidence of improvement; inspect controls and test.')


if __name__ == '__main__':
    main()
