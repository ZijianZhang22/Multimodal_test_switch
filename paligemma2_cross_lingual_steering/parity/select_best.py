#!/usr/bin/env python3
"""Pick the best (layer, alpha) per group on the tuning split, with upstream plot.py's rule:
maximise improvement (steered acc - baseline acc); among ties take the smallest alpha
(then the smallest layer, for determinism).

Group = (task, prompt_lang, style, concept, variant). For every task the concept with the
largest English tuning improvement is also recorded (`task_best_concept`).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from common import ALIGNED_CONCEPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    rows = list(csv.DictReader(a.pred.open(encoding='utf-8')))
    base = {(r['id'], r['prompt_lang'], r['style']): int(r['correct']) for r in rows if r['variant'] == 'none'}
    cells = defaultdict(list)
    for r in rows:
        if r['variant'] == 'none':
            continue
        g = (r['task'], r['prompt_lang'], r['style'], r['concept'], r['variant'])
        cells[(g, int(r['layer']), float(r['alpha']))].append(r)
    table = defaultdict(list)
    for (g, layer, alpha), rs in cells.items():
        b = [base[(r['id'], r['prompt_lang'], r['style'])] for r in rs]
        acc = sum(int(r['correct']) for r in rs) / len(rs)
        table[g].append({'layer': layer, 'alpha': alpha, 'n': len(rs), 'baseline_acc': sum(b) / len(b),
                         'acc': acc, 'improvement_pp': 100 * (acc - sum(b) / len(b))})
    sizes = {len(v) for v in table.values()}
    groups = []
    for g, opts in sorted(table.items()):
        top = max(o['improvement_pp'] for o in opts)
        winner = sorted([o for o in opts if o['improvement_pp'] == top], key=lambda o: (o['alpha'], o['layer']))[0]
        groups.append(dict(zip(('task', 'prompt_lang', 'style', 'concept', 'variant'), g), **winner,
                           grid_size=len(opts)))
    best_concept = {}
    for task in sorted({g['task'] for g in groups}):
        cand = [g for g in groups if g['task'] == task and g['prompt_lang'] == 'en' and g['variant'] == 'en']
        if cand:
            # ties -> prefer the conventional task-aligned concept, then name order
            cand.sort(key=lambda g: (-g['improvement_pp'], g['concept'] != ALIGNED_CONCEPT[task], g['concept']))
            best_concept[task] = cand[0]['concept']
    out = {'rule': 'max improvement on tune split; ties -> min alpha -> min layer (upstream plot.py)',
           'grid_sizes': sorted(sizes), 'task_best_concept': best_concept, 'groups': groups}
    a.out.write_text(json.dumps(out, indent=2))
    for g in groups:
        print(f"{g['task']:8s} {g['prompt_lang']} {g['style']:11s} {g['concept']:9s} {g['variant']:10s} "
              f"L={g['layer']:2d} a={g['alpha']:.1f} base={g['baseline_acc']:.3f} acc={g['acc']:.3f} "
              f"{g['improvement_pp']:+.2f}pp (n={g['n']})")
    print('task_best_concept:', best_concept)


if __name__ == '__main__':
    main()
