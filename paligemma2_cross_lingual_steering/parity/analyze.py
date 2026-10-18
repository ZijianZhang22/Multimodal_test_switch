#!/usr/bin/env python3
"""Paired analysis of test predictions (pure python + numpy; runs on CPU).

cells.csv   : per (task, prompt_lang, style, concept, variant, layer, alpha)
overall.csv : pooled over the four tasks, per (prompt_lang, style, variant)  [+ concept if ambiguous]
pooled.csv  : additionally pooled over the three prompt languages, per (style, variant)
All CIs: 95% bootstrap, resampling images (cluster = CV-Bench id), 2000 draws, seed 42.
Also reports invalid-answer rates and the gain restricted to rows where baseline AND steered
outputs are valid, so format effects can be separated from accuracy effects.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CFG = ('task', 'prompt_lang', 'style', 'concept', 'variant', 'layer', 'alpha')


def boot(ids, diffs, rng, n=2000):
    ids = np.asarray(ids)
    diffs = np.asarray(diffs, float)
    uniq, inv = np.unique(ids, return_inverse=True)
    sums = np.bincount(inv, weights=diffs)
    cnts = np.bincount(inv)
    draws = rng.integers(0, len(uniq), size=(n, len(uniq)))
    est = sums[draws].sum(1) / cnts[draws].sum(1)
    return 100 * float(np.quantile(est, .025)), 100 * float(np.quantile(est, .975))


def summarize(rs, base, rng):
    b = np.array([base[(r['id'], r['prompt_lang'], r['style'])]['correct'] for r in rs])
    bv = np.array([base[(r['id'], r['prompt_lang'], r['style'])]['valid'] for r in rs])
    s = np.array([int(r['correct']) for r in rs])
    sv = np.array([int(r['valid']) for r in rs])
    d = s - b
    lo, hi = boot([r['id'] for r in rs], d, rng)
    both = (bv == 1) & (sv == 1)
    return {'n': len(rs), 'n_images': len({r['id'] for r in rs}),
            'baseline_acc': round(100 * b.mean(), 2), 'acc': round(100 * s.mean(), 2),
            'gain_pp': round(100 * d.mean(), 2), 'ci_low_pp': round(lo, 2), 'ci_high_pp': round(hi, 2),
            'improved': int((d > 0).sum()), 'worsened': int((d < 0).sum()),
            'baseline_invalid': int((bv == 0).sum()), 'steered_invalid': int((sv == 0).sum()),
            'n_both_valid': int(both.sum()),
            'gain_pp_both_valid': round(100 * float(d[both].mean()), 2) if both.any() else None}


def write(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def md_table(rows, cols):
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        out.append('| ' + ' | '.join(str(r.get(c, '')) for c in cols) + ' |')
    return '\n'.join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred', required=True, type=Path, nargs='+')
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    md = ['# Test-split analysis (upstream-aligned protocol)\n',
          'Gains are paired against the unsteered baseline with the same prompt; CIs resample images.\n']
    for path in a.pred:
        label = path.stem
        rng = np.random.default_rng(42)
        rows = list(csv.DictReader(path.open(encoding='utf-8')))
        base = {(r['id'], r['prompt_lang'], r['style']): {'correct': int(r['correct']), 'valid': int(r['valid'])}
                for r in rows if r['variant'] == 'none'}
        steer = [r for r in rows if r['variant'] != 'none']
        cells = defaultdict(list)
        for r in steer:
            cells[tuple(r[k] for k in CFG)].append(r)
        cell_rows = [dict(zip(CFG, k), **summarize(v, base, rng)) for k, v in sorted(cells.items())]
        configs = defaultdict(set)
        for k in cells:
            configs[(k[0], k[1], k[2], k[4])].add(k)
        ambiguous = any(len(v) > 1 for v in configs.values())
        okeys = ('prompt_lang', 'style', 'variant') + (('concept',) if ambiguous else ())
        overall, pooled = defaultdict(list), defaultdict(list)
        for r in steer:
            overall[tuple(r[k] for k in okeys)].append(r)
            pooled[tuple(r[k] for k in okeys if k != 'prompt_lang')].append(r)
        o_rows, p_rows = [], []
        for k, v in sorted(overall.items()):
            row = dict(zip(okeys, k), **summarize(v, base, rng))
            per_task = defaultdict(list)
            for r in v:
                per_task[r['task']].append(int(r['correct']) - base[(r['id'], r['prompt_lang'], r['style'])]['correct'])
            row['macro_gain_pp'] = round(100 * float(np.mean([np.mean(x) for x in per_task.values()])), 2)
            o_rows.append(row)
        pk = tuple(k for k in okeys if k != 'prompt_lang')
        for k, v in sorted(pooled.items()):
            p_rows.append(dict(zip(pk, k), **summarize(v, base, rng)))
        base_rows = []
        for (task, pl, st) in sorted({(r['task'], r['prompt_lang'], r['style']) for r in rows if r['variant'] == 'none'}):
            vals = [r for r in rows if r['variant'] == 'none' and r['task'] == task and r['prompt_lang'] == pl and r['style'] == st]
            base_rows.append({'task': task, 'prompt_lang': pl, 'style': st, 'n': len(vals),
                              'acc': round(100 * np.mean([int(r['correct']) for r in vals]), 2),
                              'invalid': sum(1 - int(r['valid']) for r in vals)})
        d = a.out / label
        d.mkdir(exist_ok=True)
        write(d / 'cells.csv', cell_rows)
        write(d / 'overall.csv', o_rows)
        write(d / 'pooled.csv', p_rows)
        write(d / 'baseline.csv', base_rows)
        cols = ['baseline_acc', 'acc', 'gain_pp', 'ci_low_pp', 'ci_high_pp', 'baseline_invalid', 'steered_invalid', 'gain_pp_both_valid']
        md += [f'\n## {label}\n', '### Baseline\n', md_table(base_rows, ['task', 'prompt_lang', 'style', 'n', 'acc', 'invalid']),
               '\n### Per task\n', md_table(cell_rows, list(CFG) + ['n'] + cols),
               '\n### Pooled over tasks\n', md_table(o_rows, list(okeys) + ['n'] + cols + ['macro_gain_pp']),
               '\n### Pooled over tasks and prompt languages\n', md_table(p_rows, list(pk) + ['n', 'n_images'] + cols)]
    (a.out / 'SUMMARY.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
