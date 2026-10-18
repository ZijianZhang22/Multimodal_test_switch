#!/usr/bin/env python3
"""Re-score legacy predictions_{val,test}.csv (the run behind steering_summary_cvbench)
with the upstream normalize_answer, without any GPU.

Legacy parse_letter only accepted answers starting with a letter and also mis-parsed
lowercase words ("a cat" -> A, "e.g. A" -> E). Upstream additionally maps an answer
that equals a choice's text ("2", "left") back to its letter. Choices come from the
checked-in curated 840 rows (legacy ids).

usage: python rescore_legacy.py --pred path/to/predictions_test.csv [--out rescored_test]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import load_curated, score


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred', required=True, type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    out = a.out or a.pred.with_name(a.pred.stem + '_rescored')
    out.mkdir(parents=True, exist_ok=True)
    cur = load_curated()
    rows = list(csv.DictReader(a.pred.open(encoding='utf-8')))
    for r in rows:
        c = cur[r['id']]
        lang = r['prompt_lang']
        ch_l = c['choices'] if lang == 'en' else c['translated_choices'][lang]
        sc = score(r['prediction'], f"({r['answer']})", c['choices'], ch_l, lang)
        r['correct_upstream'], r['valid_upstream'] = sc['correct'], sc['valid']
        r['valid_legacy'] = int(bool(r['parsed']))
    with (out / 'predictions_rescored.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    base = {(r['id'], r['prompt_lang']): r for r in rows if r['vector_lang'] == 'none'}
    groups = defaultdict(list)
    for r in rows:
        if r['vector_lang'] != 'none':
            groups[(r['layer'], r['alpha'], r['prompt_lang'], r['vector_lang'])].append(r)
    for r in rows:
        if r['vector_lang'] != 'none':
            groups[(r['layer'], r['alpha'], 'ALL', r['vector_lang'])].append(r)
    rng = np.random.default_rng(42)
    summary = []
    for k, rs in sorted(groups.items()):
        bl = [base[(r['id'], r['prompt_lang'])] for r in rs]
        rec = dict(zip(('layer', 'alpha', 'prompt_lang', 'vector_lang'), k), n=len(rs))
        for name, col, vcol in (('legacy', 'correct', 'valid_legacy'), ('upstream', 'correct_upstream', 'valid_upstream')):
            b = np.array([int(x[col]) for x in bl])
            s = np.array([int(x[col]) for x in rs])
            d = s - b
            ids = np.array([x['id'] for x in rs])
            uniq, inv = np.unique(ids, return_inverse=True)
            sums, cnts = np.bincount(inv, weights=d), np.bincount(inv)
            dr = rng.integers(0, len(uniq), (2000, len(uniq)))
            est = sums[dr].sum(1) / cnts[dr].sum(1)
            rec.update({f'{name}_base': round(100 * b.mean(), 2), f'{name}_acc': round(100 * s.mean(), 2),
                        f'{name}_gain_pp': round(100 * d.mean(), 2),
                        f'{name}_ci': f'[{100*np.quantile(est,.025):+.2f}, {100*np.quantile(est,.975):+.2f}]',
                        f'{name}_base_invalid': int(sum(1 - int(x[vcol]) for x in bl)),
                        f'{name}_steer_invalid': int(sum(1 - int(x[vcol]) for x in rs))})
        summary.append(rec)
    with (out / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for r in summary:
        print(f"L{r['layer']} a={r['alpha']} prompt={r['prompt_lang']:3s} vec={r['vector_lang']} | "
              f"legacy {r['legacy_base']:.2f}->{r['legacy_acc']:.2f} ({r['legacy_gain_pp']:+.2f}, inv {r['legacy_base_invalid']}->{r['legacy_steer_invalid']}) | "
              f"upstream {r['upstream_base']:.2f}->{r['upstream_acc']:.2f} ({r['upstream_gain_pp']:+.2f} {r['upstream_ci']}, inv {r['upstream_base_invalid']}->{r['upstream_steer_invalid']})")
    print('Saved', out)


if __name__ == '__main__':
    main()
