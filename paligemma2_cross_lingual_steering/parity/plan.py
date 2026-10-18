#!/usr/bin/env python3
"""Write job lists for each stage.

A-tune : EN prompt x EN vector, every (task, concept, layer, alpha)   == upstream eval.sh grid
A-test : EN prompt x EN vector, each (task, concept) at its tuned best == upstream eval.py
B-test : 3 prompt langs x vector variants, each task at the stage-A setting
         (concept + layer + alpha frozen from the English tuning -> no per-language selection bias)
C-tune / C-test (optional): re-tune layer/alpha separately for every prompt_lang x vector_lang
"""
import argparse
import json
from pathlib import Path

from common import ALIGNED_CONCEPT, CONCEPTS, LANGS, TASKS


def floats(s):
    return [float(x) for x in s.split(',') if x]


def ints(s):
    out = []
    for part in s.split(','):
        if '-' in part:
            lo, hi = part.split('-')
            out += list(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def task_concept(best, task, mode):
    if mode == 'aligned':
        return ALIGNED_CONCEPT[task]
    return best['task_best_concept'][task]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['A-tune', 'A-test', 'B-test', 'C-tune', 'C-test'])
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--best', type=Path, help='best_settings json from select_best.py')
    p.add_argument('--tasks', default=','.join(TASKS))
    p.add_argument('--concepts', default=','.join(CONCEPTS))
    p.add_argument('--layers', default='5-20')
    p.add_argument('--alphas', default='0.1,0.2,0.4,0.6,0.8,1.0')
    p.add_argument('--concept-mode', default='best', choices=['best', 'aligned'])
    p.add_argument('--variants', default='en,zh,sw,zh_normen,sw_normen,random_s0')
    p.add_argument('--styles', default='native')
    p.add_argument('--vector-langs', default=','.join(LANGS))
    a = p.parse_args()
    tasks = a.tasks.split(',')
    best = json.loads(a.best.read_text()) if a.best else None
    jobs = []
    if a.stage == 'A-tune':
        for t in tasks:
            for c in a.concepts.split(','):
                for l in ints(a.layers):
                    for al in floats(a.alphas):
                        jobs.append(dict(task=t, prompt_lang='en', style='native', concept=c, variant='en', layer=l, alpha=al))
    elif a.stage == 'A-test':
        for g in best['groups']:
            if g['prompt_lang'] == 'en' and g['variant'] == 'en' and g['task'] in tasks:
                jobs.append(dict(task=g['task'], prompt_lang='en', style='native', concept=g['concept'],
                                 variant='en', layer=g['layer'], alpha=g['alpha']))
    elif a.stage == 'B-test':
        idx = {(g['task'], g['concept']): g for g in best['groups'] if g['prompt_lang'] == 'en' and g['variant'] == 'en'}
        for t in tasks:
            c = task_concept(best, t, a.concept_mode)
            g = idx[(t, c)]
            for style in a.styles.split(','):
                for pl in LANGS:
                    if pl == 'en' and style != 'native':
                        continue                        # EN prompt is identical in every style
                    for v in a.variants.split(','):
                        jobs.append(dict(task=t, prompt_lang=pl, style=style, concept=c, variant=v,
                                         layer=g['layer'], alpha=g['alpha']))
    elif a.stage == 'C-tune':
        for t in tasks:
            c = task_concept(best, t, a.concept_mode)
            for style in a.styles.split(','):
                for pl in LANGS:
                    if pl == 'en' and style != 'native':
                        continue
                    for v in a.vector_langs.split(','):
                        for l in ints(a.layers):
                            for al in floats(a.alphas):
                                jobs.append(dict(task=t, prompt_lang=pl, style=style, concept=c, variant=v, layer=l, alpha=al))
    elif a.stage == 'C-test':
        for g in best['groups']:
            if g['task'] in tasks:
                jobs.append({k: g[k] for k in ('task', 'prompt_lang', 'style', 'concept', 'variant', 'layer', 'alpha')})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(jobs, indent=1))
    print(f'{a.stage}: {len(jobs)} jobs per matching sample -> {a.out}')


if __name__ == '__main__':
    main()
