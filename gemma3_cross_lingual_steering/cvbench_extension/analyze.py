#!/usr/bin/env python3
"""Per-task accuracies, balanced four-task macro, paired gains, vector geometry."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from prepare import CONCEPTS,LANGS,TASKS


def write(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)


def cos(a,b):
    return float(torch.nn.functional.cosine_similarity(a.float().flatten()[None],b.float().flatten()[None])[0])


def analyze(data,split):
    report=data/f'analysis_{split}'
    report.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((data/'vectors'/'manifest.json').read_text())
    geom=[]
    for l in manifest['layers']:
        vectors={(c,lang):torch.load(data/'vectors'/c/lang/f'layer_{l}_feature.pt',weights_only=True,map_location='cpu')
                 for c in CONCEPTS for lang in LANGS}
        for c in CONCEPTS:
            for i,x in enumerate(LANGS):
                for y in LANGS[i+1:]:
                    geom.append(dict(layer=l,kind='cross_lang',concept_a=c,concept_b=c,
                                     lang_a=x,lang_b=y,cosine=cos(vectors[(c,x)],vectors[(c,y)])))
        for lang in LANGS:
            cs=list(CONCEPTS)
            for i,x in enumerate(cs):
                for y in cs[i+1:]:
                    geom.append(dict(layer=l,kind='cross_feature',concept_a=x,concept_b=y,
                                     lang_a=lang,lang_b=lang,cosine=cos(vectors[(x,lang)],vectors[(y,lang)])))
    write(report/'vector_geometry.csv',geom)
    with (data/f'predictions_{split}.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    base={}
    groups=defaultdict(list)
    for r in rows:
        k=(r['id'],r['task'],r['prompt_lang'])
        if r['vector_lang']=='none':
            if k in base:raise ValueError(f'Duplicate baseline: {k}')
            base[k]=int(r['correct'])
        else:
            g=(r['task'],r['prompt_lang'],r['concept'],r['vector_lang'],r['layer'],r['alpha'],r['mask'],r['normalized'])
            groups[g].append(r)
    rng=np.random.default_rng(42)
    summary=[]
    for g,rs in sorted(groups.items()):
        if len({r['id'] for r in rs})!=len(rs):raise ValueError(f'Duplicate example in {g}')
        diffs=[];correct=[]; improved=worsened=0
        for r in rs:
            k=(r['id'],r['task'],r['prompt_lang'])
            if k not in base:raise ValueError(f'Unpaired example {k}')
            b=base[k];p=int(r['correct'])
            correct.append(p);diffs.append(p-b)
            improved+=int(p>b);worsened+=int(p<b)
        d=np.array(diffs,dtype=float)
        boots=rng.choice(d,size=(2000,len(d)),replace=True).mean(axis=1)
        summary.append(dict(zip(('task','prompt_lang','concept','vector_lang','layer','alpha','mask','normalized'),g),
                            n=len(rs),baseline_accuracy=float(np.mean([base[(r['id'],r['task'],r['prompt_lang'])] for r in rs])),
                            accuracy=float(np.mean(correct)),gain_pp=round(100*float(d.mean()),4),
                            ci_low_pp=float(100*np.quantile(boots,.025)),ci_high_pp=float(100*np.quantile(boots,.975)),
                            improved=improved,worsened=worsened))
    write(report/'task_accuracy.csv',summary)
    base_summary=[]
    for task in TASKS:
        for lang in LANGS:
            arr=[v for (idx,t,l),v in base.items() if t==task and l==lang]
            if arr:base_summary.append({'task':task,'prompt_lang':lang,'n':len(arr),'baseline_accuracy':float(np.mean(arr))})
    write(report/'baseline_accuracy.csv',base_summary)
    # Equal-weight four-task macro. For default task-aligned feature selection,
    # the concept is intentionally NOT part of the grouping key, because Count
    # uses counting while Relation/Depth/Distance use spatial.
    macros=[]
    grouped=defaultdict(dict)
    for r in summary:
        key=tuple(r[k] for k in ('prompt_lang','vector_lang','layer','alpha','mask','normalized'))
        grouped[key][r['task']]=r
    for key,bytask in grouped.items():
        # If all-features mode is enabled, do not silently conflate concepts.
        # A separate fixed-concept macro can be calculated from task_accuracy.csv.
        if len(summary)!=len({(r['task'],r['prompt_lang'],r['vector_lang'],r['layer'],r['alpha'],r['mask'],r['normalized']) for r in summary}):
            break
        if set(TASKS)<=set(bytask):
            macros.append(dict(zip(('prompt_lang','vector_lang','layer','alpha','mask','normalized'),key),
                               macro_accuracy=float(np.mean([bytask[t]['accuracy'] for t in TASKS])),
                               macro_gain_pp=float(np.mean([bytask[t]['gain_pp'] for t in TASKS]))))
    write(report/'four_task_macro_accuracy.csv',macros)
    print(f'=== {split.upper()} CV-Bench (local holdout) ===')
    for r in base_summary:print(f"BASE {r['task']:8s} prompt={r['prompt_lang']} n={r['n']} acc={r['baseline_accuracy']:.3f}")
    for r in summary:
        print(f"STEER {r['task']:8s} prompt={r['prompt_lang']} vector={r['vector_lang']} "
              f"concept={r['concept']} L={r['layer']} alpha={r['alpha']} "
              f"acc={r['accuracy']:.3f} gain={r['gain_pp']:+.2f}pp n={r['n']}")
    print(f'Geometry rows: {len(geom)} | group rows: {len(summary)}')
    print('Reports:',report)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='outputs/gemma3_cvbench')
    p.add_argument('--split',choices=['val','test'],default='val')
    a=p.parse_args(); analyze(Path(a.data),a.split)
if __name__=='__main__':main()
