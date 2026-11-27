#!/usr/bin/env python3
"""Aggregate reported VLM accuracy and paired bootstrap differences by image ID."""
import argparse,csv,json
from pathlib import Path
import numpy as np


def paired_stats(baseline,intervention,seed=0,iters=3000):
    b={r['id']:int(r['correct']) for r in baseline}
    x={r['id']:int(r['correct']) for r in intervention}
    if set(b)!=set(x) or not b:raise ValueError('Missing/mismatched baseline and intervention IDs')
    ids=sorted(b)
    difference=np.array([x[k]-b[k] for k in ids],dtype=float)
    rng=np.random.default_rng(seed)
    indices=rng.integers(len(ids),size=(iters,len(ids)))
    boots=np.mean(difference[indices],axis=1)
    return {'delta_pp':float(100*difference.mean()),
            'ci95_low_pp':float(np.quantile(boots,.025)*100),
            'ci95_high_pp':float(np.quantile(boots,.975)*100),
            'fixed':int((difference==1).sum()),'harmed':int((difference==-1).sum())}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    args=p.parse_args();root=Path(args.out)
    report_path=root/'test_results.json'
    if not report_path.exists():
        print('No test results yet; validation and/or Oracle available at',root)
        return
    report=json.loads(report_path.read_text())
    baseline_tag='test_baseline_'+report['language']
    def read(tag):return list(csv.DictReader((root/'predictions'/f'{tag}.csv').open(encoding='utf-8')))
    baseline=read(baseline_tag)
    rows=[]
    for arm in report['arms']:
        stats=paired_stats(baseline,read(arm['config']))
        rows.append(dict(arm=arm['arm'],name=arm['config'],accuracy=arm['accuracy'],
                         valid_rate=arm['valid_rate'],**stats))
    path=root/'comparison.csv'
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Baseline ZH',report['baseline'],'EN',report['english_reference'],flush=True)
    print('Comparison:',path,flush=True)
    for row in rows:print(row['arm'],f"{row['accuracy']:.3f}",f"delta={row['delta_pp']:+.2f} pp",flush=True)
    print('IMPORTANT: Exploratory local CV-Bench split; multiple comparisons, no adjusted claims.')

if __name__=='__main__':main()
