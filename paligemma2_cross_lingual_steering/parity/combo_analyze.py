#!/usr/bin/env python3
"""Paired CV-Bench accuracy/margin analysis of language × concept interventions."""
import argparse,csv
from collections import defaultdict
from pathlib import Path
import numpy as np
CONDS=('lang','lang_neg','lang_rand','concept','lang+concept')
def ci(values,rng):
 x=np.asarray(values,float)
 if not len(x):return None
 picks=rng.integers(0,len(x),(2000,len(x)))
 return [round(float(z),3) for z in np.quantile(x[picks].mean(axis=1),(.025,.975))]
def main():
 p=argparse.ArgumentParser();p.add_argument('--pred',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 with a.pred.open(newline='') as f:rows=list(csv.DictReader(f))
 by=defaultdict(dict)
 for r in rows:
  k=(r['prompt_lang'],r['id'])
  if r['cond'] in by[k]:raise RuntimeError('Duplicate rows: '+str(k)+' '+r['cond'])
  by[k][r['cond']]=r
 rng=np.random.default_rng(42);summary=[]
 langs=sorted({r['prompt_lang'] for r in rows})
 for task in ('ALL','count','relation','depth','distance'):
  for lang in langs:
   records=[v for (l,i),v in by.items() if l==lang and 'base' in v and
       (task=='ALL' or v['base']['task']==task)]
   for cond in CONDS:
    valid=[v for v in records if cond in v]
    if not valid:continue
    da=[int(v[cond]['correct'])-int(v['base']['correct']) for v in valid]
    dm=[float(v[cond]['margin'])-float(v['base']['margin']) for v in valid]
    summary.append(dict(task=task,prompt_lang=lang,condition=cond,n=len(valid),
      baseline_acc=round(np.mean([int(v['base']['correct']) for v in valid])*100,2),
      accuracy=round(np.mean([int(v[cond]['correct']) for v in valid])*100,2),
      gain_pp=round(np.mean(da)*100,2),gain_ci95_pp=str([x*100 for x in ci(da,rng)]),
      margin_gain=round(float(np.mean(dm)),4),margin_ci95=str(ci(dm,rng)),
      invalid=sum(1-int(v[cond]['valid']) for v in valid),
      flips=sum(v[cond]['normalized']!=v['base']['normalized'] for v in valid)))
   for name,left,right in [('direction','lang','lang_neg'),('specificity','lang','lang_rand')]:
    valid=[v for v in records if left in v and right in v]
    if valid:
     da=[int(v[left]['correct'])-int(v[right]['correct']) for v in valid]
     dm=[float(v[left]['margin'])-float(v[right]['margin']) for v in valid]
     summary.append(dict(task=task,prompt_lang=lang,condition=name,n=len(valid),
       baseline_acc='',accuracy='',gain_pp=round(np.mean(da)*100,2),
       gain_ci95_pp=str([x*100 for x in ci(da,rng)]),margin_gain=round(float(np.mean(dm)),4),
       margin_ci95=str(ci(dm,rng)),invalid='',flips=''))
   valid=[v for v in records if all(k in v for k in ('base','lang','concept','lang+concept'))]
   if valid:
    da=[(int(v['lang+concept']['correct'])-int(v['lang']['correct']))-
        (int(v['concept']['correct'])-int(v['base']['correct'])) for v in valid]
    dm=[(float(v['lang+concept']['margin'])-float(v['lang']['margin']))-
        (float(v['concept']['margin'])-float(v['base']['margin'])) for v in valid]
    summary.append(dict(task=task,prompt_lang=lang,condition='interaction',n=len(valid),
       baseline_acc='',accuracy='',gain_pp=round(np.mean(da)*100,2),
       gain_ci95_pp=str([x*100 for x in ci(da,rng)]),margin_gain=round(float(np.mean(dm)),4),
       margin_ci95=str(ci(dm,rng)),invalid='',flips=''))
 if not summary:raise RuntimeError('No test rows')
 with (a.out/'effects.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 cols=['task','prompt_lang','condition','n','accuracy','gain_pp','gain_ci95_pp','margin_gain','margin_ci95']
 md=['# MRRE-inspired language × concept steering','',
     'Paired 95% bootstrap over question IDs. First-position bare-letter margin is an auxiliary proxy.',
     '','| '+' | '.join(cols)+' |','|'+'---|'*len(cols)]
 md+=['| '+' | '.join(str(r[k]) for k in cols)+' |' for r in summary]
 (a.out/'SUMMARY.md').write_text('\n'.join(md)+'\n')
 print('Saved',a.out/'SUMMARY.md')
if __name__=='__main__':main()
