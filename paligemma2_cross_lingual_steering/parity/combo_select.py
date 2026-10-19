#!/usr/bin/env python3
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
def main():
 p=argparse.ArgumentParser();p.add_argument('--pred',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
 rows=list(csv.DictReader(a.pred.open()));bas={(r['id'],r['prompt_lang']):r for r in rows if r['cond']=='base'}
 by=defaultdict(list)
 for r in rows:
  if r['cond']=='lang':by[(r['prompt_lang'],int(r['lang_layer']),float(r['lang_alpha']))].append(r)
 table=[]
 for (lang,layer,alpha),rs in sorted(by.items()):
  vals=[float(r['margin'])-float(bas[(r['id'],lang)]['margin']) for r in rs]
  table.append(dict(prompt_lang=lang,layer=layer,alpha=alpha,n=len(rs),margin_gain=float(np.mean(vals)),
                    margin_se=float(np.std(vals,ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else None))
 chosen={}
 for lang in sorted(set(r['prompt_lang'] for r in table)):
  xs=[r for r in table if r['prompt_lang']==lang];chosen[lang]=sorted(xs,key=lambda r:(-r['margin_gain'],r['alpha'],r['layer']))[0]
  print(lang,chosen[lang],flush=True)
 if not chosen:raise RuntimeError('No tune rows')
 a.out.write_text(json.dumps(dict(rule='maximize mean paired first-token margin',chosen=chosen,table=table),indent=2))
 with a.out.with_suffix('.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
if __name__=='__main__':main()
