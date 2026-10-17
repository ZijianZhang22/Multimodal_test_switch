"""Fast CPU contract test for extraction anchors, answer parser, CVBench reports."""
import csv
import json
import sys
import tempfile
import types
from pathlib import Path
import torch

base=Path(__file__).resolve().parent
sys.path.insert(0,str(base))
mock=types.ModuleType('make_data')
mock.ZH={'counting':[('测试一句话1','1')]*20,'spatial':[('测试一句话1','1')]*20}
sys.modules.setdefault('make_data',mock)
from anchors import SW, ZH_EXTRA
from prepare import correct_letter,CONCEPTS,LANGS,TASKS
from analyze import analyze

for label,d in [('SW',SW),('ZH_EXTRA',ZH_EXTRA)]:
 for c,pairs in d.items():
  assert len(pairs)==20,(label,c,len(pairs))
  for s,target in pairs:
   assert s.count(target)==1,(label,c,s,target)

assert correct_letter('(B)',['0','2','4'])=='B'
assert correct_letter('2',['0','2','4'])=='B'
assert correct_letter('D',['left','right','up','down'])=='D'

with tempfile.TemporaryDirectory() as td:
 data=Path(td)
 (data/'vectors').mkdir()
 (data/'vectors'/'manifest.json').write_text(json.dumps({'layers':[8]}))
 for c in CONCEPTS:
  for lang in LANGS:
   folder=data/'vectors'/c/lang
   folder.mkdir(parents=True)
   torch.save(torch.randn(25),folder/'layer_8_feature.pt')
 with (data/'predictions_val.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['id','task','prompt_lang','vector_lang','concept','layer','alpha','mask','normalized','correct'])
  w.writeheader()
  for task in TASKS:
   for lang in LANGS:
    for i in range(3):
     r={'id':f'{task}-{i}','task':task,'prompt_lang':lang,'vector_lang':'none',
        'concept':'none','layer':'-1','alpha':'0.0','mask':'both','normalized':'0','correct':1}
     w.writerow(r)
     w.writerow({**r,'vector_lang':'sw','concept':'counting' if task=='count' else 'spatial','layer':'8','alpha':'0.1','correct':int(i>0)})
 analyze(data,'val')
 macro=list(csv.DictReader((data/'analysis_val/four_task_macro_accuracy.csv').open()))
 assert len(macro)==3,macro
 assert all(abs(float(r['macro_gain_pp'])+33.3333333333)<0.01 for r in macro)
print('CPU contract tests PASSED')
