#!/usr/bin/env python3
"""Select on odd Tune half; evaluate frozen language/concept interventions on held-out CV-Bench."""
import argparse,csv,hashlib,json
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm
from common import TASKS,jsonl_read
from lang_vectors import halves
FIELDS=['id','task','split','prompt_lang','style','cond','lang_layer','lang_alpha','concept','concept_variant','concept_layer','concept_alpha','margin','p_gold','argmax_letter','argmax_correct','response','normalized','valid','correct']
def integers(s):
 out=[]
 for part in s.split(','):
  if '-' in part:
   a,b=map(int,part.split('-',1));out+=list(range(a,b+1))
  elif part.strip():out.append(int(part))
 return out
def concepts_from(path):
 obj=json.loads(path.read_text())
 if 'groups' in obj:
  chosen=obj.get('task_best_concept',{})
  return {g['task']:{'concept':g['concept'],'variant':'en','layer':g['layer'],'alpha':g['alpha']}
    for g in obj['groups'] if g['prompt_lang']=='en' and g['variant']=='en' and g['concept']==chosen.get(g['task'],g['concept'])}
 return {t:{'variant':'en',**v} for t,v in obj.items()}
def key(r):
 return tuple(str(r[k]) for k in ('id','prompt_lang','style','cond','lang_layer','lang_alpha','concept','concept_variant','concept_layer','concept_alpha'))
def main():
 p=argparse.ArgumentParser()
 for name in ('data','out'):p.add_argument('--'+name,required=True,type=Path)
 p.add_argument('--part',required=True,choices=['select','test'])
 p.add_argument('--langs',default='zh,sw');p.add_argument('--style',default='upstream_en')
 p.add_argument('--layers',default='4-22');p.add_argument('--alphas',default='0.25,0.5,1.0,1.5,2.0')
 p.add_argument('--lang-config',type=Path);p.add_argument('--concepts',type=Path)
 p.add_argument('--random-seed',default='0');p.add_argument('--limit',type=int,default=0)
 a=p.parse_args()
 if a.limit<0:p.error('limit must be nonnegative')
 if a.part=='test' and (not a.lang_config or not a.concepts):p.error('test requires --lang-config and --concepts')
 langs=[x.strip() for x in a.langs.split(',') if x.strip()]
 if not langs or len(set(langs))!=len(langs) or any(x not in ('zh','sw') for x in langs):p.error('langs must be distinct zh,sw')
 samples=jsonl_read(a.data/'visual_eval.jsonl')
 pool=halves(samples)[1] if a.part=='select' else [x for x in samples if x['split']=='test']
 if a.limit:pool=[s for t in TASKS for s in [x for x in pool if x['task']==t][:a.limit]]
 vecs={l:torch.load(a.data/'lang_vectors'/f'{l}_{a.style}.pt',map_location='cpu',weights_only=True) for l in langs}
 def lspec(lang,kind,layer,alpha):
  if kind=='none':return []
  obj=vecs[lang];v={'v':obj['vec'],'neg':-obj['vec'],'rand':obj[f'random_s{a.random_seed}']}[kind][layer]
  return [(layer,v,alpha)]
 plan=[]
 if a.part=='select':
  layers=integers(a.layers);alphas=[float(x) for x in a.alphas.split(',') if x.strip()]
  if not layers or not alphas or any(l<0 or l>=vecs[langs[0]]['vec'].shape[0] for l in layers):p.error('invalid layer grid')
  for lang in langs:
   plan.append((lang,'base','none',-1,0.,False))
   for l in layers:
    for alpha in alphas:plan.append((lang,'lang','v',l,alpha,False))
  concepts={}
 else:
  cfg=json.loads(a.lang_config.read_text())['chosen'];concepts=concepts_from(a.concepts)
  if any(t not in concepts for t in TASKS):raise RuntimeError('Concept config missing tasks')
  plan=[('en','base','none',-1,0.,False),('en','concept','none',-1,0.,True)]
  for lang in langs:
   l,al=int(cfg[lang]['layer']),float(cfg[lang]['alpha'])
   plan.extend([(lang,'base','none',-1,0.,False),(lang,'lang','v',l,al,False),
                (lang,'lang_neg','neg',l,al,False),(lang,'lang_rand','rand',l,al,False),
                (lang,'concept','none',-1,0.,True),(lang,'lang+concept','v',l,al,True)])
 a.out.parent.mkdir(parents=True,exist_ok=True)
 settings={k:str(v) for k,v in vars(a).items() if k!='out'}
 for k,path in [('dataset_sha',a.data/'dataset_manifest.json'),('vector_sha',a.data/'lang_vectors'/'manifest.json')]:
  settings[k]=hashlib.sha256(path.read_bytes()).hexdigest()
 if a.part=='test':
  settings['concept_sha']=hashlib.sha256(a.concepts.read_bytes()).hexdigest()
  settings['lang_config_sha']=hashlib.sha256(a.lang_config.read_bytes()).hexdigest()
 sp=a.out.with_suffix('.settings.json')
 if a.out.exists():
  if not sp.exists() or json.loads(sp.read_text())!=settings:raise RuntimeError('Resume settings mismatch. Use a new RUN.')
 else:sp.write_text(json.dumps(settings,indent=2))
 done=set()
 if a.out.exists():
  with a.out.open(newline='') as f:done={key(r) for r in csv.DictReader(f)}
 from combo_core import Runner
 runner=Runner(a.data)
 with a.out.open('a',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=FIELDS)
  if not done and a.out.stat().st_size==0:w.writeheader()
  for s in tqdm(pool,desc=a.part):
   image=None
   for lang,cond,kind,layer,alpha,use_c in plan:
    c=concepts.get(s['task']) if use_c else None
    row=dict(id=s['id'],task=s['task'],split=s['split'],prompt_lang=lang,style=a.style,cond=cond,
             lang_layer=layer,lang_alpha=alpha,concept=c['concept'] if c else 'none',
             concept_variant=c['variant'] if c else 'none',concept_layer=c['layer'] if c else -1,
             concept_alpha=c['alpha'] if c else 0.)
    if key(row) in done:continue
    if image is None:
     with Image.open(a.data/s['image']) as im:image=im.copy()
    row.update(runner.run(image,s,lang,a.style,lspec(lang,kind,layer,alpha) if lang!='en' else [],
                          c,do_generate=(a.part=='test')))
    w.writerow(row);done.add(key(row))
   f.flush()
 print('Saved',a.out)
if __name__=='__main__':main()
