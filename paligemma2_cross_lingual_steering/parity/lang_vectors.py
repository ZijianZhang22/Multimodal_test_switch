#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm
from common import TASKS,build_prompt,jsonl_read
from combo_core import Runner,capture_last

def halves(samples):
    return ([s for t in TASKS for s in [x for x in samples if x['task']==t and x['split']=='tune'][0::2]],
            [s for t in TASKS for s in [x for x in samples if x['task']==t and x['split']=='tune'][1::2]])

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True,type=Path)
    p.add_argument('--langs',default='zh,sw')
    p.add_argument('--styles',default='upstream_en')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--random-seeds',default='0')
    p.add_argument('--reuse-if-valid',action='store_true')
    a=p.parse_args()
    if a.limit<0:p.error('invalid limit')
    langs=[x.strip() for x in a.langs.split(',') if x.strip()]
    styles=[x.strip() for x in a.styles.split(',') if x.strip()]
    seeds=[int(x) for x in a.random_seeds.split(',') if x.strip()]
    if not langs or len(set(langs))!=len(langs) or any(x not in ('zh','sw') for x in langs):p.error('langs must be zh/sw')
    ext,_=halves(jsonl_read(a.data/'visual_eval.jsonl'))
    if a.limit:ext=[s for t in TASKS for s in [x for x in ext if x['task']==t][:a.limit]]
    if not ext:raise RuntimeError('empty extraction split')
    d=a.data/'lang_vectors';d.mkdir(parents=True,exist_ok=True)
    expected=dict(ids=[s['id'] for s in ext],langs=langs,styles=styles,seeds=seeds,
        dataset_sha256=hashlib.sha256((a.data/'dataset_manifest.json').read_bytes()).hexdigest())
    mp=d/'manifest.json'
    if a.reuse_if_valid and mp.exists():
        old=json.loads(mp.read_text())
        if any(old.get(k)!=v for k,v in expected.items()):
            raise RuntimeError('Language vector cache mismatch. Use separate DATA or recompute deliberately.')
        if all((d/(language+'_'+style+'.pt')).exists() for language in langs for style in styles):
            print('Reusing verified language vectors');return
    r=Runner(a.data)
    layers=list(range(len(r.model.language_model.model.layers)))
    meta={**expected,'files':{}}
    for style in styles:
        sums={k:None for k in ['en']+langs};norms={k:None for k in sums}
        for sample in tqdm(ext,desc='extract '+style):
            with Image.open(a.data/sample['image']) as im:image=im.copy()
            for language in sums:
                inp=r.inputs(image,build_prompt(sample,language,style))
                with torch.inference_mode(),capture_last(r.model,layers) as states:r.model(**inp)
                h=torch.stack([states[layer] for layer in layers])
                sums[language]=h if sums[language] is None else sums[language]+h
                ns=h.norm(dim=-1)
                norms[language]=ns if norms[language] is None else norms[language]+ns
        mean={k:v/len(ext) for k,v in sums.items()}
        for language in langs:
            vec=mean['en']-mean[language]
            obj={'vec':vec,'h_norm':norms[language]/len(ext),'h_norm_en':norms['en']/len(ext],
                 'style':style,'lang':language,'definition':'EN minus LANG last prompt token'}
            for seed in seeds:
                g=torch.Generator().manual_seed(seed)
                x=torch.randn(vec.shape,generator=g)
                obj['random_s'+str(seed)]=x*(vec.norm(dim=-1,keepdim=True)/x.norm(dim=-1,keepdim=True).clamp_min(1e-12))
            f=d/(language+'_'+style+'.pt');torch.save(obj,f)
            meta['files'][f.name]=[float(x) for x in vec.norm(dim=-1)]
    mp.write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
