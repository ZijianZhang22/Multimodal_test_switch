#!/usr/bin/env python3
"""MMBench A: tune maps on bilingual DEV subset, then ONE fixed held-out test.

Strict protocol:
  - Frozen Flickr8K-trained maps, no refitting.
  - Disjoint images between parameter tuning and final test.
  - Two independently tuned methods: text_visual_ot, text_ot.
  - Baseline ZH and EN references share prompts across conditions.
  - Per-condition JSONL checkpoints; no test-dependent selection.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from prepare_data import digest, read_jsonl
from prepare_mmbench import LETTERS, pixels_signature
from geometry import Alignment, load_map
from vlm import load_model, generate, question, letter
from scipy.stats import binomtest

METHODS=('text_visual_ot','text_ot')
CONDITIONS=('baseline_zh','baseline_en')


def protocol(exp, data, layers, strengths):
    exp=Path(exp).resolve();data=Path(data).resolve()
    src=exp/'settings.json'; dataset=data/'manifest.json'
    if not src.is_file() or not dataset.is_file():raise FileNotFoundError('Missing original settings.json or prepared MMBench manifest')
    settings=json.loads(src.read_text()); manifest=json.loads(dataset.read_text())
    if settings.get('language')!='zh' or settings.get('arm')!='a':
        raise ValueError('Expected Chinese original A-only maps')
    if Path(settings['out']).resolve()!=exp:raise ValueError('Incorrect original experiment location')
    for key, expected in settings.get('file_integrity',{}).items():
        if not Path(key).is_file() or digest(key)!=expected:raise ValueError(f'Original data integrity failed: {key}')
    paths={name:data/f'{name}.jsonl' for name in ('tune','test')}
    for key,p in paths.items():
        if not p.is_file() or digest(p)!=manifest['sha256'][key]:raise ValueError(f'MMBench dataset changed: {p}')
    datasets={key:read_jsonl(p) for key,p in paths.items()}
    if not datasets['tune'] or not datasets['test']:raise ValueError('Empty tune/test split')
    if not set(r['id'] for r in datasets['tune']).isdisjoint(r['id'] for r in datasets['test']):
        raise ValueError('Question leakage')
    img_tune={r['image_pixel_sha256'] for r in datasets['tune']}
    img_test={r['image_pixel_sha256'] for r in datasets['test']}
    if img_tune&img_test:raise ValueError('Image leakage between tune and test')
    for r in datasets['tune']+datasets['test']:
        from PIL import Image
        p=Path(r['image'])
        if not p.is_file():raise FileNotFoundError(p)
        with Image.open(p) as im:
            rgb=im.convert('RGB')
            h=hashlib.sha256(f'{rgb.size}'.encode()+rgb.tobytes()).hexdigest()
        if h!=r['image_pixel_sha256']:raise ValueError('Changed MMBench image '+str(p))
        if r['answer'] not in r['allowed'] or len(r['allowed'])<2:
            raise ValueError('Malformed answer/options')
    # No MMBench item may be identical to an image in the Flickr alignment pool.
    from PIL import Image
    training_hash=set()
    for g in ('source','english','visual'):
        for r in read_jsonl(Path(settings['pools'])/(g+'.jsonl')):
            with Image.open(r['image']) as im: rgb=im.convert('RGB')
            training_hash.add(hashlib.sha256(f'{rgb.size}'.encode()+rgb.tobytes()).hexdigest())
    if training_hash & (img_tune|img_test):raise ValueError('Flickr fitting image occurs in MMBench evaluation')
    if not set(layers).issubset(settings['layers']):raise ValueError('Candidate layers not present in original fitting')
    mp={}; map_shas={}
    pool_sha=digest(Path(settings['pools'])/'manifest.json')
    for method in METHODS:
        for layer in layers:
            p=exp/'maps'/f'layer{layer}_{method}.npz'
            if not p.is_file():raise FileNotFoundError(f'Missing trained map {p}')
            mp[(method,layer)],_ = load_map(p,required=dict(layer=layer,language='zh',model_id=settings['model'],pool_sha=pool_sha))
            map_shas[f'{method}/L{layer}']=digest(p)
    config=dict(model=settings['model'],settings_sha256=digest(src),
                mmb_manifest_sha256=digest(dataset),
                layer_grid=list(layers),strength_grid=list(strengths),
                map_sha256=map_shas, max_new_tokens=settings['max_tokens'],
                norm_cap=settings['max_relative_norm'],seed=manifest['seed'])
    return settings,manifest,datasets,mp,config


def eval_tag(method,layer,alpha,split):
    return f'{split}_{method}_L{layer}_a{alpha:g}'


def eval_one(model,proc,rows,out,tag,settings,mapping=None,layer=None,alpha=None,lang='zh'):
    out.mkdir(parents=True,exist_ok=True)
    p=out/(tag+'.jsonl')
    already={}
    if p.exists():
        with p.open('rb+') as f:
            raw=f.read()
            if raw and not raw.endswith(b'\n'):
                cutoff=raw.rfind(b'\n')+1
                f.seek(cutoff);f.truncate();raw=raw[:cutoff]
        expected={r['id']:r for r in rows}
        for line in raw.decode('utf-8').splitlines():
            obj=json.loads(line)
            rid=obj['id']
            if rid in already or rid not in expected:raise ValueError(f'Duplicate/unexpected checkpoint ID {rid}')
            row=expected[rid]
            check_saved_record(obj,row,tag,lang)
            already[rid]=obj
    operator=None
    if mapping is not None:
        align=Alignment(mapping,device='cuda')
        def operator(h,v):return align.apply(h,v,strength=alpha,mode='direct',max_relative_norm=settings['max_relative_norm'])
    with p.open('a',encoding='utf-8',buffering=1) as f:
        for i,row in enumerate(rows,1):
            if row['id'] in already: continue
            prompt=question(row,lang)
            output=generate(model,proc,row,prompt,layer=layer,operator=operator,
                            max_new_tokens=settings['max_tokens'])
            parsed=letter(output)
            valid=parsed in row['allowed']
            obj={'id':row['id'],'tag':tag,'answer':row['answer'],'allowed':row['allowed'],
                 'img_sha256':row['image_pixel_sha256'],
                 'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                 'prediction':output.replace('\n',' '),'parsed':parsed,
                 'valid':int(valid),'correct':int(valid and parsed==row['answer'])}
            f.write(json.dumps(obj,ensure_ascii=False)+'\n')
            already[row['id']]=obj
            if i%25==0 or i==len(rows):
                f.flush();os.fsync(f.fileno())
                print('Evaluate',tag,i,'/',len(rows),flush=True)
    if set(already)!={r['id'] for r in rows}:raise ValueError('Evaluation did not complete')
    return already


def check_saved_record(obj,row,tag,lang):
    parsed = obj.get('parsed', '')
    allowed = row['allowed']
    valid = int(parsed in allowed)
    if (obj.get('tag') != tag or obj.get('answer') != row['answer'] or
        obj.get('allowed') != allowed or
        obj.get('img_sha256') != row['image_pixel_sha256'] or
        obj.get('prompt_sha256') != hashlib.sha256(question(row,lang).encode()).hexdigest() or
        parsed != letter(obj.get('prediction', '')) or
        obj.get('valid') != valid or
        obj.get('correct') != int(valid and parsed == row['answer'])):
        raise ValueError(f'Checkpoint stale/corrupt: {tag}, {row["id"]}')


def score(rows,rec):
    return {'n':len(rows),'correct':sum(rec[r['id']]['correct'] for r in rows),
            'accuracy':sum(rec[r['id']]['correct'] for r in rows)/len(rows),
            'valid_rate':sum(rec[r['id']]['valid'] for r in rows)/len(rows)}


def choose(rows,pred_dir,layers,strengths):
    selected={};grid=[]
    for method in METHODS:
        for layer in layers:
            for alpha in strengths:
                tag=eval_tag(method,layer,alpha,'tune')
                preds=read_prediction_file(pred_dir/(tag+'.jsonl'),rows,tag)
                row={'method':method,'layer':layer,'alpha':alpha,**score(rows,preds)}
                grid.append(row)
        candidates=[r for r in grid if r['method']==method]
        # Deterministic tie: accuracy then valid rate, smallest alpha, lowest layer.
        selected[method]=sorted(candidates,key=lambda r:(-r['accuracy'],-r['valid_rate'],r['alpha'],r['layer']))[0]
    return selected,grid


def read_prediction_file(path,rows,tag,lang='zh'):
    if not path.is_file():raise FileNotFoundError(path)
    rec={}
    for obj in (json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x):
        if obj['id'] in rec or obj['tag']!=tag:raise ValueError('Bad prediction file '+str(path))
        rec[obj['id']]=obj
    if set(rec)!={r['id'] for r in rows}:raise ValueError(f'Incomplete {tag}: {len(rec)}/{len(rows)}')
    for row in rows:check_saved_record(rec[row['id']],row,tag,lang)
    return rec


def paired_stats(rows,base,comparison,seed=42,n_boot=2000):
    b=np.asarray([base[r['id']]['correct'] for r in rows],int)
    z=np.asarray([comparison[r['id']]['correct'] for r in rows],int)
    delta=z-b
    groups={}
    for i,r in enumerate(rows):groups.setdefault(r['image_pixel_sha256'],[]).append(i)
    clusters=list(groups.values());rng=np.random.default_rng(seed)
    boots=np.zeros(n_boot,dtype=float)
    for j in range(n_boot):
        selected=rng.integers(len(clusters),size=len(clusters))
        idx=[k for c in selected for k in clusters[int(c)]]
        boots[j]=100*float(np.mean(delta[idx]))
    repairs=int(np.sum(delta>0));harm=int(np.sum(delta<0))
    tot=np.array([delta[idx].sum() for idx in clusters],int)
    exceed=0
    observed=abs(tot.sum())
    for j in range(n_boot):
        signs=rng.integers(0,2,size=len(clusters))*2-1
        if abs(int(signs@tot))>=observed:exceed+=1
    return dict(n=len(rows),unique_images=len(clusters),baseline_correct=int(b.sum()),
                compared_correct=int(z.sum()),baseline_accuracy=float(b.mean()),
                compared_accuracy=float(z.mean()),delta_pp=float(100*np.mean(delta)),
                ci95_low_pp=float(np.quantile(boots,.025)),
                ci95_high_pp=float(np.quantile(boots,.975)),repaired=repairs,harmed=harm,
                baseline_valid_rate=float(np.mean([base[r['id']]['valid'] for r in rows])),
                compared_valid_rate=float(np.mean([comparison[r['id']]['valid'] for r in rows])),
                image_cluster_signflip_p=float((exceed+1)/(n_boot+1)),
                exact_mcnemar_p=float(binomtest(min(repairs,harm),repairs+harm,.5).pvalue if repairs+harm else 1.0)
                if len(clusters)==len(rows) else None)


def store_config(out,config):
    p=out/'run_config.json';out.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if json.loads(p.read_text())!=config:raise ValueError('Different protocol in same run output; use a fresh --out')
    else:p.write_text(json.dumps(config,ensure_ascii=False,indent=2))


def tune(args,settings,datasets,maps,config):
    rows=datasets['tune'];out=args.out.resolve();pred_dir=out/'predictions';pred_dir.mkdir(parents=True,exist_ok=True)
    processor,model=load_model(settings['model'])
    eval_one(model,processor,rows,pred_dir,'tune_baseline_zh',settings)
    for method in METHODS:
        for layer in config['layer_grid']:
            for alpha in config['strength_grid']:
                eval_one(model,processor,rows,pred_dir,eval_tag(method,layer,alpha,'tune'),settings,
                         mapping=maps[(method,layer)],layer=layer,alpha=alpha)
    selected,grid=choose(rows,pred_dir,config['layer_grid'],config['strength_grid'])
    new={'config_sha256':digest(out/'run_config.json'), 'selection_only':'MMBench tuning subset',
         'baseline':score(rows,read_prediction_file(pred_dir/'tune_baseline_zh.jsonl',rows,'tune_baseline_zh')),
         'selected':selected,'grid':grid}
    file=out/'selection_mmbench.json'
    if file.exists() and json.loads(file.read_text())!=new:raise ValueError('Tuning choice changed; do not overwrite selection')
    if not file.exists():file.write_text(json.dumps(new,ensure_ascii=False,indent=2))
    print('TUNING SELECTION:',json.dumps(selected,ensure_ascii=False),flush=True)


def test(args,settings,datasets,maps,config):
    out=args.out.resolve();file=out/'selection_mmbench.json'
    if not file.is_file():raise FileNotFoundError('Tuning must finish before test')
    selected=json.loads(file.read_text())
    if selected['config_sha256']!=digest(out/'run_config.json'):
        raise ValueError('Selection not from this run configuration')
    if set(selected['selected'])!=set(METHODS):raise ValueError('Incomplete selection')
    rows=datasets['test'];pred_dir=out/'predictions';pred_dir.mkdir(parents=True,exist_ok=True)
    processor,model=load_model(settings['model'])
    base=eval_one(model,processor,rows,pred_dir,'test_baseline_zh',settings)
    english=eval_one(model,processor,rows,pred_dir,'test_baseline_en',settings,lang='en')
    summary={'note':'LOCAL image-disjoint held-out split of public MMBench DEV, not official leaderboard CircularEval',
             'dataset_counts':{'tune':len(datasets['tune']),'test':len(rows)},
             'selected_from_tune':selected['selected'],
             'baseline_zh':score(rows,base),'english_reference':score(rows,english),
             'arms':{},'by_category':{}}
    for method in METHODS:
        choice=selected['selected'][method];layer=choice['layer'];alpha=choice['alpha']
        tag=eval_tag(method,layer,alpha,'test')
        rec=eval_one(model,processor,rows,pred_dir,tag,settings,
                     mapping=maps[(method,layer)],layer=layer,alpha=alpha)
        summary['arms'][method]=dict(tag=tag,**paired_stats(rows,base,rec,config['seed']))
        summary['by_category'][method]={}
        for cat in sorted({r['category'] for r in rows}):
            subset=[r for r in rows if r['category']==cat]
            if len(subset)>=5:
                summary['by_category'][method][cat]=paired_stats(subset,base,rec,config['seed'],n_boot=500)
    path=out/'test_results.json'
    if path.exists() and json.loads(path.read_text())!=summary:
        raise ValueError('Existing test results differ; refusing to overwrite')
    if not path.exists():path.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    table=[{'method':'baseline_zh',**summary['baseline_zh']},
           {'method':'baseline_en',**summary['english_reference']}]
    for method,arm in summary['arms'].items():
        table.append({'method':method,'n':arm['n'],'correct':arm['compared_correct'],
                      'accuracy':arm['compared_accuracy'], 'valid_rate':arm['compared_valid_rate'],
                      'delta_pp':arm['delta_pp'],'repaired':arm['repaired'],'harmed':arm['harmed'],
                      'ci95_low_pp':arm['ci95_low_pp'],'ci95_high_pp':arm['ci95_high_pp'],
                      'image_cluster_signflip_p':arm['image_cluster_signflip_p']})
    fields=list(dict.fromkeys(k for row in table for k in row))
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(table)
    print('FINAL TEST:',json.dumps({'baseline':summary['baseline_zh'],'english':summary['english_reference'],
                                   'arms':summary['arms']},ensure_ascii=False,indent=2),flush=True)
    print('FINAL FILE:',path,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--layers',default='8,13,18')
    p.add_argument('--alphas',default='0.15,0.3,0.6')
    p.add_argument('--stage',choices=['tune','test','all'],default='all')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    layers=[int(i) for i in a.layers.split(',')]
    strengths=[float(i) for i in a.alphas.split(',')]
    if not layers or len(set(layers))!=len(layers) or any(k<0 for k in layers):raise ValueError('Invalid layers')
    if not strengths or len(set(strengths))!=len(strengths) or any(not 0<i<=2 for i in strengths):raise ValueError('Invalid strengths')
    settings,manifest,datasets,maps,config=protocol(a.experiment,a.data,layers,strengths)
    store_config(a.out.resolve(),config)
    print(f'MMBench plan: tune={len(datasets["tune"])} test={len(datasets["test"])}; '
          f'layers={layers} alphas={strengths}; frozen maps only',flush=True)
    if a.dry_run:return
    if a.stage in ('tune','all'):tune(a,settings,datasets,maps,config)
    if a.stage in ('test','all'):test(a,settings,datasets,maps,config)

if __name__=='__main__':main()
