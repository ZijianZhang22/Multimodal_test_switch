#!/usr/bin/env python3
"""Unsupervised geometry (A) and confidence/visual gated steering (B).

Stages: extract -> fit -> validate -> test; oracle is a *separate paired
mechanistic diagnostic*, never counted as unsupervised performance.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
from geometry import fit_map,load_map,save_map,Alignment
from prepare_data import digest,read_jsonl,validate_pools
from vlm import MODEL,load_model,inputs,captured_states,generate,question,caption,letter,oracle_operator

METHODS=('mean_shift','text_ot','text_visual_ot','text_noise_ot')
MODES=('direct','confidence','visual','confidence_visual','constant_half')


def load_eval(path,language):
    rows=read_jsonl(path)
    if not rows:raise ValueError('No CV-Bench rows')
    seen=set()
    for row in rows:
        if not {'id','task','image','answer','questions'}<=row.keys():
            raise ValueError('Bad CV-Bench schema')
        if language not in row['questions'] or 'en' not in row['questions']:
            raise ValueError('Missing bilingual evaluation questions')
        if row['id'] in seen:raise ValueError('Duplicate CV-Bench id')
        seen.add(row['id'])
        if not Path(row['image']).is_file():raise FileNotFoundError(row['image'])
    return rows


def config_key(args,train_hash):
    return dict(model_id=args.model,language=args.language,layers=args.layers,
                pool_sha=train_hash,prompt_version='caption-v1-neutral-visual-v1',
                extractor='last-text-token+mean-image-prefix')


def leakage_audit(args):
    """No Flickr train-image bytes may appear in either local CV-Bench split."""
    pools=validate_pools(args.pools)
    eval_rows=load_eval(args.val,args.language)+load_eval(args.test,args.language)
    train_images={r['image'] for group in pools.values() for r in group}
    evaluation_images={r['image'] for r in eval_rows}
    if train_images&evaluation_images:
        raise ValueError('Same image file used in training and CV-Bench evaluation')
    training_sha={digest(p) for p in train_images}
    eval_sha={digest(p) for p in evaluation_images}
    if training_sha&eval_sha:
        raise ValueError('Same image content reused across Flickr training and CV-Bench validation/test')
    print('Leakage preflight OK:',len(training_sha),'train images vs',len(eval_sha),'evaluation images',flush=True)


def extract(args):
    leakage_audit(args)
    pools=validate_pools(args.pools)
    sha=digest(Path(args.pools)/'manifest.json')
    fingerprint=config_key(args,sha)
    out=Path(args.out)/'features';out.mkdir(parents=True,exist_ok=True)
    cache_meta=out/'manifest.json'
    if cache_meta.exists():
        if json.loads(cache_meta.read_text())!=fingerprint:
            raise ValueError('Activation cache metadata mismatch; use separate --out')
    else:cache_meta.write_text(json.dumps(fingerprint,indent=2))
    processor,model=load_model(args.model)
    for pool,rows in pools.items():
        group=out/pool;group.mkdir(exist_ok=True)
        for i,row in enumerate(rows):
            file=group/f'{i:05d}.npz'
            if file.exists():
                with np.load(file,allow_pickle=False) as data:
                    if str(data['id'].item())!=row['id'] or len(data['states'])!=len(args.layers):
                        raise ValueError('Stale feature file: '+str(file))
                continue
            act=captured_states(model,inputs(processor,row,caption(row)),args.layers,
                                need_visual=(pool=='visual'))
            states=np.stack([act[l]['visual' if pool=='visual' else 'text'] for l in args.layers])
            tmp=file.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,id=np.array(row['id']),states=states.astype('float32'))
            tmp.replace(file)
            if (i+1)%25==0:print('Extract',pool,i+1,'/',len(rows),flush=True)
    print('Extraction done; all model parameters remain frozen',flush=True)


def load_features(args):
    base=Path(args.out)/'features'
    sha=digest(Path(args.pools)/'manifest.json')
    if json.loads((base/'manifest.json').read_text())!=config_key(args,sha):
        raise ValueError('Cache provenance mismatch')
    groups=validate_pools(args.pools)
    data={}
    for name,rows in groups.items():
        data[name]=[]
        for i,row in enumerate(rows):
            with np.load(base/name/f'{i:05d}.npz',allow_pickle=False) as z:
                if str(z['id'].item())!=row['id']:raise ValueError('Cache order corruption')
                data[name].append(z['states'])
        data[name]=np.stack(data[name])
    return data,sha


def fit(args):
    data,sha=load_features(args)
    dest=Path(args.out)/'maps';dest.mkdir(exist_ok=True)
    for li,layer in enumerate(args.layers):
        x=data['source'][:,li,:];y=data['english'][:,li,:];z=data['visual'][:,li,:]
        for method in METHODS:
            mp=fit_map(x,y,z,method=method,dim=args.dim,iterations=args.iterations,
                       batch=args.batch,visual_weight=args.visual_weight,seed=args.seed)
            meta=dict(layer=layer,language=args.language,model_id=args.model,
                      pool_sha=sha,seed=args.seed,dim=args.dim,iterations=args.iterations,
                      batch=args.batch,visual_weight=args.visual_weight)
            destfile=dest/f'layer{layer}_{method}.npz'
            save_map(destfile,mp,meta)
            print('Fit',destfile,'last W change',mp['history'][-1] if mp['history'] else '-',flush=True)


def load_operator(args,layer,method,mode,alpha,model):
    sha=digest(Path(args.pools)/'manifest.json')
    mp,meta=load_map(Path(args.out)/'maps'/f'layer{layer}_{method}.npz',
                     required=dict(layer=layer,language=args.language,model_id=args.model,pool_sha=sha))
    wrapped=Alignment(mp,device='cuda')
    def op(h,v):return wrapped.apply(h,v,strength=alpha,mode=mode,max_relative_norm=args.max_relative_norm)
    return op


def evaluate_rows(args,rows,model,proc,lang,tag,layer=None,op=None):
    path=Path(args.out)/'predictions'/f'{tag}.csv';path.parent.mkdir(exist_ok=True)
    expected=[r['id'] for r in rows]
    if path.exists():
        old=list(csv.DictReader(path.open(encoding='utf-8',newline='')))
        if [r['id'] for r in old]!=expected or any(r.get('tag')!=tag for r in old):
            raise ValueError('Stale eval predictions; choose a fresh --out')
        return metric(old)
    results=[]
    for i,row in enumerate(rows):
        text=generate(model,proc,row,question(row,lang),layer=layer,operator=op,
                      max_new_tokens=args.max_tokens)
        pred=letter(text)
        results.append(dict(id=row['id'],task=row['task'],language=lang,tag=tag,
                            prediction=text.replace('\n',' '),parsed=pred,
                            answer=row['answer'],correct=int(pred==row['answer'])))
        if (i+1)%25==0:print('Evaluate',tag,i+1,'/',len(rows),flush=True)
    tmp=path.with_suffix('.tmp.csv')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','task','language','tag','prediction','parsed','answer','correct'])
        w.writeheader();w.writerows(results)
    tmp.replace(path)
    return metric(results)


def metric(rows):
    return dict(n=len(rows),accuracy=round(sum(int(r['correct']) for r in rows)/len(rows),6),
                valid_rate=round(sum(bool(r['parsed']) for r in rows)/len(rows),6),
                by_task={k:round(sum(int(x['correct']) for x in v)/len(v),6) for k,v in by_task(rows).items()})


def by_task(rows):
    out=defaultdict(list)
    for r in rows:out[r['task']].append(r)
    return out


def cache_oracle_states(args,rows,model,processor):
    # Paired English states are used ONLY for causal patching; never fit an OT map.
    dest=Path(args.out)/'oracle_states';dest.mkdir(exist_ok=True)
    result=[]
    for i,row in enumerate(rows):
        p=dest/f'{i:05d}.npz'
        if p.is_file():
            with np.load(p,allow_pickle=False) as v:
                if str(v['id'].item())!=row['id'] or v['english'].shape[0]!=len(args.layers):
                    raise ValueError('Oracle cache mismatch')
                states=v['english'].copy()
        else:
            hidden=captured_states(model,inputs(processor,row,question(row,'en')),args.layers)
            states=np.stack([hidden[l]['text'] for l in args.layers])
            tmp=p.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,id=np.array(row['id']),english=states)
            tmp.replace(p)
        result.append(states)
    return np.stack(result)


def oracle(args,rows,model,proc):
    baseline=evaluate_rows(args,rows,model,proc,args.language,'val_baseline_'+args.language)
    english=evaluate_rows(args,rows,model,proc,'en','val_baseline_en')
    states=cache_oracle_states(args,rows,model,proc)
    table=[]
    for index,layer in enumerate(args.layers):
        for alpha in args.oracle_strengths:
            tag=f'oracle_val_L{layer}_a{alpha:g}'
            p=Path(args.out)/'predictions'/f'{tag}.csv'
            saved=None
            if p.exists():saved=list(csv.DictReader(p.open(encoding='utf-8')))
            if saved is None:
                rec=[]
                for i,row in enumerate(rows):
                    op=oracle_operator(states[i,index],alpha,args.max_relative_norm)
                    predtext=generate(model,proc,row,question(row,args.language),layer=layer,
                                      operator=op,max_new_tokens=args.max_tokens)
                    pred=letter(predtext)
                    rec.append(dict(id=row['id'],task=row['task'],language=args.language,tag=tag,
                                    prediction=predtext.replace('\n',' '),parsed=pred,answer=row['answer'],
                                    correct=int(pred==row['answer'])))
                p.parent.mkdir(exist_ok=True)
                with p.open('w',encoding='utf-8',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rec[0]));w.writeheader();w.writerows(rec)
                saved=rec
            if [r['id'] for r in saved]!=[r['id'] for r in rows]:
                raise ValueError('Oracle prediction order mismatch')
            source=list(csv.DictReader((Path(args.out)/'predictions'/('val_baseline_'+args.language+'.csv')).open()))
            eng=list(csv.DictReader((Path(args.out)/'predictions'/'val_baseline_en.csv').open()))
            repaired=sum(int(not int(a['correct']) and int(b['correct']) and int(e['correct']))
                         for a,b,e in zip(source,saved,eng))
            harmed=sum(int(int(a['correct']) and not int(b['correct'])) for a,b in zip(source,saved))
            table.append(dict(layer=layer,alpha=alpha,**metric(saved),english_correct_zh_wrong_repaired=repaired,
                              previously_correct_harmed=harmed))
    out=Path(args.out)/'oracle_val.json'
    out.write_text(json.dumps(dict(baseline=baseline,english=english,grid=table,
                                   note='Oracle diagnostic using paired EN questions; NOT unsupervised'),indent=2))
    print('Oracle validation written',out,flush=True)


def validate(args,rows,model,proc):
    """Two-stage, validation-only hyperparameter search.

    A: all layers x four maps x alpha with direct injection.
    B: only A-selected layer per map x gating modes x alpha, with direct as
       common reference. A and B are genuinely different intervention functions.
    """
    baseline=evaluate_rows(args,rows,model,proc,args.language,'val_baseline_'+args.language)
    en=evaluate_rows(args,rows,model,proc,'en','val_baseline_en')
    scores=[]
    for layer in args.layers:
        for method in METHODS:
            for alpha in args.strengths:
                op=load_operator(args,layer,method,'direct',alpha,model)
                tag=f'val_L{layer}_{method}_direct_a{alpha:g}'
                stats=evaluate_rows(args,rows,model,proc,args.language,tag,layer,op)
                scores.append(dict(layer=layer,method=method,mode='direct',strength=alpha,**stats))
                print('VALIDATE A',tag,stats['accuracy'],flush=True)
    selected={}
    for method in METHODS:
        candidates=[r for r in scores if r['method']==method and r['mode']=='direct']
        chosen=sorted(candidates,key=lambda x:(-x['accuracy'],-x['valid_rate'],x['strength'],x['layer']))[0]
        selected[f'A/{method}']=chosen
    # A-only mode: do not run B's confidence/visual/constant gate search.
    # Direct alignment is reported only once, under the A/ prefix.
    if args.arm == 'all':
        select_b_gates(args, scores, selected, rows, model, proc)
    report=dict(validation_base=baseline,english_reference=en,grid=scores,selected=selected,
                keys='A map/layer selected on validation; optional B gates use the same validation; no test tuning')
    path=Path(args.out)/'selection_val.json'
    path.write_text(json.dumps(report,indent=2))
    print('Selection saved',path,flush=True)


def select_b_gates(args,scores,selected,rows,model,proc):
    # B uses the A-selected map/layer. Select gates independently on val.
    # No duplicate B 'dynamic' arm: ungated h+alpha(T(h)-h) IS A direct.
    for method in ('text_ot','text_visual_ot'):
        best=selected[f'A/{method}']
        for mode in ('confidence','visual','confidence_visual','constant_half'):
            candidates=[]
            for alpha in args.strengths:
                op=load_operator(args,best['layer'],method,mode,alpha,model)
                tag=f'val_L{best["layer"]}_{method}_{mode}_a{alpha:g}'
                stats=evaluate_rows(args,rows,model,proc,args.language,tag,best['layer'],op)
                rec=dict(layer=best['layer'],method=method,mode=mode,strength=alpha,**stats)
                scores.append(rec);candidates.append(rec)
                print('VALIDATE B',tag,stats['accuracy'],flush=True)
            selected[f'B/{method}/{mode}']=sorted(candidates,key=lambda x:(-x['accuracy'],-x['valid_rate'],x['strength']))[0]

def selected_for_arm(selected,arm):
    if arm not in ('a','all'):
        raise ValueError(f'Unexpected arm={arm}')
    result={key:value for key,value in selected.items() if arm=='all' or key.startswith('A/')}
    if not result or (arm=='a' and not all(key.startswith('A/') for key in result)):
        raise ValueError('No valid A method selected')
    return result


def run_test(args,rows,model,proc):
    selection_path=Path(args.out)/'selection_val.json'
    if not selection_path.is_file():raise FileNotFoundError('Run validate before test')
    report=json.loads(selection_path.read_text())
    baseline=evaluate_rows(args,rows,model,proc,args.language,'test_baseline_'+args.language)
    en=evaluate_rows(args,rows,model,proc,'en','test_baseline_en')
    test=[]
    for name,choice in selected_for_arm(report['selected'],args.arm).items():
        if choice['layer'] not in args.layers or choice['method'] not in METHODS or choice['mode'] not in MODES:
            raise ValueError('Selected method/grid not available')
        op=load_operator(args,choice['layer'],choice['method'],choice['mode'],choice['strength'],model)
        tag=f'test_L{choice["layer"]}_{choice["method"]}_{choice["mode"]}_a{choice["strength"]:g}'
        stats=evaluate_rows(args,rows,model,proc,args.language,tag,choice['layer'],op)
        test.append(dict(arm=name,config=tag,**stats))
    summary=dict(language=args.language,baseline=baseline,english_reference=en,arms=test,
                 note='User-local split of publicly available CV-Bench test; NOT official held-out benchmark')
    (Path(args.out)/'test_results.json').write_text(json.dumps(summary,indent=2))
    print('TEST results saved',Path(args.out)/'test_results.json',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('extract','fit','oracle','validate','test'),required=True)
    p.add_argument('--arm',choices=('a','all'),default='all',help='a: run direction A only; all: A plus B')
    p.add_argument('--pools',default='outputs/flickr_zh')
    p.add_argument('--val',default='outputs/cvbench/val.jsonl')
    p.add_argument('--test',default='outputs/cvbench/test.jsonl')
    p.add_argument('--out',default='outputs/experiment_zh')
    p.add_argument('--language',choices=('zh','sw'),default='zh')
    p.add_argument('--model',default=MODEL)
    p.add_argument('--layers',default='4,8,13,18,22')
    p.add_argument('--dim',type=int,default=16)
    p.add_argument('--iterations',type=int,default=30)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--visual-weight',type=float,default=.25)
    p.add_argument('--strengths',default='.15,.3,.6')
    p.add_argument('--oracle-strengths',default='.3,.6,1')
    p.add_argument('--max-relative-norm',type=float,default=.15)
    p.add_argument('--max-tokens',type=int,default=8)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    a.layers=[int(x) for x in a.layers.split(',')]
    a.strengths=[float(x) for x in a.strengths.split(',')]
    a.oracle_strengths=[float(x) for x in a.oracle_strengths.split(',')]
    if not a.layers or len(set(a.layers))!=len(a.layers) or any(l<0 for l in a.layers):
        raise ValueError('Invalid candidate layers')
    if not a.strengths or any(not 0<x<=2 for x in a.strengths):
        raise ValueError('Invalid strength grid')
    if not all(0<x<=2 for x in a.oracle_strengths):raise ValueError('Invalid oracle strength grid')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    settings={k:v for k,v in vars(a).items() if k!='stage'}
    settings['file_integrity']={str(path):digest(path) for path in
        (Path(a.pools)/'manifest.json',Path(a.val),Path(a.test)) if path.is_file()}
    settings_path=out/'settings.json'
    if settings_path.exists():
        old=json.loads(settings_path.read_text())
        # Selection cannot be silently changed between validation and test.
        if old!=settings:raise ValueError('Experiment settings mismatch; use NEW --out')
    else:settings_path.write_text(json.dumps(settings,indent=2,ensure_ascii=False))
    if a.stage=='fit':return fit(a)
    if a.stage=='extract':return extract(a)
    processor,model=load_model(a.model)
    rows=load_eval(a.val if a.stage!='test' else a.test,a.language)
    if a.stage=='oracle':return oracle(a,rows,model,processor)
    if a.stage=='validate':return validate(a,rows,model,processor)
    if a.stage=='test':return run_test(a,rows,model,processor)

if __name__=='__main__':main()
