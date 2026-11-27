#!/usr/bin/env python3
"""Only ZH original model vs preselected Text+Visual OT, on extended CV-Bench.

No map fitting, no Oracle patching, no English reference, no other OT arms.
Uses L8 alpha0.6 selected on earlier 237-item CV-Bench validation; refuses
mismatched saved selection/map/settings. Per-question durable JSONL checkpoints.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from prepare_data import digest,read_jsonl
from prepare_extended_cvbench import pixel_sha
from geometry import load_map, Alignment
from vlm import load_model,generate,question,letter

CONFIG_KEYS=('pools','val','test','out','language','model','layers','dim',
             'max_relative_norm','max_tokens','seed')


def load_protocol(experiment,extended):
    exp=Path(experiment).resolve()
    ext=Path(extended).resolve()
    settings_file=exp/'settings.json'
    selection_file=exp/'selection_val.json'
    for p in (settings_file,selection_file,ext/'manifest.json',ext/'test_extended.jsonl'):
        if not p.is_file():raise FileNotFoundError(f'Missing required file: {p}')
    settings=json.loads(settings_file.read_text(encoding='utf-8'))
    for k in CONFIG_KEYS:
        if k not in settings:raise ValueError(f'Missing original setting {k}')
    if Path(settings['out']).resolve()!=exp:raise ValueError('Settings experiment path changed')
    if settings['language']!='zh':raise ValueError('This pilot is Chinese-only')
    if settings.get('arm')!='a':raise ValueError('Expected prior validation with ARM=a')
    for file,expected in settings['file_integrity'].items():
        if not Path(file).is_file() or digest(file)!=expected:
            raise ValueError(f'Original calibration input changed: {file}')
    selected=json.loads(selection_file.read_text(encoding='utf-8'))['selected']
    config=selected.get('A/text_visual_ot')
    if not config or config.get('method')!='text_visual_ot' or config.get('mode')!='direct':
        raise ValueError('No previously selected visual OT configuration')
    layer=config['layer'];alpha=config['strength']
    if layer not in settings['layers'] or not 0<alpha<=2:
        raise ValueError('Invalid preselected layer/strength')
    manifest=json.loads((ext/'manifest.json').read_text(encoding='utf-8'))
    if (digest(ext/'test_extended.jsonl')!=manifest['extended_jsonl_sha256'] or
        digest(settings['val'])!=manifest['original_val_sha256'] or
        digest(settings['test'])!=manifest['original_test_sha256']):
        raise ValueError('Extended dataset provenance mismatch')
    rows=read_jsonl(ext/'test_extended.jsonl')
    if len(rows)!=manifest['target_total']:
        raise ValueError('Wrong number of evaluation rows')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate IDs')
    if any(r['evaluation_group'] not in ('curated_existing','new_mt_zh') or
           not r['questions'].get('zh') or r['answer'] not in tuple('ABCDEF')
           for r in rows):raise ValueError('Invalid extended row format')
    if (sum(r['evaluation_group']=='curated_existing' for r in rows)!=manifest['curated_count'] or
        sum(r['evaluation_group']=='new_mt_zh' for r in rows)!=manifest['machine_translated_count']):
        raise ValueError('Expanded dataset provenance group counts mismatch')
    # Hashes in the JSONL are assertions, not evidence that the local image
    # files still have the same contents. Rehash actual images on every start.
    file_pixels={}
    for r in rows:
        path=Path(r['image'])
        if not path.is_file():raise FileNotFoundError(path)
        if path not in file_pixels:file_pixels[path]=pixel_sha(path)
        if file_pixels[path]!=r['image_pixel_sha256']:
            raise ValueError(f'Image bytes/pixels changed after dataset construction: {path}')
    val_ids={r['id'] for r in read_jsonl(settings['val'])}
    if val_ids&{r['id'] for r in rows}:raise ValueError('Validation IDs included in test')
    # The old experiment already performed an exact-file audit. Recheck against
    # the newly extended set using *decoded pixels* to reject JPG/PNG overlaps.
    training_pixels=set()
    for part in ('source','english','visual'):
        path=Path(settings['pools'])/f'{part}.jsonl'
        for r in read_jsonl(path):training_pixels.add(pixel_sha(r['image']))
    eval_pixels={r['image_pixel_sha256'] for r in rows}
    if training_pixels&eval_pixels:
        raise ValueError('Train image overlaps extended evaluation by pixel content')
    print('Image leak audit:',len(training_pixels),'train images vs',
          len(eval_pixels),'test images, no exact-pixel overlap',flush=True)
    mp_path=exp/'maps'/f'layer{layer}_text_visual_ot.npz'
    sha=digest(Path(settings['pools'])/'manifest.json')
    mp,meta=load_map(mp_path,required=dict(layer=layer,language='zh',
                                           model_id=settings['model'],pool_sha=sha))
    return settings,config,manifest,rows,mp,mp_path


def cache_manifest(settings,config,manifest,mappath,extended):
    return {'model':settings['model'],
            'original_experiment':str(Path(settings['out']).resolve()),
            'original_settings_sha256':digest(Path(settings['out'])/'settings.json'),
            'original_selection_sha256':digest(Path(settings['out'])/'selection_val.json'),
            'map_sha256':digest(mappath),'layer':config['layer'],
            'alpha':config['strength'],
            'max_relative_norm':settings['max_relative_norm'],
            'max_new_tokens':settings['max_tokens'],
            'extended_sha256':manifest['extended_jsonl_sha256'],
            'target_total':manifest['target_total'],
            'translation_model':manifest['translation_model']}


def existing_records(path,expected_rows,tag):
    """Permit checkpoint/restart; refuse stale prompt/answer or malformed records."""
    expected={r['id']:r for r in expected_rows}
    saved={}
    if path.is_file():
        # Power loss may leave only the final record truncated. A complete
        # newline-terminated row is never discarded, and malformed complete
        # records still fail loudly. Truncate ONLY the partial trailing bytes.
        with path.open('rb+') as f:
            content=f.read()
            if content and not content.endswith(b'\n'):
                cutoff=content.rfind(b'\n')+1
                print(f'Recovering interrupted final JSONL record in {path}: '
                      f'truncating {len(content)-cutoff} bytes',flush=True)
                f.seek(cutoff);f.truncate()
                content=content[:cutoff]
        for n,line in enumerate(content.decode('utf-8').splitlines(),1):
                record=json.loads(line)
                rid=record['id']
                if rid in saved:raise ValueError('Duplicate checkpoint ID '+rid)
                if rid not in expected:raise ValueError('Unexpected checkpoint ID '+rid)
                row=expected[rid]
                if (record['tag']!=tag or record['answer']!=row['answer'] or
                    record['prompt_sha256']!=hashlib.sha256(row['questions']['zh'].encode()).hexdigest() or
                    record['image_pixel_sha256']!=row['image_pixel_sha256'] or
                    record['evaluation_group']!=row['evaluation_group']):
                    raise ValueError(f'Wrong checkpoint/protocol for {rid}')
                saved[rid]=record
    return saved


def make_record(row,tag,prediction):
    parsed=letter(prediction)
    return dict(id=row['id'],task=row['task'],evaluation_group=row['evaluation_group'],
                image_pixel_sha256=row['image_pixel_sha256'],
                prompt_sha256=hashlib.sha256(row['questions']['zh'].encode()).hexdigest(),
                tag=tag,answer=row['answer'],prediction=prediction.replace('\n',' '),
                parsed=parsed,correct=int(parsed==row['answer']))


def run_predictions(model,processor,rows,mp,config,settings,dest):
    op=Alignment(mp,device='cuda')
    def intervene(h,v):
        return op.apply(h,v,strength=config['strength'],mode='direct',
                        max_relative_norm=settings['max_relative_norm'])
    pred_dir=dest/'predictions';pred_dir.mkdir(parents=True,exist_ok=True)
    files={t:pred_dir/(t+'.jsonl') for t in ('baseline_zh','text_visual_ot')}
    saved={t:existing_records(files[t],rows,t) for t in files}
    n=len(rows)
    processed={t:len(saved[t]) for t in files}
    print('Resume',processed,'/',n,flush=True)
    handles={t:files[t].open('a',encoding='utf-8',buffering=1) for t in files}
    try:
        for i,row in enumerate(rows,1):
            for tag in ('baseline_zh','text_visual_ot'):
                if row['id'] in saved[tag]:continue
                prompt=question(row,'zh')
                if tag=='baseline_zh':
                    output=generate(model,processor,row,prompt,max_new_tokens=settings['max_tokens'])
                else:
                    output=generate(model,processor,row,prompt,
                                    layer=config['layer'],operator=intervene,
                                    max_new_tokens=settings['max_tokens'])
                record=make_record(row,tag,output)
                handles[tag].write(json.dumps(record,ensure_ascii=False)+'\n')
                handles[tag].flush()
                saved[tag][row['id']]=record
                processed[tag]+=1
            if i%25==0 or i==n:
                for f in handles.values():os.fsync(f.fileno())
                print(f'Evaluate {i}/{n}, baseline {processed["baseline_zh"]}/{n}, '
                      f'visual {processed["text_visual_ot"]}/{n}',flush=True)
    finally:
        for f in handles.values():f.close()
    return saved


def group_stats(rows,reference,predictions,seed=42,bootstrap=4000):
    """Cluster paired bootstrap by exact image pixels, not by translated text."""
    if not rows:raise ValueError('Empty group')
    b=np.asarray([reference[r['id']]['correct'] for r in rows],int)
    z=np.asarray([predictions[r['id']]['correct'] for r in rows],int)
    dif=z-b
    groups={}
    for idx,row in enumerate(rows):groups.setdefault(row['image_pixel_sha256'],[]).append(idx)
    clusters=list(groups.values());n=len(rows)
    rng=np.random.default_rng(seed)
    boots=np.empty(bootstrap,dtype=float)
    for i in range(bootstrap):
        picked=rng.integers(len(clusters),size=len(clusters))
        choices=[j for k in picked for j in clusters[int(k)]]
        boots[i]=100*float(np.mean(dif[choices]))
    from scipy.stats import binomtest
    fixed=int((dif==1).sum());harmed=int((dif==-1).sum())
    # Standard McNemar assumes independent paired questions. When multiple
    # questions share one image, report cluster-aware randomization instead.
    ordinary_p=binomtest(min(fixed,harmed),fixed+harmed,.5).pvalue if fixed+harmed else 1.0
    cluster_totals=np.array([dif[idxs].sum() for idxs in clusters],dtype=np.int32)
    observed=abs(int(cluster_totals.sum()))
    n_perm=max(bootstrap,4000)
    # Replacing sign of each image's entire paired effect retains within-image
    # dependence. Monte-Carlo p has +1 correction. Assumes exchangeability of
    # treatment labels at the image cluster level under the sharp null.
    exceed=0
    for i in range(n_perm):
        signs=rng.integers(0,2,len(clusters))*2-1
        exceed+=int(abs(int(signs@cluster_totals))>=observed)
    cluster_p=(exceed+1)/(n_perm+1)
    return dict(n=n,unique_images=len(clusters),
                baseline_correct=int(b.sum()),visual_correct=int(z.sum()),
                baseline_accuracy=float(b.mean()),visual_accuracy=float(z.mean()),
                delta_pp=float(100*np.mean(dif)),
                ci95_low_pp=float(np.quantile(boots,.025)),
                ci95_high_pp=float(np.quantile(boots,.975)),
                repaired=fixed,harmed=harmed,
                exact_mcnemar_p=float(ordinary_p) if len(clusters)==n else None,
                image_cluster_signflip_p=float(cluster_p),
                baseline_valid_rate=float(np.mean([bool(reference[r['id']]['parsed']) for r in rows])),
                visual_valid_rate=float(np.mean([bool(predictions[r['id']]['parsed']) for r in rows])))


def summarize(rows,records,out,manifest,config,seed=42):
    result={'dataset':manifest,'config':config,'groups':{},'tasks':{}}
    for group in ('all','curated_existing','new_mt_zh'):
        samples=rows if group=='all' else [r for r in rows if r['evaluation_group']==group]
        result['groups'][group]=group_stats(samples,records['baseline_zh'],
                                           records['text_visual_ot'],seed)
    for task in sorted({r['task'] for r in rows}):
        result['tasks'][task]=group_stats([r for r in rows if r['task']==task],
                                         records['baseline_zh'],records['text_visual_ot'],seed)
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'comparison.csv').open('w',encoding='utf-8',newline='') as f:
        table=[dict(group=k,**v) for k,v in result['groups'].items()]
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    print('FINAL SUMMARY',json.dumps(result['groups'],ensure_ascii=False,indent=2),flush=True)
    print('FINAL OUTPUT',out/'summary.json',out/'comparison.csv',flush=True)
    print('Interpret curated/MT groups SEPARATELY: extra translations unverified; '
          'same public CV-Bench test dataset; selection on earlier validation.',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',required=True,type=Path)
    p.add_argument('--extended',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    settings,config,manifest,rows,mp,mappath=load_protocol(a.experiment,a.extended)
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    fingerprint=cache_manifest(settings,config,manifest,mappath,a.extended)
    fp=out/'run_manifest.json'
    if fp.exists() and json.loads(fp.read_text())!=fingerprint:
        raise ValueError('Existing results refer to a different model/config/dataset; use new --out')
    if not fp.exists():fp.write_text(json.dumps(fingerprint,ensure_ascii=False,indent=2))
    print('Comparing only ZH baseline vs Text+Visual OT:',
          f'layer={config["layer"]}, alpha={config["strength"]}, N={len(rows)}',flush=True)
    print('Original 600 curated + new machine translations:',manifest['curated_count'],
          manifest['machine_translated_count'],flush=True)
    if a.dry_run:return
    processor,model=load_model(settings['model'])
    records=run_predictions(model,processor,rows,mp,config,settings,out)
    summarize(rows,records,out,manifest,config,settings['seed'])


if __name__=='__main__':main()
