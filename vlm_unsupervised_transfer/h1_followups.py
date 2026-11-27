#!/usr/bin/env python3
"""H1 follow-ups: frozen-map diagnostics + locked external bilingual replication.

D1 forward/inverse orientation; D2 partial interpolation; D3 native vs PCA;
D4 Flickr caption -> MMBench question shift; D5 genuinely new paired dataset.
No map fitting or model intervention. Existing H1 caches reused for D1-D4.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist

from prepare_data import digest, read_jsonl
from geometry import load_map
from h1_geometry import (METHODS, IMAGES, MODES, VERSION, embeddings, load_states,
                         paired_ci, relational_similarity, retrieval, unique_images, unit)

FOLLOWUP_VERSION = 'h1_followups_v1'
ALPHAS = (0., .1, .25, .5, .75, 1.)  # fixed exploratory sweep, not chosen on held-out data


def write_csv(path, rows):
    if not rows:
        raise ValueError('Empty report: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.csv')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def pixel_digest(path):
    with Image.open(path) as f:
        rgb = f.convert('RGB')
        return hashlib.sha256(f'{rgb.size}'.encode()+rgb.tobytes()).hexdigest()


def verify_existing(exp, h1, mmbench):
    stamp = json.loads((h1/'h1_manifest.json').read_text(encoding='utf-8'))
    settings_file = exp/'settings.json'
    if digest(settings_file) != stamp['original_settings_sha256']:
        raise ValueError('Original experiment settings changed')
    settings = json.loads(settings_file.read_text(encoding='utf-8'))
    if stamp['protocol'] != VERSION or settings['model'] != stamp['model']:
        raise ValueError('H1 provenance/model mismatch')
    if mmbench is not None:
        if digest(mmbench/'manifest.json') != stamp['dataset_manifest_sha256']:
            raise ValueError('Original MMBench manifest changed')
        manifest = json.loads((mmbench/'manifest.json').read_text(encoding='utf-8'))
        if digest(mmbench/'test.jsonl') != stamp['dataset_test_sha256']:
            raise ValueError('Original MMBench test split changed')
        if digest(mmbench/'tune.jsonl') != manifest['sha256']['tune']:
            raise ValueError('Original MMBench tune split changed')
        all_rows = read_jsonl(mmbench/'test.jsonl')
        selected = unique_images(all_rows, stamp['sample_size'], stamp['seed'])
        if [r['id'] for r in selected] != stamp['sample_ids']:
            raise ValueError('Sample IDs no longer match H1')
    layers = stamp['layers']
    maps = {}
    poolsha = digest(Path(settings['pools'])/'manifest.json')
    for layer in layers:
        for method in METHODS:
            key = f'{method}_L{layer}'
            path = exp/'maps'/f'layer{layer}_{method}.npz'
            if digest(path) != stamp['map_sha256'][key]:
                raise ValueError('Frozen alignment map changed: '+str(path))
            maps[layer, method], _ = load_map(path, required={
                'layer':layer, 'language':'zh', 'model_id':settings['model'],
                'pool_sha':poolsha})
    return stamp, settings, maps


def mapped_pair(zh, en, mp, direction='W'):
    b = np.asarray(mp['basis'], dtype='float64')
    mx = np.asarray(mp['mean_source'], dtype='float64')
    my = np.asarray(mp['mean_english'], dtype='float64')
    w = np.asarray(mp['W'], dtype='float64')
    if direction == 'WT': w = w.T
    if direction not in ('W','WT'):
        raise ValueError(direction)
    return unit(((zh-mx)@b)@w), unit((en-my)@b)


def partial_pair(zh, en, mp, alpha):
    """Blended PCA states (not generally an orthogonal matrix at 0<alpha<1)."""
    b = np.asarray(mp['basis'], dtype='float64')
    mx = np.asarray(mp['mean_source'], dtype='float64')
    my = np.asarray(mp['mean_english'], dtype='float64')
    w = np.asarray(mp['W'], dtype='float64')
    x = (zh-mx)@b
    return unit((1-alpha)*x + alpha*(x@w)), unit((en-my)@b)


def full_pair(zh, en, mp, variant):
    b = np.asarray(mp['basis'], dtype='float64')
    mx = np.asarray(mp['mean_source'], dtype='float64')
    my = np.asarray(mp['mean_english'], dtype='float64')
    w = np.asarray(mp['W'], dtype='float64')
    if variant == 'native_raw':
        return unit(zh), unit(en)
    if variant == 'english_center_raw':
        return unit(zh-my), unit(en-my)
    x = zh-mx
    if variant == 'center':
        return unit(x), unit(en-my)
    if variant == 'mapped':
        x = x + (((x@b)@(w-np.eye(w.shape[0])))@b.T)
        return unit(x), unit(en-my)
    raise ValueError(variant)


def one_score(x, y, extra):
    r, p = retrieval(x, y)
    return dict(**extra, **r), p


def cache_matrix(h1, stamp, image, mode, layer):
    idx = stamp['layers'].index(layer)
    ids = stamp['sample_ids']
    # load_states only needs a list of records containing 'id'.
    rows = [{'id':i} for i in ids]
    return (load_states(h1,rows,image,mode,'zh',idx).astype('float64'),
            load_states(h1,rows,image,mode,'en',idx).astype('float64'))


def run_diagnostics(exp, h1, out, stamp, settings, maps):
    orientation, partial, spaces, shifts, paired_deltas = [], [], [], [], []
    primary_check = json.loads((h1/'summary.json').read_text(encoding='utf-8'))
    original = {(r['image'],r['prompt'],r['layer'],r['method'],r['variant']):r
                for r in primary_check['results']}
    for image in IMAGES:
        for mode in MODES:
            for layer in stamp['layers']:
                zh,en = cache_matrix(h1,stamp,image,mode,layer)
                for method in METHODS:
                    mp = maps[layer,method]
                    # Query-level paired confidence intervals preserve bilingual pairs.
                    x0,y0=embeddings(zh,en,mp,'center')
                    _,base_details=retrieval(x0,y0)
                    def contrast(details, label):
                        a=.5*((details['rank_zh_en']==1).astype(float) +
                                (details['rank_en_zh']==1).astype(float))
                        b=.5*((base_details['rank_zh_en']==1).astype(float) +
                                (base_details['rank_en_zh']==1).astype(float))
                        gain,lo,hi=paired_ci(a,b,seed=stamp['seed'])
                        paired_deltas.append(dict(image=image,prompt=mode,layer=layer,method=method,
                                                   comparison=label,reference='pca_center',
                                                   delta_avg_recall1_pp=gain,ci95_low_pp=lo,ci95_high_pp=hi))
                    # D1: includes the original center/raw/W checks and inverse orientation.
                    for variant in ('raw','center','W','WT'):
                        if variant in ('raw','center'):
                            x,y = embeddings(zh,en,mp,variant)
                        else:
                            x,y = mapped_pair(zh,en,mp,variant)
                        result,details = one_score(x,y,dict(image=image,prompt=mode,layer=layer,
                                                            method=method,variant=variant))
                        orientation.append(result)
                        if variant in ('W','WT'):contrast(details,'orientation_'+variant)
                        orig_variant = 'mapped' if variant=='W' else variant
                        if orig_variant in ('raw','center','mapped'):
                            old = original[(image,mode,layer,method,orig_variant)]
                            for k in ('recall1_zh_en','recall1_en_zh','paired_cosine'):
                                if not np.isclose(result[k],old[k],atol=1e-6):
                                    raise ValueError(f'Existing H1 reproduction mismatch: {image}/{mode}/{layer}/{method}/{variant}/{k}')
                    # D2: fixed-strength continuum; not a tuned choice.
                    for alpha in ALPHAS:
                        x,y = partial_pair(zh,en,mp,alpha)
                        r,details = one_score(x,y,dict(image=image,prompt=mode,layer=layer,
                                                       method=method,alpha=alpha))
                        partial.append(r)
                        if alpha>0:contrast(details,'partial_alpha_'+str(alpha))
                    # D3: compare same representations with and without PCA.
                    for variant in ('raw','center','mapped'):
                        x,y=embeddings(zh,en,mp,variant)
                        r,_=one_score(x,y,dict(image=image,prompt=mode,layer=layer,
                                               method=method,space='pca',variant=variant))
                        spaces.append(r)
                    for variant in ('native_raw','english_center_raw','center','mapped'):
                        x,y=full_pair(zh,en,mp,variant)
                        r,details=one_score(x,y,dict(image=image,prompt=mode,layer=layer,
                                                     method=method,space='full',variant=variant))
                        spaces.append(r)
                        if variant=='center':contrast(details,'full_center_vs_pca_center')
                print(f'D1-D3 {image}/{mode} layer {layer}', flush=True)
    # D4: fit captions vs questions in original decoder space; real-image main.
    layers_orig=list(settings['layers'])
    pooldir=Path(settings['pools']); features=exp/'features'
    for lang,pool in [('zh','source'),('en','english')]:
        pool_rows=read_jsonl(pooldir/f'{pool}.jsonl')
        orig_states=[]
        for i,row in enumerate(pool_rows):
            with np.load(features/pool/f'{i:05d}.npz',allow_pickle=False) as f:
                if str(f['id'].item()) != str(row['id']):
                    raise ValueError('Flickr feature index/id mismatch')
                states=f['states']
                if states.shape[0] != len(layers_orig):
                    raise ValueError('Flickr feature layer count mismatch')
                orig_states.append(states.copy())
        orig_states=np.stack(orig_states)
        for layer in stamp['layers']:
            src=orig_states[:,layers_orig.index(layer),:].astype('float64')
            for image in ('real','blank'):
                rows=[{'id':i} for i in stamp['sample_ids']]
                q=load_states(h1,rows,image,'stem',lang,stamp['layers'].index(layer))
                shift = shift_metric(src,q,seed=stamp['seed'],permutations=199)
                shifts.append(dict(language=lang,layer=layer,image=image,
                                   train_corpus='Flickr8K_caption_with_real_image',
                                   evaluation_corpus=f'MMBench_{image}_image_question_stem',**shift))
    for filename, rows in [('direction_check.csv',orientation),('partial_ot.csv',partial),
                           ('pca_vs_full.csv',spaces),('distribution_shift.csv',shifts),
                           ('paired_deltas.csv',paired_deltas)]:
        write_csv(out/filename,rows)
    summary = {
        'version':FOLLOWUP_VERSION,
        'source_h1_manifest_sha256':digest(h1/'h1_manifest.json'),
        'primary':'layer8 blank/stem; all other settings are exploratory',
        'n':stamp['sample_size'],
        'orientation_primary':[r for r in orientation if r['layer']==8 and r['image']=='blank' and r['prompt']=='stem'],
        'partial_primary':[r for r in partial if r['layer']==8 and r['image']=='blank' and r['prompt']=='stem'],
        'space_primary':[r for r in spaces if r['layer']==8 and r['image']=='blank' and r['prompt']=='stem'],
        'distribution_shift':shifts,
        'notes':['W is the trained source->English mapping; W.T is a diagnostic, not a valid retrained mapping.',
                 'Partial linear interpolation is not necessarily orthogonal.',
                 'Flickr vs MMBench shift includes caption/question, image and prompt changes.',
                 'Same 200 H1 examples were inspected already: no post-hoc sweep is confirmatory.']}
    tmp=out/'diagnostics_summary.tmp.json'
    tmp.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    tmp.replace(out/'diagnostics_summary.json')
    print('Saved D1-D4 to',out,flush=True)


def shift_metric(train, question, seed=42, permutations=199):
    """RBF MMD (biased, nonnegative) with an item-label permutation null.

    Features are normalized to remove raw activation-norm confounds; report
    domain mean direction cosine and relative mean shift separately.
    """
    x=unit(train);y=unit(question)
    both=np.concatenate([x,y],axis=0)
    ds=cdist(both,both,metric='sqeuclidean')
    triu=ds[np.triu_indices(len(both),1)]
    positive=triu[triu>1e-12]
    bw=float(np.median(positive)) if len(positive) else 1.
    kernel=np.exp(-ds/max(bw,1e-8))
    n=len(x);m=len(y)
    def mmd(indices):
        ix=indices[:n];iy=indices[n:]
        return float(kernel[np.ix_(ix,ix)].mean()+kernel[np.ix_(iy,iy)].mean()-
                     2*kernel[np.ix_(ix,iy)].mean())
    obs=mmd(np.arange(n+m))
    rng=np.random.default_rng(seed)
    exceed=sum(mmd(rng.permutation(n+m))>=obs-1e-12 for _ in range(permutations))
    mx=x.mean(0);my=y.mean(0)
    cos=float(mx@my/(max(np.linalg.norm(mx),1e-12)*max(np.linalg.norm(my),1e-12)))
    return dict(n_train=n,n_question=m,mmd2=obs,permutation_p=(exceed+1)/(permutations+1),
                bandwidth_squared=bw,mean_direction_cosine=cos,
                mean_shift_l2=float(np.linalg.norm(mx-my)))


def external_records(path, limit, seed, previous_hashes):
    path=path.resolve()
    rows=read_jsonl(path)
    if len(rows)<20:
        raise ValueError('D5 needs >=20 independently collected bilingual question pairs')
    by_id=set();by_image=set();by_question={'zh':set(),'en':set()};clean=[]
    for r in rows:
        rid=str(r['id'])
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',rid) or rid in ('.','..'):
            raise ValueError('External id must be a safe filename: '+rid)
        image=(path.parent/r['image']).resolve()
        if rid in by_id or not image.is_file():
            raise ValueError('Duplicate id or missing external image: '+rid)
        if 'questions' not in r or not all(isinstance(r['questions'].get(l),str) and r['questions'][l].strip()
                                            for l in ('zh','en')):
            raise ValueError('Each pair requires nonempty questions.zh and questions.en')
        # Exact repeated question stems make one-positive retrieval ill-defined.
        normalized = {lang:' '.join(r['questions'][lang].casefold().split()) for lang in ('zh','en')}
        if any(normalized[lang] in by_question[lang] for lang in ('zh','en')):
            raise ValueError('Repeated question stem in external corpus: '+rid)
        if any(re.search(r'(?m)^\s*\(?[A-H]\)\s+', r['questions'][lang])
               for lang in ('zh','en')):
            raise ValueError('External input must contain question stems without answer options: '+rid)
        h=pixel_digest(image)
        if h in by_image or h in previous_hashes:
            raise ValueError('Repeated or previously used image in external corpus: '+rid)
        by_id.add(rid);by_image.add(h)
        for lang in ('zh','en'):by_question[lang].add(normalized[lang])
        clean.append(dict(id=rid,image=str(image),image_sha=h,questions=r['questions']))
    if limit<20 or limit>len(clean):
        raise ValueError(f'External sample size must be between 20 and {len(clean)}')
    rng=np.random.default_rng(seed)
    clean.sort(key=lambda r:r['id'])
    picks=rng.choice(len(clean),size=limit,replace=False)
    return [clean[int(i)] for i in picks]


def old_image_hashes(mmbench,settings):
    seen=set()
    for split in ('tune','test'):
        seen.update(r['image_pixel_sha256'] for r in read_jsonl(mmbench/f'{split}.jsonl'))
    for pool in ('source','english','visual'):
        for r in read_jsonl(Path(settings['pools'])/f'{pool}.jsonl'):
            seen.add(pixel_digest(Path(r['image'])))
    return seen


def external_test(path,limit,out,exp,h1,mmbench,stamp,settings,maps):
    if mmbench is None:
        raise ValueError('D5 requires --mmbench to check that new images do not overlap')
    rows=external_records(path,limit,stamp['seed'],old_image_hashes(mmbench,settings))
    src_hash=digest(path)
    group=out/'external'/src_hash[:12]
    group.mkdir(parents=True,exist_ok=True)
    fixed={'version':FOLLOWUP_VERSION,'external_file_sha256':src_hash,'model':settings['model'],
           'fixed_layer':8,'fixed_image':'shared_gray_448','fixed_prompt':'question_stem_without_options',
           'seed':stamp['seed'],'n':len(rows),'ids':[r['id'] for r in rows],
           'image_shas':[r['image_sha'] for r in rows],
           'map_sha256':{m:stamp['map_sha256'][f'{m}_L8'] for m in METHODS}}
    manifest=group/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8'))!=fixed:
            raise ValueError('External config changed, choose new output directory')
    else:
        if (group/'features').exists():
            raise ValueError('External features present without matching manifest')
        manifest.write_text(json.dumps(fixed,indent=2),encoding='utf-8')
    blank=group/'gray.png'
    if not blank.exists():Image.new('RGB',(448,448),(127,127,127)).save(blank)
    from vlm import VQA_PREFIX, inputs, captured_states, load_model
    # Only one locked layer; if all features exist, skip loading GPU model.
    missing=[]
    for row in rows:
        for lang in ('zh','en'):
            f=group/'features'/lang/(row['id']+'.npz')
            if not f.is_file():missing.append((row,lang,f))
    if missing:
        proc,model=load_model(settings['model'])
        for j,(r,lang,file) in enumerate(missing,1):
            file.parent.mkdir(parents=True,exist_ok=True)
            prompt=VQA_PREFIX+r['questions'][lang].strip()
            state=captured_states(model,inputs(proc,{'image':str(blank)},prompt),[8])[8]['text']
            if not np.isfinite(state).all():raise ValueError('Nonfinite external activation')
            temp=file.with_suffix('.tmp.npz')
            np.savez_compressed(temp,state=state.astype('float32'),
                                key=np.array(hashlib.sha256((src_hash+r['id']+lang+prompt).encode()).hexdigest()))
            os.replace(temp,file)
            if j%20==0 or j==len(missing):print('D5 extract',j,'/',len(missing),flush=True)
        del proc,model
    vectors={}
    for lang in ('zh','en'):
        vals=[]
        for r in rows:
            with np.load(group/'features'/lang/(r['id']+'.npz'),allow_pickle=False) as f:
                key=hashlib.sha256((src_hash+r['id']+lang+VQA_PREFIX+r['questions'][lang].strip()).encode()).hexdigest()
                if str(f['key'].item())!=key:raise ValueError('External feature cache mismatch')
                vals.append(f['state'].copy())
        vectors[lang]=np.stack(vals)
    out_rows=[]
    for method in METHODS:
        mp=maps[8,method]
        for variant in ('raw','center','W'):
            a,b=embeddings(vectors['zh'],vectors['en'],mp,variant if variant!='W' else 'mapped')
            score,pair=one_score(a,b,dict(method=method,variant=variant,layer=8,image='blank',prompt='stem'))
            rsa=relational_similarity(a,b,seed=stamp['seed'],permutations=399)
            out_rows.append(dict(**score,**rsa))
        # Paired improvement CI for fixed W vs center on locked external set.
        ac,bc=embeddings(vectors['zh'],vectors['en'],mp,'center')
        am,bm=embeddings(vectors['zh'],vectors['en'],mp,'mapped')
        _,p0=retrieval(ac,bc);_,p1=retrieval(am,bm)
        for direction in ('zh_en','en_zh'):
            gain,lo,hi=paired_ci((p1['rank_'+direction]==1).astype(float),
                                  (p0['rank_'+direction]==1).astype(float),stamp['seed'])
            out_rows[-1][f'delta_{direction}_pp']=gain
            out_rows[-1][f'delta_{direction}_ci95_low_pp']=lo
            out_rows[-1][f'delta_{direction}_ci95_high_pp']=hi
    # Ensure fixed columns across raw/center/W conditions.
    keys=list(dict.fromkeys(k for row in out_rows for k in row))
    write_csv(group/'external_locked.csv',[{k:r.get(k) for k in keys} for r in out_rows])
    print('Saved locked independent-replication results:',group/'external_locked.csv',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,default=Path('/workspace/vlm_A_zh_seed42/experiment'))
    p.add_argument('--h1',type=Path,default=Path('/workspace/vlm_A_zh_seed42/h1_geometry_n200'))
    p.add_argument('--mmbench',type=Path,default=Path('/workspace/vlm_A_zh_seed42/mmbench_dev_bilingual'))
    p.add_argument('--out',type=Path,default=Path('/workspace/vlm_A_zh_seed42/h1_followups'))
    p.add_argument('--stage',choices=('diagnostics','external','all'),default='diagnostics')
    p.add_argument('--external-pairs',type=Path)
    p.add_argument('--external-n',type=int,default=200)
    a=p.parse_args()
    exp,h1,out=(x.resolve() for x in (a.experiment,a.h1,a.out))
    mmbench=a.mmbench.resolve()
    stamp,settings,maps=verify_existing(exp,h1,mmbench)
    out.mkdir(parents=True,exist_ok=True)
    if a.stage in ('external','all') and a.external_pairs is None:
        raise ValueError('External replication requires --external-pairs (new, disjoint bilingual JSONL)')
    if a.stage in ('diagnostics','all'):
        run_diagnostics(exp,h1,out,stamp,settings,maps)
    if a.stage in ('external','all'):
        if a.external_pairs is None:
            raise ValueError('External replication requires --external-pairs (new, disjoint bilingual JSONL)')
        external_test(a.external_pairs,a.external_n,out,exp,h1,mmbench,stamp,settings,maps)


if __name__=='__main__':main()