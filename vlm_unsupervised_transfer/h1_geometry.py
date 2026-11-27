#!/usr/bin/env python3
"""H1: Test whether *frozen* unpaired mappings align paired EN/ZH VLM states.

Primary endpoint: bidirectional paired-question R@1 on MMBench local-test,
question-stem prompts, and an identical gray image for every question.
Secondary diagnostics: real images and original full multiple-choice prompts.
No fitting, model steering, label scoring, or parameter selection is performed.
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

from prepare_data import digest, read_jsonl
from geometry import load_map
from vlm import captured_states, inputs, load_model, question, VQA_PREFIX

METHODS = ("mean_shift", "text_ot", "text_visual_ot", "text_noise_ot")
MODES = ("stem", "full")
IMAGES = ("blank", "real")
VERSION = "h1_geometry_v1_unique_image_test_only"


def sha_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def unique_images(rows, n, seed):
    """One paired question per image; rotations never become retrieval collisions."""
    chosen = {}
    for r in sorted(rows, key=lambda x: x['id']):
        key = r['image_pixel_sha256']
        if key not in chosen:
            chosen[key] = r
    rng = np.random.default_rng(seed)
    keys = sorted(chosen)
    if n <= 0 or n > len(keys):
        raise ValueError(f'Need 1..{len(keys)} unique images, requested {n}')
    picks = rng.choice(len(keys), size=n, replace=False)
    return [chosen[keys[int(i)]] for i in picks]


def stem_prompt(row, lang):
    """Discard instruction and option lines; preserve original question + hint."""
    full = row['questions'][lang]
    if '\n' not in full:
        raise ValueError('MMBench row does not have instruction and question lines')
    remainder = full.split('\n', 1)[1]
    m = re.search(r'\n\([A-F]\)\s', remainder)
    if m is None:
        raise ValueError('Could not locate original MMBench option block')
    text = remainder[:m.start()].strip()
    if not text:
        raise ValueError('Empty question stem')
    return VQA_PREFIX + text


def model_prompt(row, lang, prompt_mode):
    if prompt_mode == 'full':
        return question(row, lang)
    if prompt_mode == 'stem':
        return stem_prompt(row, lang)
    raise ValueError(prompt_mode)


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def embeddings(source, english, mapping, variant):
    """All comparisons done in the map's *own* frozen common PCA basis.

    Identity baseline: project Chinese with the EN center, not source center.
    Center control: translate Chinese mean to English, but W=I.
    Mapped: source center + frozen rotation W. No output norm clipping here.
    """
    b = np.asarray(mapping['basis'], dtype=np.float64)
    mx = np.asarray(mapping['mean_source'], dtype=np.float64)
    my = np.asarray(mapping['mean_english'], dtype=np.float64)
    w = np.asarray(mapping['W'], dtype=np.float64)
    target = unit((english-my) @ b)
    if variant == 'raw':
        origin = (source-my) @ b
    elif variant == 'center':
        origin = (source-mx) @ b
    elif variant == 'mapped':
        origin = ((source-mx) @ b) @ w
    else:
        raise ValueError(variant)
    return unit(origin), target


def retrieval(source, target):
    """Symmetric retrieval: every row i has exactly one positive at column i."""
    x, y = unit(source), unit(target)
    if x.shape != y.shape or x.ndim != 2 or len(x) < 3:
        raise ValueError('Require matching 2D bilingual matrices with >=3 examples')
    sim = x @ y.T
    n = len(x)
    diag = sim[np.arange(n), np.arange(n)]
    # Dense sorting is acceptable for n up to ~1,000 and keeps ties deterministic.
    sorted_xy = np.argsort(-sim, axis=1, kind='stable')
    sorted_yx = np.argsort(-sim.T, axis=1, kind='stable')
    rank_xy = np.argmax(sorted_xy == np.arange(n)[:, None], axis=1) + 1
    rank_yx = np.argmax(sorted_yx == np.arange(n)[:, None], axis=1) + 1
    mask = np.eye(n, dtype=bool)
    off = np.where(mask, -np.inf, sim)
    off_diag = off.max(axis=1)
    result = {
        'n': n,
        'recall1_zh_en': float(np.mean(rank_xy == 1)),
        'recall1_en_zh': float(np.mean(rank_yx == 1)),
        'recall5_zh_en': float(np.mean(rank_xy <= min(n,5))),
        'recall5_en_zh': float(np.mean(rank_yx <= min(n,5))),
        'mrr_zh_en': float(np.mean(1/rank_xy)),
        'mrr_en_zh': float(np.mean(1/rank_yx)),
        'paired_cosine': float(np.mean(diag)),
        'paired_margin_vs_hardest_negative': float(np.mean(diag-off_diag)),
        'chance_recall1': 1/n,
    }
    pair = dict(positive_cosine=diag, rank_zh_en=rank_xy, rank_en_zh=rank_yx,
                margin=diag-off_diag)
    return result, pair


def relational_similarity(source, target, seed=42, permutations=0):
    """RSA/Mantel-like Spearman of within-language pairwise cosine geometry.

    Any orthogonal W preserves the Chinese Gram matrix, so this probes
    shared structural geometry independently of the particular learned W.
    Randomly permuting English *item labels* gives a paired-order null.
    """
    from scipy.stats import rankdata
    a, b = unit(source), unit(target)
    if a.shape != b.shape or len(a)<4:
        raise ValueError('RSA requires >=4 paired examples')
    n=len(a)
    tr=np.triu_indices(n,1)
    sa=(a@a.T)[tr]
    sb=(b@b.T)[tr]
    ra=rankdata(sa)
    def correlation(v):
        rv=rankdata(v)
        d1=ra-ra.mean();d2=rv-rv.mean()
        denom=np.linalg.norm(d1)*np.linalg.norm(d2)
        return float((d1@d2)/denom) if denom>1e-12 else float('nan')
    observed=correlation(sb)
    if permutations<=0:
        return dict(rsa_spearman=observed, rsa_permutation_p=None)
    rng=np.random.default_rng(seed)
    gram=b@b.T
    exceed=0
    for _ in range(permutations):
        idx=rng.permutation(n)
        shuffled=gram[np.ix_(idx,idx)][tr]
        null=correlation(shuffled)
        exceed+=int(np.isfinite(null) and abs(null)>=abs(observed))
    return dict(rsa_spearman=observed,
                rsa_permutation_p=(exceed+1)/(permutations+1))


def paired_ci(a, b, seed=42, repeats=4000):
    """Paired bootstrap of per-example recall1 improvements (distinct images)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 3:
        raise ValueError('Expected paired query-level 1-D arrays')
    dif = a-b
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats)
    for i in range(repeats):
        take = rng.integers(len(dif), size=len(dif))
        boot[i] = 100*dif[take].mean()
    return (float(100*dif.mean()), float(np.quantile(boot,.025)),
            float(np.quantile(boot,.975)))


def protocol(experiment, dataset, layers, sample_size, seed):
    exp = Path(experiment).resolve()
    dat = Path(dataset).resolve()
    settings_file = exp/'settings.json'
    dataset_manifest_file = dat/'manifest.json'
    if not settings_file.is_file() or not dataset_manifest_file.is_file():
        raise FileNotFoundError('Need original experiment/settings.json and prepared MMBench manifest')
    settings = json.loads(settings_file.read_text(encoding='utf-8'))
    manifest = json.loads(dataset_manifest_file.read_text(encoding='utf-8'))
    if settings.get('language') != 'zh' or settings.get('arm') != 'a':
        raise ValueError('Requires original A-only Chinese alignment maps')
    if Path(settings['out']).resolve() != exp:
        raise ValueError('Experiment output path mismatch')
    if 8 not in layers:
        raise ValueError('Predeclared primary layer 8 must be included')
    if not set(layers).issubset(set(settings['layers'])) or len(layers) != len(set(layers)):
        raise ValueError('Need unique layers previously fitted in experiment A')
    if manifest.get('protocol') != 'paired_mmbench_dev_v1.0_image_disjoint_v2_with_hint':
        raise ValueError('Only reviewed MMBench v2 paired data with hints is supported')
    for p, v in settings.get('file_integrity', {}).items():
        if not Path(p).is_file() or digest(p) != v:
            raise ValueError('Original alignment training/eval manifest changed: '+p)
    for split in ('tune','test'):
        f = dat/f'{split}.jsonl'
        if not f.is_file() or digest(f) != manifest['sha256'][split]:
            raise ValueError('MMBench '+split+' JSONL changed')
    tune_rows = read_jsonl(dat/'tune.jsonl')
    test_rows = read_jsonl(dat/'test.jsonl')
    if ({r['image_pixel_sha256'] for r in tune_rows} &
        {r['image_pixel_sha256'] for r in test_rows}):
        raise ValueError('MMBench test and prior tuning share an image')
    rows = unique_images(test_rows, sample_size, seed)
    mp = {}; map_shas = {}
    pool_sha = digest(Path(settings['pools'])/'manifest.json')
    for layer in layers:
        for method in METHODS:
            path = exp/'maps'/f'layer{layer}_{method}.npz'
            if not path.is_file():
                raise FileNotFoundError(f'Missing fitted {method} layer {layer} map: {path}')
            mp[(layer,method)],_ = load_map(path,required={
                'layer':layer,'language':'zh','model_id':settings['model'],'pool_sha':pool_sha})
            map_shas[f'{method}_L{layer}'] = digest(path)
    training_pixels = set()
    for group in ('source','english','visual'):
        for record in read_jsonl(Path(settings['pools'])/(group+'.jsonl')):
            with Image.open(record['image']) as f:
                pix = f.convert('RGB')
                training_pixels.add(hashlib.sha256(f'{pix.size}'.encode()+pix.tobytes()).hexdigest())
    if training_pixels & {r['image_pixel_sha256'] for r in rows}:
        raise ValueError('Training/evaluation image overlap by decoded pixels')
    for row in rows:
        with Image.open(row['image']) as f:
            im = f.convert('RGB')
            pixel = hashlib.sha256(f'{im.size}'.encode()+im.tobytes()).hexdigest()
        if pixel != row['image_pixel_sha256']:
            raise ValueError('Changed test image '+row['id'])
        if not row['questions'].get('zh') or not row['questions'].get('en'):
            raise ValueError('Unpaired or missing EN/ZH question '+row['id'])
    stamp = {
        'protocol': VERSION, 'original_settings_sha256':digest(settings_file),
        'dataset_manifest_sha256':digest(dataset_manifest_file),
        'dataset_test_sha256':manifest['sha256']['test'],
        'map_sha256': map_shas, 'model':settings['model'],
        'layers': list(layers), 'sample_size':sample_size, 'seed':seed,
        'sample_ids':[r['id'] for r in rows],
        'primary_endpoint':'blank/stem, layer 8, RSA plus bidirectional Recall@1; other settings diagnostic',
    }
    return settings, rows, mp, stamp


def extract_embeddings(rows, settings, layers, out, stamp):
    """Idempotent feature extraction: one forward pass extracts all layers."""
    blank = out/'neutral_gray_448.png'
    if not blank.is_file():
        Image.new('RGB', (448,448), (127,127,127)).save(blank)
    proc, model = load_model(settings['model'])
    total = len(rows)*len(IMAGES)*len(MODES)*2
    finished = 0
    for image_mode in IMAGES:
        for prompt_mode in MODES:
            for row in rows:
                for lang in ('zh','en'):
                    prompt = model_prompt(row,lang,prompt_mode)
                    pic = blank if image_mode == 'blank' else Path(row['image'])
                    d = out/'features'/image_mode/prompt_mode/lang
                    d.mkdir(parents=True,exist_ok=True)
                    path = d/(row['id']+'.npz')
                    key = sha_json({'id':row['id'],'prompt':prompt,'picture_sha':
                                    'blank_rgb127_448' if image_mode == 'blank' else row['image_pixel_sha256'],
                                    'model':settings['model'],'layers':layers,'version':VERSION})
                    if path.is_file():
                        with np.load(path,allow_pickle=False) as f:
                            cached_key = str(f['key'].item())
                            states = f['states']
                        if cached_key != key or states.shape[0] != len(layers) or not np.isfinite(states).all():
                            raise ValueError('Stale or corrupt feature cache: '+str(path))
                    else:
                        payload = {'image':str(pic)}
                        states_dict = captured_states(model, inputs(proc,payload,prompt),layers)
                        states = np.stack([states_dict[l]['text'] for l in layers]).astype('float32')
                        if not np.isfinite(states).all():raise ValueError('Nonfinite feature '+row['id'])
                        tmp = path.with_suffix('.partial.npz')
                        np.savez_compressed(tmp,key=np.array(key),states=states)
                        os.replace(tmp,path)
                    finished += 1
                    if finished % 25 == 0 or finished == total:
                        print('Extract H1',finished,'/',total, 'image='+image_mode,
                              'prompt='+prompt_mode,flush=True)
    del model, proc


def load_states(out, rows, image_mode, prompt_mode, lang, layer_index):
    result=[]
    for r in rows:
        path=out/'features'/image_mode/prompt_mode/lang/(r['id']+'.npz')
        with np.load(path,allow_pickle=False) as f:
            result.append(f['states'][layer_index].copy())
    return np.stack(result)


def score_all(rows, maps, layers, out, seed):
    report=[]; per_pair=[]
    baseline_cache={}
    direct_comparisons=[]
    for image_mode in IMAGES:
        for prompt_mode in MODES:
            for j,layer in enumerate(layers):
                chinese=load_states(out,rows,image_mode,prompt_mode,'zh',j)
                english=load_states(out,rows,image_mode,prompt_mode,'en',j)
                mapped_ranks={}
                for method in METHODS:
                    mp=maps[(layer,method)]
                    store={}
                    centered_zh,centered_en=embeddings(chinese,english,mp,'center')
                    rsa=relational_similarity(centered_zh,centered_en,seed,
                        permutations=399 if (image_mode=='blank' and prompt_mode=='stem' and layer==8) else 0)
                    for variant in ('raw','center','mapped'):
                        zh,en=embeddings(chinese,english,mp,variant)
                        stats, details=retrieval(zh,en)
                        store[variant]=details
                        report.append(dict(image=image_mode,prompt=prompt_mode,
                                           layer=layer,method=method,variant=variant,**stats,**rsa))
                        for k,row in enumerate(rows):
                            per_pair.append(dict(image=image_mode,prompt=prompt_mode,
                                                 layer=layer,method=method,variant=variant,
                                                 id=row['id'],image_sha=row['image_pixel_sha256'],
                                                 rank_zh_en=int(details['rank_zh_en'][k]),
                                                 rank_en_zh=int(details['rank_en_zh'][k]),
                                                 cosine=float(details['positive_cosine'][k]),
                                                 margin=float(details['margin'][k])))
                    mapped_ranks[method]=store['mapped']
                    # Each fitted method must improve over its OWN raw/center baseline.
                    for ref in ('raw','center'):
                        for direction in ('zh_en','en_zh'):
                            a=(store['mapped']['rank_'+direction]==1).astype(float)
                            b=(store[ref]['rank_'+direction]==1).astype(float)
                            gain,lo,hi=paired_ci(a,b,seed)
                            baseline_cache[(image_mode,prompt_mode,layer,method,ref,direction)]=(gain,lo,hi)
                for direction in ('zh_en','en_zh'):
                    a=(mapped_ranks['text_visual_ot']['rank_'+direction]==1).astype(float)
                    b=(mapped_ranks['text_ot']['rank_'+direction]==1).astype(float)
                    gain,lo,hi=paired_ci(a,b,seed)
                    direct_comparisons.append(dict(image=image_mode,prompt=prompt_mode,layer=layer,
                                                   direction=direction,visual_minus_text_recall1_pp=gain,
                                                   ci95_low_pp=lo,ci95_high_pp=hi))
    path=out/'results.csv'
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(report[0]));writer.writeheader();writer.writerows(report)
    with (out/'per_pair.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_pair[0]));writer.writeheader();writer.writerows(per_pair)
    primary = [x for x in report if x['image']=='blank' and x['prompt']=='stem']
    summary = {
        'protocol':VERSION,'n_unique_image_pairs':len(rows),
        'primary_endpoint':'blank/stem, layer 8, RSA and R@1 averaged both retrieval directions',
        'results':report,
        'paired_bootstrap_gain_vs_controls':[
            dict(image=k[0],prompt=k[1],layer=k[2],method=k[3],reference=k[4],
                 direction=k[5],gain_pp=v[0],ci95_low_pp=v[1],ci95_high_pp=v[2])
            for k,v in baseline_cache.items()],
        'primary_results':[x for x in primary if x['layer']==8],
        'visual_vs_text_paired_recall1':direct_comparisons,
        'interpretation':'Exploratory H1 probing; paired MMBench labels ONLY evaluate frozen Flickr-fitted maps. '
                         'Never infer causal improvement in VLM reasoning from retrieval alone. '
                         'Real-image scores can be inflated by image identity.'
    }
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('H1 PRIMARY RESULTS (blank + stem):',flush=True)
    for x in primary:
        if x['layer'] == 8 and x['variant'] in ('center','mapped'):
            print(f"  L{x['layer']} {x['method']} {x['variant']}: "
                  f"R@1 zh->en={x['recall1_zh_en']:.3f}, en->zh={x['recall1_en_zh']:.3f}, "
                  f"cos={x['paired_cosine']:.3f}",flush=True)
    print('Saved:',out/'summary.json',out/'results.csv',out/'per_pair.csv',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--mmbench',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--n',type=int,default=200,help='Unique image/EN-ZH paired questions; 0 is invalid')
    p.add_argument('--layers',default='8,13,18')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--stage',choices=('all','extract','score'),default='all')
    a=p.parse_args()
    layers=[int(s) for s in a.layers.split(',')]
    settings,rows,maps,stamp=protocol(a.experiment,a.mmbench,layers,a.n,a.seed)
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=out/'h1_manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != stamp:
            raise ValueError('H1 config has changed; choose a NEW --out instead of mixing caches')
    else:
        if (out/'features').exists() or (out/'results.csv').exists():
            raise ValueError('Unmanifested H1 output exists, use a clean --out')
        manifest.write_text(json.dumps(stamp,indent=2),encoding='utf-8')
    if a.stage in ('all','extract'):
        extract_embeddings(rows,settings,layers,out,stamp)
    if a.stage in ('all','score'):
        score_all(rows,maps,layers,out,a.seed)


if __name__=='__main__':
    main()