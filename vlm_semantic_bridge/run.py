#!/usr/bin/env python3
"""Frozen CLIP-Rosetta3 -> English retrieval -> frozen PaliGemma2 VQA.

Training: none. The matched English bank is Flickr8K train only. Query images,
questions, answers are MMBench Dev (image-disjoint local tune/test). No paired
EN question or label participates in retrieval. The English reference arm uses
its official translation for an UPPER-BOUND diagnostic, not fair low-resource
inference. This is a pilot, not official MMBench CircularEval.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bridge_core import (ARMS, MAP_ARMS, build_prompt, bank_captions, extract_question_for_embedding,
                         image_cluster_bootstrap, jsonl, map_vectors, parse_flickr_captions,
                         random_rankings, rankings, sha256)


def files_and_rows(args):
    data_dir = args.data_dir.resolve()
    manifest = json.loads((data_dir / 'manifest.json').read_text(encoding='utf-8'))
    if 'image_disjoint' not in manifest.get('protocol', ''):
        raise ValueError('Expected image-disjoint MMBench preparation; use prepare_mmbench.py')
    for name in ('tune', 'test'):
        path = data_dir / f'{name}.jsonl'
        if sha256(path) != manifest['sha256'][name]:
            raise ValueError(f'Prepared MMBench {name} has changed; use the original dataset')
    rows = jsonl(data_dir / f'{args.split}.jsonl')
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError('No evaluation questions selected')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicated question IDs in evaluation')
    for row in rows:
        if not {'en', 'zh'} <= set(row['questions']):
            raise ValueError('Missing bilingual question for '+row['id'])
        if row['answer'] not in row['allowed']:
            raise ValueError('Invalid gold option label')
        if not Path(row['image']).is_file():
            raise FileNotFoundError(row['image'])
    if args.split == 'test':
        tune = jsonl(data_dir / 'tune.jsonl')
        if {r['image_pixel_sha256'] for r in rows} & {r['image_pixel_sha256'] for r in tune}:
            raise ValueError('MMBench tune/test image overlap')
    return rows


def signatures(args, rows):
    p = args.clip_work.resolve()
    result_dir = args.map_dir.resolve()
    paths = {'features': p/'features.npz', 'encoder_lock': p/'model_revisions.json',
             'rosetta_report': result_dir/'report.json',
             'bank_captions': p/'raw'/'flickr8kenc.caption.txt',
             'mmbench_tune': args.data_dir.resolve()/'tune.jsonl',
             'mmbench_test': args.data_dir.resolve()/'test.jsonl'}
    for method in args.arms:
        if method in MAP_ARMS:
            paths[method+'_checkpoint'] = result_dir/f'{method}_{args.map_seed}.npz'
    missing = [str(v) for v in paths.values() if not v.is_file()]
    if missing:
        raise FileNotFoundError('Required previously computed files are missing:\n  ' + '\n  '.join(missing))
    return {name: sha256(path) for name,path in paths.items()}


def settings(args, signatures, rows):
    return {'version': 1, 'data_dir': str(args.data_dir.resolve()),
            'clip_work': str(args.clip_work.resolve()), 'map_dir': str(args.map_dir.resolve()),
            'split': args.split, 'limit': args.limit, 'ids': [r['id'] for r in rows],
            'top_k': args.top_k, 'map_seed': args.map_seed, 'random_seed': args.random_seed,
            'arms': args.arms, 'model': args.model, 'max_new_tokens': args.max_new_tokens,
            'input_sha256': signatures,
            'note': 'training Flickr bank only; question embeddings have NO English paired question/answer; plain cosine retrieval'}


def save_or_verify_config(out, config, create):
    path = out/'config.json'
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        if old != config:
            raise ValueError(f'Output already exists with incompatible settings: {path}. Use a new --out')
    elif create:
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    else:
        raise FileNotFoundError(f'Missing {path}. Run --stage prepare first')


def _verify_model_provenance(work):
    raw = json.loads((work/'model_revisions.json').read_text(encoding='utf-8'))
    names = raw.get('models', {})
    revisions = raw.get('revisions', {})
    if not (names.get('e') == names.get('i') and 'clip' in names.get('e','').lower()):
        raise ValueError('English and image features must come from SAME frozen CLIP model')
    if not names.get('l') or not all(revisions.get(k) for k in ('l','e','i')):
        raise ValueError('Missing pinned L/E/I checkpoint provenance')
    return names,revisions


def _check_bank_image_disjoint(bank_ids, args, rows):
    # Compute exact decoded-pixel signatures, same rule as prepare_mmbench.py.
    hashes = set()
    for identifier in bank_ids:
        path = args.clip_work/'raw'/'images'/f'{identifier}.jpg'
        if not path.is_file():
            raise FileNotFoundError(f'Bank image not available for leakage audit: {path}')
        with Image.open(path) as img:
            rgb = img.convert('RGB')
            digest = hashlib.sha256(f'{rgb.size}'.encode()+rgb.tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError('Duplicate image content in retrieval bank')
        hashes.add(digest)
    eval_hashes = {r['image_pixel_sha256'] for r in rows}
    if hashes & eval_hashes:
        raise ValueError('Retrieved bank image overlaps evaluation image pixels')


def load_maps_and_features(args, rows):
    work = args.clip_work.resolve()
    names, revisions = _verify_model_provenance(work)
    report = json.loads((args.map_dir/'report.json').read_text(encoding='utf-8'))
    proto = report.get('config', {})
    if proto.get('bridge') != 'identity':
        raise ValueError('This pilot uses CLIP-native identity bridge, not DINOv2 projection')
    shared, joint = bool(proto.get('shared_ei_pca')), bool(proto.get('joint_ei_pca'))
    if shared == joint:
        raise ValueError('Expected exactly one CLIP shared/joint PCA mode in Rosetta report')
    dim = int(proto['dim'])
    if dim <= 0:
        raise ValueError('Invalid PCA dimension')
    if int(proto.get('budget',-1)) <= 0:
        raise ValueError('This version requires a positive paired anchor budget')
    if report.get('pca_protocol') not in (None, 'shared_separate_means' if shared else 'joint_single_mean'):
        raise ValueError('Rosetta report PCA protocol inconsistent with config')
    # Original encoder/PCA preprocessing MUST be reconstructed from its TRAIN data.
    sys.path.insert(0, str(ROOT/'visual_anchor'))
    from core import fit_space_pcas
    with np.load(work/'features.npz', allow_pickle=False) as src:
        arrays = {key: src[key] for key in ('l_fit','e_fit','i_fit','e_bank','e_bank_ids','l_pair')}
    pcas = fit_space_pcas(arrays, dim, float(proto.get('whiten',0)), shared_ei=shared, joint_ei=joint)
    bank_ids = [str(x) for x in arrays['e_bank_ids']]
    _check_bank_image_disjoint(bank_ids, args, rows)
    captions = bank_captions(bank_ids, parse_flickr_captions(work/'raw'/'flickr8kenc.caption.txt'))
    bank_z = pcas['e'].transform(arrays['e_bank'])
    maps = {}
    budget = int(proto['budget'])
    for method in args.arms:
        if method not in MAP_ARMS:
            continue
        ckpt = args.map_dir/f'{method}_{args.map_seed}.npz'
        with np.load(ckpt, allow_pickle=False) as data:
            values = {k: data[k] for k in ('W','mean_x','mean_y','V_IE','anchor_indices','lambda_y','lambda_z')}
        idx = np.random.default_rng(args.map_seed).choice(len(arrays['l_pair']), budget, replace=False)
        if not np.array_equal(values['anchor_indices'], idx):
            raise ValueError(f'Anchor identities differ from Rosetta3 seed/budget for {method}')
        if not np.allclose(values['V_IE'], np.eye(dim), atol=1e-5):
            raise ValueError(f'{method} is not a CLIP-native identity-bridge checkpoint')
        matches = [x for x in report.get('runs',[]) if x.get('method')==method and x.get('seed')==args.map_seed]
        if len(matches) != 1:
            raise ValueError(f'Missing one report run for {method}/{args.map_seed}')
        for key in ('lambda_y','lambda_z'):
            if not np.isclose(float(values[key]),float(matches[0][key])):
                raise ValueError(f'Checkpoint {method} has different {key} than report')
        maps[method] = values
    return names, revisions, pcas['l'], bank_z, captions, bank_ids, maps


def prepare(args, rows, cfg):
    if args.top_k < 1:
        raise ValueError('top_k must be >= 1')
    names,revs,l_pca,bank_z,captions,bank_ids,maps = load_maps_and_features(args,rows)
    if args.top_k > len(captions):
        raise ValueError('top_k larger than training bank')
    stems = [extract_question_for_embedding(r['questions']['zh']) for r in rows]
    queries = None
    if any(arm in MAP_ARMS for arm in args.arms):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(names['l'], revision=revs['l'], device=args.encoder_device,
                                    trust_remote_code=False)
        model.max_seq_length=128  # exactly the feature-extraction protocol
        parts=[]
        for start in range(0,len(stems),args.encoder_batch):
            batch=stems[start:start+args.encoder_batch]
            parts.append(model.encode(batch, batch_size=len(batch), show_progress_bar=False,
                                      normalize_embeddings=False, convert_to_numpy=True))
        queries = l_pca.transform(np.vstack(parts))
        del model
    idxs={}
    similarities={}
    for arm, mapping in maps.items():
        source,target=map_vectors(queries,bank_z,mapping)
        ids, scores=rankings(source,target,args.top_k)
        idxs[arm]=ids
        similarities[arm]=scores
    if 'random' in args.arms:
        idxs['random']=random_rankings(len(rows),len(captions),args.top_k,
                                      [r['id'] for r in rows], args.random_seed)
    lines=[]
    for i,row in enumerate(rows):
        for arm in args.arms:
            idx = idxs.get(arm)
            picked = []
            if idx is not None:
                for j,pos in enumerate(idx[i]):
                    item = {'bank_id':bank_ids[int(pos)],'caption':captions[int(pos)]}
                    if arm in similarities:
                        item['cosine'] = float(similarities[arm][i,j])
                    picked.append(item)
            prompt=build_prompt(row,arm,[p['caption'] for p in picked])
            lines.append({'id':row['id'],'arm':arm,'image_pixel_sha256':row['image_pixel_sha256'],
                          'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                          'retrieved':picked})
    path=args.out/'prompts.jsonl'
    if path.exists():
        old=jsonl(path)
        if old != lines:
            raise ValueError(f'Existing prompts differ: {path}; use a new output folder')
    else:
        with path.open('w',encoding='utf-8') as f:
            for line in lines:
                f.write(json.dumps(line,ensure_ascii=False)+'\n')
    print('Prepared retrieval prompts:',len(rows),'questions x',len(args.arms),'arms',flush=True)
    print('Saved:',path,flush=True)


def _stored_predictions(path, prompt_map, rows):
    records={}
    if not path.exists():
        return records
    for rec in jsonl(path):
        rid=rec['id']
        if rid in records or rid not in prompt_map:
            raise ValueError('Duplicate/unknown prediction ID in '+str(path))
        p=prompt_map[rid]
        row=rows[rid]
        if (rec.get('prompt_sha256') != p['prompt_sha256'] or
            rec.get('image_pixel_sha256') != row['image_pixel_sha256'] or
            rec.get('answer') != row['answer']):
            raise ValueError('Stale prediction: '+rid)
        records[rid]=rec
    return records


def evaluate(args, rows):
    prompts=jsonl(args.out/'prompts.jsonl')
    expected={(r['id'],a) for r in rows for a in args.arms}
    if set((r['id'],r['arm']) for r in prompts)!=expected or len(prompts)!=len(expected):
        raise ValueError('Incomplete or duplicate prompt table')
    row_map={r['id']:r for r in rows}
    by_arm={a:{} for a in args.arms}
    for rec in prompts:
        if (rec['image_pixel_sha256']!=row_map[rec['id']]['image_pixel_sha256'] or
            hashlib.sha256(rec['prompt'].encode()).hexdigest()!=rec['prompt_sha256']):
            raise ValueError('Corrupt prompt table')
        by_arm[rec['arm']][rec['id']]=rec
    out=args.out/'predictions';out.mkdir(exist_ok=True)
    saved={arm:_stored_predictions(out/f'{arm}.jsonl',by_arm[arm],row_map) for arm in args.arms}
    if any(len(saved[a])<len(rows) for a in args.arms):
        sys.path.insert(0,str(ROOT/'vlm_unsupervised_transfer'))
        from vlm import load_model, generate, letter
        processor,model=load_model(args.model)
        for arm in args.arms:
            filename=out/f'{arm}.jsonl'
            with filename.open('a',encoding='utf-8') as f:
                for n,row in enumerate(rows):
                    if row['id'] in saved[arm]:
                        continue
                    prompt=by_arm[arm][row['id']]
                    answer=generate(model,processor,row,prompt['prompt'],max_new_tokens=args.max_new_tokens)
                    parsed=letter(answer)
                    rec={'id':row['id'],'arm':arm,'answer':row['answer'],
                         'image_pixel_sha256':row['image_pixel_sha256'],
                         'prompt_sha256':prompt['prompt_sha256'],'prediction':answer,
                         'parsed':parsed,'valid':int(parsed in row['allowed']),
                         'correct':int(parsed in row['allowed'] and parsed==row['answer'])}
                    f.write(json.dumps(rec,ensure_ascii=False)+'\n')
                    saved[arm][row['id']]=rec
                    if n%10==0 or n==len(rows)-1:
                        f.flush();os.fsync(f.fileno())
                    if (n+1)%25==0 or n+1==len(rows):
                        print(f'{arm}: {n+1}/{len(rows)}',flush=True)
    summary=[]
    for arm in args.arms:
        d=saved[arm]
        if len(d)!=len(rows):
            raise ValueError(f'Incomplete evaluation for {arm}')
        accuracy=sum(d[r['id']]['correct'] for r in rows)/len(rows)
        record={'arm':arm,'n':len(rows),'correct':int(round(len(rows)*accuracy)),
                'accuracy':accuracy,'valid_rate':sum(d[r['id']]['valid'] for r in rows)/len(rows)}
        if arm!='baseline_zh' and 'baseline_zh' in saved:
            record.update(image_cluster_bootstrap(rows,saved['baseline_zh'],d,seed=args.random_seed))
        summary.append(record)
    with (args.out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        fields=list(dict.fromkeys(k for row in summary for k in row))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summary)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    for row in summary:
        print(f"{row['arm']}: {row['correct']}/{row['n']} = {row['accuracy']:.2%}"
              + (f" (vs ZH {row['delta_pp']:+.2f} pp)" if 'delta_pp' in row else ''),flush=True)
    print('FINAL:',args.out/'summary.csv',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('prepare','eval','all'),default='all')
    p.add_argument('--data-dir',type=Path,required=True,help='prepared bilingual MMBench directory')
    p.add_argument('--clip-work',type=Path,required=True,help='visual_anchor/real_work_clip_native')
    p.add_argument('--map-dir',type=Path,required=True,help='Rosetta3 CLIP result dir with report and method_seed maps')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',default='google/paligemma2-3b-mix-448')
    p.add_argument('--split',choices=('tune','test'),default='tune')
    p.add_argument('--limit',type=int,default=30,help='0=full split; otherwise first N fixed items')
    p.add_argument('--top-k',type=int,default=2)
    p.add_argument('--map-seed',type=int,default=0)
    p.add_argument('--random-seed',type=int,default=42)
    p.add_argument('--arms',nargs='+',default=['baseline_zh','baseline_en','pairs_only','text_only','joint','random'])
    p.add_argument('--encoder-device',default='cpu',help='CPU avoids conflicting with VLM GPU')
    p.add_argument('--encoder-batch',type=int,default=32)
    p.add_argument('--max-new-tokens',type=int,default=8)
    a=p.parse_args()
    if not a.arms or len(a.arms)!=len(set(a.arms)) or set(a.arms)-set(ARMS):
        p.error('Invalid/duplicate method names; allowed: '+', '.join(ARMS))
    if not (0<=a.limit and a.top_k>=1 and a.encoder_batch>=1 and a.max_new_tokens>=1):
        p.error('limit>=0, top-k/encoder-batch/max-new-tokens>=1 required')
    a.data_dir=a.data_dir.resolve();a.clip_work=a.clip_work.resolve()
    a.map_dir=a.map_dir.resolve();a.out=a.out.resolve()
    rows=files_and_rows(a)
    sig=signatures(a,rows)
    cfg=settings(a,sig,rows)
    save_or_verify_config(a.out,cfg,create=a.stage!='eval')
    if a.stage in ('prepare','all'):
        prepare(a,rows,cfg)
    if a.stage in ('eval','all'):
        evaluate(a,rows)


if __name__=='__main__':
    main()