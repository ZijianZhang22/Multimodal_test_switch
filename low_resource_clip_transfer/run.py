#!/usr/bin/env python3
"""Zero-shot Chinese CIFAR-10 classification through frozen CLIP-Rosetta3.

No CIFAR-10 train data, labels, or images enter Rosetta/CLIP fitting. The only
learned low-resource mapping is a pre-existing Flickr8K-CN Rosetta checkpoint.
All class templates, evaluation rows and hyperparameters are fixed in code.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import urllib.request
import urllib.error
import http.client
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "visual_anchor"))
from transfer_core import (CIFAR10_EN, CIFAR10_ZH, EN_TEMPLATES, ZH_TEMPLATES,
                  CIFAR10_MD5, CIFAR10_URL, class_prototypes, classify,
                  make_prompts, md5_file, paired_bootstrap, project_images,
                  project_map, read_cifar_test, select_balanced_indices,
                  sha256_file, unit)

METHODS = ("pairs_only", "text_only", "joint", "image_only", "caption_only",
           "joint_caption", "joint_noise", "joint_extra_english")


def download_cifar(path):
    """Verified official CIFAR archive; interruptions keep a resumable .partial."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if md5_file(path) != CIFAR10_MD5:
            raise ValueError(f"Existing CIFAR-10 file has wrong MD5: {path}")
        print("Reusing verified CIFAR-10 archive:", path, flush=True)
        return

    tmp = path.with_name(path.name + ".partial")
    if tmp.is_file() and md5_file(tmp) == CIFAR10_MD5:
        tmp.replace(path)
        print("Completed previous CIFAR-10 download:", path, flush=True)
        return

    print(f"Downloading official CIFAR-10 archive (~162 MiB): {CIFAR10_URL}", flush=True)
    for attempt in range(1, 7):
        start = tmp.stat().st_size if tmp.is_file() else 0
        req = urllib.request.Request(
            CIFAR10_URL, headers={"User-Agent": "Rosetta3-Transfer-Research/1.0"})
        if start:
            req.add_header("Range", f"bytes={start}-")
            print(f"Resuming CIFAR-10 from {start / 1048576:.1f} MiB...", flush=True)
        try:
            with urllib.request.urlopen(req, timeout=60) as source:
                status = source.getcode()
                if start and status == 206:
                    content_range = source.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {start}-"):
                        raise ValueError(
                            f"Server returned a mismatched byte range: {content_range}")
                    mode = "ab"
                elif status == 200:
                    if start:
                        print("Server does not support resume; restarting download.", flush=True)
                    mode = "wb"
                else:
                    raise ValueError(f"Unexpected HTTP status for CIFAR archive: {status}")
                written = start if mode == "ab" else 0
                last_reported = written // (16 * 1048576)
                with tmp.open(mode) as f:
                    while chunk := source.read(1024 * 1024):
                        f.write(chunk)
                        written += len(chunk)
                        now = written // (16 * 1048576)
                        if now > last_reported:
                            print(f"CIFAR-10 downloaded: {written / 1048576:.1f} MiB",
                                  flush=True)
                            last_reported = now
            if md5_file(tmp) != CIFAR10_MD5:
                raise ValueError(
                    f"CIFAR-10 archive MD5 mismatch: {tmp}. "
                    "Delete this .partial file to retry from zero.")
            tmp.replace(path)
            print("CIFAR-10 verified:", path, flush=True)
            return
        except (urllib.error.URLError, TimeoutError, OSError, http.client.IncompleteRead) as exc:
            if attempt == 6:
                raise RuntimeError(
                    f"CIFAR-10 download failed after {attempt} attempts; "
                    f"partial bytes retained at {tmp}") from exc
            print(f"Download interrupted ({type(exc).__name__}); retry "
                  f"{attempt}/6, keeping .partial...", flush=True)
            time.sleep(min(2 * attempt, 10))
        except KeyboardInterrupt:
            retained = tmp.stat().st_size if tmp.is_file() else 0
            print(f"\nDownload paused. Saved {retained / 1048576:.1f} MiB at {tmp}. "
                  "Rerun the same command to resume.", flush=True)
            raise

def load_original_assets(args):
    report_file = args.map_dir / 'report.json'
    features_file = args.clip_work / 'features.npz'
    lock_file = args.clip_work / 'model_revisions.json'
    needed = [report_file, features_file, lock_file] + [args.map_dir / f'{m}_{args.map_seed}.npz' for m in args.methods]
    missing = [str(f) for f in needed if not f.is_file()]
    if missing:
        raise FileNotFoundError("Missing CLIP-Rosetta3 training artifacts:\n  " + "\n  ".join(missing)
                                + "\nRun the existing vlm_semantic_bridge/run_oneclick.sh to fit the 50-pair maps first.")

    report = json.loads(report_file.read_text(encoding='utf-8'))
    protocol = report.get('config', {})
    if protocol.get('bridge') != 'identity':
        raise ValueError('Require the CLIP-native identity bridge (not DINOv2 learned bridge)')
    shared = bool(protocol.get('shared_ei_pca'))
    joint = bool(protocol.get('joint_ei_pca'))
    if shared == joint:
        raise ValueError('Expected exactly one of shared-ei-pca or joint-ei-pca')
    dim = int(protocol.get('dim', 0))
    if dim <= 0 or int(protocol.get('budget', -1)) != args.budget:
        raise ValueError('Rosetta checkpoint dimension/budget do not match requested run')
    names = json.loads(lock_file.read_text(encoding='utf-8'))
    models = names.get('models', {})
    revs = names.get('revisions', {})
    if not (models.get('e') and models['e'] == models.get('i') and 'clip' in models['e'].lower()):
        raise ValueError('English/image features must use one identical CLIP checkpoint')
    if not models.get('l') or not all(revs.get(k) for k in ('l','e','i')):
        raise ValueError('Missing pinned Chinese/English/image encoder revisions')
    if 'text2vec' not in models['l'].lower():
        raise ValueError('Pilot Chinese prompts require the original Chinese monolingual text2vec encoder')

    from core import fit_space_pcas  # visual_anchor/core.py (same as training)
    with np.load(features_file, allow_pickle=False) as f:
        arrays = {k:f[k] for k in ('l_fit', 'e_fit', 'i_fit', 'l_pair', 'i_bank')}
    pcas = fit_space_pcas(arrays, dim, float(protocol.get('whiten',0)),
                          shared_ei=shared, joint_ei=joint)
    train_image_mean = pcas['i'].transform(arrays['i_bank']).mean(axis=0, keepdims=True)
    checkpoints = {}
    for method in args.methods:
        fp = args.map_dir / f'{method}_{args.map_seed}.npz'
        with np.load(fp, allow_pickle=False) as z:
            keys = ('W', 'mean_x', 'mean_y', 'image_mean', 'anchor_indices',
                    'V_IE', 'lambda_y', 'lambda_z')
            absent = set(keys) - set(z.files)
            if absent:
                raise ValueError(f'Incomplete checkpoint {fp}: {sorted(absent)}')
            chk = {k:z[k] for k in keys}
        expected = np.random.default_rng(args.map_seed).choice(len(arrays['l_pair']),args.budget,replace=False)
        if not np.array_equal(chk['anchor_indices'], expected):
            raise ValueError(f'Checkpoint {method} uses different supervised anchors')
        if not np.allclose(chk['V_IE'], np.eye(dim), atol=1e-5):
            raise ValueError(f'Checkpoint {method} uses learned image bridge (unsupported)')
        if not np.allclose(chk['image_mean'], train_image_mean, atol=1e-5):
            raise ValueError(f'Checkpoint {method} image mean differs from training-image-bank mean')
        seen = [r for r in report.get('runs', []) if r.get('method')==method and r.get('seed')==args.map_seed]
        if len(seen) != 1 or any(not np.isclose(float(chk[k]), float(seen[0][k]))
                                 for k in ('lambda_y','lambda_z')):
            raise ValueError(f'Checkpoint {method} and Rosetta report disagree')
        checkpoints[method] = chk
    return models, revs, pcas, train_image_mean, checkpoints, report


def config_and_hashes(args, indices, models, revs, report):
    return {
        'protocol': 'cifar10_test_frozen_rosetta3_zero_shot_v1',
        'benchmark': 'official_cifar10_test_binary',
        'cifar_md5': CIFAR10_MD5, 'sample_indices_sha256': hashlib.sha256(indices.tobytes()).hexdigest(),
        'n_images': len(indices), 'random_seed': args.seed,
        'language': 'zh', 'map_seed': args.map_seed, 'map_budget': args.budget,
        'methods': list(args.methods), 'model_revision_lock': revs,
        'model_names': models, 'rosetta_config': report['config'],
        'map_file_sha256': {m:sha256_file(args.map_dir/f'{m}_{args.map_seed}.npz') for m in args.methods},
        'features_file_sha256': sha256_file(args.clip_work/'features.npz'),
        'english_templates': list(EN_TEMPLATES), 'chinese_templates': list(ZH_TEMPLATES),
        'english_labels': list(CIFAR10_EN), 'chinese_labels': list(CIFAR10_ZH),
        'clip_work': str(args.clip_work), 'map_dir': str(args.map_dir),
        'description': 'CIFAR images/labels do not train Rosetta; English CLIP reference is an upper benchmark, not low-pair transfer.'
    }


def save_config(out, config):
    out.mkdir(parents=True, exist_ok=True)
    path = out/'config.json'
    if path.is_file():
        old = json.loads(path.read_text(encoding='utf-8'))
        if old != config:
            raise ValueError(f'Incompatible existing run {out}. Use a fresh OUT for new settings.')
    else:
        path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')


def encode_or_restore(args, config, images, models, revs):
    """Cache raw features only, never downstream predictions or fitted mappings."""
    cache = args.out/'raw_encodings.npz'
    sig = hashlib.sha256(json.dumps(config,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    if cache.is_file():
        with np.load(cache,allow_pickle=False) as z:
            if str(z['signature']) != sig:
                raise ValueError('Feature cache does not match current config')
            keys=('images','en_text','zh_clip_text','zh_mono_text')
            out={k:z[k] for k in keys}
        if out['images'].shape[0] != len(images):
            raise ValueError('Stale or incomplete image cache')
        print('Reusing cached CLIP/text2vec embeddings:',cache,flush=True)
        return out

    import clip_encoders
    model, tokenizer, processor = clip_encoders.load_clip(models['e'],revs['e'],args.device)
    en_prompts = make_prompts(CIFAR10_EN,EN_TEMPLATES)
    zh_prompts = make_prompts(CIFAR10_ZH,ZH_TEMPLATES)
    print('Encoding frozen CLIP class prototypes and CIFAR-10 test images...', flush=True)
    en_text = clip_encoders.encode_texts(model,tokenizer,en_prompts,args.device)
    zh_clip_text = clip_encoders.encode_texts(model,tokenizer,zh_prompts,args.device)
    parts=[]
    for start in range(0,len(images),args.batch_size):
        batch = [Image.fromarray(img) for img in images[start:start+args.batch_size]]
        parts.append(clip_encoders.encode_images(model,processor,batch,args.device))
        if (start//args.batch_size)%5==0 or start+args.batch_size >= len(images):
            print('CLIP images:',min(start+args.batch_size,len(images)),'/',len(images),flush=True)
    image_embs = np.concatenate(parts,axis=0)
    del model, tokenizer, processor
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except ImportError:
        pass

    print('Encoding Chinese class prompts with original monolingual text2vec...',flush=True)
    from sentence_transformers import SentenceTransformer
    zh_model = SentenceTransformer(models['l'],revision=revs['l'],device=args.device,trust_remote_code=False)
    zh_model.max_seq_length = 128
    zh_mono = zh_model.encode(zh_prompts,batch_size=args.batch_size,
                              normalize_embeddings=False,convert_to_numpy=True,show_progress_bar=False)
    del zh_model
    gc.collect()
    out = dict(images=image_embs,en_text=en_text,zh_clip_text=zh_clip_text,zh_mono_text=zh_mono)
    for key,value in out.items():
        if not np.isfinite(value).all():
            raise ValueError(f'Nonfinite features from encoder: {key}')
    tmp=cache.with_suffix('.partial.npz')
    np.savez_compressed(tmp,signature=np.asarray(sig),**out)
    tmp.replace(cache)
    print('Cached:',cache,flush=True)
    return out


def evaluate(args, labels, indices, raw, pcas, image_mean, maps, config):
    labels=np.asarray(labels,dtype=np.int64)
    image_raw=raw['images']
    # CLIP's original pixel/text space: direct diagnostic (does not use Rosetta).
    en_ref = class_prototypes(raw['en_text'],len(EN_TEMPLATES))
    zh_clip = class_prototypes(raw['zh_clip_text'],len(ZH_TEMPLATES))
    preds={}
    preds['english_clip_reference'],_=classify(image_raw,en_ref)
    preds['chinese_clip_direct'],_=classify(image_raw,zh_clip)

    # Rosetta protocol uses normalized train-fitted PCA, then map-specific
    # source centering and fixed visual-bank target centering for ALL methods.
    l_z=pcas['l'].transform(raw['zh_mono_text'])
    i_z=project_images(image_raw,pcas['i'],image_mean)
    for method,chk in maps.items():
        source=project_map(l_z,chk)
        proto=class_prototypes(source,len(ZH_TEMPLATES))
        preds[method],_=classify(i_z,proto)
    rows=[]
    for arm,pred in preds.items():
        correct=int((pred==labels).sum())
        row={'arm':arm,'n':len(labels),'correct':correct,'accuracy':float(correct/len(labels))}
        if arm != 'pairs_only' and 'pairs_only' in preds:
            row['delta_vs_pairs_only_pp']=float(100*((pred==labels).mean()-(preds['pairs_only']==labels).mean()))
        if arm != 'text_only' and 'text_only' in preds:
            row['delta_vs_text_only_pp']=float(100*((pred==labels).mean()-(preds['text_only']==labels).mean()))
        rows.append(row)
    target = args.out/'summary.csv'
    with target.open('w',newline='',encoding='utf-8') as f:
        names=list(dict.fromkeys(k for row in rows for k in row))
        w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)

    detailed = []
    for j,(idx,truth) in enumerate(zip(indices,labels)):
        record={'test_index':int(idx),'label':int(truth),'label_en':CIFAR10_EN[truth],
                'label_zh':CIFAR10_ZH[truth]}
        for arm,pred in preds.items():
            record[f'pred_{arm}']=int(pred[j])
            record[f'correct_{arm}']=int(pred[j]==truth)
        detailed.append(record)
    with (args.out/'per_example.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(detailed[0]));w.writeheader();w.writerows(detailed)

    byclass=[]
    for arm,pred in preds.items():
        for cls in range(10):
            m = labels == cls
            byclass.append({'arm':arm,'class_id':cls,'class_en':CIFAR10_EN[cls],
                            'n':int(m.sum()),'accuracy':float((pred[m]==labels[m]).mean())})
    with (args.out/'per_class.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(byclass[0]));w.writeheader();w.writerows(byclass)

    pairs=[]
    # A specific comparison of visual training versus text-only/supervised
    # map checks whether benefits survive on novel visual *classification*.
    for compared in preds:
        for base in ('pairs_only','text_only','chinese_clip_direct'):
            if base not in preds or compared == base:
                continue
            stat=paired_bootstrap(labels,preds[base],preds[compared],seed=args.seed,
                                  n_boot=args.bootstraps)
            pairs.append({'compared':compared,'baseline':base,**stat})
    report={'config':config,'summary':rows,'paired_differences':pairs,
            'scope':'Frozen CIFAR-10 test zero-shot classification; CLIP reference sees English category names; neither is VQA.',
            'cautions':['CIFAR-10 is a 10-class English-origin object classification benchmark, not a cultural or multilingual VQA benchmark.',
                        'English labels are supplied as a reference upper baseline; no CIFAR labels or examples were used to fit Rosetta.',
                        'CIFAR-10 32x32 images are upscaled by CLIP processor. Domain shift can be substantial.',
                        'Confidence intervals resample images within each class. No multi-seed uncertainty unless separate Rosetta seeds are evaluated.',
                        'Source label names and natural language templates differ semantically across languages; no template tuning on held-out test.']}
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for row in rows:
        delta=row.get('delta_vs_pairs_only_pp')
        print(f"{row['arm']}: {row['correct']}/{row['n']} = {row['accuracy']:.2%}"
              + (f' (vs pairs_only {delta:+.2f} pp)' if delta is not None else ''),flush=True)
    for e in pairs:
        if e['compared']=='joint' and e['baseline'] in ('pairs_only','text_only'):
            print(f"joint - {e['baseline']}: {e['delta_pp']:+.2f} pp; "
                  f"image-bootstrap CI [{e['ci95_pp'][0]:+.2f}, {e['ci95_pp'][1]:+.2f}] pp",flush=True)
    print('Saved:',target,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clip-work',type=Path,required=True)
    p.add_argument('--map-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--budget',type=int,default=50)
    p.add_argument('--map-seed',type=int,default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--num-images',type=int,default=500,help='class-balanced test images; 0 = all 10000')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--bootstraps',type=int,default=2000)
    p.add_argument('--device',default='cuda')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=['pairs_only','text_only','joint'])
    a=p.parse_args()
    if len(set(a.methods))!=len(a.methods) or not a.methods:
        p.error('Methods must be distinct and nonempty')
    if a.budget<1 or a.map_seed<0 or a.num_images<0 or a.batch_size<1 or a.bootstraps<1:
        p.error('Budget, seeds, sample size, batch size or bootstrap count invalid')
    for k in ('clip_work','map_dir','out','data_dir'):
        setattr(a,k,getattr(a,k).resolve())
    models,revs,pcas,image_mean,maps,report=load_original_assets(a)
    archive=a.data_dir/'cifar-10-binary.tar.gz'
    download_cifar(archive)
    images,labels=read_cifar_test(archive)
    indices=select_balanced_indices(labels,a.num_images,a.seed)
    cfg=config_and_hashes(a,indices,models,revs,report)
    save_config(a.out,cfg)
    raw=encode_or_restore(a,cfg,images[indices],models,revs)
    evaluate(a,labels[indices],indices,raw,pcas,image_mean,maps,cfg)


if __name__=='__main__':
    main()