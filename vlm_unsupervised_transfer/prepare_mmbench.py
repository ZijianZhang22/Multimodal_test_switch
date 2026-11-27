#!/usr/bin/env python3
"""Prepare a locally held-out bilingual MMBench v1.0 DEV evaluation.

No source/target supervision is used to refit maps. Image-cluster disjoint
validation and test subsets are made before any VLM scores are observed.
MMBench's public DEV is not an official secret held-out TEST split.
"""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
from prepare_data import digest, read_jsonl, write_jsonl

URLS = {
    'en': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN.tsv',
    'zh': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_CN.tsv',
}
# MMBench v1.0 DEV VLMEvalKit TSV checksums from VLMEvalKit ImageMCQDataset.
EXPECTED_MD5 = {'en': 'b6caf1133a01c6bb705cf753bb527ed8',
                'zh': '08b8fc3324a5ed74155350f57be69fbd'}
LETTERS = 'ABCDEF'


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def download_tsv(url, path):
    path = Path(path)
    if path.exists(): return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=120) as response, tmp.open('wb') as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk: break
                out.write(chunk)
        if tmp.stat().st_size < 1024: raise ValueError('Downloaded suspiciously small TSV')
        tmp.replace(path)
    finally:
        if tmp.exists(): tmp.unlink()


def load_tsv(path):
    try: csv.field_size_limit(sys.maxsize)
    except OverflowError: csv.field_size_limit(2**31-1)
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        keys = set(reader.fieldnames or [])
        if not {'index', 'image', 'question', 'answer', 'A', 'B'} <= keys:
            raise ValueError(f'{path}: unexpected MMBench columns: {keys}')
        out = {}
        for r in reader:
            idx = str(r['index']).strip()
            if not idx or idx in out: raise ValueError(f'Duplicate/empty MMBench index: {idx}')
            out[idx] = r
    if not out: raise ValueError(f'No records in {path}')
    return out


def resolve_base64(data, records):
    # Image in a VLMEvalKit TSV is base64 OR another row's exact index.
    seen = set()
    while data in records:
        if data in seen: raise ValueError(f'Image reference cycle: {data}')
        seen.add(data)
        data = str(records[data]['image']).strip()
    if not data: raise ValueError('Missing image payload')
    if data.startswith('data:image/'):
        data = data.split(',',1)[1]
    try: decoded = base64.b64decode(data, validate=True)
    except Exception as exc: raise ValueError(f'Unrecognized base64 image/reference: {data[:30]!r}') from exc
    try:
        with Image.open(io.BytesIO(decoded)) as im:
            im.verify()
    except Exception as exc: raise ValueError('Invalid base64 image') from exc
    return decoded


def pixels_signature(image_bytes):
    with Image.open(io.BytesIO(image_bytes)) as im:
        rgb = im.convert('RGB')
        return hashlib.sha256(f'{rgb.size}'.encode()+rgb.tobytes()).hexdigest(), rgb.copy()


def clean(value):
    if value is None: return ''
    value = str(value).strip()
    return '' if value.casefold() in ('nan','none','null') else value


def row_question(r, lang):
    q = clean(r['question'])
    if not q: raise ValueError('Empty question')
    options = {}
    for key in LETTERS:
        value = clean(r.get(key))
        if value: options[key] = value
        elif options:  # prohibit gaps between MCQ option letters
            if any(clean(r.get(rest)) for rest in LETTERS[LETTERS.index(key)+1:]):
                raise ValueError('Discontinuous option letters')
    if len(options) < 2: raise ValueError('Too few options')
    answer = clean(r['answer']).upper()
    if answer not in options: raise ValueError(f'Answer {answer!r} not in options {tuple(options)}')
    header = ('请根据图片回答以下选择题。只输出一个大写英文字母，必须属于这些选项：' + '/'.join(options) + '。'
              if lang == 'zh' else
              'Answer the multiple-choice question using only one capital letter from: ' + '/'.join(options) + '.')
    hint = clean(r.get('hint'))
    # Official VLMEvalKit includes an optional Hint before Question. Omitting it
    # changes the evaluated problem for examples that depend on visual context.
    stem = ((('提示：' if lang == 'zh' else 'Hint: ') + hint + '\n') if hint else '') + q
    return header + '\n' + stem + '\n' + '\n'.join(f'({key}) {value}' for key,value in options.items()), answer, tuple(options)


def prepare_pairs(en_rows, cn_rows, images):
    if set(en_rows) != set(cn_rows):
        raise ValueError(f'EN/CN index mismatch: EN-only {len(set(en_rows)-set(cn_rows))}, CN-only {len(set(cn_rows)-set(en_rows))}')
    images.mkdir(parents=True, exist_ok=True)
    pairs=[]
    for idx in sorted(en_rows, key=lambda x:(int(x) if x.isdigit() else float('inf'),x)):
        e, z = en_rows[idx], cn_rows[idx]
        q_en, ans_en, opts_en = row_question(e,'en')
        q_zh, ans_zh, opts_zh = row_question(z,'zh')
        if (ans_en,opts_en)!=(ans_zh,opts_zh):
            raise ValueError(f'EN/CN answers or choice letters disagree for index {idx}')
        payload_en = resolve_base64(clean(e['image']), en_rows)
        payload_cn = resolve_base64(clean(z['image']), cn_rows)
        hash_en, pic = pixels_signature(payload_en)
        hash_cn,_ = pixels_signature(payload_cn)
        if hash_en != hash_cn: raise ValueError(f'EN/CN images differ for index {idx}')
        filename = images / f'{hash_en}.png'
        if not filename.exists(): pic.save(filename)
        pairs.append(dict(id=f'mm_{idx}', original_index=idx, image=str(filename.resolve()),
                          image_pixel_sha256=hash_en, answer=ans_en,
                          allowed=list(opts_en), questions={'en':q_en,'zh':q_zh},
                          category=clean(z.get('category')) or clean(e.get('category')) or 'unknown',
                          l2_category=clean(z.get('l2-category')) or clean(e.get('l2-category')) or 'unknown'))
    return pairs


def split_image_groups(rows, tune_size, seed):
    if tune_size < 2 or tune_size >= len(rows): raise ValueError('Invalid tune_size')
    by_image=defaultdict(list)
    for row in rows: by_image[row['image_pixel_sha256']].append(row)
    keys=sorted(by_image)
    if len(keys)<2: raise ValueError('Cannot split dataset: only one image')
    random.Random(seed).shuffle(keys)
    tune_keys=[]; count=0
    for key in keys:
        if count < tune_size and len(tune_keys) < len(keys)-1:
            tune_keys.append(key);count+=len(by_image[key])
        else:break
    ids=set(tune_keys)
    tune=[r for r in rows if r['image_pixel_sha256'] in ids]
    test=[r for r in rows if r['image_pixel_sha256'] not in ids]
    if not tune or not test:raise ValueError('Empty split')
    assert set(x['image_pixel_sha256'] for x in tune).isdisjoint(x['image_pixel_sha256'] for x in test)
    return tune,test


def check_existing_dataset(out, requested):
    man=out/'manifest.json'
    if not man.is_file(): return False
    old=json.loads(man.read_text())
    for k,v in requested.items():
        if old.get(k)!=v: raise ValueError(f'Existing prepared MMBench has different {k}; use a new --out')
    for k in ('tune','test'):
        file=out/f'{k}.jsonl'
        if not file.is_file() or digest(file)!=old['sha256'][k]: raise ValueError(f'Corrupt existing prepared {file}')
        for row in read_jsonl(file):
            p=Path(row['image'])
            if not p.is_file(): raise FileNotFoundError(p)
            with Image.open(p) as im: rgb=im.convert('RGB')
            h=hashlib.sha256(f'{rgb.size}'.encode()+rgb.tobytes()).hexdigest()
            if h != row['image_pixel_sha256']: raise ValueError(f'Image content changed: {p}')
    print('Using verified existing bilingual MMBench:',old['counts'],flush=True)
    return True


def prepare(out, en_tsv, cn_tsv, tune_size=200, seed=42, verify_official=True):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    en_tsv=Path(en_tsv).resolve();cn_tsv=Path(cn_tsv).resolve()
    requested={'seed':seed,'tune_target':tune_size,'en_tsv_sha256':digest(en_tsv),
               'cn_tsv_sha256':digest(cn_tsv),'protocol':'paired_mmbench_dev_v1.0_image_disjoint_v2_with_hint'}
    if check_existing_dataset(out,requested): return
    # Do not silently mix data from a prior aborted preparation.
    for name in ('tune.jsonl','test.jsonl','manifest.json'):
        if (out/name).exists(): raise FileExistsError(f'Partial prior dataset {out/name}; use a clean folder')
    if verify_official:
        for lang, path in [('en',en_tsv),('zh',cn_tsv)]:
            if md5(path)!=EXPECTED_MD5[lang]:
                raise ValueError(f'Official MMBench {lang} file MD5 mismatch: {md5(path)}. Supply --allow-unpinned if you intentionally use a different TSV release.')
    en,zh=load_tsv(en_tsv),load_tsv(cn_tsv)
    pairs=prepare_pairs(en,zh,out/'images')
    tune,test=split_image_groups(pairs,tune_size,seed)
    if not (set(r['id'] for r in tune).isdisjoint(r['id'] for r in test)):
        raise ValueError('Question-ID overlap')
    write_jsonl(out/'tune.jsonl',tune);write_jsonl(out/'test.jsonl',test)
    manifest=dict(**requested,counts={'all':len(pairs),'tune':len(tune),'test':len(test)},
                  unique_images={'all':len({r['image_pixel_sha256'] for r in pairs}),
                                 'tune':len({r['image_pixel_sha256'] for r in tune}),
                                 'test':len({r['image_pixel_sha256'] for r in test})},
                  sha256={n:digest(out/f'{n}.jsonl') for n in ('tune','test')},
                  note='Local image-disjoint split of PUBLIC MMBench DEV; NOT official secret held-out TEST')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print('Bilingual MMBench ready:',manifest['counts'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--en-tsv',type=Path,default=None)
    p.add_argument('--cn-tsv',type=Path,default=None)
    p.add_argument('--tune-size',type=int,default=200)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--allow-unpinned',action='store_true',help='Explicitly override official MMBench v1.0 MD5 pin')
    a=p.parse_args()
    raw=a.out/'raw';raw.mkdir(parents=True,exist_ok=True)
    en_tsv=a.en_tsv or raw/'MMBench_DEV_EN.tsv'
    cn_tsv=a.cn_tsv or raw/'MMBench_DEV_CN.tsv'
    if not a.en_tsv:download_tsv(URLS['en'],en_tsv)
    if not a.cn_tsv:download_tsv(URLS['zh'],cn_tsv)
    prepare(a.out,en_tsv,cn_tsv,a.tune_size,a.seed,not a.allow_unpinned)

if __name__=='__main__': main()
