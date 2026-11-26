#!/usr/bin/env python3
"""Prepare strictly image-disjoint EN / non-EN / visual Flickr8K pools.

Only the official Flickr8K training split is sampled. No parallel EN--ZH pairs
or same-image bilingual pairs are used in training. Each source caption may be
paired with its OWN image; this is cross-lingually unpaired, not image-free SSL.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

REV = 'a19b9aa4a7ad2a42741460d9f47385637ae270cd'
BASE = f'https://raw.githubusercontent.com/li-xirong/flickr8kcn/{REV}/data/'
IMAGE_URL = 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip'
CAPTIONS = ('flickr8kenc.caption.txt','flickr8kzhg.caption.txt','flickr8ktrain.txt',
            'flickr8kval.txt','flickr8ktest.txt')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + '.downloading')
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'vlm-unaligned-research/1.0'}), timeout=120) as src, open(tmp,'wb') as output:
        shutil.copyfileobj(src, output)
    tmp.replace(dest)
    return dest


def parse_captions(path):
    rows={}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        key, caption = line.split(maxsplit=1)
        image, number = key.rsplit('#',1)
        identifier=(Path(image).stem,int(number))
        if identifier in rows:
            raise ValueError(f'Duplicate caption {identifier}')
        rows[identifier]=caption.strip()
    return rows


def split_ids(path):
    return {Path(line.strip()).stem for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()}


def swahili_captions(path):
    out={}
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader=csv.DictReader(f)
        if not {'image','caption_SW'} <= set(reader.fieldnames or []):
            raise ValueError('Expected SwaFlickr8K CSV columns image,caption_SW')
        for row in reader:
            key=Path(row['image'].strip()).stem
            if key and row['caption_SW'].strip() and key not in out:
                out[key]=row['caption_SW'].strip()
    return out


def disjoint_groups(ids, n_source, n_target, n_visual, seed, source_available=None):
    """Split by image ID BEFORE constructing any source/target examples."""
    ids=sorted(set(ids))
    rng=random.Random(seed)
    rng.shuffle(ids)
    need=n_source+n_target+n_visual
    if any(x<2 for x in (n_source,n_target,n_visual)) or len(ids)<need:
        raise ValueError(f'Need {need} image-disjoint training IDs, found {len(ids)}')
    if source_available is not None:
        ids=[i for i in ids if i in source_available]
        if len(ids)<need: raise ValueError(f'Only {len(ids)} language-eligible IDs; need {need}')
    groups={'source': ids[:n_source],
            'english': ids[n_source:n_source+n_target],
            'visual': ids[n_source+n_target:need]}
    vals=[set(x) for x in groups.values()]
    assert not any(vals[i]&vals[j] for i in range(3) for j in range(i+1,3))
    return groups


def ensure_images(raw, stems, archive=None, download=True):
    raw=Path(raw)
    images=raw/'images';images.mkdir(parents=True,exist_ok=True)
    missing={stem for stem in stems if not (images/(stem+'.jpg')).is_file()}
    if missing:
        zpath=Path(archive) if archive else raw/'Flickr8k_Dataset.zip'
        if not zpath.is_file():
            if not download:raise FileNotFoundError(f'Image archive missing: {zpath}')
            fetch(IMAGE_URL,zpath)
        with zipfile.ZipFile(zpath) as z:
            for member in z.infolist():
                name=Path(member.filename).name
                if Path(name).stem in missing and name.lower().endswith('.jpg'):
                    dest=images/name
                    if not dest.exists():
                        with z.open(member) as src, dest.open('wb') as dst:
                            shutil.copyfileobj(src,dst)
        missing={stem for stem in stems if not (images/(stem+'.jpg')).is_file()}
        if missing:raise FileNotFoundError(f'Missing image files: {sorted(missing)[:5]}')
    # Verify duplicate image bytes cannot silently cross pools.
    return {stem:str((images/(stem+'.jpg')).resolve()) for stem in stems}


def write_jsonl(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def validate_pools(directory):
    path=Path(directory)
    groups={name:read_jsonl(path/f'{name}.jsonl') for name in ('source','english','visual')}
    ids=[{r['id'] for r in rows} for rows in groups.values()]
    if any(ids[i]&ids[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Cross-language/cross-modality image-ID pairing detected')
    manifest_path=path/'manifest.json'
    if manifest_path.is_file():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        for name,expected in manifest.get('sha256',{}).items():
            if digest(path/(name+'.jsonl'))!=expected:
                raise ValueError('Prepared corpus changed after manifest creation: '+name)
    sha_sets=[]
    for name,rows in groups.items():
        if not rows:raise ValueError(f'Empty {name} pool')
        if any(not Path(r['image']).is_file() for r in rows):raise FileNotFoundError('Missing training image')
        if name!='visual' and any(not r.get('caption','').strip() for r in rows):
            raise ValueError(f'Empty caption in {name}')
        sha_sets.append({digest(r['image']) for r in rows})
    if any(sha_sets[i]&sha_sets[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Identical image bytes cross pools')
    return groups


def prepare(args):
    raw=Path(args.raw).resolve();raw.mkdir(parents=True,exist_ok=True)
    if args.download:
        for filename in CAPTIONS:fetch(BASE+filename,raw/filename)
    else:
        for filename in CAPTIONS:
            if not (raw/filename).is_file():raise FileNotFoundError(raw/filename)
    en=parse_captions(raw/'flickr8kenc.caption.txt')
    zh=parse_captions(raw/'flickr8kzhg.caption.txt')
    train=split_ids(raw/'flickr8ktrain.txt')
    val=split_ids(raw/'flickr8kval.txt')
    test=split_ids(raw/'flickr8ktest.txt')
    if train&(val|test) or val&test:raise ValueError('Flickr8K upstream split overlap')
    if args.language=='sw' and not args.swahili_csv:
        raise ValueError('--swahili-csv is mandatory for --language sw')
    sw=swahili_captions(args.swahili_csv) if args.language=='sw' else {}
    have={i for i in train if (i,0) in en and ((i,0) in zh if args.language=='zh' else i in sw)}
    groups=disjoint_groups(have,args.n_source,args.n_english,args.n_visual,args.seed)
    images=ensure_images(raw,set().union(*[set(x) for x in groups.values()]),args.image_archive,args.download)
    all_rows={}
    for name,ids in groups.items():
        all_rows[name]=[]
        for identifier in ids:
            caption=(zh[identifier,0] if args.language=='zh' else sw[identifier]) if name=='source' else (
                en[identifier,0] if name=='english' else '')
            all_rows[name].append(dict(id=identifier,image=images[identifier],caption=caption,language=(
                args.language if name=='source' else ('en' if name=='english' else 'visual'))))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if any((out/f'{name}.jsonl').exists() for name in all_rows):
        raise FileExistsError(f'Refusing to overwrite a prepared dataset at {out}; use a clean --out')
    for name,rows in all_rows.items():write_jsonl(out/f'{name}.jsonl',rows)
    validate_pools(out)
    manifest={'language':args.language,'seed':args.seed,'source':'Flickr8K-CN pinned '+REV,
              'source_is_translation_for_zh': args.language=='zh',
              'unpaired_definition':'No shared training image ID across source/English/visual pools',
              'uses_monolingual_image_caption_pairs':True,
              'train_split_only':True,'counts':{k:len(v) for k,v in all_rows.items()},
              'pool_image_ids':groups,
              'sha256':{name:digest(out/f'{name}.jsonl') for name in all_rows},
              'notes':['Flickr captions and EN translations may have been observed during encoder pretraining.',
                       'No bilingual correspondences are used by the fitting algorithm.',
                       'Visual pool is an independent image-only pool with neutral English instruction.']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Prepared',manifest['counts'],'language',args.language,'->',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='outputs/flickr_zh')
    p.add_argument('--raw',default='../visual_anchor/real_work_clip_native/raw')
    p.add_argument('--language',choices=('zh','sw'),default='zh')
    p.add_argument('--swahili-csv',default=None)
    p.add_argument('--image-archive',default=None)
    p.add_argument('--n-source',type=int,default=256)
    p.add_argument('--n-english',type=int,default=256)
    p.add_argument('--n-visual',type=int,default=256)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--download',action=argparse.BooleanOptionalAction,default=True)
    prepare(p.parse_args())

if __name__=='__main__':main()