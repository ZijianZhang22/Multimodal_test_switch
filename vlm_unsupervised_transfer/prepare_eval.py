#!/usr/bin/env python3
"""Reuse exact curated 840-item multilingual local CV-Bench evaluation cohort.

No CV-Bench examples enter unpaired fitting. Original public CV-Bench 'test'
is split locally into val/test; neither is a fresh official blind benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from prepare_data import digest,read_jsonl,write_jsonl


def from_existing(src):
    rows=read_jsonl(src)
    root=Path(src).resolve().parent
    for row in rows:
        pic=Path(row['image'])
        if not pic.is_absolute():pic=(root/pic).resolve()
        if not pic.is_file():raise FileNotFoundError(pic)
        row['image']=str(pic)
    return rows


def from_hf(images):
    from datasets import load_dataset
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'paligemma2_cross_lingual_steering'))
    from apply_curated import load_translations
    def correct_letter(answer,choices):
        import re
        a=str(answer).strip()
        m=re.fullmatch(r'\(?([A-Ha-h])\)?[.)]?',a)
        if m and (ord(m.group(1).upper())-65)<len(choices):return m.group(1).upper()
        for j,choice in enumerate(choices):
            if a.casefold()==str(choice).strip().casefold():return chr(65+j)
        if a.isdigit() and 0<=int(a)<len(choices):return chr(65+int(a))
        raise ValueError('Unknown CV-Bench answer format')
    curated=load_translations()
    dataset=load_dataset('nyu-visionx/CV-Bench',split='test')
    found={};images.mkdir(parents=True,exist_ok=True)
    for raw in dataset:
        key=f'cv_{int(raw["idx"])}'
        if key not in curated:continue
        source=curated[key]
        choices=[str(x) for x in raw['choices']]
        if (str(raw['task']).strip().lower()!=source['task']
                or str(raw['question'])!=source['question']
                or choices!=source['choices']
                or correct_letter(raw['answer'],choices)!=source['answer']):
            raise ValueError(f'Curated English source/answer mismatch for {key}')
        if key in found:raise ValueError('Duplicate official CV-Bench id '+key)
        dest=images/(key+'.png')
        if not dest.is_file():raw['image'].convert('RGB').save(dest)
        found[key]=dict(id=key,image=str(dest.resolve()),task=source['task'],
                        split=source['split'],answer=source['answer'],
                        source=raw['source'],questions=source['questions'],
                        question=source['question'],choices=source['choices'])
    if set(found)!=set(curated):
        raise ValueError(f'Missing curated CV-Bench rows: {len(set(curated)-set(found))}; refuse partial cohort')
    return list(found.values())


def split_eval(rows):
    if any(row.get('split') not in ('val','test') for row in rows):
        raise ValueError('Expected curated CV-Bench local split val/test')
    if len({row['id'] for row in rows})!=len(rows):
        raise ValueError('Duplicate CV-Bench sample id')
    # Assign by FILE CONTENT, not just task/ID; keep public test rows only in test.
    sha={row['image']:digest(row['image']) for row in rows}
    test_sha={sha[r['image']] for r in rows if r['split']=='test'}
    groups={'val':[],'test':[]}
    for row in rows:
        if row['split']=='val' and sha[row['image']] in test_sha:continue
        groups[row['split']].append(row)
    # Ensure no equal image content in the two local splits.
    if {sha[r['image']] for r in groups['val']}&{sha[r['image']] for r in groups['test']}:
        raise ValueError('Image content overlap between validation and test')
    if not all(groups.values()):raise ValueError('Empty evaluation split')
    return groups


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='outputs/cvbench')
    p.add_argument('--source',default=None,help='Existing paligemma2_cross_lingual_steering/.../visual_eval.jsonl')
    a=p.parse_args();out=Path(a.out)
    if (out/'manifest.json').exists():
        print('Using existing frozen evaluation cohort',out,flush=True)
        return
    rows=from_existing(a.source) if a.source else from_hf(out/'images')
    groups=split_eval(rows)
    out.mkdir(parents=True,exist_ok=True)
    for name,group in groups.items():write_jsonl(out/(name+'.jsonl'),group)
    meta=dict(source=a.source if a.source else 'nyu-visionx/CV-Bench public test',
              note='Local CV-Bench split, not official blind-test performance; translations not human verified',
              counts={k:len(v) for k,v in groups.items()},
              sha256={k:digest(out/(k+'.jsonl')) for k in groups})
    (out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('Eval cohort:',meta['counts'],flush=True)

if __name__=='__main__':main()
