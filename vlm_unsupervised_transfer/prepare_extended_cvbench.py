#!/usr/bin/env python3
"""Add Chinese MT questions to the original fixed 600-item CV-Bench test.

The validation cohort never enters this extension. All old test questions are
reused byte-for-byte. Extra questions are from the remaining official CV-Bench
public test split, translated with pinned NLLB, and tagged as machine-translated.
This is an exploratory *local* extension, not an official Chinese benchmark.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from PIL import Image
from prepare_data import read_jsonl, write_jsonl, digest

TASKS=('count','relation','depth','distance')
NLLB='facebook/nllb-200-distilled-600M'
ZH_INSTRUCTION='请只回答正确选项对应的一个大写英文字母（A、B、C 等）。'
# Fixed translation for answer options with important spatial semantics.
FIXED_OPTIONS={'left':'左边','right':'右边','above':'上方','below':'下方',
               'yes':'是','no':'否','true':'正确','false':'错误',
               'front':'前方','back':'后方','behind':'后方',
               'closer':'更近','farther':'更远','same':'相同'}


def pixel_sha(image):
    """Compare decoded pixels, so JPEG/PNG file encodings don't bypass exclusion."""
    if isinstance(image,(str,Path)):
        with Image.open(image) as f:
            rgb=f.convert('RGB')
    else:
        rgb=image.convert('RGB')
    h=hashlib.sha256()
    h.update(f'{rgb.size[0]}x{rgb.size[1]}:RGB:'.encode())
    h.update(rgb.tobytes())
    return h.hexdigest()


def answer_letter(answer,choices):
    a=str(answer).strip()
    match=re.fullmatch(r'\(?([A-Ha-h])\)?[.)]?',a)
    if match and ord(match.group(1).upper())-65<len(choices):
        return match.group(1).upper()
    for j,c in enumerate(choices):
        if a.casefold()==str(c).strip().casefold():return chr(65+j)
    if a.isdigit() and 0<=int(a)<len(choices):return chr(65+int(a))
    raise ValueError(f'Unknown answer {answer!r} choices={choices!r}')


def copy_existing(rows,tag):
    out=[]
    for row in rows:
        r=dict(row)
        if not Path(r['image']).is_file():raise FileNotFoundError(r['image'])
        if not isinstance(r['questions'].get('zh'),str) or not r['questions']['zh'].strip():
            raise ValueError(f'Missing curated Chinese question {r["id"]}')
        r['evaluation_group']=tag
        r['image_pixel_sha256']=pixel_sha(r['image'])
        out.append(r)
    return out


def choose_extra(dataset,val,test,needed,seed):
    import random
    if needed<0:raise ValueError('target total is below existing test size')
    occupied_ids={r['id'] for r in val+test}
    # Exclude validation images; test questions may share an image and are
    # then clustered by decoded-image hash in the final paired statistics.
    val_pixels={r['image_pixel_sha256'] for r in val}
    occupied_pixels={r['image_pixel_sha256'] for r in test}
    candidates=defaultdict(list)
    skipped={'used_id':0,'pixel_duplicate':0,'wrong_task':0,'invalid':0}
    for raw in dataset:
        task=str(raw['task']).strip().lower()
        if task not in TASKS:skipped['wrong_task']+=1;continue
        id_=f'cv_{int(raw["idx"])}'
        if id_ in occupied_ids:skipped['used_id']+=1;continue
        choices=[str(c) for c in raw['choices']]
        if not 2<=len(choices)<=6:skipped['invalid']+=1;continue
        try:ans=answer_letter(raw['answer'],choices)
        except ValueError:skipped['invalid']+=1;continue
        px=pixel_sha(raw['image'])
        if px in val_pixels:skipped['pixel_duplicate']+=1;continue
        candidates[task].append(dict(id=id_,task=task,answer=ans,
                                     question=str(raw['question']),choices=choices,
                                     source=str(raw.get('source','')),
                                     image_object=raw['image'],image_pixel_sha256=px))
    rng=random.Random(seed)
    for task in TASKS:
        candidates[task].sort(key=lambda x:x['id'])
        rng.shuffle(candidates[task])
    chosen=[];seen=occupied_pixels.copy()
    # Prefer previously unseen test images, but permit extra *questions* on
    # the same test image when needed. Validation image pixels stay excluded.
    deferred={task:[] for task in TASKS}
    while len(chosen)<needed:
        progressed=False
        for task in TASKS:
            while candidates[task]:
                row=candidates[task].pop()
                if row['image_pixel_sha256'] in seen:
                    deferred[task].append(row)
                    continue
                chosen.append(row);seen.add(row['image_pixel_sha256']);progressed=True
                break
            if len(chosen)>=needed:break
        if not progressed:break
    # Exhaust unique-image candidates before allowing repeated images within test.
    while len(chosen)<needed:
        progressed=False
        for task in TASKS:
            if deferred[task]:
                chosen.append(deferred[task].pop());progressed=True
            if len(chosen)>=needed:break
        if not progressed:break
    if len(chosen)!=needed:
        raise ValueError(f'Only {len(chosen)} new image-disjoint candidates; requested {needed}. '
                         f'Try a smaller --total. skipped={skipped}')
    return chosen,skipped


def fixed_option(text):
    trimmed=text.strip()
    if re.fullmatch(r'[-+]?\d+(?:\.\d+)?',trimmed):return trimmed
    return FIXED_OPTIONS.get(trimmed.lower().rstrip('.?!'),None)


class Translator:
    def __init__(self,model_name=NLLB,batch=8,revision=None):
        import torch
        from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
        self.torch=torch
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer=AutoTokenizer.from_pretrained(model_name,revision=revision,src_lang='eng_Latn')
        self.model=AutoModelForSeq2SeqLM.from_pretrained(model_name,revision=revision).eval().to(self.device)
        self.bos=self.tokenizer.convert_tokens_to_ids('zho_Hans')
        if self.bos is None or self.bos==self.tokenizer.unk_token_id:
            raise ValueError('NLLB zho_Hans target token is absent')
        self.batch=batch

    def translate(self,texts):
        out=[]
        for i in range(0,len(texts),self.batch):
            inp=self.tokenizer(texts[i:i+self.batch],return_tensors='pt',
                               padding=True,truncation=True,max_length=256).to(self.device)
            with self.torch.inference_mode():
                generated=self.model.generate(**inp,forced_bos_token_id=self.bos,
                                              max_new_tokens=160,num_beams=3)
            out+=self.tokenizer.batch_decode(generated,skip_special_tokens=True)
        return out


def translate_extra(rows,cache_file,model_name=NLLB,batch=8,revision=None):
    """Cache by exact source string and model; never translate existing curated rows."""
    cache={}
    if cache_file.is_file():
        saved=json.loads(cache_file.read_text(encoding='utf-8'))
        if saved.get('translation_model')!=model_name or saved.get('revision')!=revision:
            raise ValueError('Translation cache model/revision changed')
        cache=saved['text']
    source={r['question'] for r in rows}
    for r in rows:
        for option in r['choices']:
            if fixed_option(option) is None:source.add(option)
    missing=[s for s in sorted(source) if s not in cache]
    if missing:
        t=Translator(model_name,batch,revision)
        try:
            for i in range(0,len(missing),64):
                part=missing[i:i+64]
                result=t.translate(part)
                if len(part)!=len(result):raise RuntimeError('Missing MT outputs')
                for en,zh in zip(part,result):
                    if not zh.strip():raise ValueError(f'Empty translation for {en}')
                    cache[en]=zh.strip()
                cache_file.parent.mkdir(parents=True,exist_ok=True)
                cache_file.write_text(json.dumps({'translation_model':model_name,'revision':revision,'text':cache},
                                                 ensure_ascii=False),encoding='utf-8')
                print('Translated',min(i+64,len(missing)),'/',len(missing),flush=True)
        finally:
            del t
    return cache


def build_questions(rows,cache,img_dir):
    img_dir.mkdir(parents=True,exist_ok=True)
    result=[]
    for row in rows:
        choices_zh=[]
        for option in row['choices']:
            translated=fixed_option(option)
            choices_zh.append(translated if translated is not None else cache[option])
        question_zh=cache[row['question']]
        if not question_zh or any(not a for a in choices_zh):
            raise ValueError(f'Empty translation for {row["id"]}')
        choices_en='\n'.join(f'({chr(65+i)}) {text}' for i,text in enumerate(row['choices']))
        choices_cn='\n'.join(f'({chr(65+i)}) {text}' for i,text in enumerate(choices_zh))
        image=img_dir/(row['id']+'.png')
        if image.is_file():
            if pixel_sha(image)!=row['image_pixel_sha256']:
                raise ValueError(f'Existing image changed: {image}')
        else:
            row['image_object'].convert('RGB').save(image)
        result.append(dict(id=row['id'],image=str(image.resolve()),task=row['task'],
                           answer=row['answer'],split='test',source=row['source'],
                           question=row['question'],choices=row['choices'],
                           evaluation_group='new_mt_zh',
                           image_pixel_sha256=row['image_pixel_sha256'],
                           questions={'en':row['question']+'\n'+choices_en,
                                      'zh':ZH_INSTRUCTION+'\n'+question_zh+'\n'+choices_cn}))
    return result


def summarize_audit(rows,out):
    with (out/'translation_audit.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','task','answer','question_en','question_zh','full_zh'])
        w.writeheader()
        for r in rows:
            w.writerow(dict(id=r['id'],task=r['task'],answer=r['answer'],
                            question_en=r['question'],
                            question_zh=r['questions']['zh'].split('\n')[1],
                            full_zh=r['questions']['zh']))


def build(val_path,test_path,out,total,seed=42,translation_model=NLLB,mt_batch=8,
          dataset=None,translator_cache=None,allow_full=False):
    out=Path(out)
    if mt_batch<1:raise ValueError('mt_batch must be positive')
    if (out/'manifest.json').exists():
        manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        if (manifest.get('target_total')!=total or manifest.get('seed')!=seed or
            manifest.get('original_val_sha256')!=digest(val_path) or
            manifest.get('original_test_sha256')!=digest(test_path) or
            manifest.get('translation_model')!=translation_model):
            raise ValueError('Extended dataset exists with different settings. Use another --out')
        if digest(out/'test_extended.jsonl')!=manifest['extended_jsonl_sha256']:
            raise ValueError('Extended test data changed; refusing to reuse')
        print('Reusing verified extended test:',out/'test_extended.jsonl',flush=True)
        return manifest
    val=copy_existing(read_jsonl(val_path),'validation_excluded')
    test=copy_existing(read_jsonl(test_path),'curated_existing')
    if {r['id'] for r in val}&{r['id'] for r in test}:
        raise ValueError('Validation/test ID overlap')
    if {r['image_pixel_sha256'] for r in val}&{r['image_pixel_sha256'] for r in test}:
        raise ValueError('Validation/test image overlap')
    if total<=len(test):
        raise ValueError(f'Choose >{len(test)} total questions for expansion')
    if total>2638 or (total>2401 and not allow_full):
        raise ValueError('Requested >2401 questions; one must reserve validation examples')
    if dataset is None:
        from datasets import load_dataset
        dataset=load_dataset('nyu-visionx/CV-Bench',split='test')
    additions,skipped=choose_extra(dataset,val,test,total-len(test),seed)
    out.mkdir(parents=True,exist_ok=True)
    # Resolve immutable HF revision, record exact pretrained translator weights.
    from huggingface_hub import model_info
    revision=model_info(translation_model).sha
    if not isinstance(revision,str) or len(revision)<20:raise ValueError('Unable to pin MT revision')
    cache_path=Path(translator_cache) if translator_cache else out/'translation_cache_zh.json'
    cache=translate_extra(additions,cache_path,translation_model,mt_batch,revision)
    mt=build_questions(additions,cache,out/'new_images')
    rows=test+mt
    if len(rows)!=total or len({r['id'] for r in rows})!=total:
        raise AssertionError('Wrong item count/duplicate IDs')
    # Existing curated rows may contain more than one question per same image.
    # They are retained unchanged; report cluster bootstrap by image pixels.
    if {r['image_pixel_sha256'] for r in rows}&{r['image_pixel_sha256'] for r in val}:
        raise ValueError('Image leakage between extended test and validation')
    write_jsonl(out/'test_extended.jsonl',rows)
    summarize_audit(mt,out)
    by_task={task:sum(r['task']==task for r in rows) for task in TASKS}
    manifest=dict(target_total=total,curated_count=len(test),machine_translated_count=len(mt),
                  task_counts=by_task,seed=seed,
                  translation_model=translation_model,translation_revision=revision,translation_verified=False,
                  original_val_sha256=digest(val_path),original_test_sha256=digest(test_path),
                  extended_jsonl_sha256=digest(out/'test_extended.jsonl'),
                  original_source='nyu-visionx/CV-Bench public test',
                  note='Original curated 600 + NLLB MT extra, val 237 excluded; exploratory mixed-translation benchmark, NOT a new official CV-Bench split',
                  skipped=skipped)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Created extended test:',total,'curated',len(test),'MT',len(mt),'by_task',by_task,flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-val',required=True,type=Path)
    p.add_argument('--original-test',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--total',type=int,default=1500,help='1500 recommended; maximum with original validation excluded ~2401')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--translation-model',default=NLLB)
    p.add_argument('--mt-batch',type=int,default=8)
    a=p.parse_args()
    build(a.original_val,a.original_test,a.out,a.total,a.seed,a.translation_model,a.mt_batch)


if __name__=='__main__':main()
