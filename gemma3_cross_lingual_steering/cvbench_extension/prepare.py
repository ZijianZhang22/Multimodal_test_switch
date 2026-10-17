#!/usr/bin/env python3
"""Prepare held-out multilingual CV-Bench (Count / Relation / Depth / Distance).

CV-Bench ships ONE public test split. We make deterministic LOCAL validation
and held-out evaluation subsets, NOT a new official CV-Bench split. Translation
uses NLLB (machine generated, unverified): audit before publication.
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
sys.path.insert(0, str(PARENT))
from make_data import ZH
from anchors import ZH_EXTRA, SW

CONCEPTS = {'counting':'counting','spatial':'spatial_relationship',
            'attribute':'attribute','entity':'entity'}
TASKS = ('count','relation','depth','distance')
LANGS = ('en','zh','sw')
TARGET_LANG = {'zh':'zho_Hans', 'sw':'swh_Latn'}
INSTRUCTION = {
    'en': 'Answer the multiple-choice question with only one capital letter (A, B, C, D, or E).',
    'zh': '请只回答正确选项的一个大写英文字母（A、B、C、D 或 E）。',
    'sw': 'Jibu swali la kuchagua kwa herufi moja kubwa tu (A, B, C, D au E).'
}


def jsonl_write(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')


def build_anchors(out):
    if not (PARENT/'.upstream/get_features/taxonomy_sentences.py').exists():
        raise FileNotFoundError('Run the parent run.sh once first to clone upstream')
    sys.path.insert(0, str(PARENT/'.upstream/get_features'))
    from taxonomy_sentences import generate_taxonomy_sentences
    for concept, taxonomy in CONCEPTS.items():
        original = generate_taxonomy_sentences(taxonomy)
        chinese = ZH[concept] if concept in ZH else ZH_EXTRA[concept]
        swahili = SW[concept]
        if not all(len(v)==20 for v in (original,chinese,swahili)):
            raise ValueError(f'Expected 20 anchors for {concept} and every language')
        for lang,pairs in [('en',original),('zh',chinese),('sw',swahili)]:
            for sentence, target in pairs:
                if sentence.count(target)!=1:
                    raise ValueError(f'Ambiguous target: {lang}/{concept}: {sentence!r}/{target!r}')
            jsonl_write(out/'corpus'/f'{concept}_{lang}.jsonl',
                        [{'id':i,'sentence':s,'target':t} for i,(s,t) in enumerate(pairs)])


def correct_letter(answer, choices):
    import re
    a=str(answer).strip()
    m=re.fullmatch(r'\(?([A-Ha-h])\)?[.)]?',a)
    if m and (ord(m.group(1).upper())-65)<len(choices):
        return m.group(1).upper()
    for j, choice in enumerate(choices):
        if a.casefold()==str(choice).strip().casefold():
            return chr(65+j)
    if a.isdigit() and 0<=int(a)<len(choices):
        return chr(65+int(a))
    raise ValueError(f'Cannot identify CV-Bench correct choice: {answer!r} {choices!r}')


class NLLBTranslator:
    def __init__(self, model_id='facebook/nllb-200-distilled-600M', batch=8):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch=torch
        self.tok=AutoTokenizer.from_pretrained(model_id, src_lang='eng_Latn')
        self.model=AutoModelForSeq2SeqLM.from_pretrained(model_id).eval()
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.batch=batch
        self.model_id=model_id
        print(f'Translation model {model_id} on {self.device}',flush=True)

    def translate(self, texts, lang):
        if lang=='en': return texts
        tgt_id=self.tok.convert_tokens_to_ids(TARGET_LANG[lang])
        outs=[]
        for start in range(0,len(texts),self.batch):
            segment=texts[start:start+self.batch]
            x=self.tok(segment,return_tensors='pt',padding=True,truncation=True,max_length=512).to(self.device)
            with self.torch.inference_mode():
                y=self.model.generate(**x, forced_bos_token_id=tgt_id,max_new_tokens=180,num_beams=3)
            outs.extend(self.tok.batch_decode(y,skip_special_tokens=True))
        return outs


def prepare(data, n_val, n_test, seed, translate=True, translation_model='facebook/nllb-200-distilled-600M'):
    from datasets import load_dataset
    import torch
    data.mkdir(parents=True,exist_ok=True)
    build_anchors(data)
    print('Loading official CV-Bench dataset: nyu-visionx/CV-Bench (public test split)',flush=True)
    dataset=load_dataset('nyu-visionx/CV-Bench',split='test')
    bytask={task:[] for task in TASKS}
    for row in dataset:
        task=str(row['task']).strip().lower()
        if task in bytask: bytask[task].append(row)
    if any(not bytask[task] for task in TASKS):
        raise ValueError('Unexpected CV-Bench task names: '+str(Counter(str(r['task']) for r in dataset)))
    rng=random.Random(seed)
    samples=[]
    for task in TASKS:
        pool=sorted(bytask[task],key=lambda r:int(r['idx']))
        rng.shuffle(pool)
        needed=n_val+n_test
        if n_test==0: needed=len(pool)
        if n_val>=len(pool) or needed>len(pool):
            raise ValueError(f'{task} has {len(pool)} samples; need val={n_val}, test={n_test}')
        selection=pool if n_test==0 else pool[:needed]
        for index,row in enumerate(selection):
            split='val' if index<n_val else 'test'
            choices=[str(c) for c in row['choices']]
            answer=correct_letter(row['answer'],choices)
            image_name=f'{task}_{int(row["idx"]):06d}.png'
            dest=data/'images'/image_name
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():
                row['image'].convert('RGB').save(dest)
            samples.append({'id':f'cv_{int(row["idx"])}','task':task,
                            'split':split,'image':f'images/{image_name}',
                            'source':row['source'],'answer':answer,
                            'question':str(row['question']),'choices':choices})
    cache_path=data/'translation_cache.json'
    cache=json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {}
    if translate:
        t=None
        try:
            for lang in ('zh','sw'):
                unique=sorted({s['question'] for s in samples}|{c for s in samples for c in s['choices']})
                missing=[x for x in unique if f'{lang}\t{x}' not in cache]
                if missing:
                    if t is None: t=NLLBTranslator(translation_model)
                    print(f'Translating {len(missing)} distinct strings to {lang}',flush=True)
                    for i in range(0,len(missing),64):
                        part=missing[i:i+64]
                        translated=t.translate(part,lang)
                        if len(translated)!=len(part): raise RuntimeError('Translation count mismatch')
                        for a,b in zip(part,translated):
                            if not b.strip():raise ValueError(f'Empty {lang} translation of {a!r}')
                            cache[f'{lang}\t{a}']=b
                        cache_path.write_text(json.dumps(cache,ensure_ascii=False),encoding='utf-8')
        finally:
            if t is not None:
                del t
                if torch.cuda.is_available():torch.cuda.empty_cache()
    for sample in samples:
        sample['questions']={}
        for lang in LANGS:
            question=sample['question'] if lang=='en' else cache.get(f'{lang}\t{sample["question"]}')
            if not question: raise RuntimeError(f'Missing translation for {lang}: {sample["id"]}. Rerun without --no-translate')
            choices=[c if lang=='en' else cache.get(f'{lang}\t{c}') for c in sample['choices']]
            if any(not c for c in choices):raise RuntimeError(f'Missing choice translation: {sample["id"]}/{lang}')
            choices_block='\n'.join(f'({chr(65+j)}) {c}' for j,c in enumerate(choices))
            sample['questions'][lang]=f'{INSTRUCTION[lang]}\n{question}\n{choices_block}'
    jsonl_write(data/'visual_eval.jsonl',samples)
    # Save a human-readable sample of actual translations for quality review.
    with (data/'translation_audit.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','task','answer','en','zh','sw'])
        w.writeheader()
        for s in samples[:30]:w.writerow({k:s[k] if k in ('id','task','answer') else s['questions'][k] for k in w.fieldnames})
    manifest={'source_dataset':'nyu-visionx/CV-Bench','original_split':'test',
              'local_split':'deterministic seeded shuffle per task','seed':seed,
              'val_per_task':n_val,'test_per_task':n_test,'task_counts':dict(Counter((s['task'],s['split']) for s in samples)),
              'translation_model':translation_model,'translation_verified':False,
              'languages':list(LANGS),'concepts':list(CONCEPTS)}
    manifest['task_counts']={f'{task}_{split}':sum(x['task']==task and x['split']==split for x in samples)
                             for task in TASKS for split in ('val','test')}
    (data/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Wrote {len(samples)} unique images, three parallel prompts each; inspect {data}/translation_audit.csv',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='outputs/gemma3_cvbench')
    p.add_argument('--n-val',type=int,default=60)
    p.add_argument('--n-test',type=int,default=150,help='Per task. 0 uses every remaining sample')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--no-translate',action='store_true')
    a=p.parse_args()
    prepare(Path(a.data),a.n_val,a.n_test,a.seed,translate=not a.no_translate)

if __name__=='__main__':main()
