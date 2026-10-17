#!/usr/bin/env python3
"""PaliGemma2 3B four CV-Bench tasks x three languages; independent Gemma2 vectors.
Uses the author's unchanged PaliGemmaMeanShiftManipulator.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"gemma3_cross_lingual_steering"/"cvbench_extension"))
sys.path.insert(0,str(ROOT))
from vectors import MODEL_ID,REV
from prepare import LANGS,TASKS,CONCEPTS
from vectors import corpus_hash

# Preset: the two 3D spatial subtasks use spatial concept; all-four feature
# intervention can be requested to study cross-feature transfer.
TASK_CONCEPT={'count':'counting','relation':'spatial','depth':'spatial','distance':'spatial'}
FIELDS=['id','split','task','source','prompt_lang','concept','vector_lang','layer',
        'alpha','mask','normalized','prediction','parsed','answer','correct']


def parse_letter(text):
    s=str(text).strip()
    # Require a letter in an explicit answer context. Do not search arbitrary
    # prose (e.g. spelling "A" in a paragraph) for the first A-E character.
    for pattern in [r'^\s*\(?([A-Fa-f])\)?(?=\W|$)',
                    r'^(?:answer|option|choice|jibu|答案|选项)\s*[:：]?\s*\(?([A-Fa-f])\)?(?=\W|$)']:
        m=re.search(pattern,s,re.I)
        if m:return m.group(1).upper()
    return ''


def key(row):
    return tuple(str(row[f]) for f in ('id','prompt_lang','concept','vector_lang','layer','alpha','mask','normalized'))


def run(data,split,layers,alphas,mask,limit,normalized,all_features,max_tokens):
    if not (ROOT/'.upstream/models/gemma3_meanshift_model.py').exists():
        raise RuntimeError('Run parent run.sh first (upstream missing)')
    sys.path.insert(0,str(ROOT/'.upstream'))
    from models.paligemma_meanshift_model import PaliGemmaMeanShiftManipulator,MeanShiftConfig
    from transformers import PaliGemmaProcessor,PaliGemmaForConditionalGeneration,__version__
    m=json.loads((data/'vectors'/'manifest.json').read_text())
    if (m['model']!=MODEL_ID or m['upstream_revision']!=REV or
        m['transformers']!=__version__ or m['corpus_sha256']!=corpus_hash(data)):
        raise ValueError('Vector manifest mismatch; run FORCE_EXTRACT=1 for this dataset')
    if not set(layers)<=set(m['layers']):raise ValueError('Requested layers not in vector manifest')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    samples=[json.loads(x) for x in (data/'visual_eval.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    selected=[]
    for task in TASKS:
        subset=[s for s in samples if s['task']==task and s['split']==split]
        selected.extend(subset[:limit] if limit else subset)
    if not selected:raise ValueError(f'No dataset rows for {split}')
    pred=data/f'predictions_{split}.csv'
    settings={'split':split,'layers':layers,'alphas':alphas,'mask':mask,'limit':limit,
              'normalized':normalized,'all_features':all_features,'max_tokens':max_tokens,
              'dataset_manifest':json.loads((data/'dataset_manifest.json').read_text()),
              'vector_sha':m['corpus_sha256']}
    settings_path=data/f'predictions_{split}_settings.json'
    if pred.exists():
        if not settings_path.exists():raise ValueError('Predictions exist but no settings manifest: use a NEW data directory')
        if json.loads(settings_path.read_text())!=settings:
            raise ValueError('Evaluation settings changed for existing CSV; move it or use another DATA path')
    else:
        settings_path.write_text(json.dumps(settings,indent=2,ensure_ascii=False),encoding='utf-8')
    done=set()
    if pred.exists():
        with pred.open(encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f):done.add(key(r))
    processor=PaliGemmaProcessor.from_pretrained(MODEL_ID)
    model=PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().to('cuda')
    wrappers={}
    with pred.open('a',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        if not done:w.writeheader()
        for s in tqdm(selected,desc=f'CV-Bench multilingual {split}'):
            with Image.open(data/s['image']) as source: im=source.convert('RGB')
            for prompt_lang in LANGS:
                concepts=CONCEPTS if all_features else {TASK_CONCEPT[s['task']]:None}
                # Baseline one time per example/language, saved independent of vector.
                for concept in concepts:
                    for layer in layers:
                        for alpha in alphas:
                            for vl in LANGS:
                                path=data/'vectors'/concept/vl/f'layer_{layer}_feature.pt'
                                if not path.exists():raise FileNotFoundError(path)
                                k=(s['id'],prompt_lang,concept,vl,str(layer),str(float(alpha)),mask,str(int(normalized)))
                                if k in done:continue
                                wk=(concept,vl,layer,alpha)
                                if wk not in wrappers:
                                    cfg=MeanShiftConfig(feature_paths={layer:str(path)},
                                                        manipulation_values={layer:alpha},
                                                        normalize_features=normalized)
                                    wrappers[wk]=PaliGemmaMeanShiftManipulator(
                                        model=model,processor=processor,model_name=MODEL_ID,
                                        device='cuda',config=cfg,debug=False)
                                baseline_key=(s['id'],prompt_lang,'none','none','-1','0.0',mask,str(int(normalized)))
                                need_baseline=baseline_key not in done
                                result=wrappers[wk].generate_with_manipulation(
                                    image=im,text_prompt="answer en Answer the multiple choice question by only responding the letter of the correct answer. " + s['questions'][prompt_lang],
                                    masks=mask,max_new_tokens=max_tokens,do_sample=False,
                                    output_baseline=need_baseline,save_distribution_plot=False)
                                if need_baseline:
                                    original=result.get('original_text') or ''
                                    parsed=parse_letter(original)
                                    w.writerow(dict(id=s['id'],split=split,task=s['task'],source=s['source'],
                                                    prompt_lang=prompt_lang,concept='none',vector_lang='none',
                                                    layer=-1,alpha=0.0,mask=mask,normalized=int(normalized),
                                                    prediction=original,parsed=parsed,answer=s['answer'],
                                                    correct=int(parsed==s['answer'])))
                                    done.add(baseline_key)
                                text=result.get('manipulated_text') or ''
                                parsed=parse_letter(text)
                                w.writerow(dict(id=s['id'],split=split,task=s['task'],source=s['source'],
                                                prompt_lang=prompt_lang,concept=concept,vector_lang=vl,
                                                layer=layer,alpha=float(alpha),mask=mask,normalized=int(normalized),
                                                prediction=text,parsed=parsed,answer=s['answer'],
                                                correct=int(parsed==s['answer'])))
                                done.add(k)
                                f.flush()
    print(f'Saved: {pred}',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='outputs/gemma3_cvbench')
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--layers',default='8,16')
    p.add_argument('--alphas',default='0.1')
    p.add_argument('--mask',choices=['image_token','text_token','both'],default='both')
    p.add_argument('--limit',type=int,default=0,help='0=all; per task')
    p.add_argument('--normalized',type=int,choices=[0,1],default=0)
    p.add_argument('--all-features',action='store_true',help='Intervene all four concept vectors on each task')
    p.add_argument('--max-tokens',type=int,default=20)
    a=p.parse_args()
    run(Path(a.data),a.split,[int(x) for x in a.layers.split(',')],
        [float(x) for x in a.alphas.split(',')],a.mask,a.limit,bool(a.normalized),
        a.all_features,a.max_tokens)
if __name__=='__main__':main()
