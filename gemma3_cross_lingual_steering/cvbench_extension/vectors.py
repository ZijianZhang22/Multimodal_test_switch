#!/usr/bin/env python3
"""Extract 4 taxonomies x (EN/ZH/SW) with the established Gemma3 mean-shift code."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from extract import MODEL_ID,REV,mean_shift_from_hidden,read_jsonl
from prepare import CONCEPTS, LANGS


def corpus_hash(data):
    h=hashlib.sha256()
    for c in CONCEPTS:
        for lang in LANGS:
            h.update(f'{c}/{lang}\n'.encode())
            h.update((data/'corpus'/f'{c}_{lang}.jsonl').read_bytes())
    return h.hexdigest()


def extract(data,layers):
    from transformers import AutoProcessor,Gemma3ForConditionalGeneration,__version__
    if not torch.cuda.is_available():raise RuntimeError('Gemma3 requires CUDA')
    processor=AutoProcessor.from_pretrained(MODEL_ID)
    model=Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().to('cuda')
    num=len(model.language_model.model.layers)
    if not layers or any(i<0 or i>=num for i in layers):
        raise ValueError(f'Layer indices must be within [0,{num-1}]')
    for c in CONCEPTS:
        for lang in LANGS:
            rows=read_jsonl(data/'corpus'/f'{c}_{lang}.jsonl')
            tensors=mean_shift_from_hidden(rows,processor.tokenizer,model,layers)
            target=data/'vectors'/c/lang
            target.mkdir(parents=True,exist_ok=True)
            for layer in layers:
                torch.save(tensors[layer],target/f'layer_{layer}_feature.pt')
                print(f'{c}/{lang}/layer_{layer}: norm={tensors[layer].norm().item():.3f}',flush=True)
    m={'model':MODEL_ID,'transformers':__version__,'upstream_revision':REV,
       'layers':layers,'concepts':list(CONCEPTS),'languages':list(LANGS),
       'corpus_sha256':corpus_hash(data),'extraction':'mean highlighted token - mean other token; Gemma3 decoder'}
    (data/'vectors'/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    print('Saved multilingual vectors manifest',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='outputs/gemma3_cvbench')
    p.add_argument('--layers',default='8,16')
    a=p.parse_args()
    extract(Path(a.data),[int(i) for i in a.layers.split(',')])

if __name__=='__main__':main()
