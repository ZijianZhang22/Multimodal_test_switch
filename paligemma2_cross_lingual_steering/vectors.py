#!/usr/bin/env python3
"""Extract Mean Shift from independent text-only google/gemma-2-2b.

Upstream math: pooled target-token mean minus pooled remaining non-BOS token mean.
Tokenizer-offset target alignment is the only extraction adaptation needed for ZH/SW.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import torch
from tqdm import tqdm

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"gemma3_cross_lingual_steering"/"cvbench_extension"))
from prepare import CONCEPTS,LANGS
MODEL_ID="google/paligemma2-3b-mix-448"
TEXT_ID="google/gemma-2-2b"
REV="ce1c0f0cffed9b859901ce347f9d2b869a512178"

def corpus_hash(data):
    h=hashlib.sha256()
    for c in CONCEPTS:
        for lang in LANGS:
            h.update(f"{c}/{lang}\n".encode())
            h.update((data/"corpus"/f"{c}_{lang}.jsonl").read_bytes())
    return h.hexdigest()

def mean_shift(model,tok,rows,layers):
    # Exactly the original Mean Shift pooling, with tokenizer character offsets
    # for reliable multi-token target spans in non-English input.
    total={l:{"t":None,"c":None,"nt":0,"nc":0} for l in layers}
    for row in tqdm(rows,desc="Text-only concept extraction"):
        sentence,target=row["sentence"],row["target"]
        if sentence.count(target)!=1:raise ValueError((sentence,target))
        batch=tok(sentence,return_tensors="pt",return_offsets_mapping=True)
        offsets=batch.pop("offset_mapping")[0].tolist()
        start=sentence.index(target); end=start+len(target)
        valid=[i for i,(a,b) in enumerate(offsets) if b>a]
        tgt=[i for i in valid if offsets[i][0]<end and offsets[i][1]>start]
        ctl=[i for i in valid if i not in tgt]
        if not tgt or not ctl:raise RuntimeError(f"Missing target/control: {sentence}")
        batch=batch.to(model.device)
        with torch.inference_mode():
            states=model(**batch,output_hidden_states=True,use_cache=False).hidden_states
        for l in layers:
            h=states[l+1][0].detach().float().cpu()
            t=h[tgt].double().sum(0); c=h[ctl].double().sum(0)
            st=total[l]
            st["t"]=t if st["t"] is None else st["t"]+t
            st["c"]=c if st["c"] is None else st["c"]+c
            st["nt"]+=len(tgt); st["nc"]+=len(ctl)
    return {l:(s["t"]/s["nt"]-s["c"]/s["nc"]).float() for l,s in total.items()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--layers",default="8,16")
    a=p.parse_args()
    data=Path(a.data)
    layers=[int(v) for v in a.layers.split(",")]
    from transformers import AutoTokenizer,AutoModelForCausalLM,__version__
    tok=AutoTokenizer.from_pretrained(TEXT_ID,use_fast=True)
    model=AutoModelForCausalLM.from_pretrained(TEXT_ID,torch_dtype=torch.bfloat16,attn_implementation="sdpa").eval().to("cuda")
    if min(layers)<0 or max(layers)>=len(model.model.layers):raise ValueError("Invalid decoder layer")
    for c in CONCEPTS:
        for lang in LANGS:
            rows=[json.loads(s) for s in (data/"corpus"/f"{c}_{lang}.jsonl").read_text(encoding="utf-8").splitlines() if s]
            vectors=mean_shift(model,tok,rows,layers)
            dest=data/"vectors"/c/lang;dest.mkdir(parents=True,exist_ok=True)
            for l,v in vectors.items():
                torch.save(v,dest/f"layer_{l}_feature.pt")
                print(f"{c}/{lang} layer={l} norm={v.norm().item():.4f}",flush=True)
    manifest={"model":MODEL_ID,"text_model":TEXT_ID,"upstream_revision":REV,"transformers":__version__,
              "layers":layers,"concepts":list(CONCEPTS),"languages":list(LANGS),
              "corpus_sha256":corpus_hash(data),"method":"upstream pooled mean-shift; offset span alignment"}
    (data/"vectors"/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("Saved text-only vectors",flush=True)

if __name__=="__main__":main()
