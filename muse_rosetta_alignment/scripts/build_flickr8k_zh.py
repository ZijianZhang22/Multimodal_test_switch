#!/usr/bin/env python3
import argparse,csv
from pathlib import Path
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from tqdm.auto import tqdm

def subset(split,n,out,source):
    ds=load_dataset(source,split=split,streaming=True)
    rows=[]
    for entry in tqdm(ds,total=n,desc=split):
        if len(rows)>=n:break
        cap=str(entry["caption_0"]).strip()
        if not cap:continue
        path=Path("images")/f"{split}_{len(rows):05d}.jpg"
        img=entry["image"]
        if not isinstance(img,Image.Image):raise TypeError(type(img))
        img.convert("RGB").save(out/path,format="JPEG")
        rows.append({"image_path":str(path),"en":cap,"split":split})
    if len(rows)<n:raise RuntimeError(f"Only {len(rows)} {split} rows found")
    return rows

@torch.inference_mode()
def translate(rows,name,batch):
    tok=AutoTokenizer.from_pretrained(name)
    model=AutoModelForSeq2SeqLM.from_pretrained(name).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    tag=">>cmn_Hans<<"
    if tok.convert_tokens_to_ids(tag)==tok.unk_token_id:raise RuntimeError("Translation model lacks Simplified Chinese target token")
    for k in tqdm(range(0,len(rows),batch),desc="translate"):
        part=rows[k:k+batch]
        inputs=tok([tag+" "+x["en"] for x in part],return_tensors="pt",padding=True,truncation=True,max_length=192).to(model.device)
        txt=tok.batch_decode(model.generate(**inputs,num_beams=3,max_new_tokens=128),skip_special_tokens=True)
        for row,t in zip(part,txt):
            if not t.strip():raise RuntimeError("Empty Chinese translation")
            row["zh"]=t.strip()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("data"))
    p.add_argument("--train-limit",type=int,default=3000)
    p.add_argument("--test-limit",type=int,default=500)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--source",default="intro/flickr8k")
    p.add_argument("--translator",default="Helsinki-NLP/opus-mt-en-zh")
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);(a.out/"images").mkdir(exist_ok=True)
    rows=subset("train",a.train_limit,a.out,a.source)+subset("test",a.test_limit,a.out,a.source)
    translate(rows,a.translator,a.batch_size)
    with (a.out/"manifest.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["image_path","en","zh","split"]);w.writeheader();w.writerows(rows)
    print("Saved",a.out/"manifest.csv","(machine-translated Chinese; not culturally validated)")
if __name__=="__main__":main()
