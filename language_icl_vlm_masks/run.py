#!/usr/bin/env python3
"""Kirtane language-vector CV-Bench ablation on PaliGemma2, independent of Gan vectors."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
from collections import defaultdict
import torch
from PIL import Image
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"paligemma2_cross_lingual_steering"/"parity"))
from common import MODEL_ID,build_prompt,jsonl_read,score
from core import partition,sample_pairs,hash_pairs,extract_vectors,token_masks,steer_decoder,demos_text,paired_metrics
FIELDS=["id","task","split","lang","mask","layer","alpha","response","normalized","valid","correct","tokens_edited"]

def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--lang",choices=["zh","sw"],default="zh")
    ap.add_argument("--layers",default="8,12,16")
    ap.add_argument("--alphas",default="0.5,1.0,2.0")
    ap.add_argument("--masks",default="text,image,both")
    ap.add_argument("--k",type=int,default=6)
    ap.add_argument("--shots",type=int,default=6)
    ap.add_argument("--extract-limit",type=int,default=12)
    ap.add_argument("--limit-val",type=int,default=3)
    ap.add_argument("--limit-test",type=int,default=3)
    ap.add_argument("--max-new-tokens",type=int,default=20)
    args=ap.parse_args()
    layers=[int(x) for x in args.layers.split(",")]
    alphas=[float(x) for x in args.alphas.split(",")]
    masks=args.masks.split(",")
    assert all(x in ("text","image","both") for x in masks)
    assert args.k>0 and args.shots>=0
    data=args.data.resolve();out=args.out.resolve()
    samples=list(jsonl_read(data/"visual_eval.jsonl"))
    split=partition(samples)
    settings=dict(vars(args))
    settings["data"]=str(data);settings["out"]=str(out)
    settings["dataset_sha"]=hashlib.sha256((data/"visual_eval.jsonl").read_bytes()).hexdigest()
    out.mkdir(parents=True,exist_ok=True)
    p=out/"settings.json"
    if p.exists() and json.loads(p.read_text())!=settings:
        raise RuntimeError("Settings changed; choose NEW OUT")
    p.write_text(json.dumps(settings,indent=2))
    from transformers import PaliGemmaForConditionalGeneration,PaliGemmaProcessor
    processor=PaliGemmaProcessor.from_pretrained(MODEL_ID)
    model=PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID,torch_dtype=torch.bfloat16).eval().to("cuda")
    assert max(layers)<len(model.language_model.model.layers)
    pairs=sample_pairs(split["compute"],args.lang,args.k,args.extract_limit)
    vec_path=out/"language_vectors.pt";meta_path=out/"vector_manifest.json"
    meta=dict(model=MODEL_ID,layers=layers,pairs_sha256=hash_pairs(pairs))
    if vec_path.exists():
        if not meta_path.exists() or json.loads(meta_path.read_text())!=meta:
            raise RuntimeError("Vector cache mismatch")
        vectors=torch.load(vec_path,map_location="cpu",weights_only=True)
    else:
        vectors=extract_vectors(model,processor.tokenizer,pairs,layers,"cuda")
        torch.save(vectors,vec_path);meta_path.write_text(json.dumps(meta,indent=2))
    print("Language vectors computed/loaded:",{k:round(v.norm().item(),4) for k,v in vectors.items()})
    def generate(image,prompt,mask="baseline",layer=-1,alpha=0.):
        inp=processor(text=prompt,images=image,return_tensors="pt").to(torch.bfloat16).to("cuda")
        n=inp["input_ids"].shape[-1]
        if mask=="baseline":
            with torch.inference_mode():
                output=model.generate(**inp,max_new_tokens=args.max_new_tokens,do_sample=False)
            return processor.decode(output[0][n:],skip_special_tokens=True).strip(),0
        cmask=token_masks(inp["input_ids"],model.config.image_token_index,
                          [model.config.pad_token_id,model.config.bos_token_id,model.config.eos_token_id])[mask]
        with torch.inference_mode(),steer_decoder(model,layer,vectors[layer],alpha,cmask) as edits:
            output=model.generate(**inp,max_new_tokens=args.max_new_tokens,do_sample=False)
        if not edits["calls"]:
            raise RuntimeError("Steering hook never fired")
        return processor.decode(output[0][n:],skip_special_tokens=True).strip(),edits["tokens"]
    def evaluate(rows,stage,configs,limit):
        tasks=defaultdict(list)
        for s in rows:tasks[s["task"]].append(s)
        selected=[s for task in sorted(tasks) for s in (tasks[task][:limit] if limit else tasks[task])]
        output=[]
        for i,s in enumerate(selected):
            with Image.open(data/s["image"]) as im:image=im.copy()
            demo=demos_text(split["compute"],s["task"],args.shots)
            prompt=demo+build_prompt(s,args.lang,"native")
            for mask,layer,alpha in [("baseline",-1,0.)]+configs:
                response,edited=generate(image,prompt,mask,layer,alpha)
                sc=score(response,s["answer"],s["choices"],s["choices_i18n"][args.lang],args.lang)
                output.append(dict(id=s["id"],task=s["task"],split=stage,lang=args.lang,
                                   mask=mask,layer=layer,alpha=alpha,response=response,
                                   tokens_edited=edited,**sc))
            if (i+1)%5==0:print(stage,i+1,"/",len(selected),flush=True)
        return output
    chosen_path=out/"chosen.json"
    if not chosen_path.exists():
        grid=[(m,l,a) for m in masks for l in layers for a in alphas]
        val=evaluate(split["val"],"val",grid,args.limit_val)
        write_csv(out/"validation.csv",val)
        chosen={}
        for mask in masks:
            candidates=[(l,a) for l in layers for a in alphas]
            def metric(pair):
                rows=[r for r in val if r["mask"]==mask and r["layer"]==pair[0] and r["alpha"]==pair[1]]
                return (sum(int(r["correct"]) for r in rows)/len(rows),-pair[1],-pair[0])
            best=max(candidates,key=metric)
            chosen[mask]=dict(layer=best[0],alpha=best[1])
        chosen_path.write_text(json.dumps(chosen,indent=2))
    else:chosen=json.loads(chosen_path.read_text())
    if (out/"test.csv").exists():
        raise RuntimeError("Held-out test already exists; do not overwrite")
    configs=[(m,int(chosen[m]["layer"]),float(chosen[m]["alpha"])) for m in masks]
    result=evaluate(split["test"],"test",configs,args.limit_test)
    write_csv(out/"test.csv",result)
    summary=paired_metrics(result)
    lines=["# Kirtane-style language steering: VLM image/text/both ablation",
           "Language: "+args.lang,"Frozen validation settings: "+json.dumps(chosen),
           "| Mask | n | Baseline | Steered | Delta pp | Fixed | Broken | Disagreement |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for (lang,mask),v in summary.items():
        lines.append("| {} | {} | {:.3f} | {:.3f} | {:+.2f} | {} | {} | {:.3f} |".format(
             mask,v["n"],v["baseline_accuracy"],v["steered_accuracy"],v["delta_pp"],
             v["fixed"],v["broken"],v["discordance"]))
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
if __name__=="__main__":main()
