#!/usr/bin/env python3
"""MMRB first-four image-to-caption intervention. Run --help for commands.

Six schedules act only on FIRST FOUR images; image 5+ stays ORIGINAL.
The four-character schedule is not the whole input's modality mixture.
"""
import argparse, collections, io, json, os, random, re, string, sys, time
from pathlib import Path

SCHEDULES=("VVVV","TTTT","VTTT","TVTT","TTVT","TTTV")
CAPTION_PROMPT=("Describe this image in detail. Include information about the setting, objects, "
                "people, clothing, actions, facial expressions, colors, numbers, and any notable "
                "background elements. Be as specific and thorough as possible.")
DATASET_ID="HarrytheOrange/MMRB"

def load_jsonl(path):
    if not Path(path).exists(): return []
    with open(path,encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]

def add_jsonl(path,item):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,"a",encoding="utf-8") as f:f.write(json.dumps(item,ensure_ascii=False)+"\n")

def image_items(row):
    """Ordered inputs only. Reject ambiguously arranged image data rather than guessing."""
    for key in ("images","image_paths","image_list","image_files","image"):
        v=row.get(key)
        if isinstance(v,list) and v: return v,key
    # Numbered image columns, sorted numerically.
    numbered=[]
    for k,v in row.items():
        m=re.fullmatch(r"(?:image|img)[_\-]?(\d+)",k)
        if m and v is not None: numbered.append((int(m[1]),v))
    if numbered: return [v for _,v in sorted(numbered)],"numbered"
    return [],"unknown"

def source_value(x):
    if isinstance(x,dict):
        if x.get("bytes") is not None:return x["bytes"]
        if x.get("path"):return x["path"]
        if x.get("image") is not None:return source_value(x["image"])
    return x

def materialize(image,output,roots):
    """Save an HF decoded PIL image / bytes / existing path as PNG for deterministic reuse."""
    from PIL import Image
    x=source_value(image)
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
    if hasattr(x,"save"):
        src=x
    elif isinstance(x,bytes):
        src=Image.open(io.BytesIO(x))
    elif isinstance(x,str):
        loc=Path(x)
        if not loc.is_file():
            found=[Path(root)/x for root in roots if (Path(root)/x).is_file()]
            if not found: raise FileNotFoundError("Image path not resolved: "+repr(x))
            loc=found[0]
        src=Image.open(loc)
    else:
        raise TypeError("Unsupported HF image field: "+str(type(x)))
    src.convert("RGB").save(p,format="PNG")
    return str(p.resolve())

def select(args):
    from datasets import load_dataset,DatasetDict
    print("Loading official HF dataset:",DATASET_ID,flush=True)
    ds=load_dataset(DATASET_ID,"all")
    print("Splits:",list(ds.keys()) if isinstance(ds,DatasetDict) else "single",flush=True)
    if isinstance(ds,DatasetDict):
        if "train" in ds: ds=ds["train"]
        else: ds=next(iter(ds.values()))
    print("Rows:",len(ds),"columns:",ds.column_names,flush=True)
    print("First row field types:",{k:type(v).__name__ for k,v in ds[0].items()},flush=True)
    rng=random.Random(args.seed)
    ids=list(range(len(ds)));rng.shuffle(ids)
    rows=[]; counts=collections.Counter();errors=collections.Counter();roots=[".",args.asset_root]
    out=Path(args.output_dir);(out/"images").mkdir(parents=True,exist_ok=True)
    for i in ids:
        row=ds[i]
        items,source=image_items(row);counts[len(items)]+=1
        if len(items)<4:continue
        if args.max_images and len(items)>args.max_images:continue
        q=row.get("question") or row.get("query") or row.get("prompt")
        if not isinstance(q,str) or not q.strip():errors["missing_question"]+=1;continue
        opts=row.get("options") or row.get("choices") or []
        if isinstance(opts,str):
            try:opts=json.loads(opts)
            except Exception:opts=[opts]
        if isinstance(opts,dict):opts=[f"{k}. {v}" for k,v in opts.items()]
        if not isinstance(opts,list):errors["bad_options"]+=1;continue
        if any(not isinstance(o,(str,int,float)) for o in opts):
            errors["non_text_options"]+=1;continue
        answer=row.get("answer",row.get("golden_answer",row.get("correct_answer")))
        if answer is None:errors["missing_answer"]+=1;continue
        # Determine a stable identifier. The same question+image order is used in all conditions.
        rid=f'{row.get("subtask",row.get("task","mmrb"))}_{row.get("index",i)}'
        rid=re.sub(r"[^A-Za-z0-9_-]","_",str(rid))
        imagepaths=[]
        try:
            for j,x in enumerate(items):
                imagepaths.append(materialize(x,out/"images"/f"{rid}_{j+1}.png",roots))
        except Exception as exc:
            errors["unresolved_image"]+=1
            print("SKIP",rid,str(exc)[:180],flush=True)
            for p in imagepaths:Path(p).unlink(missing_ok=True)
            continue
        rows.append({"qid":rid,"dataset_index":i,"question":q,
                     "options":[str(x) for x in opts],"answer":str(answer),
                     "question_type":str(row.get("question_type","")),
                     "subtask":str(row.get("subtask","")),
                     "original_image_count":len(imagepaths),
                     "image_paths":imagepaths,"caption_targets":imagepaths[:4],
                     "source_image_column":source,
                     "reasoning_steps_available":bool(row.get("reasoning_steps"))})
        if len(rows)>=args.n:break
    print("Observed image counts:",dict(sorted(counts.items())),flush=True)
    print("Rejected reasons:",dict(errors),flush=True)
    print("Selected:",len(rows),"requested:",args.n,flush=True)
    if len(rows)<args.n:raise RuntimeError(
        f"Only {len(rows)} eligible MMRB questions; please inspect schema and --max-images")
    dest=out/"manifest.jsonl"
    with open(dest,"w",encoding="utf-8") as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print("Manifest:",dest,flush=True)

def load_model(name,captioner=False):
    import torch
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(name)
    if captioner:
        from transformers import LlavaForConditionalGeneration
        cls=LlavaForConditionalGeneration
    else:
        from transformers import Qwen3VLForConditionalGeneration
        cls=Qwen3VLForConditionalGeneration
    model=cls.from_pretrained(name,torch_dtype="auto",device_map="auto")
    model.eval()
    return model,processor

def generate(model,processor,content,max_new_tokens):
    import torch
    messages=[{"role":"user","content":content}]
    batch=processor.apply_chat_template(
        messages,add_generation_prompt=True,tokenize=True,
        return_dict=True,return_tensors="pt")
    batch=batch.to(next(model.parameters()).device)
    prompt_length=batch["input_ids"].shape[1]
    with torch.inference_mode():
        ids=model.generate(**batch,do_sample=False,max_new_tokens=max_new_tokens)
    new=ids[0,prompt_length:]
    text=processor.decode(new,skip_special_tokens=True)
    return text.strip(),int(new.numel()),len(new)>=max_new_tokens

def caption(args):
    from PIL import Image
    manifest=load_jsonl(args.manifest)
    existing={x["path"]:x["caption"] for x in load_jsonl(args.output)
              if x.get("caption") and x.get("path")}
    targets=list(dict.fromkeys(path for row in manifest for path in row["caption_targets"]))
    todo=[p for p in targets if p not in existing]
    print("Captions cached:",len(existing),"pending:",len(todo),flush=True)
    if not todo:return
    model,processor=load_model(args.model,captioner=True)
    for i,path in enumerate(todo,1):
        try:
            with Image.open(path) as f: picture=f.convert("RGB")
            content=[{"type":"image","image":picture},{"type":"text","text":CAPTION_PROMPT}]
            result,ntok,truncated=generate(model,processor,content,200)
            if not result:raise ValueError("empty caption")
            add_jsonl(args.output,{"path":path,"caption":result,"tokens":ntok,
                                   "truncated":truncated,"model":args.model,
                                   "prompt":CAPTION_PROMPT})
            print(f"[CAPTION {i}/{len(todo)}] {Path(path).name} tokens={ntok}",flush=True)
        except Exception as exc:
            print("CAPTION ERROR",path,repr(exc),flush=True)
    print("Caption complete, restartable via same command",flush=True)

def build_content(row,schedule,captions):
    from PIL import Image
    if schedule not in SCHEDULES:raise ValueError(schedule)
    paths=row["image_paths"]
    if len(paths)<4:raise ValueError("Need >=4 images")
    out=[]
    def put(s):out.append({"type":"text","text":s})
    put("Use the evidence in order. Answer the question based on ALL provided evidence. "
        "Some evidence is supplied as original images and some as image descriptions.\n")
    for i,path in enumerate(paths):
        put(f"\n[Evidence image {i+1}]\n")
        if i<4 and schedule[i]=="T":
            if path not in captions:raise ValueError("Missing caption: "+path)
            put("Description: "+captions[path]+"\n")
        else:
            put("Original image:\n")
            with Image.open(path) as f:out.append({"type":"image","image":f.convert("RGB")})
    question="\nQuestion: "+row["question"]+"\n"
    if row.get("options"):
        question+="Options:\n"+"\n".join(str(x) for x in row["options"])+"\n"
    question+="Respond with your final answer in the exact format Answer[<answer>]."
    put(question)
    return out

def parse_answer(s):
    hits=re.findall(r"Answer\s*\[\s*([^\]]+?)\s*\]",s,re.I|re.S)
    return hits[-1].strip() if hits else s.splitlines()[-1].strip()

def norm(s):
    s=str(s).lower().strip()
    return " ".join(re.sub(r"[^a-z0-9]+"," ",s).split())

def check_correct(output,reference):
    x=norm(parse_answer(output));y=norm(reference)
    if x==y:return True
    # For multi-choice, ground truth might be stored as "A" or "(A)".
    if len(y)==1 and y in "abcde":
        found=re.match(r"^\(?([a-e])\)?(?:\s|$)",x)
        return bool(found and found.group(1)==y)
    return False

def evaluate(args):
    manifest=load_jsonl(args.manifest)
    captions={x["path"]:x["caption"] for x in load_jsonl(args.captions)
              if x.get("caption")}
    done={(x["qid"],x["schedule"]) for x in load_jsonl(args.output) if x.get("status")=="ok"}
    todo=[(row,sch) for row in manifest for sch in SCHEDULES if (row["qid"],sch) not in done]
    print("Evaluation jobs pending:",len(todo),"/",len(manifest)*len(SCHEDULES),flush=True)
    if not todo:return
    missing=set()
    for row,sch in todo:
        for i,p in enumerate(row["caption_targets"]):
            if sch[i]=="T" and p not in captions:missing.add(p)
    if missing:raise RuntimeError(f"Missing {len(missing)} captions; run caption first")
    model,processor=load_model(args.model)
    for j,(row,sch) in enumerate(todo,1):
        try:
            content=build_content(row,sch,captions)
            nvision=sum(x["type"]=="image" for x in content)
            assert nvision == row["original_image_count"]-sch.count("T")
            start=time.time()
            response,tokens,trunc=generate(model,processor,content,args.max_new_tokens)
            record={"status":"ok","qid":row["qid"],"schedule":sch,
                    "original_image_count":row["original_image_count"],
                    "vision_count":nvision,"text_caption_count":sch.count("T"),
                    "answer":row["answer"],"prediction":parse_answer(response),
                    "correct":check_correct(response,row["answer"]),
                    "output":response,"output_tokens":tokens,"truncated":trunc,
                    "seconds":round(time.time()-start,2)}
            add_jsonl(args.output,record)
            print(f"[{j}/{len(todo)}] {row['qid']} {sch} {nvision}V correct={record['correct']}",flush=True)
        except Exception as exc:
            add_jsonl(args.errors,{"qid":row["qid"],"schedule":sch,"error":repr(exc)})
            print("EVAL ERROR",row["qid"],sch,repr(exc),flush=True)

def summary(args):
    import csv
    good={(x["qid"],x["schedule"]):x for x in load_jsonl(args.predictions)
          if x.get("status")=="ok"}
    manifest=load_jsonl(args.manifest)
    complete=[r for r in manifest if all((r["qid"],s) in good for s in SCHEDULES)]
    print("Fully paired questions",len(complete),"/",len(manifest))
    if not complete:return
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    metrics=[]
    for sch in SCHEDULES:
        xs=[good[(r["qid"],sch)] for r in complete]
        row={"schedule":sch,"paired_n":len(xs),"accuracy":sum(z["correct"] for z in xs)/len(xs),
             "mean_output_tokens":sum(z["output_tokens"] for z in xs)/len(xs),
             "total_truncated":sum(z["truncated"] for z in xs)}
        metrics.append(row);print(row)
    with open(out/"summary.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    with open(out/"flips.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["qid","original_image_count","oneV_schedule","allV_correct",
                                    "allT_correct","oneV_correct","oneV_minus_allT"])
        for r in complete:
            base=good[r["qid"],"VVVV"]["correct"]
            alt=good[r["qid"],"TTTT"]["correct"]
            for sch in SCHEDULES[2:]:
                one=good[r["qid"],sch]["correct"]
                w.writerow([r["qid"],r["original_image_count"],sch,base,alt,one,int(one)-int(alt)])
    size_counts=collections.Counter(r["original_image_count"] for r in complete)
    print("Complete paired samples by original image count:",dict(sorted(size_counts.items())))
    for n,ct in sorted(size_counts.items()):
        print(f"images={n} n={ct} "+
              " ".join(s+":"+str(round(sum(good[r["qid"],s]["correct"] for r in complete
                 if r["original_image_count"]==n)/ct,3)) for s in SCHEDULES))

def selftest(_):
    assert SCHEDULES==("VVVV","TTTT","VTTT","TVTT","TTVT","TTTV")
    for n in (4,5,6,8):
        for sch in SCHEDULES:
            assert n-sch.count("T") >=0
            assert sum(1 for i in range(n) if i>=4 or sch[i]=="V")==n-sch.count("T")
    assert check_correct("Reasoning...\nAnswer[B]","B")
    assert not check_correct("Answer[A]","B")
    print("Selftest passed: all 6 schedules and 4+ image conservation.")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("prepare");p.add_argument("--n",type=int,default=50)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--max-images",type=int,default=10,help="Limit total images to prevent OOM; 0 = no limit")
    p.add_argument("--asset-root",default="data")
    p.add_argument("--output-dir",default="data/pilot")
    p.set_defaults(func=select)
    p=commands.add_parser("caption");p.add_argument("--manifest",default="data/pilot/manifest.jsonl")
    p.add_argument("--output",default="results/captions.jsonl")
    p.add_argument("--model",default="llava-hf/llava-1.5-7b-hf")
    p.set_defaults(func=caption)
    p=commands.add_parser("evaluate");p.add_argument("--manifest",default="data/pilot/manifest.jsonl")
    p.add_argument("--captions",default="results/captions.jsonl")
    p.add_argument("--model",default="Qwen/Qwen3-VL-8B-Thinking")
    p.add_argument("--max-new-tokens",type=int,default=1536)
    p.add_argument("--output",default="results/predictions.jsonl")
    p.add_argument("--errors",default="results/errors.jsonl")
    p.set_defaults(func=evaluate)
    p=commands.add_parser("summarize")
    p.add_argument("--manifest",default="data/pilot/manifest.jsonl")
    p.add_argument("--predictions",default="results/predictions.jsonl")
    p.add_argument("--output-dir",default="results")
    p.set_defaults(func=summary)
    commands.add_parser("selftest").set_defaults(func=selftest)
    args=parser.parse_args();args.func(args)

if __name__=="__main__":main()
