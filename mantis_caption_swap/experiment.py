#!/usr/bin/env python3
"""Mantis-Eval paired variable-length image-to-caption swaps.

Sample 50 from the complete 217-item test split, regardless of image count (2–5).
For N images test all V, all T, and one-at-a-time V->T only.
Do not evaluate or summarize keep-one-V configurations.
"""
import argparse
import collections
import csv
import json
import random
import re
import string
import time
from pathlib import Path

DATASET="TIGER-Lab/Mantis-Eval"
SPLIT="test"
CAPTION_PROMPT=(
    "Describe this image in detail. Include information about the setting, objects, "
    "people, clothing, actions, facial expressions, colors, numbers, and any notable "
    "background elements. Be as specific and thorough as possible."
)
IMAGE_MARKER=re.compile(r"<image\s*/?>",re.IGNORECASE)


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path,"r",encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path,record):
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a",encoding="utf-8") as f:
        f.write(json.dumps(record,ensure_ascii=False)+"\n")


def require_manifest(path):
    """Fail loudly instead of reporting 0 pending work as a successful experiment."""
    p=Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {p}. Run: python -u experiment.py prepare --n 50 --seed 42")
    manifest=read_jsonl(p)
    if not manifest:
        raise RuntimeError(
            f"Manifest empty: {p}. Re-run prepare; no QA experiment has been performed.")
    if any(not x.get("image_paths") for x in manifest):
        raise RuntimeError("Manifest contains samples without images; inspect prepare output")
    return manifest


def schedules(n):
    """Return ordered dict: distinct V/T schedule -> all matching experimental labels."""
    if n<1:
        raise ValueError("Question must have at least one image")
    out={}
    def add(s,label):
        out.setdefault(s,[]).append(label)
    add("V"*n,"all_V")
    add("T"*n,"all_T")
    for i in range(n):
        add("V"*i+"T"+"V"*(n-i-1),f"T_at_{i+1}")
    return out


def plan_indices(size, target, seed, previous=None, extend=False):
    """Preserve the prior 50-question pilot exactly; append only unseen dataset rows."""
    if not 0 < target <= size:
        raise ValueError(f"Requested {target} examples; dataset contains {size}")
    old=previous or []
    if old and not extend:
        raise RuntimeError("Manifest exists. Use --extend to preserve earlier samples/results.")
    if extend and not old:
        raise RuntimeError("--extend requires an existing nonempty manifest")
    old_ids=[r["source_index"] for r in old]
    if len(old_ids)!=len(set(old_ids)):
        raise RuntimeError("Existing manifest contains duplicate dataset rows")
    if any(not isinstance(i,int) or i<0 or i>=size for i in old_ids):
        raise RuntimeError("Existing manifest source_index is invalid")
    if len(old_ids)>target:
        raise RuntimeError(f"Cannot shrink manifest from {len(old_ids)} to {target} questions")
    random_indices=list(range(size))
    random.Random(seed).shuffle(random_indices)
    existing_set=set(old_ids)
    extra=[i for i in random_indices if i not in existing_set][:target-len(old_ids)]
    return old_ids+extra


def prepare(args):
    import os
    from datasets import load_dataset
    ds=load_dataset(DATASET,split=SPLIT)
    target=len(ds) if args.all else args.n
    print(f"Loaded {len(ds)} official {DATASET} questions; columns={ds.column_names}",flush=True)
    out=Path(args.output_dir)
    imgs_dir=out/"images"
    imgs_dir.mkdir(parents=True,exist_ok=True)
    manifest_path=out/"manifest.jsonl"
    if manifest_path.exists():
        if not args.extend:
            raise RuntimeError(f"Manifest already exists at {manifest_path}. "
                               "Use --extend to preserve the existing 50 and their results.")
        previous=require_manifest(manifest_path)
    else:
        previous=[]
        if args.extend:
            raise RuntimeError("--extend requested, but manifest.jsonl does not exist")
    for old in previous:
        source_index=old["source_index"]
        record=ds[source_index]
        expected_qid=str(record.get("id") or f"test_{source_index}")
        if old["qid"]!=expected_qid or old["image_count"]!=len(record["images"]):
            raise RuntimeError(f"Original dataset changed for row {source_index}; refusing to reuse results")
    indices=plan_indices(len(ds),target,args.seed,previous,extend=args.extend)
    new_indices=indices[len(previous):]
    manifest=list(previous)
    image_counts=collections.Counter(x["image_count"] for x in previous)
    mismatches=sum(not x.get("image_placeholders_matched",True) for x in previous)
    print(f"Existing questions={len(previous)}, adding={len(new_indices)}, target={target}",flush=True)
    for j,index in enumerate(new_indices,1):
        r=ds[index]
        question=r["question"]
        pictures=r["images"]
        if not isinstance(question,str) or not isinstance(pictures,(list,tuple)) or not pictures:
            raise ValueError(f"Unsupported image/question schema at dataset row {index}")
        qid=str(r.get("id") or f"test_{index}")
        safe=re.sub(r"[^a-zA-Z0-9_-]","_",qid)
        image_paths=[]
        for i,picture in enumerate(pictures):
            path=imgs_dir/f"{safe}_{index}_{i+1}.png"
            if not path.exists():
                if isinstance(picture,dict) and picture.get("bytes") is not None:
                    import io
                    from PIL import Image
                    picture=Image.open(io.BytesIO(picture["bytes"]))
                elif isinstance(picture,dict) and picture.get("path"):
                    from PIL import Image
                    picture=Image.open(picture["path"])
                if not hasattr(picture,"convert"):
                    raise TypeError(f"Unable to decode image {i+1} in {qid}: {type(picture)}")
                picture.convert("RGB").save(path,format="PNG")
            image_paths.append(str(path.resolve()))
        marker_count=len(IMAGE_MARKER.findall(question))
        matched=marker_count==len(image_paths)
        if not matched:
            mismatches+=1
            print(f"WARNING {qid}: placeholders={marker_count} images={len(image_paths)}; "
                  "using ordered prefix fallback",flush=True)
        options=r.get("options") or []
        if not isinstance(options,(list,tuple)):
            raise TypeError(f"Unsupported options for {qid}: {type(options)}")
        manifest.append({
            "qid":qid,"source_index":index,"question_type":r.get("question_type"),
            "question":question,"options":[str(v) for v in options],
            "answer":str(r.get("answer","")),
            "category":r.get("category"),"data_source":r.get("data_source"),
            "image_paths":image_paths,"image_count":len(image_paths),
            "image_placeholders_matched":matched,
        })
        image_counts[len(image_paths)]+=1
        print(f"[PREPARE {j}/{len(new_indices)}] {qid} images={len(image_paths)} "
              f"conditions={len(schedules(len(image_paths)))}",flush=True)
    ids=[r["qid"] for r in manifest]
    if len(set(ids))!=len(ids):
        raise RuntimeError("Duplicate qids detected; manifest left untouched")
    if len(manifest)!=target:
        raise RuntimeError(f"Expected {target}, got {len(manifest)}; manifest left untouched")
    # Atomic replace avoids partial writes if interrupted.
    tmp=out/"manifest.jsonl.tmp"
    with tmp.open("w",encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
    os.replace(tmp,manifest_path)
    total_schedules=sum(len(schedules(r["image_count"])) for r in manifest)
    print("SELECTED:",len(manifest),"image-count distribution:",dict(sorted(image_counts.items())),flush=True)
    print("QA generations planned:",total_schedules,
          "previous results remain cached; placeholder mismatches:",mismatches,flush=True)
    print("Manifest:",manifest_path,flush=True)



def load_model(model_name,captioner=False):
    import torch
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(model_name)
    if captioner:
        from transformers import LlavaForConditionalGeneration
        model_cls=LlavaForConditionalGeneration
    else:
        from transformers import Qwen3VLForConditionalGeneration
        model_cls=Qwen3VLForConditionalGeneration
    model=model_cls.from_pretrained(model_name,device_map="auto",dtype="auto")
    model.eval()
    return model,processor


def generate(model,processor,content,max_tokens):
    import torch
    batch=processor.apply_chat_template(
        [{"role":"user","content":content}],
        tokenize=True,return_dict=True,return_tensors="pt",add_generation_prompt=True)
    batch=batch.to(next(model.parameters()).device)
    prefix=batch["input_ids"].shape[-1]
    with torch.inference_mode():
        output=model.generate(**batch,max_new_tokens=max_tokens,do_sample=False)
    tail=output[0,prefix:]
    decoded=processor.decode(tail,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    return decoded.strip(),int(tail.shape[-1]),int(tail.shape[-1])>=max_tokens


def caption(args):
    from PIL import Image
    selected=require_manifest(args.manifest)
    existing={x["path"]:x["caption"] for x in read_jsonl(args.output)
              if x.get("path") and x.get("caption")}
    images=list(dict.fromkeys(p for r in selected for p in r["image_paths"]))
    pending=[p for p in images if p not in existing]
    print(f"Caption targets={len(images)}, cached={len(images)-len(pending)}, pending={len(pending)}",flush=True)
    if not pending:
        return
    model,processor=load_model(args.model,captioner=True)
    failed=[]
    for idx,p in enumerate(pending,1):
        try:
            with Image.open(p) as f: img=f.convert("RGB")
            content=[{"type":"image","image":img},{"type":"text","text":CAPTION_PROMPT}]
            response,token_count,truncated=generate(model,processor,content,200)
            if not response:raise ValueError("Empty caption")
            write_jsonl(args.output,{"path":p,"caption":response,
                                    "tokens":token_count,"truncated":truncated,
                                    "model":args.model,"prompt":CAPTION_PROMPT})
            print(f"[CAPTION {idx}/{len(pending)}] {Path(p).name} tokens={token_count}",flush=True)
        except Exception as ex:
            failed.append((p,str(ex)))
            print("CAPTION ERROR",p,repr(ex),flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} caption(s) failed. Rerun 'caption' to resume.")


def build_content(q,sch,captions):
    """Keep question/image interleaving. Image slot i always refers to original image i."""
    from PIL import Image
    paths=q["image_paths"]
    if len(sch)!=len(paths) or any(c not in "VT" for c in sch):
        raise ValueError("Schedule mismatches images")
    content=[]
    def put(text):
        if text:content.append({"type":"text","text":text})
    def put_slot(index):
        put(f"\n[Image {index+1}] ")
        path=paths[index]
        if sch[index]=="T":
            if path not in captions:raise ValueError("Missing caption for "+path)
            put("Image description: "+captions[path]+"\n")
        else:
            with Image.open(path) as inp:
                content.append({"type":"image","image":inp.convert("RGB")})
            put("\n")
    put("Answer using all evidence, including any image descriptions. "
        "Images and descriptions are indexed in original order.\n")
    pieces=IMAGE_MARKER.split(q["question"])
    count=len(pieces)-1
    if count==len(paths):
        for i,path in enumerate(paths):
            put(pieces[i])
            put_slot(i)
        put(pieces[-1])
    else:
        # Preserve original slot sequence. Explicitly record mismatch in manifest.
        for i in range(len(paths)):put_slot(i)
        put("\nQuestion: "+IMAGE_MARKER.sub("",q["question"]))
    if q["options"]:
        put("\nOptions:\n"+"\n".join(q["options"]))
    put("\nProvide a concise final answer formatted exactly as Answer[<your answer>].")
    return content


def answer_from_output(text):
    # Last explicit final answer has priority over text in preceding reasoning.
    m=re.findall(r"Answer\s*\[\s*([^\]]+?)\s*\]",text,flags=re.IGNORECASE|re.DOTALL)
    if m:return m[-1].strip()
    m=re.findall(r"(?:final answer|answer)\s*:\s*(.+)",text,flags=re.I)
    if m:return m[-1].splitlines()[0].strip()
    return text.splitlines()[-1].strip() if text.strip() else ""


def normalized(v):
    s=str(v).casefold().strip()
    s=re.sub(r"^[\(\[]?([a-e])[\)\]\.:]\s*$",r"\1",s)
    s=re.sub(r"[^a-z0-9\s]"," ",s)
    return " ".join(s.split())


def grade(answer,reference,options):
    guess=normalized(answer)
    expected=normalized(reference)
    if not guess:return False
    if guess==expected:return True
    # Map a model's full option text to the correct option letter for MCQ.
    if expected in "abcde" and len(expected)==1 and options:
        for i,opt in enumerate(options):
            if normalized(opt)==guess and chr(ord("a")+i)==expected:
                return True
        prefix=re.match(r"^([a-e])(?:\s|$)",guess)
        if prefix and prefix.group(1)==expected:return True
    return False


def evaluate(args):
    manifest=require_manifest(args.manifest)
    captions={x["path"]:x["caption"] for x in read_jsonl(args.captions)
              if x.get("path") and x.get("caption")}
    previous={}
    for record in read_jsonl(args.output):
        if record.get("status")=="ok":
            previous[record["qid"],record["schedule"]]=record
    def needs_eval(qid,sch):
        old=previous.get((qid,sch))
        if old is None:return True
        if not old.get("truncated"):return False
        old_cap=int(old.get("max_new_tokens") or old.get("tokens") or 0)
        return old_cap<args.max_new_tokens
    todo=[(q,s,aliases) for q in manifest for s,aliases in schedules(q["image_count"]).items()
          if needs_eval(q["qid"],s)]
    retries=sum(1 for q,s,_ in todo if (q["qid"],s) in previous)
    print(f"Pending inference jobs={len(todo)} (retry truncated={retries}; cap={args.max_new_tokens})",flush=True)
    if not todo:return
    missing={path for q,s,_ in todo for i,path in enumerate(q["image_paths"])
             if s[i]=="T" and path not in captions}
    if missing:raise RuntimeError(f"Missing {len(missing)} captions; run caption first")
    model,processor=load_model(args.model,captioner=False)
    failures=[]
    for count,(q,s,aliases) in enumerate(todo,1):
        try:
            content=build_content(q,s,captions)
            actual=sum(item["type"]=="image" for item in content)
            assert actual==s.count("V"),(actual,s)
            start=time.monotonic()
            output,tokens,truncated=generate(model,processor,content,args.max_new_tokens)
            prediction=answer_from_output(output)
            item={"status":"ok","qid":q["qid"],"schedule":s,"aliases":aliases,
                  "image_count":q["image_count"],"vision_count":s.count("V"),
                  "caption_count":s.count("T"),"question_type":q["question_type"],
                  "category":q["category"],"reference":q["answer"],
                  "prediction":prediction,"correct":grade(prediction,q["answer"],q["options"]),
                  "raw_output":output,"tokens":tokens,"truncated":truncated,
                  "max_new_tokens":args.max_new_tokens,
                  "seconds":round(time.monotonic()-start,2)}
            write_jsonl(args.output,item)
            print(f"[QA {count}/{len(todo)}] {q['qid']} {s} "
                  f"correct={item['correct']} truncated={truncated}",flush=True)
        except Exception as ex:
            failures.append((q["qid"],s,str(ex)))
            write_jsonl(args.errors,{"qid":q["qid"],"schedule":s,"error":repr(ex)})
            print("QA ERROR",q["qid"],s,repr(ex),flush=True)
    if failures:raise RuntimeError(f"{len(failures)} inference job(s) failed; rerun evaluate to resume.")


def summarize(args):
    manifest=require_manifest(args.manifest)
    results={(r["qid"],r["schedule"]):r for r in read_jsonl(args.predictions)
             if r.get("status")=="ok"}
    paired=[q for q in manifest if all((q["qid"],s) in results for s in schedules(q["image_count"]))]
    print("Fully paired questions:",len(paired),"/",len(manifest),flush=True)
    if not paired:return
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    sizes=sorted({q["image_count"] for q in paired})
    for image_count in ["all"]+sizes:
        selected=[q for q in paired if image_count=="all" or q["image_count"]==image_count]
        groups=collections.defaultdict(list)
        for q in selected:
            for schedule,aliases in schedules(q["image_count"]).items():
                for alias in aliases:
                    groups[alias].append(results[q["qid"],schedule])
        ordered=["all_V","all_T"]+[f"T_at_{i}" for i in range(1,6)]
        for alias in ordered:
            rs=groups.get(alias)
            if not rs:continue
            row={"image_count":image_count,"condition":alias,"n":len(rs),
                 "accuracy":round(sum(int(r["correct"]) for r in rs)/len(rs),5),
                 "truncated":sum(int(r["truncated"]) for r in rs),
                 "mean_output_tokens":round(sum(r["tokens"] for r in rs)/len(rs),2)}
            rows.append(row);print(row,flush=True)
    with (out/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    flips=[]
    for q in paired:
        r=results
        for s,aliases in schedules(q["image_count"]).items():
            if s in ("V"*q["image_count"],"T"*q["image_count"]):continue
            current=r[q["qid"],s]
            flips.append({"qid":q["qid"],"image_count":q["image_count"],"schedule":s,
                          "aliases":";".join(aliases),
                          "all_V_correct":r[q["qid"],"V"*q["image_count"]]["correct"],
                          "all_T_correct":r[q["qid"],"T"*q["image_count"]]["correct"],
                          "condition_correct":current["correct"]})
    if flips:
        with (out/"flips.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(flips[0]))
            writer.writeheader();writer.writerows(flips)


def selftest(_):
    previous=[{"source_index":2},{"source_index":7}]
    assert plan_indices(10,10,42,previous,extend=True)[:2]==[2,7]
    assert len(set(plan_indices(10,10,42,previous,extend=True)))==10
    try:
        plan_indices(10,10,42,previous,extend=False)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Must not overwrite pilot when --extend is missing")
    for n in (2,3,4,5):
        d=schedules(n)
        assert len(d)==n+2
        assert len(set(d))==len(d)
        assert all(len(s)==n for s in d)
        assert len(d["V"*n])==1
        assert len(d["T"*n])==1
        for i in range(n):
            assert "T_at_"+str(i+1) in d["V"*i+"T"+"V"*(n-i-1)]
    assert not any(label.startswith("V_at_") for aliases in d.values() for label in aliases)
    assert grade("Answer[C]","C",["(A) One","(B) Two","(C) Three"])
    assert not grade("Answer[A]","C",["(A) One","(B) Two","(C) Three"])
    assert grade("Answer[Yes]","yes",[])
    print("SELFTEST PASSED: all image lengths, unique schedules and answer parsing",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("prepare")
    p.add_argument("--n",type=int,default=50);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--all",action="store_true",help="Select the entire dataset (217 questions)")
    p.add_argument("--extend",action="store_true",help="Append missing questions, preserving existing pilot")
    p.add_argument("--output-dir",default="data/pilot")
    p.set_defaults(func=prepare)
    p=commands.add_parser("caption")
    p.add_argument("--manifest",default="data/pilot/manifest.jsonl")
    p.add_argument("--output",default="results/captions.jsonl")
    p.add_argument("--model",default="llava-hf/llava-1.5-7b-hf")
    p.set_defaults(func=caption)
    p=commands.add_parser("evaluate")
    p.add_argument("--manifest",default="data/pilot/manifest.jsonl")
    p.add_argument("--captions",default="results/captions.jsonl")
    p.add_argument("--output",default="results/predictions.jsonl")
    p.add_argument("--errors",default="results/errors.jsonl")
    p.add_argument("--model",default="Qwen/Qwen3-VL-8B-Thinking")
    p.add_argument("--max-new-tokens",type=int,default=8192)
    p.set_defaults(func=evaluate)
    p=commands.add_parser("summarize")
    p.add_argument("--manifest",default="data/pilot/manifest.jsonl")
    p.add_argument("--predictions",default="results/predictions.jsonl")
    p.add_argument("--output-dir",default="results")
    p.set_defaults(func=summarize)
    commands.add_parser("selftest").set_defaults(func=selftest)
    args=parser.parse_args()
    args.func(args)


if __name__=="__main__":
    main()
