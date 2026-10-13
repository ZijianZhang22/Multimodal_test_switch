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

def open_official_archive(args):
    """Use the official ZIP, NOT datasets.load_dataset's imagefolder auto-conversion."""
    import zipfile
    archive=Path(args.archive)
    if not archive.is_file():
        print("Downloading official MMRB 6GB archive; cached across runs:",archive,flush=True)
        from huggingface_hub import hf_hub_download
        archive=Path(hf_hub_download(repo_id=DATASET_ID,repo_type="dataset",
            filename="data_MMRB_20250531.zip",local_dir=str(archive.parent)))
    if not zipfile.is_zipfile(archive):
        raise RuntimeError("Not a valid official MMRB ZIP: "+str(archive))
    return zipfile.ZipFile(archive)


def find_annotation(z):
    """Find the actual task JSON, rather than treating each image as a QA row."""
    names=[n for n in z.namelist() if not n.endswith("/")]
    choices=[n for n in names if n.lower().endswith("mmrb_data.json")]
    if not choices:
        choices=[n for n in names if n.lower().endswith(".json")
                 and any(t in n.lower() for t in ("annotation","mmrb","data"))]
    for candidate in sorted(choices,key=lambda n:(not n.lower().endswith("mmrb_data.json"),len(n))):
        print("Inspecting annotation:",candidate,flush=True)
        try:
            with z.open(candidate) as f: raw=json.load(f)
            rows=raw if isinstance(raw,list) else (raw.get("data") or raw.get("questions") or raw.get("annotations"))
            if isinstance(rows,list) and rows and isinstance(rows[0],dict):
                if "question" in rows[0] and "image_paths" in rows[0]:
                    return candidate,rows
        except (ValueError,KeyError,UnicodeDecodeError,TypeError) as exc:
            print("Not task annotations:",str(exc)[:150],flush=True)
    print("Archive file examples:",names[:15],flush=True)
    print("JSON candidates:",choices[:30],flush=True)
    raise RuntimeError("Couldn't locate a list of MMRB question records with question and image_paths")


def index_images(z):
    """Match original paths exactly by suffix; fall back to basename only if unique."""
    from collections import defaultdict
    by_basename=defaultdict(list)
    names=set()
    for n in z.namelist():
        if n.endswith("/"):continue
        names.add(n)
        by_basename[Path(n).name].append(n)
    return names,by_basename


def locate_image(path,archive_names,by_basename):
    p=str(path).replace("\\","/").lstrip("./")
    if not p or p.startswith("/") or ".." in Path(p).parts:
        raise ValueError("Unsafe/empty image path: "+repr(path))
    if p in archive_names: return p
    found=[x for x in by_basename[Path(p).name] if x.endswith("/"+p)]
    if len(found)==1:return found[0]
    if len(found)>1: raise ValueError("Multiple matching archive paths for "+p)
    fallback=by_basename[Path(p).name]
    if len(fallback)==1:return fallback[0]
    raise FileNotFoundError("Image path unavailable or ambiguous: "+p)


def select(args):
    from PIL import Image
    from collections import Counter
    with open_official_archive(args) as z:
        ann,rows=find_annotation(z)
        image_count=Counter()
        for r in rows: image_count[len(r.get("image_paths") or [])]+=1
        print("Official task annotations:",len(rows),"entry:",ann,flush=True)
        print("Image-count distribution:",dict(sorted(image_count.items())),flush=True)
        qualifying=sum(n for k,n in image_count.items() if k>=4 and (not args.max_images or k<=args.max_images))
        print("Potential >=4 images within max limit:",qualifying,flush=True)
        if qualifying<args.n:
            raise RuntimeError(f"Only {qualifying} candidates, need {args.n}. Increase --max-images.")
        names,by_basename=index_images(z)
        ids=list(range(len(rows)));random.Random(args.seed).shuffle(ids)
        out=Path(args.output_dir)
        out.mkdir(parents=True,exist_ok=True)
        (out/"images").mkdir(exist_ok=True)
        selected=[];reasons=Counter();unique_ids=set()
        for idx in ids:
            r=rows[idx]
            paths=r.get("image_paths") or []
            if len(paths)<4 or (args.max_images and len(paths)>args.max_images):continue
            question=r.get("question")
            answer=r.get("answer")
            if not isinstance(question,str) or not question.strip() or answer is None:
                reasons["missing_question_or_answer"]+=1;continue
            opts=r.get("options") or []
            if isinstance(opts,str):
                try: opts=json.loads(opts)
                except json.JSONDecodeError:opts=[opts]
            if isinstance(opts,dict):opts=[f"{k}. {v}" for k,v in opts.items()]
            if not isinstance(opts,list) or any(not isinstance(v,(str,int,float)) for v in opts):
                reasons["unsupported_options"]+=1;continue
            rid=str(r.get("subtask","mmrb"))+"_"+str(r.get("index",idx))
            rid=re.sub(r"[^A-Za-z0-9_-]","_",rid)
            if rid in unique_ids:rid+=f"_row{idx}"
            try:
                locations=[locate_image(p,names,by_basename) for p in paths]
            except (ValueError,FileNotFoundError) as exc:
                reasons["image_not_found"]+=1
                print("SKIP",rid,str(exc)[:150],flush=True);continue
            output_paths=[]
            try:
                for j,zname in enumerate(locations):
                    dest=out/"images"/f"{rid}_{j+1}.png"
                    if not dest.exists():
                        with z.open(zname) as f:
                            img=Image.open(f).convert("RGB")
                            img.save(dest,format="PNG")
                    output_paths.append(str(dest.resolve()))
            except Exception as exc:
                reasons["image_decode_failed"]+=1
                print("SKIP",rid,repr(exc)[:180],flush=True)
                for p in output_paths:Path(p).unlink(missing_ok=True)
                continue
            unique_ids.add(rid)
            selected.append({
                "qid":rid,"dataset_index":idx,"question":question,
                "options":[str(o) for o in opts],"answer":str(answer),
                "question_type":str(r.get("question_type","")),
                "subtask":str(r.get("subtask","")),
                "original_image_count":len(output_paths),
                "image_paths":output_paths,
                "caption_targets":output_paths[:4],
                "original_archive_image_paths":paths,
                "reasoning_steps_available":bool(r.get("reasoning_steps"))})
            if len(selected)==args.n:break
    print("Rejected reasons:",dict(reasons),flush=True)
    print("Selected:",len(selected),"requested:",args.n,flush=True)
    if len(selected)<args.n:raise RuntimeError(f"Only {len(selected)} valid MMRB examples, need {args.n}")
    manifest=out/"manifest.jsonl"
    with manifest.open("w",encoding="utf-8") as f:
        for row in selected:f.write(json.dumps(row,ensure_ascii=False)+"\n")
    print("Manifest:",manifest,flush=True)


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
    p.add_argument("--archive",default="data/data_MMRB_20250531.zip")
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
