#!/usr/bin/env python3
"""MMQA four-candidate-image, six-condition paired ablation.

Exactly 50 dev questions: VVVV TTTT VTTT TVTT TTVT TTTV.
V = original image; T = question-blind caption from LLaVA-1.5-7B.
Image slot order is NOT the causal logical-hop order.
"""
import argparse
import collections
import csv
import gzip
import json
import random
import re
import string
import time
import zipfile
from pathlib import Path

CAPTION_PROMPT = ("Describe this image in detail. Include information about the setting, objects, "
                  "people, clothing, actions, facial expressions, colors, numbers, and any notable "
                  "background elements. Be as specific and thorough as possible.")
SCHEDULES = ("VVVV", "TTTT", "VTTT", "TVTT", "TTVT", "TTTV")

def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for ln in f:
            if ln.strip(): yield json.loads(ln)

def append(path, row):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def table_to_text(record):
    table = record.get("table", {})
    heads = [c.get("column_name", "") for c in table.get("header", [])]
    rows = [[str(c.get("text","") if isinstance(c,dict) else c).replace("\n"," ").replace("|","/")
             for c in row] for row in table.get("table_rows", [])]
    return "Table title: " + record.get("title","") + "\n" + "\n".join(
        [" | ".join(heads)] + [" | ".join(row) for row in rows])

def candidates(dataset, limit, max_chars):
    data=Path(dataset)
    qfile=data/"MMQA_dev.jsonl.gz"
    for fname in ("MMQA_dev.jsonl.gz","MMQA_images.jsonl.gz","MMQA_texts.jsonl.gz","MMQA_tables.jsonl.gz"):
        if not (data/fname).exists(): raise FileNotFoundError(data/fname)
    image={v["id"]:v for v in read_jsonl(data/"MMQA_images.jsonl.gz")}
    texts={v["id"]:v for v in read_jsonl(data/"MMQA_texts.jsonl.gz")}
    tables={v["id"]:v for v in read_jsonl(data/"MMQA_tables.jsonl.gz")}
    eligible=[]
    image_number=collections.Counter()
    supporting_number=collections.Counter()
    for q in read_jsonl(qfile):
        m=q.get("metadata") or {}
        ids=m.get("image_doc_ids") or []
        image_number[len(ids)]+=1
        support=q.get("supporting_context") or []
        support_images=[c["doc_id"] for c in support if c.get("doc_part")=="image" and c["doc_id"] in ids]
        support_modes={c.get("doc_part") for c in support}
        if len(ids)!=4 or len(set(ids))!=4 or any(i not in image for i in ids): continue
        if len(support)<2 or len(support_modes)<2 or not support_images: continue
        if "image" not in (m.get("modalities") or []): continue
        supporting_number[len(support_images)]+=1
        tdocs=[texts[i] for i in m.get("text_doc_ids",[]) if i in texts]
        tab=tables.get(m.get("table_id"))
        tabtext=table_to_text(tab) if tab else ""
        if len(tabtext)+sum(len(x.get("text","")) for x in tdocs)>max_chars: continue
        answers=[x if isinstance(x,str) else str(x.get("answer","")) for x in q.get("answers",[])]
        eligible.append({
            "qid":q["qid"], "question":q["question"],"answers":list(dict.fromkeys(answers)),
            "type":m.get("type",""),"support_image_ids":support_images,
            "supporting_context":support, "table":tabtext,
            "texts":[{"id":x["id"],"title":x.get("title",""),"text":x.get("text","")} for x in tdocs],
            "images":[{"id":i,"path":image[i]["path"],"title":image[i].get("title",""),
                       "support":i in support_images} for i in ids]})
    print("Dev candidate-image count distribution:",dict(sorted(image_number.items())))
    print("Eligible 4-image, multi-hop, image-supported questions:",len(eligible))
    print("Supporting images among eligible:",dict(sorted(supporting_number.items())))
    if len(eligible)<limit: raise ValueError(f"Need {limit} eligible examples, found {len(eligible)}")
    return eligible

def prepare(args):
    cand=candidates(args.dataset_dir,args.n,args.max_context_chars)
    zip_path=Path(args.images_zip)
    if not zip_path.exists(): raise FileNotFoundError(zip_path)
    rng=random.Random(args.seed)
    rng.shuffle(cand)
    outdir=Path(args.output_dir)
    outdir.mkdir(parents=True,exist_ok=True)
    chosen=[]
    with zipfile.ZipFile(zip_path) as z:
        # Account for a leading folder in the official image archive.
        lookup=collections.defaultdict(list)
        for name in z.namelist():
            if not name.endswith("/"): lookup[Path(name).name].append(name)
        resolved={}
        for q in cand:
            names=[]
            for img in q["images"]:
                path=img["path"].replace("\\","/").lstrip("./")
                found=[name for name in lookup[Path(path).name]
                       if name==path or name.endswith("/"+path)]
                if len(found)!=1: break
                names.append(found[0])
            if len(names)!=4: continue
            for img,zname in zip(q["images"],names):
                suffix=Path(zname).suffix.lower() or ".jpg"
                target=outdir/"images"/(img["id"]+suffix)
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    with z.open(zname) as src, target.open("wb") as dst:
                        import shutil
                        shutil.copyfileobj(src,dst)
                img["local_path"]=str(target.resolve())
                resolved[img["id"]]=str(target)
            chosen.append(q)
            if len(chosen)==args.n: break
    if len(chosen)!=args.n:
        raise ValueError(f"Found only {len(chosen)} fully located 4-image examples, need exactly {args.n}")
    manifest=outdir/"manifest.jsonl"
    with manifest.open("w",encoding="utf-8") as f:
        for q in chosen: f.write(json.dumps(q,ensure_ascii=False)+"\n")
    print("Selected EXACTLY",len(chosen),"questions; unique image files:",len(resolved))
    print("Manifest:",manifest)

def conditions(q):
    imgs=q["images"]
    ids=[v["id"] for v in imgs]
    if len(ids)!=4 or len(set(ids))!=4: raise ValueError("Require four distinct image IDs")
    for sched in SCHEDULES:
        replaced={ids[i] for i in range(4) if sched[i]=="T"}
        keep=next((ids[i] for i in range(4) if sched[i]=="V"),None) if sched.count("V")==1 else None
        yield sched,replaced,keep

def model_load(name,dtype):
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(name)
    if "llava" in name.lower():
        from transformers import LlavaForConditionalGeneration
        cls=LlavaForConditionalGeneration
    else:
        from transformers import Qwen3VLForConditionalGeneration
        cls=Qwen3VLForConditionalGeneration
    model=cls.from_pretrained(name,device_map="auto",dtype=dtype)
    model.eval()
    return model,processor

def generate(model,processor,content,max_tokens):
    import torch
    messages=[{"role":"user","content":content}]
    batch=processor.apply_chat_template(messages,add_generation_prompt=True,
                                        tokenize=True,return_dict=True,return_tensors="pt")
    batch=batch.to(model.device)
    offset=batch["input_ids"].shape[-1]
    with torch.inference_mode():
        result=model.generate(**batch,do_sample=False,max_new_tokens=max_tokens)
    ids=result[0][offset:]
    text=processor.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    return text.strip(),len(ids),len(ids)>=max_tokens

def caption(args):
    import torch
    from PIL import Image
    questions=list(read_jsonl(args.manifest))
    path=Path(args.output)
    existing={r["image_id"] for r in read_jsonl(path)} if path.exists() else set()
    targets={}
    for q in questions:
        for img in q["images"]:
            if img["id"] not in existing: targets[img["id"]]=img["local_path"]
    if not targets:
        print("All captions already cached");return
    model,processor=model_load(args.model,torch.float16)
    for index,(iid,imgpath) in enumerate(targets.items(),1):
        try:
            with Image.open(imgpath) as inp: picture=inp.convert("RGB")
            content=[{"type":"image","image":picture},{"type":"text","text":CAPTION_PROMPT}]
            out,ntok,truncated=generate(model,processor,content,200)
            text=re.sub(r"<\|.*?\|>","",out).strip()
            if not text: raise ValueError("Empty caption")
            append(path,{"image_id":iid,"caption":text,"generated_tokens":ntok,
                         "truncated":truncated,"model":args.model,
                         "prompt":CAPTION_PROMPT,"do_sample":False,"max_new_tokens":200})
            print(f"Caption {index}/{len(targets)} {iid[:8]} tokens={ntok} truncated={truncated}")
        except Exception as exc:
            print("Caption failed",iid,repr(exc))
    print("Caption output:",path)

def build_input(q,replaced,captions):
    from PIL import Image
    result=[]
    def put(s): result.append({"type":"text","text":s})
    put("Answer the question using the following independent evidence items. "
        "Give a short final answer.\n\n")
    if q["table"]: put("[TABLE EVIDENCE]\n"+q["table"]+"\n\n")
    for i,t in enumerate(q["texts"],1):
        put(f"[TEXT EVIDENCE {i}: {t['title']}]\n{t['text']}\n\n")
    for i,img in enumerate(q["images"],1):
        put(f"[IMAGE EVIDENCE {i}: {img['title']}]\n")
        if img["id"] in replaced:
            put("Image description: "+captions[img["id"]]+"\n\n")
        else:
            put("Original image:\n")
            result.append({"type":"image","image":Image.open(img["local_path"]).convert("RGB")})
            put("\n\n")
    put("Question: "+q["question"]+"\nFinal answer:")
    return result

def normalize(s):
    s=str(s).lower().strip()
    s="".join(c if c not in string.punctuation else " " for c in s)
    return " ".join(re.sub(r"\b(a|an|the)\b"," ",s).split())

def final_answer(text):
    text=re.sub(r"<think>.*?</think>","",text,flags=re.I|re.S)
    if "</think>" in text: text=text.rsplit("</think>",1)[-1]
    text=re.split("final answer:|answer:",text,flags=re.I)[-1]
    text=re.sub(r"<\|(?:im_end|endoftext|eot_id)\|>","",text).strip()
    return text.splitlines()[0].strip() if text else ""

def evaluate(args):
    import torch
    manifest=list(read_jsonl(args.manifest))
    captions={x["image_id"]:x["caption"] for x in read_jsonl(args.captions)}
    predpath=Path(args.output)
    done={(x["qid"],x["schedule"]) for x in read_jsonl(predpath)} if predpath.exists() else set()
    todo=[]
    for q in manifest:
        for label,replaced,kept in conditions(q):
            if (q["qid"],label) in done: continue
            if not replaced.issubset(captions):
                raise ValueError(f"Missing caption(s) for {q['qid']} / {label}: {replaced-set(captions)}")
            todo.append((q,label,replaced,kept))
    print("Pending generations:",len(todo)," expected total:",len(manifest)*6)
    if not todo: return
    model,processor=model_load(args.model,torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    for idx,(q,label,replaced,kept) in enumerate(todo,1):
        content=None
        try:
            tic=time.time()
            content=build_input(q,replaced,captions)
            nimg=sum(x["type"]=="image" for x in content)
            assert nimg==label.count("V"),(label,nimg)
            answer,n,trunc=generate(model,processor,content,args.max_new_tokens)
            pred=final_answer(answer)
            correct=bool(normalize(pred)) and any(normalize(pred)==normalize(x) for x in q["answers"])
            pos=next((i+1 for i,img in enumerate(q["images"]) if img["id"]==kept),None)
            append(predpath,{"qid":q["qid"],"schedule":label,"question":q["question"],
                             "gold":q["answers"],"prediction":pred,"raw_response":answer,
                             "correct_em":correct,"generated_tokens":n,"truncated":trunc,
                             "kept_visual_id":kept,"visual_input_position":pos,
                             "visual_is_support":kept in q["support_image_ids"] if kept else None,
                             "support_image_ids":q["support_image_ids"],"question_type":q["type"],
                             "seconds":round(time.time()-tic,2),"model":args.model})
            print(f"{idx}/{len(todo)} {label} EM={int(correct)} truncated={trunc} qid={q['qid'][:8]}")
        except Exception as exc:
            append(args.errors,{"qid":q["qid"],"schedule":label,"error":repr(exc)})
            print("ERROR",label,q["qid"],repr(exc))
        finally:
            if content:
                for x in content:
                    if x["type"]=="image":x["image"].close()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

def summarize(args):
    rows=list(read_jsonl(args.predictions))
    byq=collections.defaultdict(dict)
    for r in rows: byq[r["qid"]][r["schedule"]]=r
    matched={qid:c for qid,c in byq.items()
             if all(label in c and not c[label]["truncated"] for label in SCHEDULES)}
    print("Completed non-truncated six-condition matches:",len(matched))
    if not matched: return
    report=[]
    for label in SCHEDULES:
        correct=sum(bool(c[label]["correct_em"]) for c in matched.values())
        rate=correct/len(matched)
        report.append({"schedule":label,"n":len(matched),"correct":correct,"accuracy_em":rate})
        print(f"{label}: {correct}/{len(matched)} = {rate:.3f}")
    pairs=[]
    for label in SCHEDULES[2:]:
        delta_t=[];delta_v=[];support=0
        for qid,c in matched.items():
            cur=c[label]; vv=c["VVVV"];tt=c["TTTT"]
            dt=int(cur["correct_em"])-int(tt["correct_em"])
            dv=int(cur["correct_em"])-int(vv["correct_em"])
            delta_t.append(dt);delta_v.append(dv)
            support+=bool(cur["visual_is_support"])
            if dt or dv:
                pairs.append({"qid":qid,"schedule":label,
                              "visual_input_position":cur["visual_input_position"],
                              "visual_is_support":cur["visual_is_support"],
                              "delta_vs_TTTT":dt,"delta_vs_VVVV":dv,
                              "question":cur["question"],"gold":cur["gold"],
                              "VVVV":vv["prediction"],"TTTT":tt["prediction"],
                              "oneV":cur["prediction"]})
        print(f"{label}: delta vs TTTT={sum(delta_t)/len(delta_t):+.3f}; "
              f"vs VVVV={sum(delta_v)/len(delta_v):+.3f}; "
              f"kept supporting={support}/{len(matched)}")
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    with (out/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["schedule","n","correct","accuracy_em"])
        w.writeheader();w.writerows(report)
    with (out/"flips.jsonl").open("w",encoding="utf-8") as f:
        for x in pairs:f.write(json.dumps(x,ensure_ascii=False)+"\n")
    print("CAUTION: image position is a PHYSICAL slot, not a logical hop; candidates may be distractors.")

def selftest(_):
    q={"qid":"test","images":[{"id":str(i)} for i in range(4)]}
    c=list(conditions(q))
    assert [x[0] for x in c]==list(SCHEDULES)
    for label,repl,kept in c:
        assert len(repl)==label.count("T")
        assert len(set(x["id"] for x in q["images"])-repl)==label.count("V")
    print("Six exact four-image schedules verified.")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest="command",required=True)
    t=sp.add_parser("prepare")
    t.add_argument("--dataset-dir",default="data/multimodalqa/dataset")
    t.add_argument("--images-zip",default="data/images.zip")
    t.add_argument("--output-dir",default="data/pilot")
    t.add_argument("--n",type=int,default=50)
    t.add_argument("--seed",type=int,default=42)
    t.add_argument("--max-context-chars",type=int,default=14000)
    t.set_defaults(func=prepare)
    t=sp.add_parser("caption")
    t.add_argument("--manifest",default="data/pilot/manifest.jsonl")
    t.add_argument("--output",default="results/captions.jsonl")
    t.add_argument("--model",default="llava-hf/llava-1.5-7b-hf")
    t.set_defaults(func=caption)
    t=sp.add_parser("evaluate")
    t.add_argument("--manifest",default="data/pilot/manifest.jsonl")
    t.add_argument("--captions",default="results/captions.jsonl")
    t.add_argument("--model",default="Qwen/Qwen3-VL-8B-Thinking")
    t.add_argument("--max-new-tokens",type=int,default=1536)
    t.add_argument("--output",default="results/predictions.jsonl")
    t.add_argument("--errors",default="results/errors.jsonl")
    t.set_defaults(func=evaluate)
    t=sp.add_parser("summarize")
    t.add_argument("--predictions",default="results/predictions.jsonl")
    t.add_argument("--output-dir",default="results")
    t.set_defaults(func=summarize)
    sp.add_parser("selftest").set_defaults(func=selftest)
    args=p.parse_args()
    args.func(args)

if __name__=="__main__":
    main()
