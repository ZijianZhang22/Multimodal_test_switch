#!/usr/bin/env python3
"""
Generate an arbitrary-hop full factorial Text/Vision schedule dataset.

For hop count H, every logical fact is independently carried as Text (T) or
Vision (V), giving 2^H modality schedules. This generalizes the original 4-hop
all-16 design to H=5 (32 schedules), H=6 (64 schedules), etc.

Important parity note:
- For odd H, a unit-step cardinal walk cannot end at SAME=(0,0).
- Therefore H=5 balances over the 8 reachable directional answer classes.
"""

import argparse
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DIRECTIONS={"N":(0,-1),"S":(0,1),"E":(1,0),"W":(-1,0)}
DIR_WORD={"N":"north of","S":"south of","E":"east of","W":"west of"}
ALL_LABELS=["NORTH","SOUTH","EAST","WEST","NORTHEAST","NORTHWEST","SOUTHEAST","SOUTHWEST","SAME"]

def vector_to_answer(x,y):
    if x==0 and y==0: return "SAME"
    v="NORTH" if y<0 else "SOUTH" if y>0 else ""
    h="EAST" if x>0 else "WEST" if x<0 else ""
    return v+h if v and h else v or h

def load_font(size):
    for path in ["DejaVuSans-Bold.ttf","/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        try: return ImageFont.truetype(path,size)
        except Exception: pass
    return ImageFont.load_default()

def draw_relation_image(reference,subject,direction,out_path,size=384):
    img=Image.new("RGB",(size,size),"white"); d=ImageDraw.Draw(img)
    center=(size//2,size//2); offset=size//4
    dx,dy=DIRECTIONS[direction]
    ref=center; subj=(center[0]+dx*offset,center[1]+dy*offset); r=34
    d.line([ref,subj],fill="black",width=5)
    for xy,txt in [(ref,reference),(subj,subject)]:
        d.ellipse([xy[0]-r,xy[1]-r,xy[0]+r,xy[1]+r],fill="white",outline="black",width=4)
        font=load_font(30); box=d.textbbox((0,0),txt,font=font)
        tw=box[2]-box[0]; th=box[3]-box[1]
        d.text((xy[0]-tw/2,xy[1]-th/2-2),txt,fill="black",font=font)
    out_path.parent.mkdir(parents=True,exist_ok=True); img.save(out_path)

def make_problem(rng,pid,h):
    ents=[f"N{i}" for i in range(h+1)]
    dirs=[rng.choice(list(DIRECTIONS)) for _ in range(h)]
    x=y=0; facts=[]
    for i,direction in enumerate(dirs,1):
        dx,dy=DIRECTIONS[direction]; x+=dx; y+=dy
        facts.append({
            "step":i,"logical_step":i,"reference":ents[i-1],"subject":ents[i],
            "direction":direction,
            "text":f"{ents[i]} is {DIR_WORD[direction]} {ents[i-1]}.",
            "state_x":x,"state_y":y,
        })
    return {
        "problem_id":f"h{h}_p{pid:05d}","hop_count":h,"entities":ents,
        "directions":dirs,"facts":facts,
        "question":f"Where is {ents[-1]} relative to {ents[0]}?",
        "answer":vector_to_answer(x,y),
    }

def reachable_labels(h):
    # SAME is impossible for odd H due to parity of unit cardinal steps.
    return [x for x in ALL_LABELS if not (h%2==1 and x=="SAME")]

def balanced_latents(n,seed,h):
    rng=random.Random(seed+1009*h)
    labels=reachable_labels(h)
    quota={lab:math.ceil(n/len(labels)) for lab in labels}
    counts=Counter(); out=[]; attempts=0
    while len(out)<n:
        attempts+=1
        if attempts>n*10000:
            raise RuntimeError(f"Could not balance H={h}; counts={dict(counts)}")
        p=make_problem(rng,len(out),h); a=p["answer"]
        if a not in quota or counts[a]>=quota[a]: continue
        counts[a]+=1; p["problem_id"]=f"h{h}_p{len(out):05d}"; out.append(p)
    return out,counts

def schedule_meta(s):
    return {
        "schedule":s,
        "switch_count":sum(s[i]!=s[i+1] for i in range(len(s)-1)),
        "vision_count":s.count("V"),
        "text_count":s.count("T"),
        "starts_with":s[0],"ends_with":s[-1],
        "vision_fraction":s.count("V")/len(s),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out_dir",type=Path,default=Path("data_factorial"))
    ap.add_argument("--hops",type=int,default=5)
    ap.add_argument("--n_problems",type=int,default=100)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    h=args.hops
    if h<2: raise ValueError("--hops must be >=2")
    args.out_dir.mkdir(parents=True,exist_ok=True)
    image_dir=args.out_dir/"images"; image_dir.mkdir(parents=True,exist_ok=True)

    latents,counts=balanced_latents(args.n_problems,args.seed,h)
    schedules=["".join(bits) for bits in itertools.product("TV",repeat=h)]
    with (args.out_dir/"latent_problems.jsonl").open("w",encoding="utf-8") as lf, \
         (args.out_dir/"examples.jsonl").open("w",encoding="utf-8") as ef:
        for p in latents:
            for fact in p["facts"]:
                rel=Path("images")/f"{p['problem_id']}_step{fact['step']}.png"
                fact["image"]=str(rel)
                draw_relation_image(fact["reference"],fact["subject"],fact["direction"],args.out_dir/rel)
            lf.write(json.dumps(p,ensure_ascii=False)+"\n")
            for s in schedules:
                facts=[{**fact,"source":s[i]} for i,fact in enumerate(p["facts"])]
                ex={
                    "task_type":"factorial_schedule",
                    "example_id":f"{p['problem_id']}_{s}",
                    "problem_id":p["problem_id"],"hop_count":h,
                    **schedule_meta(s),
                    "logical_schedule":s,"presentation_schedule":s,
                    "facts":facts,"question":p["question"],"answer":p["answer"],
                }
                ef.write(json.dumps(ex,ensure_ascii=False)+"\n")

    print(f"H={h}; problems={len(latents)}; schedules={len(schedules)}; examples={len(latents)*len(schedules)}")
    print("Answer counts:",dict(counts))
    print("Examples:",args.out_dir/"examples.jsonl")

if __name__=="__main__":
    main()
