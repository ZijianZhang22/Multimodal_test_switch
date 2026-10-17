#!/usr/bin/env python3
"""Run 30 matched Idis originals x 4 variants x 2 conditions = 240 generations."""
import argparse
import json
import random
from pathlib import Path
from PIL import Image
from run import PERCEPTION_BASE

MODES=("visual_only","visual_plus_text")
ORDER=("clean","aligned","conflicting","irrelevant")


def prompt_for(mode):
    if mode=="visual_only":
        return PERCEPTION_BASE
    if mode=="visual_plus_text":
        return (PERCEPTION_BASE + "\n\n"
                "[Independent Question 2] What is 17 + 23?\n"
                "Answer both questions. Keep the image category in <answer>X</answer> "
                "and provide the arithmetic answer in <q2>NUMBER</q2>. "
                "Think step by step as needed.")
    raise ValueError(mode)


def tokens_for(processor,image_path,prompt,device):
    from qwen_vl_utils import process_vision_info
    im=Image.open(image_path).convert("RGB").resize((512,512))
    message=[{"role":"user","content":[{"type":"image","image":im},{"type":"text","text":prompt}]}]
    chat=processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
    images,videos=process_vision_info(message)
    return processor(text=[chat],images=images,videos=videos,padding=True,return_tensors="pt").to(device)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",default="idis30/data/manifest.jsonl")
    ap.add_argument("--out",default="idis30/results/predictions.jsonl")
    ap.add_argument("--model",default="Qwen/Qwen3-VL-8B-Thinking")
    ap.add_argument("--samples",type=int,default=1)
    ap.add_argument("--max-new-tokens",type=int,default=4096)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--temperature",type=float,default=0.7)
    ap.add_argument("--top-p",type=float,default=0.95)
    args=ap.parse_args()
    rows=[json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("Empty manifest")
    for row in rows:
        if tuple(sorted(row["images"]))!=tuple(sorted(ORDER)):
            raise ValueError(f"Four matched conditions required: {row['id']}")
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    done=set()
    if out.exists():
        for ln in out.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                x=json.loads(ln)
                done.add((x["id"],x["variant"],x["condition"],x["rep"]))
    tasks=[(row,v,mode,rep) for row in rows for v in ORDER
           for rep in range(args.samples) for mode in MODES
           if (row["id"],v,mode,rep) not in done]
    print(f"Originals: {len(rows)}; all planned: {len(rows)*4*2*args.samples}; pending: {len(tasks)}",flush=True)
    if not tasks:
        return
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; run on RunPod")
    processor=AutoProcessor.from_pretrained(args.model)
    model=Qwen3VLForConditionalGeneration.from_pretrained(args.model,dtype=torch.bfloat16,device_map="auto").eval()
    device=next(model.parameters()).device
    # Condition-specific seed is deterministic; repetitions remain independent.
    for idx,(row,v,mode,rep) in enumerate(tasks,1):
        random.seed(args.seed+rep)
        torch.manual_seed(args.seed+rep)
        torch.cuda.manual_seed_all(args.seed+rep)
        prompt=prompt_for(mode)
        inputs=tokens_for(processor,row["images"][v],prompt,device)
        sample=args.temperature>0
        with torch.inference_mode():
            output=model.generate(**inputs,max_new_tokens=args.max_new_tokens,
                                  do_sample=sample,**({"temperature":args.temperature,"top_p":args.top_p} if sample else {}))
        output_tokens=output[0][inputs["input_ids"].shape[1]:]
        prediction=processor.decode(output_tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        rec={
            "id":row["id"],"variant":v,"condition":mode,"rep":rep,
            "q1_gold":row["label"],"q2_gold":"40" if mode=="visual_plus_text" else None,
            "image":row["images"][v],"prompt":prompt,"pred":prediction,
            "num_input_tokens":int(inputs["input_ids"].shape[1]),
            "num_new_tokens":int(len(output_tokens)),
            "hit_max_token":bool(len(output_tokens)>=args.max_new_tokens),
            "model":args.model,"seed":args.seed+rep,"temperature":args.temperature,
            "top_p":args.top_p}
        with out.open("a",encoding="utf-8") as f:
            f.write(json.dumps(rec,ensure_ascii=False)+"\n")
        print(f"[{idx}/{len(tasks)}] {row['id']} {v} {mode} rep={rep} tokens={len(output_tokens)}",flush=True)
    print("COMPLETE",out)


if __name__=="__main__":
    main()
