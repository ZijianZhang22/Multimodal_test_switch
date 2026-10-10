#!/usr/bin/env python3
"""Paired Idis-math pilot: same image/question, with/without an easy FIRST question.

Output length is counted using generated token IDs, separately for Qwen3-VL's
<think>...</think> and final answer; never infer thinking count from whitespace.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image

EASY_QUESTION = "What is 17 + 23?"
EASY_GOLD = "40"


def math_prompt(question):
    # Identical suffix in both conditions. Only the easy question prefix differs.
    return (
        "Question 2 (visual mathematics):\n"
        f"{question}\n"
        "Solve Question 2 carefully using the provided image and its relevant details. "
        "Reason step by step as needed. At the end, write the final answer to "
        "Question 2 inside <math_answer>...</math_answer>."
    )


def prompt_for(question, condition):
    if condition == "original":
        return math_prompt(question)
    if condition == "easy_first":
        return (
            f"Question 1 (independent, solve FIRST): {EASY_QUESTION}\n"
            "This easy question is entirely unrelated to the image and Question 2. "
            "Provide its answer inside <warmup>...</warmup>. "
            "Then solve the separate visual mathematics question below.\n\n"
            + math_prompt(question)
        )
    raise ValueError(f"Unexpected condition: {condition}")


def build_inputs(processor, image_file, prompt, device, max_image_side):
    from qwen_vl_utils import process_vision_info
    im = Image.open(image_file).convert("RGB")
    if max(im.size) > max_image_side:
        im.thumbnail((max_image_side, max_image_side), Image.Resampling.LANCZOS)
    messages = [{"role":"user","content":[
        {"type":"image","image":im},{"type":"text","text":prompt},
    ]}]
    txt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos=process_vision_info(messages)
    return processor(text=[txt],images=images,videos=videos,
                     padding=True,return_tensors="pt").to(device)


def split_token_counts(tokenizer, ids):
    """Find the actual Qwen3 '</think>' token, preserving exact generation counts."""
    marker_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if not marker_ids:
        return None, None, False
    for i in range(len(ids)-len(marker_ids)+1):
        if ids[i:i+len(marker_ids)] == marker_ids:
            # thinking count excludes </think> itself; include final EOS in final count.
            return i, len(ids)-(i+len(marker_ids)), True
    return None,None,False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="idis_math10/data/manifest.jsonl")
    p.add_argument("--out", default="idis_math10/results/predictions.jsonl")
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Thinking")
    p.add_argument("--max-new-tokens",type=int,default=8192)
    p.add_argument("--max-image-side",type=int,default=1536,
                   help="Avoid shrinking MathVerse diagrams to 512px")
    p.add_argument("--samples",type=int,default=1)
    p.add_argument("--temperature",type=float,default=0.7)
    p.add_argument("--top-p",type=float,default=0.95)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    if args.samples<1:p.error("samples must be >=1")
    if args.max_image_side<512:p.error("max-image-side should be >=512")
    manifest=Path(args.manifest).expanduser().resolve()
    rows=[json.loads(s) for s in manifest.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not rows:raise RuntimeError("No math questions in manifest")
    out=Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    completed=set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r=json.loads(line)
                k=(r["sample_index"],r["condition"],int(r["rep"]))
                if k in completed:raise ValueError(f"Duplicate record {k}")
                if r.get("model")!=args.model:
                    raise ValueError("Existing output uses a different model. Set a new OUT.")
                completed.add(k)
    tasks=[]
    for r in rows:
        if not Path(r["image"]).is_file():
            raise FileNotFoundError(r["image"])
        for rep in range(args.samples):
            for condition in ("original","easy_first"):
                if (r["sample_index"],condition,rep) not in completed:
                    tasks.append((r,condition,rep))
    print(f"Selected math questions: {len(rows)}; planned generations: {len(rows)*2*args.samples}; "
          f"pending: {len(tasks)}",flush=True)
    if not tasks:
        from score import main as score_main
        score_main(out)
        return
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Qwen3-VL inference")
    processor=AutoProcessor.from_pretrained(args.model)
    model=Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,dtype=torch.bfloat16,device_map="auto"
    ).eval()
    device=next(model.parameters()).device
    for index,(r,condition,rep) in enumerate(tasks,1):
        # Same seed for both conditions of the same question/rep.
        salt=int(hashlib.sha256((r["sample_index"]+"|"+str(rep)).encode()).hexdigest()[:8],16)
        seed=(args.seed+salt)%(2**31)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        prompt=prompt_for(r["question"],condition)
        inputs=build_inputs(processor,r["image"],prompt,device,args.max_image_side)
        sample=args.temperature>0
        with torch.inference_mode():
            outputs=model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=sample,
                **({"temperature":args.temperature,"top_p":args.top_p} if sample else {})
            )
        generated=outputs[0][inputs["input_ids"].shape[1]:].tolist()
        n_thinking,n_final,has_end=split_token_counts(processor.tokenizer,generated)
        raw=processor.tokenizer.decode(
            generated, skip_special_tokens=False,clean_up_tokenization_spaces=False
        )
        rec={
            "sample_index":r["sample_index"],"problem_index":r["problem_index"],
            "problem_version":r.get("problem_version"),"condition":condition,"rep":rep,
            "variant":r["variant"],"n_distractors":r["n_distractors"],
            "question":r["question"],"question_for_eval":r["question_for_eval"],
            "gold":r["answer"],"easy_gold":EASY_GOLD if condition=="easy_first" else None,
            "image":r["image"],"prompt":prompt,"prediction":raw,
            "num_input_tokens":int(inputs["input_ids"].shape[1]),
            "num_new_tokens":len(generated),
            "thinking_tokens":n_thinking,"final_tokens":n_final,
            "has_thinking_end":has_end,
            "hit_max_token":len(generated)>=args.max_new_tokens,
            "model":args.model,"seed":seed,"temperature":args.temperature,
            "top_p":args.top_p,"max_new_tokens":args.max_new_tokens,
            "max_image_side":args.max_image_side,
        }
        with out.open("a",encoding="utf-8") as file:
            file.write(json.dumps(rec,ensure_ascii=False)+"\n")
        print(f"[{index}/{len(tasks)}] {r['sample_index']} {condition} "
              f"total={len(generated)} thinking={n_thinking} final={n_final} "
              f"truncated={rec['hit_max_token']}",flush=True)
    from score import main as score_main
    score_main(out)


if __name__=="__main__":
    main()
