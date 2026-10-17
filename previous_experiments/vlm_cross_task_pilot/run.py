#!/usr/bin/env python3
"""Run Q2-only, Q1-only, image+Q2, Q1->Q2 and Q2->Q1 via same Qwen3-VL checkpoint."""
import argparse
import json
import random
import re
from pathlib import Path
from PIL import Image

PERCEPTION_BASE = ("[Question] Which category best describes the main object in the image? "
                   "Choose exactly one from: Dog, Bird, Vehicle, Reptile, Carnivore, Insect, "
                   "Instrument, Primate, Fish.\n"
                   "Use a thinking process to analyze the problem step-by-step.\n"
                   "At the end, provide your answer and clearly indicate it using <answer>X</answer> format.")
Q1 = ("Which category best describes the MAIN object in the image? Choose exactly one from: "
      "Dog, Bird, Vehicle, Reptile, Carnivore, Insect, Instrument, Primate, Fish.")


def build_task(row, variant, condition):
    q2 = row['q2_question']
    if condition == 'q2_only':
        return None, (f"Solve the following math problem step by step.\n{q2}\n"
                      "Put your final answer inside <q2>...</q2>." )
    image_path = row['images'][variant]
    if condition == 'q1_only':
        return image_path, PERCEPTION_BASE    # Exact published base prompt
    if condition == 'image_q2':
        return image_path, ("Solve the following math problem. The attached image is not needed.\n"
                            f"{q2}\nPut your final answer inside <q2>...</q2>.")
    if condition == 'joint_vt':
        return image_path, (f"Answer both INDEPENDENT questions, in the specified order.\n"
                            f"Question 1 (use image): {Q1}\n"
                            f"Question 2 (use ONLY Question 2 text): {q2}\n"
                            "Reason step by step as needed. End with <q1>CLASS</q1> <q2>FINAL_ANSWER</q2>.")
    if condition == 'joint_tv':
        return image_path, (f"Answer both INDEPENDENT questions, in the specified order.\n"
                            f"Question 2 (use ONLY Question 2 text): {q2}\n"
                            f"Question 1 (use image): {Q1}\n"
                            "Reason step by step as needed. End with <q2>FINAL_ANSWER</q2> <q1>CLASS</q1>.")
    raise ValueError(condition)


def inputs_for(processor, image_path, prompt, device):
    parts = []
    if image_path:
        im = Image.open(image_path).convert('RGB').resize((512,512))
        parts.append({'type':'image','image':im})
    parts.append({'type':'text','text':prompt})
    chat = [{'role':'user','content':parts}]
    chat_txt = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    if image_path:
        from qwen_vl_utils import process_vision_info
        ims, vids = process_vision_info(chat)
        payload = {'text':[chat_txt],'images':ims,'videos':vids}
    else:
        payload = {'text':[chat_txt]}
    return processor(**payload,return_tensors='pt',padding=True).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs',default='pilot_data/pairs.jsonl')
    ap.add_argument('--out',default='pilot_results/predictions.jsonl')
    ap.add_argument('--model',default='Qwen/Qwen3-VL-8B-Thinking')
    ap.add_argument('--variants',default='all',help='all or comma-separated variants present in pairs')
    ap.add_argument('--samples',type=int,default=1,help='Set 5 to match original Idis repetition count')
    ap.add_argument('--temperature',type=float,default=0.7)
    ap.add_argument('--top-p',type=float,default=0.95)
    ap.add_argument('--max-new-tokens',type=int,default=1536)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--conditions',default='q2_only,q1_only,image_q2,joint_vt')
    args = ap.parse_args()
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise SystemExit('A CUDA GPU is required for Qwen3-VL-8B-Thinking. Run this on RunPod.')
    rows = [json.loads(s) for s in Path(args.pairs).read_text(encoding='utf-8').splitlines() if s.strip()]
    torch.manual_seed(args.seed); random.seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model,dtype=torch.bfloat16,device_map='auto').eval()
    device = next(model.parameters()).device
    out = Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text(encoding='utf-8').splitlines():
            if line.strip():
                r=json.loads(line)
                done.add((r['id'],r['variant'],r['condition'],r['rep']))
    conditions=[c.strip() for c in args.conditions.split(',')]
    # One Q2-only run per pair: avoid duplicating the no-image baseline for each variant.
    tasks = []
    for row in rows:
        if 'q2_only' in conditions:
            for rep in range(args.samples): tasks.append((row,'none','q2_only',rep))
        variants = row['images'] if args.variants == 'all' else [s.strip() for s in args.variants.split(',')]
        for variant in variants:
            if variant not in row['images']: continue
            for condition in conditions:
                if condition == 'q2_only': continue
                for rep in range(args.samples): tasks.append((row,variant,condition,rep))
    print('Total tasks',len(tasks),'already done',len(done),flush=True)
    for k,(row,variant,condition,rep) in enumerate(tasks,1):
        if (row['id'],variant,condition,rep) in done: continue
        image_path,prompt=build_task(row,variant,condition)
        inputs=inputs_for(processor,image_path,prompt,device)
        sampling = args.temperature > 0
        with torch.inference_mode():
            seq=model.generate(**inputs,max_new_tokens=args.max_new_tokens,
                               do_sample=sampling,
                               **({'temperature':args.temperature,'top_p':args.top_p} if sampling else {}))
        prefix=inputs['input_ids'].shape[1]
        generated=seq[0][prefix:]
        pred=processor.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        eos_ids={model.generation_config.eos_token_id}
        if isinstance(model.generation_config.eos_token_id,list): eos_ids=set(model.generation_config.eos_token_id)
        # Some thinking models emit <|im_end|> and use multiple EOS ids, so report full-budget as a warning only.
        maybe_budget=bool(len(generated)>=args.max_new_tokens)
        rec={'id':row['id'],'variant':variant,'condition':condition,'rep':rep,
             'q1_gold':row['q1_label'],'q2_gold':row['q2_gold'],'q2_id':row['q2_id'],
             'q2_benchmark':row['q2_benchmark'],'q2_level':row.get('q2_level'),
             'pred':pred,'num_new_tokens':len(generated),'hit_max_token':maybe_budget,
             'model':args.model,'temperature':args.temperature,'top_p':args.top_p}
        with out.open('a',encoding='utf-8') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
        print(f'{k}/{len(tasks)} {row["id"]} {variant} {condition} rep={rep} tokens={len(generated)}',flush=True)
    print('COMPLETE:',out)

if __name__=='__main__':main()
