import argparse,hashlib,json,random
from pathlib import Path
MODEL="Qwen/Qwen3-VL-8B-Thinking"
def target_question(p):
 return "Question 2 (text-only mathematics):\n"+p+"\nSolve Question 2 carefully. Reason step by step as needed. At the end give its final answer in <math_answer>...</math_answer>."
def prompt_for(problem,condition):
 base=target_question(problem)
 if condition=="A":return base
 if condition=="B":return base+"\n\nQuestion 3 (independent, answer after Question 2): What is 17 + 23?\nBoth questions must be answered. Finish Question 2 FIRST and then Question 3. Give the separate answer to Question 3 in <warmup>...</warmup>. Do not stop after answering Question 2."
 if condition=="C":return "Question 1 (independent, answer FIRST): What is 17 + 23?\nAnswer Question 1 first inside <warmup>...</warmup>, then solve the separate Question 2 below. Both questions must be answered.\n\n"+base
 raise ValueError(condition)
def split_counts(tokenizer,ids):
 marker=tokenizer.encode("</think>",add_special_tokens=False)
 for i in range(len(ids)-len(marker)+1):
  if ids[i:i+len(marker)]==marker:return i,len(ids)-i-len(marker),True
 return None,None,False
def main():
 p=argparse.ArgumentParser()
 for name,default in [("manifest","text_math50/data/manifest.jsonl"),("out","text_math50/results/predictions.jsonl"),("model",MODEL)]:p.add_argument("--"+name,default=default)
 for name,default in [("samples",1),("max-new-tokens",12288),("seed",1043)]:p.add_argument("--"+name,type=int,default=default)
 for name,default in [("temperature",.7),("top-p",.95)]:p.add_argument("--"+name,type=float,default=default)
 p.add_argument("--conditions",nargs="+",choices=["A","B","C"],default=["A","B","C"])
 a=p.parse_args()
 rows=[json.loads(x) for x in Path(a.manifest).read_text().splitlines() if x.strip()]
 if not rows or a.samples<1 or len(set(a.conditions))!=len(a.conditions):raise ValueError("Bad task configuration")
 output=Path(a.out);output.parent.mkdir(parents=True,exist_ok=True);done=set()
 if output.exists():
  for line in output.read_text().splitlines():
   if not line.strip():continue
   r=json.loads(line);key=(r["id"],int(r["rep"]),r["condition"])
   if key in done:raise ValueError(f"Duplicate: {key}")
   if (r["model"],r["base_seed"],r["max_new_tokens"],r["temperature"],r["top_p"])!=(a.model,a.seed,a.max_new_tokens,a.temperature,a.top_p):raise ValueError("Resume settings changed; select another DATA directory")
   done.add(key)
 tasks=[(r,rep,c) for r in rows for rep in range(a.samples) for c in a.conditions if (r["id"],rep,c) not in done]
 print(f"Questions {len(rows)}; pending {len(tasks)} / {len(rows)*a.samples*len(a.conditions)}",flush=True)
 if not tasks:return
 import torch
 from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
 if not torch.cuda.is_available():raise RuntimeError("GPU required")
 processor=AutoProcessor.from_pretrained(a.model)
 model=Qwen3VLForConditionalGeneration.from_pretrained(a.model,dtype=torch.bfloat16,device_map="auto").eval()
 dev=next(model.parameters()).device
 for ix,(row,rep,c) in enumerate(tasks,1):
  seed=(a.seed+int(hashlib.sha256((row["id"]+"|"+str(rep)).encode()).hexdigest()[:8],16))%(2**31)
  random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  prompt=prompt_for(row["problem"],c)
  messages=[{"role":"user","content":[{"type":"text","text":prompt}]}]
  inp=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt").to(dev)
  opts=dict(max_new_tokens=a.max_new_tokens,do_sample=a.temperature>0)
  if a.temperature>0:opts.update(temperature=a.temperature,top_p=a.top_p)
  with torch.inference_mode():ids=model.generate(**inp,**opts)[0][inp["input_ids"].shape[1]:].tolist()
  thinking,final,has_end=split_counts(processor.tokenizer,ids)
  rec=dict(id=row["id"],source_index=row["source_index"],level=row["level"],subject=row["subject"],problem=row["problem"],gold=row["gold"],rep=rep,condition=c,model=a.model,prompt=prompt,prediction=processor.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),num_input_tokens=inp["input_ids"].shape[1],num_new_tokens=len(ids),thinking_tokens=thinking,final_tokens=final,has_thinking_end=has_end,hit_max_token=len(ids)>=a.max_new_tokens,seed=seed,base_seed=a.seed,temperature=a.temperature,top_p=a.top_p,max_new_tokens=a.max_new_tokens)
  with output.open("a",encoding="utf8") as f:f.write(json.dumps(rec,ensure_ascii=False)+"\n")
  print(f"[{ix}/{len(tasks)}] level={row['level']} {row['id']} {c} total={len(ids)} thinking={thinking} truncated={rec['hit_max_token']}",flush=True)
if __name__=="__main__":main()
