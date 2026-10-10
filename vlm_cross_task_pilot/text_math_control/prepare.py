import argparse,json,random,re
from pathlib import Path
from collections import defaultdict,Counter
def is_text_only(p):
 return not bool(re.search(r'\[asy\]|\\begin\{asy\}|\b(?:diagram|figure|image|picture)\b|\b(?:shown|graphed|pictured|illustrated)\s+(?:above|below)\b',p,re.I))
def select_rows(dataset,count,seed):
 if count<5:raise ValueError("Need at least 5 questions")
 buckets=defaultdict(list)
 for i,x in enumerate(dataset):
  p=str(x.get("problem") or "").strip();gold=str(x.get("answer") or "").strip()
  level=int(x.get("level") or 0)
  if p and gold and 1<=level<=5 and is_text_only(p):
   buckets[level].append(dict(id=str(x.get("unique_id") or f"math500_{i}"),source_index=i,level=level,subject=str(x.get("subject") or ""),problem=p,gold=gold))
 rng=random.Random(seed);chosen=[]
 for level in range(1,6):
  rng.shuffle(buckets[level]);n=count//5+int(level<=count%5)
  if len(buckets[level])<n:raise ValueError(f"Not enough level {level} samples")
  chosen.extend(buckets[level][:n])
 rng.shuffle(chosen)
 if len({r["id"] for r in chosen})!=len(chosen):raise ValueError("Duplicate IDs")
 return chosen
def main():
 p=argparse.ArgumentParser();p.add_argument("--out",default="text_math50/data/manifest.jsonl");p.add_argument("--count",type=int,default=50);p.add_argument("--seed",type=int,default=1043);a=p.parse_args()
 path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
  if len(rows)!=a.count:raise ValueError("Manifest count differs")
 else:
  from datasets import load_dataset
  rows=select_rows(load_dataset("HuggingFaceH4/MATH-500",split="test"),a.count,a.seed)
  temp=path.with_suffix(".tmp");temp.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf8");temp.replace(path)
 print("Ready:",path,"levels:",dict(Counter(x["level"] for x in rows)))
if __name__=="__main__":main()
