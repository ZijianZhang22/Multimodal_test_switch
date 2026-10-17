import argparse,csv,json,random,re,statistics,sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from idis_math10.score import auto_score,final_tag,box_extract,decimal_of
TO_TARGET=re.compile(r'(?im)(?:^|\n)\s*(?:now\s+)?(?:question|problem|task)\s*2\s*[:.\-–]|\b(?:now|next|then|moving on|turn(?:ing)? to|let.s (?:do|solve|address))[^\n]{0,70}\b(?:question|problem|task)\s*2\b')
TO_WARMUP=re.compile(r'(?im)(?:^|\n)\s*(?:now\s+)?(?:question|problem|task)\s*3\s*[:.\-–]|\b(?:now|next|then|moving on|turn(?:ing)? to|let.s (?:do|solve|address))[^\n]{0,70}\b(?:question|problem|task)\s*3\b')
ARITHMETIC=re.compile(r'17\s*\+\s*23|seventeen plus twenty|question\s*3',re.I)
def parts(r):
 txt=r.get("prediction","")
 if "</think>" not in txt:return None,""
 a,b=txt.split("</think>",1)
 return a.removeprefix("<think>"),b
def segments(r,tokenizer):
 think,_=parts(r)
 if think is None or r.get("hit_max_token"):return None,"truncated_or_incomplete"
 if r["condition"]=="A":return r["thinking_tokens"],"exact"
 if r["condition"]=="C":
  m=TO_TARGET.search(think)
  if not m:return None,"ambiguous_switch"
  target=think[m.start():]
  if re.search(r'(?i)(back|return|revisit).{0,15}question\s*1',target):return None,"back_to_q1"
  return len(tokenizer.encode(target,add_special_tokens=False)),"estimated_target"
 if r["condition"]=="B":
  m=TO_WARMUP.search(think)
  if m:return len(tokenizer.encode(think[:m.start()],add_special_tokens=False)),"estimated_target"
  if ARITHMETIC.search(think):return None,"mixed"
  return r["thinking_tokens"],"no_arithmetic_in_thinking"
 raise ValueError(r["condition"])
def mean(x):return sum(x)/len(x) if x else None
def ci(xs,n=3000):
 if len(xs)<2:return (None,None)
 rng=random.Random(42);means=sorted(mean([xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(n))
 return means[int(.025*(n-1))],means[int(.975*(n-1))]
def write(p,rows,cols):
 with open(p,"w",newline="",encoding="utf8") as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
def evaluate(predictions,out_dir):
 from transformers import AutoTokenizer
 raw=[json.loads(x) for x in Path(predictions).read_text().splitlines() if x.strip()]
 if not raw:raise ValueError("No predictions")
 out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
 models={x["model"] for x in raw}
 if len(models)!=1:raise ValueError("Mixed models")
 tokenizer=AutoTokenizer.from_pretrained(next(iter(models)),use_fast=True)
 ids=sorted({(r["id"],int(r["rep"])) for r in raw})
 label_path=out/"manual_labels_abc.csv"
 if not label_path.exists():write(label_path,[dict(id=i,rep=rep,A_correct="",B_correct="",C_correct="",notes="") for i,rep in ids],["id","rep","A_correct","B_correct","C_correct","notes"])
 with label_path.open(newline="",encoding="utf8") as f:manual={(x["id"],int(x["rep"])):x for x in csv.DictReader(f)}
 rows=[];index={}
 for r in raw:
  key=(r["id"],int(r["rep"]),r["condition"])
  if key in index:raise ValueError(f"Duplicate {key}")
  _,final=parts(r)
  pred=final_tag(final,"math_answer") or box_extract(final)
  warmup=final_tag(final,"warmup") if r["condition"]!="A" else None
  auto,method=auto_score(pred,r["gold"])
  label=str(manual.get(key[:2],{}).get(r["condition"]+"_correct","")).strip()
  correctness=int(label) if label in ("0","1") else (1 if auto is True else None)
  target,status=segments(r,tokenizer)
  record=dict(id=r["id"],rep=int(r["rep"]),level=r["level"],condition=r["condition"],gold=r["gold"],answer=pred,correct=correctness,auto_method=method,warmup=warmup,warmup_correct=(int(decimal_of(warmup)==decimal_of("40")) if warmup is not None else None),both_answer_tags=(bool(pred) and bool(warmup) if r["condition"]!="A" else bool(pred)),total_tokens=r["num_new_tokens"],thinking_tokens=r["thinking_tokens"],final_tokens=r.get("final_tokens"),target_thinking_tokens_est=target,target_segmentation=status,truncated=r["hit_max_token"],has_thinking_end=r["has_thinking_end"])
  rows.append(record);index[key]=record
 pairs=[]
 for rid,rep in ids:
  triple=[index.get((rid,rep,c)) for c in ("A","B","C")]
  if any(r is None for r in triple):raise ValueError(f"Incomplete: {rid}/{rep}")
  a,b,c=triple
  good=not any(x["truncated"] or not x["has_thinking_end"] for x in triple)
  targets=[r["target_thinking_tokens_est"] if good else None for r in triple]
  def diff(i,j):return targets[j]-targets[i] if targets[i] is not None and targets[j] is not None else None
  row=dict(id=rid,rep=rep,level=a["level"],A_total=a["total_tokens"],B_total=b["total_tokens"],C_total=c["total_tokens"],A_target=targets[0],B_target=targets[1],C_target=targets[2],B_minus_A_target=diff(0,1),C_minus_A_target=diff(0,2),B_minus_C_target=diff(2,1),B_minus_A_total=b["total_tokens"]-a["total_tokens"],C_minus_A_total=c["total_tokens"]-a["total_tokens"],B_minus_C_total=b["total_tokens"]-c["total_tokens"],B_completed=b["both_answer_tags"],C_completed=c["both_answer_tags"],A_correct=a["correct"],B_correct=b["correct"],C_correct=c["correct"],A_seg_status=a["target_segmentation"],B_seg_status=b["target_segmentation"],C_seg_status=c["target_segmentation"],any_truncated=not good)
  pairs.append(row)
 write(out/"scored.csv",rows,list(rows[0]));write(out/"pairs.csv",pairs,list(pairs[0]))
 needs=[r for r in pairs if r["any_truncated"] or any(r[k+"_correct"] is None for k in "ABC") or not r["B_completed"] or not r["C_completed"] or r["B_target"] is None or r["C_target"] is None]
 write(out/"needs_review.csv",needs,list(pairs[0]))
 stats=[]
 for level in ["ALL",1,2,3,4,5]:
  subset=[r for r in pairs if level=="ALL" or r["level"]==level]
  for x,y in [("A","B"),("A","C"),("C","B")]:
   label=y+"_minus_"+x;deltas=[r[label+"_target"] for r in subset if r[label+"_target"] is not None];totals=[r[label+"_total"] for r in subset];lo,hi=ci(deltas)
   stats.append(dict(level=level,comparison=label,n=len(subset),target_valid=len(deltas),target_mean_delta=mean(deltas),target_median_delta=statistics.median(deltas) if deltas else None,target_shorter=sum(v<0 for v in deltas),ci_95_low=lo,ci_95_high=hi,total_mean_delta=mean(totals),total_median_delta=statistics.median(totals) if totals else None))
 write(out/"summary.csv",stats,list(stats[0]))
 print("Triples:",len(pairs),"B complete:",sum(x["B_completed"] for x in pairs),"C complete:",sum(x["C_completed"] for x in pairs))
 for r in stats[:3]:print(r)
 print("Results:",out,"; uncertain correctness requires manual review")
def main():
 p=argparse.ArgumentParser();p.add_argument("--predictions",default="text_math50/results/predictions.jsonl");p.add_argument("--out-dir",default="text_math50/results/analysis");a=p.parse_args();evaluate(a.predictions,a.out_dir)
if __name__=="__main__":main()
