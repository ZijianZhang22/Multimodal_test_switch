#!/usr/bin/env python3
"""Idis-math paired report: accurate token accounting + cautious MathVerse answer review.

Auto-evaluation confirms only clear exact/numeric/symbolic equivalence; otherwise
mark 'review', not 'wrong'. Edit manual_labels.csv (0/1) for benchmark accuracy.
"""
import csv
import json
import re
from pathlib import Path
from decimal import Decimal,InvalidOperation
from collections import defaultdict


def final_tag(text,name):
    m=re.findall(r"<"+re.escape(name)+r"\s*>(.*?)</"+re.escape(name)+r"\s*>",
                 text,flags=re.IGNORECASE|re.DOTALL)
    return m[-1].strip() if m else None


def box_extract(text):
    i=text.rfind("\\boxed{")
    if i<0:return None
    start=i+len("\\boxed{")
    depth=1
    out=[]
    for ch in text[start:]:
        if ch=="{":depth+=1
        elif ch=="}":
            depth-=1
            if depth==0:return "".join(out).strip()
        out.append(ch)
    return None


def norm(text):
    if text is None:return ""
    x=str(text).strip().replace("−","-").replace("–","-")
    x=x.replace("\\left","").replace("\\right","").replace("$","")
    x=re.sub(r"\\boxed\{(.+)\}",r"\1",x)
    x=re.sub(r"\s+","",x)
    return x.lower()


def decimal_of(text):
    x=norm(text).replace(",","").replace("\\%","%")
    x=x.rstrip("%")
    try:
        if re.fullmatch(r"[-+]?\d+(\.\d+)?",x):
            return Decimal(x)
        if re.fullmatch(r"[-+]?\d+/\d+",x):
            a,b=x.split("/")
            return Decimal(a)/Decimal(b)
    except (InvalidOperation,ZeroDivisionError):
        pass
    return None


def auto_score(pred,gold):
    if pred is None or not pred.strip():
        return None,"missing_answer"
    if norm(pred)==norm(gold):
        return True,"exact"
    a,b=decimal_of(pred),decimal_of(gold)
    if a is not None and b is not None and a==b:
        return True,"numeric_equivalence"
    try:
        from math_verify import parse,verify
        good=verify(parse("$"+str(gold)+"$"),parse("$"+str(pred)+"$"))
        if good:return True,"math_verify_equivalence"
    except Exception:
        pass
    return None,"manual_review_required"


def write(path,rows,fields):
    with Path(path).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def mean(xs):
    return sum(xs)/len(xs) if xs else None


def main(pred_path):
    pred_path=Path(pred_path)
    raw=[json.loads(x) for x in pred_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not raw:raise RuntimeError("Predictions are empty")
    out=pred_path.parent
    lookup={}
    rows=[]
    for r in raw:
        key=(r["sample_index"],int(r["rep"]),r["condition"])
        if key in lookup:raise ValueError(f"Duplicate record {key}")
        pred=final_tag(r["prediction"],"math_answer")
        if pred is None:pred=box_extract(r["prediction"])
        auto,method=auto_score(pred,r["gold"])
        warmup=final_tag(r["prediction"],"warmup")
        q1_correct=(decimal_of(warmup)==Decimal("40")) if r["condition"]=="easy_first" else None
        item={
            "sample_index":r["sample_index"],"rep":int(r["rep"]),
            "condition":r["condition"],"problem_version":r.get("problem_version",""),
            "variant":r["variant"],"gold":r["gold"],"math_prediction":pred,
            "auto_math_correct":auto,"auto_method":method,
            "warmup_prediction":warmup,"warmup_correct":q1_correct,
            "total_tokens":r["num_new_tokens"],"thinking_tokens":r.get("thinking_tokens"),
            "final_tokens":r.get("final_tokens"),"truncated":r.get("hit_max_token",False),
            "has_thinking_end":r.get("has_thinking_end",False),
        }
        rows.append(item)
        lookup[key]=item
    write(out/"scored.csv",rows,list(rows[0]))

    # 10-row template for manual grading; keep prior annotation on rerun.
    label_file=out/"manual_labels.csv"
    if not label_file.exists():
        pair_ids=sorted({(r["sample_index"],r["rep"]) for r in rows})
        templates=[{"sample_index":sid,"rep":rep,
                    "original_correct":"","easy_first_correct":"","notes":""}
                   for sid,rep in pair_ids]
        write(label_file,templates,["sample_index","rep","original_correct",
                                    "easy_first_correct","notes"])
    manual={}
    with label_file.open(encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            manual[(row["sample_index"],int(row["rep"]))]=row
    pairs=[]
    for (sid,rep,mode),base in lookup.items():
        if mode!="original":continue
        multi=lookup.get((sid,rep,"easy_first"))
        if not multi:continue
        labels=manual.get((sid,rep),{})
        def judged(row,name):
            value=str(labels.get(name,"")).strip()
            if value in ("1","0"):return int(value)
            if row["auto_math_correct"] is True:return 1
            return None
        bg=judged(base,"original_correct")
        mg=judged(multi,"easy_first_correct")
        thinking_delta=(multi["thinking_tokens"]-base["thinking_tokens"]
                        if multi["thinking_tokens"] is not None and
                        base["thinking_tokens"] is not None else None)
        pairs.append({
            "sample_index":sid,"rep":rep,"variant":base["variant"],
            "problem_version":base["problem_version"],
            "original_tokens":base["total_tokens"],
            "easy_first_tokens":multi["total_tokens"],
            "total_token_delta":multi["total_tokens"]-base["total_tokens"],
            "total_compression_fraction":(
                1-multi["total_tokens"]/base["total_tokens"]
                if base["total_tokens"] else None),
            "original_thinking_tokens":base["thinking_tokens"],
            "easy_first_thinking_tokens":multi["thinking_tokens"],
            "thinking_token_delta":thinking_delta,
            "thinking_compression_fraction":(
                1-multi["thinking_tokens"]/base["thinking_tokens"]
                if thinking_delta is not None and base["thinking_tokens"] else None),
            "original_truncated":base["truncated"],"easy_first_truncated":multi["truncated"],
            "original_math_answer":base["math_prediction"],
            "easy_first_math_answer":multi["math_prediction"],
            "gold":base["gold"],"original_correct":bg,"easy_first_correct":mg,
            "accuracy_delta":mg-bg if bg is not None and mg is not None else None,
            "warmup_correct":multi["warmup_correct"],
        })
    keys=list(pairs[0]) if pairs else ["sample_index","rep"]
    write(out/"pairs.csv",pairs,keys)
    flagged=[]
    for r in pairs:
        if r["original_correct"] is None or r["easy_first_correct"] is None or \
            r["original_truncated"] or r["easy_first_truncated"]:
            flagged.append(r)
    write(out/"needs_review.csv",flagged,keys)
    def getf(k):
        return [float(r[k]) for r in pairs if r[k] is not None]
    valid=[r for r in pairs if r["accuracy_delta"] is not None]
    summary=[{
        "questions_selected":len(set(r["sample_index"] for r in raw)),
        "completed_predictions":len(raw),"complete_pairs":len(pairs),
        "mean_total_token_delta":mean(getf("total_token_delta")),
        "mean_thinking_token_delta":mean(getf("thinking_token_delta")),
        "mean_total_compression_fraction":mean(getf("total_compression_fraction")),
        "mean_thinking_compression_fraction":mean(getf("thinking_compression_fraction")),
        "thinking_pairs":len(getf("thinking_token_delta")),
        "shorter_total":sum(x["total_token_delta"]<0 for x in pairs),
        "shorter_thinking":sum(x["thinking_token_delta"] is not None and
                               x["thinking_token_delta"]<0 for x in pairs),
        "truncated_pairs":sum(x["original_truncated"] or x["easy_first_truncated"] for x in pairs),
        "q2_fully_scored_pairs":len(valid),
        "q2_accuracy_delta_scored":mean([r["accuracy_delta"] for r in valid]),
        "needs_review":len(flagged),
    }]
    write(out/"summary.csv",summary,list(summary[0]))
    print("\n=== Idis-math10: easy-first vs original ===")
    for k,v in summary[0].items():print(f"{k}: {v}")
    print("\nQ2 grading: only exact/equivalent answers counted automatically as correct.")
    print("For unknown answers, fill manual_labels.csv with 0/1 and rerun score.py.")
    print("Output folder:",out)
    return summary[0]


if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",default="idis_math10/results/predictions.jsonl")
    a=ap.parse_args()
    main(Path(a.predictions))
