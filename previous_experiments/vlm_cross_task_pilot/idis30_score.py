#!/usr/bin/env python3
"""Analyze each variant's Q1 accuracy and token compression with exact paired images."""
import argparse,csv,json,re
from collections import defaultdict
from pathlib import Path
from score import tag,numeric

VARIANTS=("clean","aligned","conflicting","irrelevant")
LABELS=("dog","bird","vehicle","reptile","carnivore","insect","instrument","primate","fish")


def parse_visual(pred):
    ans=tag(pred,"answer")
    if not ans:return None
    ans=re.sub(r"[^a-z ]","",ans.lower()).strip()
    if ans in ("wheeled vehicle","vehicle"):return "vehicle"
    if ans in ("musical instrument","instrument"):return "instrument"
    return ans if ans in LABELS else None


def score_record(r):
    pred=r["pred"]
    visual=parse_visual(pred)
    q2=tag(pred,"q2") if r["condition"]=="visual_plus_text" else None
    thought=pred.split("</think>")[0] if "</think>" in pred else pred
    return {"id":r["id"],"variant":r["variant"],"condition":r["condition"],"rep":r["rep"],
            "q1_gold":r["q1_gold"],"q1_pred":visual,"q1_correct":int(visual==r["q1_gold"].lower()),
            "q1_tag_present":int(visual is not None),
            "q2_pred":q2,"q2_correct":int(numeric(q2)==numeric("40")) if q2 is not None else "",
            "q2_tag_present":int(q2 is not None) if r["condition"]=="visual_plus_text" else "",
            "output_tokens":r["num_new_tokens"],"input_tokens":r.get("num_input_tokens",""),
            "thinking_text_words":len(thought.split()),
            "has_think_close":int("</think>" in pred),
            "truncated":int(r.get("hit_max_token",False))}


def write(path,data,cols):
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(data)


def avg(vals):
    return sum(vals)/len(vals) if vals else None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",default="idis30/results/predictions.jsonl")
    args=ap.parse_args()
    path=Path(args.predictions)
    raw=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data=[score_record(r) for r in raw]
    if not data:raise RuntimeError("No predictions")
    out=path.parent
    write(out/"scored.csv",data,list(data[0]))
    grouped=defaultdict(list)
    key={}
    for r in data:
        k=(r["id"],r["variant"],r["rep"],r["condition"])
        if k in key:raise ValueError(f"Duplicate prediction {k}")
        key[k]=r;grouped[(r["variant"],r["condition"])].append(r)
    summary=[]
    for (variant,condition),items in sorted(grouped.items()):
        summary.append({"variant":variant,"condition":condition,"n":len(items),
                        "q1_accuracy":avg([r["q1_correct"] for r in items]),
                        "q2_accuracy":avg([r["q2_correct"] for r in items if isinstance(r["q2_correct"],int)]),
                        "avg_output_tokens":avg([r["output_tokens"] for r in items]),
                        "avg_thinking_words":avg([r["thinking_text_words"] for r in items]),
                        "truncated":sum(r["truncated"] for r in items)})
    write(out/"summary.csv",summary,list(summary[0]))
    pairs=[]
    for (id_,variant,rep,mode),single in key.items():
        if mode!="visual_only":continue
        multi=key.get((id_,variant,rep,"visual_plus_text"))
        if not multi:continue
        pairs.append({"id":id_,"variant":variant,"rep":rep,
                      "single_correct":single["q1_correct"],"multi_correct":multi["q1_correct"],
                      "accuracy_delta":multi["q1_correct"]-single["q1_correct"],
                      "single_tokens":single["output_tokens"],"multi_tokens":multi["output_tokens"],
                      "token_delta":multi["output_tokens"]-single["output_tokens"],
                      "compression_fraction":(single["output_tokens"]-multi["output_tokens"])/single["output_tokens"] if single["output_tokens"] else "",
                      "single_thinking_words":single["thinking_text_words"],
                      "multi_thinking_words":multi["thinking_text_words"],
                      "q2_correct":multi["q2_correct"],"single_truncated":single["truncated"],
                      "multi_truncated":multi["truncated"]})
    cols=["id","variant","rep","single_correct","multi_correct","accuracy_delta",
          "single_tokens","multi_tokens","token_delta","compression_fraction",
          "single_thinking_words","multi_thinking_words","q2_correct","single_truncated","multi_truncated"]
    write(out/"paired.csv",pairs,cols)
    stats=[]
    for variant in VARIANTS:
        values=[p for p in pairs if p["variant"]==variant]
        if not values:continue
        stats.append({"variant":variant,"paired_n":len(values),
                      "accuracy_delta":avg([v["accuracy_delta"] for v in values]),
                      "avg_token_delta":avg([v["token_delta"] for v in values]),
                      "mean_compression_fraction":avg([v["compression_fraction"] for v in values if v["compression_fraction"]!=""]),
                      "improved":sum(v["accuracy_delta"]==1 for v in values),
                      "degraded":sum(v["accuracy_delta"]==-1 for v in values),
                      "truncated_single":sum(v["single_truncated"] for v in values),
                      "truncated_multi":sum(v["multi_truncated"] for v in values)})
    cols=["variant","paired_n","accuracy_delta","avg_token_delta","mean_compression_fraction",
          "improved","degraded","truncated_single","truncated_multi"]
    write(out/"paired_summary.csv",stats,cols)
    clean=next((r for r in stats if r["variant"]=="clean"),None)
    if clean:
        for row in stats:
            row["accuracy_interaction_vs_clean"]=row["accuracy_delta"]-clean["accuracy_delta"]
    write(out/"interaction_vs_clean.csv",stats,cols+["accuracy_interaction_vs_clean"] if clean else cols)
    print("Q1: visual classification, Q2: arithmetic answer")
    for r in stats:
        print(f"{r['variant']:12} N={r['paired_n']:3} Δacc={r['accuracy_delta']:+.3f} Δtokens={r['avg_token_delta']:+.1f} compression={r['mean_compression_fraction']:.1%} improved={r['improved']} degraded={r['degraded']}")
    print("Wrote",out)


if __name__=="__main__":main()
