#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict

def load(path):
    rows=[]
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def cond_name(r):
    if r.get("condition") == "text_baseline":
        return "TTTT"
    if r.get("condition") == "single_visual":
        return f"V@{r.get('visual_logical_step')}"
    return r.get("condition", "other")

def summarize(rows, title):
    groups=defaultdict(list)
    for r in rows:
        groups[cond_name(r)].append(r)

    print(f"\n=== {title} ===")
    for name in ["TTTT","V@1","V@4"]:
        g=groups.get(name,[])
        if not g:
            print(f"{name}: N=0")
            continue
        correct=sum(bool(x.get("correct")) for x in g)
        valid=sum(x.get("prediction") is not None for x in g)
        toks=sum(x.get("output_tokens",0) for x in g)/len(g)
        lat=sum(x.get("latency_sec",0.0) for x in g)/len(g)
        print(f"{name}: {correct}/{len(g)} = {correct/len(g):.1%} | valid={valid}/{len(g)} | mean_tokens={toks:.1f} | mean_latency={lat:.2f}s")

    base={r["problem_id"]:r for r in groups.get("TTTT",[])}
    for name in ["V@1","V@4"]:
        deltas=[]
        flips01=flips10=0
        for r in groups.get(name,[]):
            b=base.get(r["problem_id"])
            if b is None:
                continue
            bv=int(bool(b["correct"]))
            rv=int(bool(r["correct"]))
            deltas.append(rv-bv)
            if bv==0 and rv==1: flips01+=1
            if bv==1 and rv==0: flips10+=1
        if deltas:
            print(f"{name} - TTTT matched delta: {sum(deltas)/len(deltas):+.1%} | T wrong->V right={flips01} | T right->V wrong={flips10}")

    return groups

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--direct", default="results_4hop_trace/qwen3vl8b_4hop_direct_same.jsonl")
    ap.add_argument("--trace", default="results_4hop_trace/qwen3vl8b_4hop_trace.jsonl")
    args=ap.parse_args()

    direct=load(args.direct)
    trace=load(args.trace)
    dg=summarize(direct, "DIRECT")
    tg=summarize(trace, "COORDINATE TRACE")

    print("\n=== Same-example DIRECT vs TRACE ===")
    for name in ["TTTT","V@1","V@4"]:
        d={r["problem_id"]:r for r in dg.get(name,[])}
        t={r["problem_id"]:r for r in tg.get(name,[])}
        ids=sorted(set(d)&set(t))
        if not ids:
            continue
        da=sum(bool(d[i]["correct"]) for i in ids)/len(ids)
        ta=sum(bool(t[i]["correct"]) for i in ids)/len(ids)
        improved=sum((not d[i]["correct"]) and t[i]["correct"] for i in ids)
        worsened=sum(d[i]["correct"] and (not t[i]["correct"]) for i in ids)
        print(f"{name}: direct={da:.1%} trace={ta:.1%} trace-direct={ta-da:+.1%} | direct wrong->trace right={improved} | direct right->trace wrong={worsened}")

if __name__ == "__main__":
    main()
