#!/usr/bin/env python3
import json
from collections import defaultdict

DIRECT="results_4hop_trace/qwen3vl8b_4hop_direct_same.jsonl"
TRACE="results_4hop_trace/qwen3vl8b_4hop_trace.jsonl"

def load(path):
    rows=[]
    with open(path,encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r=json.loads(line)
            cond="TTTT" if r["condition"]=="text_baseline" else f"V@{r['visual_logical_step']}"
            r["_cond"]=cond
            rows.append(r)
    return rows

def summarize(rows,name):
    print(f"\n=== {name} ===")
    groups=defaultdict(list)
    for r in rows:
        groups[r["_cond"]].append(r)
    for cond in ["TTTT","V@1","V@4"]:
        g=groups.get(cond,[])
        if not g:
            continue
        acc=sum(bool(x["correct"]) for x in g)/len(g)
        valid=sum(x.get("prediction") is not None for x in g)/len(g)
        toks=sum(x.get("output_tokens",0) for x in g)/len(g)
        print(f"{cond}: acc={acc:.1%} valid={valid:.1%} mean_tokens={toks:.1f}")
    base={r["problem_id"]:r for r in groups.get("TTTT",[])}
    for cond in ["V@1","V@4"]:
        g=groups.get(cond,[])
        ds=[]; wr=rw=0
        for r in g:
            b=base.get(r["problem_id"])
            if b is None: continue
            bv=int(bool(b["correct"])); rv=int(bool(r["correct"]))
            ds.append(rv-bv)
            if bv==0 and rv==1: wr+=1
            if bv==1 and rv==0: rw+=1
        if ds:
            print(f"{cond}-TTTT delta={sum(ds)/len(ds):+.1%} | T wrong->V right={wr} | T right->V wrong={rw}")

def matched_mode_flips(direct,trace):
    print("\n=== Same-example Direct vs Trace ===")
    D={(r["problem_id"],r["_cond"]):r for r in direct}
    T={(r["problem_id"],r["_cond"]):r for r in trace}
    for cond in ["TTTT","V@1","V@4"]:
        keys=[k for k in D if k[1]==cond and k in T]
        if not keys: continue
        dacc=sum(D[k]["correct"] for k in keys)/len(keys)
        tacc=sum(T[k]["correct"] for k in keys)/len(keys)
        d0t1=sum((not D[k]["correct"]) and T[k]["correct"] for k in keys)
        d1t0=sum(D[k]["correct"] and (not T[k]["correct"]) for k in keys)
        print(f"{cond}: direct={dacc:.1%}, trace={tacc:.1%}, trace-direct={tacc-dacc:+.1%}, direct wrong->trace right={d0t1}, direct right->trace wrong={d1t0}")

direct=load(DIRECT)
trace=load(TRACE)
summarize(direct,"DIRECT")
summarize(trace,"COORDINATE TRACE")
matched_mode_flips(direct,trace)
