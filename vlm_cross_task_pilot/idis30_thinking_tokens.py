#!/usr/bin/env python3
"""Estimate Thinking/Final token counts for SAVED Idis-perception predictions.
Re-tokenizes decoded text with Qwen's original tokenizer. Does NOT load VLM weights.
Saved text cannot reproduce exact generation token IDs. The original
num_new_tokens remains the exact total generation-token count.
"""
import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

VARIANTS=("clean","aligned","conflicting","irrelevant")


def avg(values):
    return sum(values)/len(values) if values else None


def save(path,rows):
    if not rows:return
    with open(path,"w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ci_cluster(rows,seed,n=3000):
    # Resample underlying source-image IDs rather than treating 4 variants
    # derived from one image as independent observations.
    grouped=defaultdict(list)
    for r in rows:grouped[r["id"]].append(r["thinking_delta"])
    ids=sorted(grouped)
    rng=random.Random(seed)
    sampled=[]
    for _ in range(n):
        deltas=[]
        for _ in ids:deltas.extend(grouped[ids[rng.randrange(len(ids))]])
        sampled.append(avg(deltas))
    sampled.sort()
    return sampled[int((len(sampled)-1)*0.025)],sampled[int((len(sampled)-1)*0.975)]


def run(predictions,out_dir,tokenizer=None,n_boot=3000,seed=42):
    from transformers import AutoTokenizer
    raw=[json.loads(s) for s in Path(predictions).read_text(encoding="utf-8").splitlines() if s.strip()]
    if not raw:raise RuntimeError("No records")
    models=set(r["model"] for r in raw)
    if len(models)!=1:raise ValueError("Model mismatch")
    model=next(iter(models))
    tokenizer=tokenizer or AutoTokenizer.from_pretrained(model,use_fast=True)
    out=Path(out_dir)
    out.mkdir(parents=True,exist_ok=True)
    outputs=[]
    lookup={}
    for r in raw:
        key=(r["id"],r["variant"],int(r["rep"]),r["condition"])
        if key in lookup:raise ValueError(f"Duplicate: {key}")
        if r["pred"].count("</think>")!=1:
            raise ValueError(f"Missing/duplicate </think>: {key}")
        think,final=r["pred"].split("</think>",1)
        think=think.removeprefix("<think>")
        n_think=len(tokenizer.encode(think,add_special_tokens=False))
        n_final=len(tokenizer.encode(final,add_special_tokens=False))
        reenc=len(tokenizer.encode(r["pred"],add_special_tokens=False))
        exact_total=int(r["num_new_tokens"])
        row=dict(
            id=r["id"],variant=r["variant"],rep=int(r["rep"]),
            condition=r["condition"],thinking_tokens_est=n_think,
            final_tokens_est=n_final,original_total_tokens=exact_total,
            reencoded_total_tokens=reenc,reencoding_gap=reenc-exact_total,
            thinking_words=len(think.split()),final_words=len(final.split()),
            truncated=str(r["hit_max_token"]).lower()=="true",
        )
        outputs.append(row)
        lookup[key]=row
    pairs=[]
    for (id_,variant,rep,condition),single in lookup.items():
        if condition!="visual_only":continue
        multi=lookup.get((id_,variant,rep,"visual_plus_text"))
        if multi is None:raise ValueError(f"Unpaired: {id_}, {variant}")
        a=single["thinking_tokens_est"]
        b=multi["thinking_tokens_est"]
        pairs.append(dict(
            id=id_,variant=variant,rep=rep,
            single_thinking=a,multi_thinking=b,
            thinking_delta=b-a,
            thinking_compression_fraction=(1-b/a if a else None),
            single_final=single["final_tokens_est"],
            multi_final=multi["final_tokens_est"],
            final_delta=multi["final_tokens_est"]-single["final_tokens_est"],
            original_total_delta=multi["original_total_tokens"]-single["original_total_tokens"],
            truncated=single["truncated"] or multi["truncated"],
        ))
    if len(pairs)*2!=len(raw):
        raise ValueError(f"Missing pairs: {len(raw)} records, {len(pairs)} pairs")
    if any(p["truncated"] for p in pairs):
        raise ValueError("Truncation detected: report separately before interpreting compression")
    summaries=[]
    for variant in (*VARIANTS,"ALL"):
        group=[p for p in pairs if p["variant"]==variant] if variant!="ALL" else pairs
        if not group:continue
        ci=ci_cluster(group,seed,n_boot)
        summaries.append(dict(
            variant=variant,n_pairs=len(group),
            mean_thinking_only=avg([p["single_thinking"] for p in group]),
            mean_thinking_with_q2=avg([p["multi_thinking"] for p in group]),
            mean_thinking_delta=avg([p["thinking_delta"] for p in group]),
            median_thinking_delta=statistics.median(p["thinking_delta"] for p in group),
            reduction_of_mean_thinking=(1-sum(p["multi_thinking"] for p in group)/
                                         sum(p["single_thinking"] for p in group)),
            mean_individual_compression=avg([p["thinking_compression_fraction"]
                                             for p in group if p["thinking_compression_fraction"] is not None]),
            shorter_thinking=sum(p["thinking_delta"]<0 for p in group),
            longer_thinking=sum(p["thinking_delta"]>0 for p in group),
            mean_final_delta=avg([p["final_delta"] for p in group]),
            mean_total_delta=avg([p["original_total_delta"] for p in group]),
            thinking_delta_ci_95_low=ci[0],thinking_delta_ci_95_high=ci[1],
        ))
    save(out/"retokenized_outputs.csv",outputs)
    save(out/"retokenized_pairs.csv",pairs)
    save(out/"retokenized_summary.csv",summaries)
    save(out/"top_expansions.csv",sorted(pairs,key=lambda r:r["thinking_delta"],reverse=True)[:15])
    save(out/"top_compressions.csv",sorted(pairs,key=lambda r:r["thinking_delta"])[:15])
    gaps=[abs(r["reencoding_gap"]) for r in outputs]
    print(f"Records={len(raw)}  Pairs={len(pairs)}  Source images={len(set(p['id'] for p in pairs))}")
    print(f"Tokenizer={model}; reencoding original-total gap: mean={avg(gaps):.2f}, median={statistics.median(gaps)}, max={max(gaps)}")
    for s in summaries:
        print(f"{s['variant']:12} N={s['n_pairs']:3}  Thinking {s['mean_thinking_only']:.1f} -> {s['mean_thinking_with_q2']:.1f}  "
              f"delta={s['mean_thinking_delta']:+.1f}  shorter={s['shorter_thinking']}/{s['n_pairs']}  "
              f"95% cluster bootstrap CI=[{s['thinking_delta_ci_95_low']:+.1f},{s['thinking_delta_ci_95_high']:+.1f}]")
    print(f"Saved results: {out}")
    print("Caution: Thinking token counts are re-encoded estimates (original token IDs not saved).")
    print("With-Q2 thinking contains reasoning for BOTH Q1 and Q2; Q1-only compression is not identified.")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--predictions",default="idis30/results/predictions.jsonl")
    p.add_argument("--out-dir",default="idis30/results/thinking_token_analysis")
    p.add_argument("--bootstrap",type=int,default=3000)
    a=p.parse_args()
    run(a.predictions,a.out_dir,n_boot=a.bootstrap)


if __name__=="__main__":
    main()
