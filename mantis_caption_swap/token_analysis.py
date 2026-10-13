#!/usr/bin/env python3
"""Analyze paired image-to-caption interventions: accuracy versus generated-token budget.

Uses only persisted model outputs; never loads models or consumes GPU time.
The primary 'untruncated' cohort excludes any question with a truncated condition,
rather than treating incomplete reasoning as an ordinary wrong answer.
"""
import argparse
import collections
import csv
import json
import math
import statistics
from pathlib import Path

from experiment import require_manifest, read_jsonl, schedules


def mean(values):
    return round(statistics.mean(values), 4) if values else ""


def median(values):
    return round(statistics.median(values), 4) if values else ""


def pearson(xs, ys):
    if len(xs) < 3:
        return ""
    mx, my = statistics.mean(xs), statistics.mean(ys)
    xx = sum((x-mx)**2 for x in xs)
    yy = sum((y-my)**2 for y in ys)
    if xx <= 0 or yy <= 0:
        return ""
    return round(sum((x-mx)*(y-my) for x,y in zip(xs,ys)) / math.sqrt(xx*yy), 4)


def write_csv(path, rows, fields):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:",p, "rows:",len(rows),flush=True)


def compare(gold, other):
    before=bool(gold["correct"])
    after=bool(other["correct"])
    if before and not after: return "correct_to_wrong"
    if not before and after: return "wrong_to_correct"
    if before and after: return "correct_to_correct"
    return "wrong_to_wrong"


def token_bin(delta):
    if delta <= -1001:return "<=-1001"
    if delta <= -101:return "-1000..-101"
    if delta <= -1:return "-100..-1"
    if delta==0:return "0"
    if delta <= 100:return "+1..+100"
    if delta <= 1000:return "+101..+1000"
    return ">=+1001"


def analyze(manifest_path, predictions_path, output_dir):
    manifest=require_manifest(manifest_path)
    records=read_jsonl(predictions_path)
    if not records:raise RuntimeError("No prediction records yet; run experiment.py evaluate first.")
    # An interrupted/resumed run may append several rows for the same condition.
    # Use the last successful record as the authoritative outcome.
    latest={}
    duplicates=0
    for item in records:
        if item.get("status")!="ok":
            continue
        key=(item["qid"],item["schedule"])
        if key in latest:
            duplicates+=1
        latest[key]=item
    expected={(q["qid"],s) for q in manifest for s in schedules(q["image_count"])}
    unexpected=len(set(latest)-expected)
    valid={key:r for key,r in latest.items() if key in expected}
    completed=[q for q in manifest if all((q["qid"],s) in valid for s in schedules(q["image_count"]))]
    untruncated=[q for q in completed if all(
        not valid[q["qid"],s].get("truncated",False)
        for s in schedules(q["image_count"]))]
    print(f"Manifest items={len(manifest)}, expected conditions={len(expected)}, "
          f"completed conditions={len(valid)}, duplicates={duplicates}, "
          f"unexpected={unexpected}")
    print(f"Fully paired questions={len(completed)}; fully untruncated={len(untruncated)}")
    if not completed:raise RuntimeError("No fully paired questions to analyze yet.")
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    detail=[]
    summary=[]
    flips=[]
    cohorts=[("all_complete",completed),("fully_untruncated",untruncated)]
    for cohort,questions in cohorts:
        if not questions:
            continue
        sizes=["all"]+sorted({q["image_count"] for q in questions})
        for size in sizes:
            group=[q for q in questions if size=="all" or q["image_count"]==size]
            by_condition=collections.defaultdict(list)
            by_delta=collections.defaultdict(list)
            by_flip=collections.defaultdict(list)
            for q in group:
                n=q["image_count"]
                qid=q["qid"]
                base=valid[qid,"V"*n]
                for schedule, aliases in schedules(n).items():
                    result=valid[qid,schedule]
                    # Collect each condition under its position-normalized alias.
                    for alias in aliases:
                        by_condition[alias].append(result)
                    if schedule=="V"*n:
                        continue
                    for alias in aliases:
                        delta=int(result["tokens"])-int(base["tokens"])
                        score_delta=int(bool(result["correct"]))-int(bool(base["correct"]))
                        pair={
                            "cohort":cohort,"image_count_group":size,
                            "qid":qid,"original_image_count":n,
                            "condition":alias,"schedule":schedule,
                            "all_v_correct":int(bool(base["correct"])),
                            "condition_correct":int(bool(result["correct"])),
                            "accuracy_delta":score_delta,
                            "all_v_tokens":base["tokens"],
                            "condition_tokens":result["tokens"],
                            "generated_token_delta":delta,
                            "generated_token_ratio":round(result["tokens"]/base["tokens"],4)
                                if base["tokens"] else "",
                            "all_v_truncated":bool(base.get("truncated",False)),
                            "condition_truncated":bool(result.get("truncated",False)),
                            "flip":compare(base,result),
                            "token_delta_bin":token_bin(delta),
                        }
                        by_delta[alias].append(pair)
                        by_flip[(alias,pair["flip"])].append(pair)
                        # Store one detail record per cohort for the ALL image-count group.
                        if size=="all":detail.append(pair)
            condition_order=["all_V","all_T"]+[f"T_at_{i}" for i in range(1,16)]
            for alias in condition_order:
                rs=by_condition.get(alias)
                if not rs:continue
                token_lengths=[int(x["tokens"]) for x in rs]
                correct_tokens=[int(x["tokens"]) for x in rs if x["correct"]]
                incorrect_tokens=[int(x["tokens"]) for x in rs if not x["correct"]]
                differences=by_delta.get(alias,[])
                deltas=[p["generated_token_delta"] for p in differences]
                changes=[p["accuracy_delta"] for p in differences]
                output_row={
                    "cohort":cohort,"image_count":size,"condition":alias,"n":len(rs),
                    "accuracy":round(sum(bool(x["correct"]) for x in rs)/len(rs),5),
                    "mean_generated_tokens":mean(token_lengths),
                    "median_generated_tokens":median(token_lengths),
                    "min_generated_tokens":min(token_lengths),
                    "max_generated_tokens":max(token_lengths),
                    "truncated_count":sum(bool(x.get("truncated")) for x in rs),
                    "mean_tokens_correct":mean(correct_tokens),
                    "mean_tokens_incorrect":mean(incorrect_tokens),
                    "mean_generated_token_delta_vs_all_V":mean(deltas),
                    "median_token_delta_vs_all_V":median(deltas),
                    "mean_accuracy_delta_vs_all_V":round(statistics.mean(changes),5)
                        if changes else "",
                    "token_delta_accuracy_delta_correlation":pearson(deltas,changes),
                    "correct_to_wrong":sum(p["flip"]=="correct_to_wrong" for p in differences),
                    "wrong_to_correct":sum(p["flip"]=="wrong_to_correct" for p in differences),
                }
                summary.append(output_row)
            for (alias,flip),items in sorted(by_flip.items()):
                delta=[p["generated_token_delta"] for p in items]
                flips.append({
                    "cohort":cohort,"image_count":size,"condition":alias,
                    "flip":flip,"n":len(items),
                    "mean_token_delta":mean(delta),"median_token_delta":median(delta),
                    "mean_all_v_tokens":mean([p["all_v_tokens"] for p in items]),
                    "mean_condition_tokens":mean([p["condition_tokens"] for p in items]),
                })
    write_csv(output/"token_accuracy_summary.csv",summary,[
        "cohort","image_count","condition","n","accuracy",
        "mean_generated_tokens","median_generated_tokens",
        "min_generated_tokens","max_generated_tokens","truncated_count",
        "mean_tokens_correct","mean_tokens_incorrect",
        "mean_generated_token_delta_vs_all_V","median_token_delta_vs_all_V",
        "mean_accuracy_delta_vs_all_V","token_delta_accuracy_delta_correlation",
        "correct_to_wrong","wrong_to_correct"])
    write_csv(output/"paired_token_deltas.csv",detail,[
        "cohort","image_count_group","qid","original_image_count",
        "condition","schedule","all_v_correct","condition_correct","accuracy_delta",
        "all_v_tokens","condition_tokens","generated_token_delta",
        "generated_token_ratio","all_v_truncated","condition_truncated",
        "flip","token_delta_bin"])
    write_csv(output/"token_accuracy_flips.csv",flips,[
        "cohort","image_count","condition","flip","n",
        "mean_token_delta","median_token_delta",
        "mean_all_v_tokens","mean_condition_tokens"])
    print("\nPRIMARY REPORT: fully_untruncated cohort / image_count=all")
    for r in summary:
        if r["cohort"]=="fully_untruncated" and r["image_count"]=="all":
            print(f"{r['condition']:>10} n={r['n']:>3} "
                  f"acc={r['accuracy']:.3f} mean_tokens={r['mean_generated_tokens']} "
                  f"delta_vs_AllV={r['mean_generated_token_delta_vs_all_V']}")
    print("NOTE: generated token count includes reasoning + answer. "
          "Do not interpret output length as pure thinking time.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",default="data/pilot/manifest.jsonl")
    p.add_argument("--predictions",default="results/predictions.jsonl")
    p.add_argument("--output-dir",default="results")
    a=p.parse_args()
    analyze(a.manifest,a.predictions,a.output_dir)


if __name__=="__main__":
    main()
