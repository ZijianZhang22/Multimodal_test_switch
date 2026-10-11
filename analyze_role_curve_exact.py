#!/usr/bin/env python3
"""Exact matched tests for the 4-hop single-Visual role curve."""

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


def load_jsonl(path):
    rows=[]
    with Path(path).open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def exact_mcnemar(n01, n10):
    n=n01+n10
    if n==0:
        return 1.0
    return binomtest(min(n01,n10), n=n, p=0.5, alternative="two-sided").pvalue


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results",required=True)
    ap.add_argument("--out_dir",type=Path,default=Path("analysis_full_roles"))
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)

    df=load_jsonl(args.results)
    df["correct"]=df["correct"].astype(int)
    base=df[df["condition"]=="text_baseline"][
        ["problem_id","order_id","hop_count","correct"]
    ].rename(columns={"correct":"text_correct"})

    single=df[df["condition"]=="single_visual"].copy()
    matched=single.merge(
        base,
        on=["problem_id","order_id","hop_count"],
        how="inner",
        validate="many_to_one",
    )

    rows=[]
    for role,g in matched.groupby("visual_logical_step"):
        n=len(g)
        vacc=g["correct"].mean()
        tacc=g["text_correct"].mean()
        n01=int(((g["text_correct"]==0)&(g["correct"]==1)).sum())
        n10=int(((g["text_correct"]==1)&(g["correct"]==0)).sum())
        rows.append({
            "visual_logical_step":int(role),
            "n":n,
            "text_accuracy":tacc,
            "single_visual_accuracy":vacc,
            "vision_minus_text":vacc-tacc,
            "text_wrong_vision_right":n01,
            "text_right_vision_wrong":n10,
            "discordant_pairs":n01+n10,
            "mcnemar_exact_p":exact_mcnemar(n01,n10),
        })

    out=pd.DataFrame(rows).sort_values("visual_logical_step")
    out.to_csv(args.out_dir/"exact_mcnemar_by_role.csv",index=False)
    print(out.to_string(index=False))

    if len(out)>=2:
        first=out.iloc[0]
        last=out.iloc[-1]
        print(
            "\nEarly-to-late penalty change:",
            f"{last['vision_minus_text']-first['vision_minus_text']:+.1%}",
        )


if __name__=="__main__":
    main()
