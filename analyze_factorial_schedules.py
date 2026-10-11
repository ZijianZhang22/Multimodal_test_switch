#!/usr/bin/env python3
"""Analyze arbitrary-hop full-factorial Text/Vision schedule experiments."""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

def load(path):
    rows=[]
    with Path(path).open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results",required=True)
    ap.add_argument("--out_dir",type=Path,default=Path("analysis_factorial"))
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    df=load(args.results)
    df["correct"]=df["correct"].astype(int)
    if "schedule" not in df.columns:
        df["schedule"]=df.get("logical_schedule",df.get("presentation_schedule"))
    h=int(df["hop_count"].iloc[0]) if "hop_count" in df else len(df["schedule"].iloc[0])
    for k in range(1,h+1):
        df[f"V{k}"]=df["schedule"].str[k-1].eq("V").astype(int)
    if "switch_count" not in df:
        df["switch_count"]=df["schedule"].map(lambda s:sum(s[i]!=s[i+1] for i in range(len(s)-1)))
    if "vision_count" not in df:
        df["vision_count"]=df["schedule"].str.count("V")

    by_sched=(df.groupby(["schedule","switch_count","vision_count"],as_index=False)
              .agg(n=("correct","size"),accuracy=("correct","mean"))
              .sort_values(["vision_count","switch_count","schedule"]))
    by_sched.to_csv(args.out_dir/"accuracy_by_schedule.csv",index=False)
    by_switch=(df.groupby("switch_count",as_index=False).agg(n=("correct","size"),accuracy=("correct","mean")))
    by_switch.to_csv(args.out_dir/"accuracy_by_switch_count.csv",index=False)
    by_vc=(df.groupby("vision_count",as_index=False).agg(n=("correct","size"),accuracy=("correct","mean")))
    by_vc.to_csv(args.out_dir/"accuracy_by_vision_count.csv",index=False)

    X=df[[f"V{k}" for k in range(1,h+1)]+["switch_count"]].astype(float)
    X=sm.add_constant(X); y=df["correct"]
    try:
        res=sm.GLM(y,X,family=sm.families.Binomial()).fit(
            cov_type="cluster",cov_kwds={"groups":df["problem_id"]})
        tab=pd.DataFrame({
            "coef_log_odds":res.params,"std_err_clustered":res.bse,
            "z":res.tvalues,"p_value":res.pvalues,
            "odds_ratio":np.exp(res.params),
            "ci_low_odds_ratio":np.exp(res.conf_int()[0]),
            "ci_high_odds_ratio":np.exp(res.conf_int()[1]),
        })
        tab.to_csv(args.out_dir/"clustered_logit_role_plus_switch.csv")
    except Exception as e:
        print("WARNING regression failed:",e)

    # Matched marginal effect of making each logical role Vision while averaging over all other schedules.
    rows=[]
    for k in range(1,h+1):
        per=(df.groupby(["problem_id",f"V{k}"])["correct"].mean().unstack().dropna())
        effect=(per[1]-per[0]).mean()
        rows.append({
            "visual_logical_step":k,"n_problems":len(per),
            "mean_accuracy_if_text":per[0].mean(),
            "mean_accuracy_if_vision":per[1].mean(),
            "paired_vision_minus_text":effect,
        })
    pos=pd.DataFrame(rows)
    pos.to_csv(args.out_dir/"position_effects.csv",index=False)

    print("\n=== Accuracy by schedule ===")
    print(by_sched.to_string(index=False))
    print("\n=== Position-specific marginal Vision effects ===")
    print(pos.to_string(index=False))
    print("\n=== Accuracy by switch count ===")
    print(by_switch.to_string(index=False))
    print("\nSaved:",args.out_dir)

if __name__=="__main__":
    main()
