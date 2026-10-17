#!/usr/bin/env python3
"""Vector geometry and paired accuracy for paper-native Gemma3 steering."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np


def cos(a,b):
    x,y=a.float().flatten(),b.float().flatten()
    return float(torch.nn.functional.cosine_similarity(x[None],y[None])[0])


def write_csv(path, data):
    if not data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


def analyze(data, split):
    report=data / ("analysis_"+split)
    report.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((data/"vectors"/"manifest.json").read_text())
    geo=[]
    for l in manifest["layers"]:
        vs={(c,lang):torch.load(data/"vectors"/c/lang/f"layer_{l}_feature.pt",
                                weights_only=True,map_location="cpu")
            for c in ("counting","spatial") for lang in ("en","zh")}
        for c in ("counting","spatial"):
            geo.append({"concept":c,"layer":l,
                        "cos_en_zh":cos(vs[(c,"en")],vs[(c,"zh")]),
                        "norm_en":float(vs[(c,"en")].norm()),
                        "norm_zh":float(vs[(c,"zh")].norm())})
        for lang in ("en","zh"):
            geo.append({"concept":"counting_vs_spatial","layer":l,
                        "lang":lang,"cos_en_zh":cos(vs[("counting",lang)],
                                                   vs[("spatial",lang)])})
    write_csv(report/"vector_geometry.csv",geo)

    with (data/f"predictions_{split}.csv").open(encoding="utf-8",newline="") as fp:
        rows=list(csv.DictReader(fp))
    baselines={}
    groups=defaultdict(list)
    for r in rows:
        key=(r["id"],r["task"],r["prompt_lang"])
        if r["vector_lang"]=="none":
            if key in baselines:
                raise ValueError(f"Duplicate baseline: {key}")
            baselines[key]=int(r["correct"])
        else:
            g=(r["task"],r["prompt_lang"],r["vector_lang"],r["layer"],r["alpha"],r["mask"])
            groups[g].append(r)
    summary=[]
    rng=np.random.default_rng(42)
    for key,values in sorted(groups.items()):
        diffs=[]
        points=[]
        seen=set()
        for v in values:
            k=(v["id"],v["task"],v["prompt_lang"])
            if k in seen or k not in baselines:
                raise ValueError(f"Unpaired or duplicate example {k}")
            seen.add(k)
            pred=int(v["correct"])
            points.append(pred)
            diffs.append(pred-baselines[k])
        arr=np.array(diffs,dtype=float)
        boots=np.array([rng.choice(arr,size=len(arr),replace=True).mean()
                        for _ in range(1000)])
        summary.append(dict(zip(("task","prompt_lang","vector_lang","layer","alpha","mask"),key),
                            n=len(values),accuracy=float(np.mean(points)),
                            gain_pp=float(100*arr.mean()),
                            ci_low_pp=float(100*np.quantile(boots,0.025)),
                            ci_high_pp=float(100*np.quantile(boots,0.975))))
    write_csv(report/"paired_accuracy.csv",summary)
    print("Geometry:",json.dumps(geo,indent=2))
    print(f"Saved {report}/vector_geometry.csv and paired_accuracy.csv")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="outputs/gemma3")
    p.add_argument("--split",choices=["val","test"],default="val")
    a=p.parse_args()
    analyze(Path(a.data),a.split)


if __name__=="__main__":
    main()
