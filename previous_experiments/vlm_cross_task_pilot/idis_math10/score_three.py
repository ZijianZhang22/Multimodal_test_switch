#!/usr/bin/env python3
"""Compare Idis-math A(original), B(visual_first), C(easy_first).

Reads old 50-question A/C file and ONLY newly generated B file. Never alters
old predictions. Exact thinking_tokens from generation; excludes truncation
from valid thinking comparisons. Accuracy is conservative and reviewable.
"""
import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from score import final_tag,box_extract,auto_score,decimal_of

CONDITIONS=("original","visual_first","easy_first")
SHORT={"original":"A","visual_first":"B","easy_first":"C"}


def records(path):
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing original predictions or new Condition B: {path}")
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip()]


def write(path, rows, fields):
    with Path(path).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def avg(xs):
    return sum(xs)/len(xs) if xs else None


def bootstrap_ci(vals,seed=42,n=3000):
    if len(vals)<2:return None,None
    rng=random.Random(seed)
    means=[]
    for _ in range(n):
        means.append(avg([vals[rng.randrange(len(vals))] for _ in vals]))
    means.sort()
    return means[int((n-1)*.025)],means[int((n-1)*.975)]


def final_segment(raw):
    text=raw.get("prediction","")
    # Only score answer tags after the end of the thinking region;
    # do not count a speculative answer mentioned inside CoT.
    if "</think>" in text:
        return text.split("</think>",1)[1]
    return ""


def extracted_answer(raw):
    final=final_segment(raw)
    return final_tag(final,"math_answer") or box_extract(final)


def warmup_answer(raw):
    final=final_segment(raw)
    return final_tag(final,"warmup")


def score_three(existing, visual_first, out_dir, seed=42):
    baseline=records(existing)
    extra=records(visual_first)
    out=Path(out_dir)
    out.mkdir(parents=True,exist_ok=True)
    if not baseline or not extra:
        raise ValueError("Both existing A/C and new B predictions must be nonempty")
    all_records=baseline+extra
    key_map={}
    for r in all_records:
        key=(str(r["sample_index"]),int(r.get("rep",0)),r["condition"])
        if key in key_map:raise ValueError(f"Duplicate record: {key}")
        key_map[key]=r
    if any(r["condition"] not in ("original","easy_first") for r in baseline):
        raise ValueError("Existing 50-question results must contain only original and easy_first")
    if any(r["condition"]!="visual_first" for r in extra):
        raise ValueError("New predictions must contain only visual_first")
    keys={(str(r["sample_index"]),int(r.get("rep",0))) for r in baseline}
    if keys != {(str(r["sample_index"]),int(r.get("rep",0))) for r in extra}:
        raise ValueError("Old and new prediction IDs do not match exactly")
    for sid,rep in keys:
        triple=[key_map.get((sid,rep,c)) for c in CONDITIONS]
        if any(r is None for r in triple):
            raise ValueError(f"Incomplete triple: sample={sid} rep={rep}")
        a,b,c=triple
        for other in (b,c):
            # Ensure identical underlying image/question/model/sampling conditions;
            # prompt and answer structure are *intentionally* different.
            for field in ("image","question","gold","model","temperature",
                          "top_p","max_new_tokens","max_image_side","seed"):
                if a.get(field)!=other.get(field):
                    raise ValueError(f"{sid} rep={rep}: {field} differs between {a['condition']} and {other['condition']}")

    manual_file=out/"manual_labels_abc.csv"
    existing_labels=Path(existing).parent/"manual_labels.csv"
    if not manual_file.exists():
        saved={}
        if existing_labels.is_file():
            with existing_labels.open(newline="",encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    saved[(str(r["sample_index"]),int(r["rep"]))]=r
        templ=[]
        for sid,rep in sorted(keys):
            prev=saved.get((sid,rep),{})
            templ.append(dict(
                sample_index=sid,rep=rep,
                original_correct=prev.get("original_correct",""),
                visual_first_correct="",
                easy_first_correct=prev.get("easy_first_correct",""),
                notes=prev.get("notes","")
            ))
        write(manual_file,templ,["sample_index","rep","original_correct",
                                "visual_first_correct","easy_first_correct","notes"])
    manual={}
    with manual_file.open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            manual[(str(row["sample_index"]),int(row["rep"]))]=row

    output_rows=[]
    pair_rows=[]
    review_rows=[]
    for sid,rep in sorted(keys):
        group={c:key_map[(sid,rep,c)] for c in CONDITIONS}
        labels=manual.get((sid,rep),{})
        graded={}
        answers={}
        for condition in CONDITIONS:
            r=group[condition]
            pred=extracted_answer(r)
            auto,method=auto_score(pred,r["gold"])
            label=str(labels.get(condition+"_correct","")).strip()
            if label in ("0","1"):
                correct=int(label)
                method="manual"
            elif auto is True:
                correct=1
            else:
                correct=None
            graded[condition]=correct
            answers[condition]=pred
            warmup=warmup_answer(r) if condition!="original" else None
            warmup_correct=(
                int(decimal_of(warmup)==decimal_of("40"))
                if warmup is not None and decimal_of(warmup) is not None else None
            )
            output_rows.append(dict(
                sample_index=sid,rep=rep,condition=condition,
                problem_version=r.get("problem_version",""),
                gold=r["gold"],answer=pred,auto_method=method,correct=correct,
                warmup=warmup,warmup_correct=warmup_correct,
                thinking_tokens=r.get("thinking_tokens"),
                total_tokens=r["num_new_tokens"],
                final_tokens=r.get("final_tokens"),
                truncated=r.get("hit_max_token",False),
                has_thinking_end=r.get("has_thinking_end",False)
            ))
        a,b,c=(group[k] for k in CONDITIONS)
        ta,tb,tc=(r.get("thinking_tokens") for r in (a,b,c))
        ok_thinking=all(t is not None and not r.get("hit_max_token",False)
                        for t,r in ((ta,a),(tb,b),(tc,c)))
        def thinking_delta(x,y):
            if x is None or y is None:return None
            return y-x
        # Old A/C may have truncations. Only report complete three-way
        # thinking comparisons so pair-level comparisons use same samples.
        if not ok_thinking:ta=tb=tc=None
        record=dict(
            sample_index=sid,rep=rep,problem_version=a.get("problem_version",""),
            gold=a["gold"],
            A_total=a["num_new_tokens"],B_total=b["num_new_tokens"],C_total=c["num_new_tokens"],
            A_thinking=ta,B_thinking=tb,C_thinking=tc,
            B_minus_A_thinking=thinking_delta(ta,tb),
            C_minus_A_thinking=thinking_delta(ta,tc),
            B_minus_C_thinking=thinking_delta(tc,tb),
            B_vs_A_compression=(1-tb/ta if ta else None),
            C_vs_A_compression=(1-tc/ta if ta else None),
            any_truncated=any(r.get("hit_max_token",False) for r in (a,b,c)),
            A_answer=answers["original"],B_answer=answers["visual_first"],
            C_answer=answers["easy_first"],
            A_correct=graded["original"],B_correct=graded["visual_first"],
            C_correct=graded["easy_first"]
        )
        pair_rows.append(record)
        if any(graded[co] is None for co in CONDITIONS) or record["any_truncated"]:
            review_rows.append(record)

    if not pair_rows:raise RuntimeError("Zero complete triples")
    write(out/"predictions_abc_scored.csv",output_rows,list(output_rows[0]))
    write(out/"paired_abc.csv",pair_rows,list(pair_rows[0]))
    write(out/"needs_review_abc.csv",review_rows,list(pair_rows[0]))

    complete=[x for x in pair_rows if not x["any_truncated"]]
    summary=[]
    for label in ("B_minus_A","C_minus_A","B_minus_C"):
        deltas=[r[label+"_thinking"] for r in complete if r[label+"_thinking"] is not None]
        lo,hi=bootstrap_ci(deltas,seed)
        summary.append(dict(
            contrast=label,questions=len(pair_rows),valid_thinking_pairs=len(deltas),
            mean_thinking_delta=avg(deltas),
            median_thinking_delta=statistics.median(deltas) if deltas else None,
            thinking_shorter=sum(d<0 for d in deltas),
            thinking_longer=sum(d>0 for d in deltas),
            ci_95_low=lo,ci_95_high=hi,
            mean_total_token_delta=avg([r[label[0]+"_total"]-r[label[-1]+"_total"] for r in pair_rows])
        ))
    write(out/"summary_abc.csv",summary,list(summary[0]))

    fully_graded=[r for r in pair_rows if all(r[x+"_correct"] is not None for x in ("A","B","C"))]
    print(f"Dataset: {len(pair_rows)} same-image questions; 3 conditions each.")
    print(f"Math accuracy: {len(fully_graded)} fully graded triples; "
          f"{len(review_rows)} triples still need review/truncation handling.")
    if fully_graded:
        for cond in ("A","B","C"):
            print(f"  {cond} accuracy (graded triple subset): {avg([r[cond+'_correct'] for r in fully_graded]):.1%}")
    else:
        print("  Accuracy unknown until manual_labels_abc.csv is reviewed")
    for r in summary:
        print(f"{r['contrast']:12} thinking delta {r['mean_thinking_delta']:+.1f} "
              f"median {r['median_thinking_delta']:+.1f}, "
              f"shorter={r['thinking_shorter']}/{r['valid_thinking_pairs']}; "
              f"CI [{r['ci_95_low']:+.1f}, {r['ci_95_high']:+.1f}]"
              if r["valid_thinking_pairs"] else f"{r['contrast']}: no valid thinking pairs")
    print(f"Files: {out}")
    print("Interpretation: thinking spans ALL tasks, not isolated visual-math reasoning.")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--existing",default="idis_math50_new/results/predictions.jsonl")
    p.add_argument("--visual-first",default="idis_math50_new/results/visual_first_predictions.jsonl")
    p.add_argument("--out-dir",default="idis_math50_new/results/abc_comparison")
    args=p.parse_args()
    score_three(args.existing,args.visual_first,args.out_dir)


if __name__=="__main__":
    main()
