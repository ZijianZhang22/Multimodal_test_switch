#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict

def answer_from_xy(x, y):
    if x == 0 and y == 0:
        return "SAME"
    v = "NORTH" if y < 0 else "SOUTH" if y > 0 else ""
    h = "EAST" if x > 0 else "WEST" if x < 0 else ""
    return v + h if v and h else v or h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default="data_4hop_trace/latent_problems.jsonl")
    ap.add_argument("--results", default="results_4hop_trace/qwen3vl8b_4hop_trace.jsonl")
    args = ap.parse_args()

    latents = {}
    with open(args.latents, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                latents[p["problem_id"]] = p

    rows = []
    with open(args.results, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            cond = "TTTT" if r["condition"] == "text_baseline" else f"V@{r['visual_logical_step']}"
            p = latents[r["problem_id"]]
            gold = {
                int(fact["logical_step"]): (int(fact["state_x"]), int(fact["state_y"]))
                for fact in p["facts"]
            }

            coords = {}
            for m in re.finditer(r"N(\d+)\s*=\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", r["raw_output"]):
                coords[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))

            row = {
                "condition": cond,
                "problem_id": r["problem_id"],
                "answer_correct": int(r["correct"]),
            }
            for k in range(1, 5):
                row[f"s{k}_parsed"] = int(k in coords)
                row[f"s{k}_correct"] = int(k in coords and coords[k] == gold[k])

            if 4 in coords:
                implied = answer_from_xy(*coords[4])
                row["final_coord_answer_correct"] = int(implied == r["answer"])
                row["readout_consistent"] = int(implied == r["prediction"])
            else:
                row["final_coord_answer_correct"] = 0
                row["readout_consistent"] = 0
            rows.append(row)

    for cond in ["TTTT", "V@1", "V@4"]:
        g = [r for r in rows if r["condition"] == cond]
        print(f"\n=== {cond} ===")
        for k in range(1, 5):
            n = sum(r[f"s{k}_correct"] for r in g)
            print(f"s{k}: {n}/{len(g)} = {n/len(g):.1%}")
        print(
            "final coordinate implies correct answer:",
            f"{sum(r['final_coord_answer_correct'] for r in g)/len(g):.1%}"
        )
        print(
            "FINAL consistent with own coordinate:",
            f"{sum(r['readout_consistent'] for r in g)/len(g):.1%}"
        )
        print(
            "actual final accuracy:",
            f"{sum(r['answer_correct'] for r in g)/len(g):.1%}"
        )

if __name__ == "__main__":
    main()
