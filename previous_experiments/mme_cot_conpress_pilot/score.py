#!/usr/bin/env python3
"""Paired baseline vs +1+1 statistics; strict MC grading and readable case review."""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def extract_tag(text, tag):
    # Only accept a complete final answer tag, never mine numbers from reasoning.
    occurrences = re.findall(
        rf"<{tag}\s*>(.*?)</{tag}\s*>", text or "",
        flags=re.DOTALL | re.IGNORECASE,
    )
    return occurrences[-1].strip() if occurrences else None


def clean(s):
    return re.sub(r"\s+", " ", str(s or "").strip()).casefold()


def score_target(answer, gold, options):
    """Returns (correct_or_null, grading_type). Null = requires manual judge."""
    if answer is None:
        return None, "missing_target_tag"
    result = clean(answer).strip(" .")
    reference = clean(gold).strip(" .")
    if options and re.fullmatch("[a-l]", reference):
        # Exact letter or "B. some option"; not a random 'B' within explanation.
        found = re.match(r"^\s*([a-l])(?:[\.\):]|\b)\s*", result)
        if found:
            return found.group(1) == reference, "multiple_choice"
        return None, "ambiguous_multiple_choice"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", result.replace(",", "")) and re.fullmatch(
        r"[-+]?\d+(?:\.\d+)?", reference.replace(",", "")
    ):
        from decimal import Decimal
        return Decimal(result.replace(",", "")) == Decimal(reference.replace(",", "")), "numeric"
    # Open-ended exact equality is a useful but conservative proxy, NOT the
    # official semantic-judge correctness metric.
    return (result == reference), "strict_text_match_unofficial"


def safe_rate(arr):
    valid = [float(x) for x in arr if x is not None]
    return (sum(valid) / len(valid)) if valid else None


def mean(arr):
    return sum(arr) / len(arr) if arr else None


def write_csv(path, rows, headers):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in headers})


def build_rows(records):
    scored = []
    completed = {}
    for rec in records:
        if rec.get("stop_reason") != "model_eos":
            continue
        ans = extract_tag(rec.get("text_output", ""), "target_answer")
        extra = extract_tag(rec.get("text_output", ""), "arithmetic_answer")
        correct, scoring_type = score_target(
            ans, rec["gold_answer"], rec.get("options", []),
        )
        scored_rec = {
            "id": rec["id"],
            "category": rec["category"],
            "subcategory": rec.get("subcategory", ""),
            "condition": rec["condition"],
            "gold_answer": rec["gold_answer"],
            "target_pred": ans,
            "target_correct": correct,
            "target_grade_type": scoring_type,
            "arithmetic_pred": extra,
            "arithmetic_correct": (clean(extra) in ("2", "2.0", "two")) if extra is not None else None,
            "generation_tokens": rec["generation_tokens"],
            "thinking_tokens": rec.get("thinking_tokens"),
            "has_thinking_tags": rec.get("has_thinking_tags", False),
            "stop_reason": rec["stop_reason"],
            "seconds": rec.get("seconds"),
        }
        scored.append(scored_rec)
        # Latest completed record for each pair/condition wins, avoiding stale retries.
        completed[(str(rec["id"]), rec["condition"])] = scored_rec
    paired = []
    for (id_, condition), base in completed.items():
        if condition != "baseline":
            continue
        other = completed.get((id_, "plus_arithmetic"))
        if other is None:
            continue
        a = base["thinking_tokens"]
        b = other["thinking_tokens"]
        if a is not None and b is not None and a > 0:
            change = (b - a) / a * 100
            compression = -change
            t_delta = b - a
        else:
            change = compression = t_delta = None
        paired.append({
            "id": id_, "category": base["category"], "subcategory": base["subcategory"],
            "baseline_thinking_tokens": a, "joint_thinking_tokens": b,
            "thinking_token_delta": t_delta, "thinking_pct_change": change,
            "compression_pct": compression,
            "baseline_output_tokens": base["generation_tokens"],
            "joint_output_tokens": other["generation_tokens"],
            "output_token_delta": other["generation_tokens"] - base["generation_tokens"],
            "baseline_target_pred": base["target_pred"],
            "joint_target_pred": other["target_pred"],
            "gold_answer": base["gold_answer"],
            "baseline_correct": base["target_correct"],
            "joint_correct": other["target_correct"],
            "grade_type": base["target_grade_type"],
            "arithmetic_pred": other["arithmetic_pred"],
            "arithmetic_correct": other["arithmetic_correct"],
            "baseline_ended_normally": base["stop_reason"] == "model_eos",
            "joint_ended_normally": other["stop_reason"] == "model_eos",
        })
    return scored, paired


def build_summary(paired):
    groups = defaultdict(list)
    for row in paired:
        groups["ALL"].append(row)
        groups[row["category"]].append(row)
    summary = []
    for category, rows in sorted(groups.items()):
        thought_pairs = [r for r in rows if r["compression_pct"] is not None]
        exact_mc_pairs = [r for r in rows if r["grade_type"] == "multiple_choice" and
                          r["baseline_correct"] is not None and r["joint_correct"] is not None]
        summary.append({
            "category": category,
            "pairs": len(rows),
            "pairs_with_thinking": len(thought_pairs),
            "avg_baseline_thinking_tokens": mean([r["baseline_thinking_tokens"] for r in thought_pairs]),
            "avg_joint_thinking_tokens": mean([r["joint_thinking_tokens"] for r in thought_pairs]),
            "mean_paired_compression_pct": mean([r["compression_pct"] for r in thought_pairs]),
            "median_paired_compression_pct": (sorted(r["compression_pct"] for r in thought_pairs)
                                              [len(thought_pairs)//2] if thought_pairs else None),
            "mean_output_token_delta": mean([r["output_token_delta"] for r in rows]),
            "mc_pairs": len(exact_mc_pairs),
            "mc_baseline_accuracy": safe_rate([r["baseline_correct"] for r in exact_mc_pairs]),
            "mc_joint_accuracy": safe_rate([r["joint_correct"] for r in exact_mc_pairs]),
            "plus_1plus1_accuracy": safe_rate([r["arithmetic_correct"] for r in rows]),
        })
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="pilot_results/predictions.jsonl")
    ap.add_argument("--out", default="pilot_results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = [json.loads(x) for x in Path(args.input).read_text(encoding="utf-8").splitlines()
               if x.strip()]
    scored, paired = build_rows(records)
    fields_scored = ["id", "category", "subcategory", "condition", "gold_answer",
                     "target_pred", "target_correct", "target_grade_type",
                     "arithmetic_pred", "arithmetic_correct", "generation_tokens",
                     "thinking_tokens", "has_thinking_tags", "stop_reason", "seconds"]
    fields_pairs = ["id", "category", "subcategory", "baseline_thinking_tokens",
                    "joint_thinking_tokens", "thinking_token_delta", "thinking_pct_change",
                    "compression_pct", "baseline_output_tokens", "joint_output_tokens",
                    "output_token_delta", "baseline_target_pred", "joint_target_pred",
                    "gold_answer", "baseline_correct", "joint_correct", "grade_type",
                    "arithmetic_pred", "arithmetic_correct",
                    "baseline_ended_normally", "joint_ended_normally"]
    write_csv(out / "scored.csv", scored, fields_scored)
    write_csv(out / "paired.csv", paired, fields_pairs)
    summary = build_summary(paired)
    fields_summary = ["category", "pairs", "pairs_with_thinking",
                      "avg_baseline_thinking_tokens", "avg_joint_thinking_tokens",
                      "mean_paired_compression_pct", "median_paired_compression_pct",
                      "mean_output_token_delta", "mc_pairs", "mc_baseline_accuracy",
                      "mc_joint_accuracy", "plus_1plus1_accuracy"]
    write_csv(out / "summary.csv", summary, fields_summary)
    errors = [r for r in records if r.get("stop_reason") != "model_eos"]
    print(f"Complete predictions: {len(scored)}, paired comparisons: {len(paired)}, errors: {len(errors)}")
    for s in summary:
        v = s["mean_paired_compression_pct"]
        label = "N/A" if v is None else f"{v:+.1f}%"
        print(f"  {s['category']}: {s['pairs']} pairs, mean thinking compression={label}, "
              f"MC baseline={s['mc_baseline_accuracy']}, MC +1+1={s['mc_joint_accuracy']}")
    print("Saved: predictions.jsonl, scored.csv, paired.csv, summary.csv")
    print("IMPORTANT: free-text strict matches are NOT official MME-CoT judge accuracy.")


if __name__ == "__main__":
    main()
