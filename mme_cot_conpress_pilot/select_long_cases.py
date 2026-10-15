#!/usr/bin/env python3
"""Choose long MME-CoT questions using baseline-only screening."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--min-thinking", type=int, default=3000)
    p.add_argument("--max-long", type=int, default=24)
    p.add_argument("--short-controls", type=int, default=8)
    p.add_argument("--short-max", type=int, default=1000)
    a = p.parse_args()
    if a.min_thinking <= 0 or a.max_long <= 0 or a.short_controls < 0:
        p.error("Invalid selection thresholds/counts")

    examples = load_jsonl(a.data)
    latest = {}
    for rec in load_jsonl(a.predictions):
        if rec.get("condition") != "baseline":
            continue
        key = str(rec.get("id"))
        # Favor completed screening generations over earlier capped retries.
        if (key not in latest or rec.get("stop_reason") == "model_eos"
                or latest[key].get("stop_reason") != "model_eos"):
            latest[key] = rec

    longs, shorts, counts = [], [], Counter()
    for row in examples:
        rec = latest.get(str(row["id"]))
        if not rec:
            counts["not_run"] += 1
            continue
        status = rec.get("stop_reason")
        thinking = rec.get("thinking_tokens")
        if status == "token_limit":
            # Right-censored, not a completed CoT token count.
            longs.append(("screen_capped", int(rec.get("generation_tokens") or 0), row, rec))
            counts["screen_capped"] += 1
        elif status == "model_eos" and isinstance(thinking, (int, float)):
            length = int(thinking)
            if length >= a.min_thinking:
                longs.append(("long_eos", length, row, rec))
                counts["long_eos"] += 1
            elif length <= a.short_max:
                shorts.append(("short_control", length, row, rec))
                counts["short_eligible"] += 1
            else:
                counts["middle_length"] += 1
        else:
            counts["unusable_baseline"] += 1

    # Prioritize capped cases, then the longest completed reasoning traces.
    longs.sort(key=lambda r: (r[0] == "screen_capped", r[1]), reverse=True)
    picked_long = longs[:a.max_long]
    shorts.sort(key=lambda r: (r[2]["category"], r[1], str(r[2]["id"])))
    controls = []
    while shorts and len(controls) < a.short_controls:
        seen = {r[2]["category"] for r in controls}
        choice = next((i for i, r in enumerate(shorts)
                       if r[2]["category"] not in seen), 0)
        controls.append(shorts.pop(choice))

    selected = picked_long + controls
    if not picked_long:
        raise SystemExit("No long questions: increase SCREEN_EXAMPLES or lower MIN_THINKING.")
    dest, manifest = Path(a.out), Path(a.manifest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as outf, manifest.open("w", encoding="utf-8", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=[
            "id", "category", "cohort", "screen_stop_reason",
            "screen_thinking_tokens", "screen_generation_tokens"])
        writer.writeheader()
        for cohort, _, row, rec in selected:
            outf.write(json.dumps(row, ensure_ascii=False) + "\n")
            writer.writerow({
                "id": str(row["id"]), "category": row["category"],
                "cohort": cohort, "screen_stop_reason": rec.get("stop_reason"),
                "screen_thinking_tokens": rec.get("thinking_tokens"),
                "screen_generation_tokens": rec.get("generation_tokens"),
            })
    print(f"Screened {len(latest)} of {len(examples)} questions; statuses: {dict(counts)}")
    print(f"Selected: {len(picked_long)} long, {len(controls)} short controls")
    print(f"Saved {dest} and {manifest}")

if __name__ == "__main__":
    main()
