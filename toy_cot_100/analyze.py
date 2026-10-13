"""Report all outputs and paired comparisons, including a cap-free sensitivity check."""
import argparse
import csv
import json
import statistics
from pathlib import Path
from prepare import SCHEDULES

def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def analyze(rows):
    keyed = {}
    for r in rows:
        if r['schedule'] in SCHEDULES:
            keyed[(r['problem_id'], r['schedule'])] = r
    summary = []
    for s in SCHEDULES:
        rs = [r for (pid, schedule), r in keyed.items() if schedule == s]
        if not rs:
            continue
        summary.append(dict(schedule=s, n=len(rs), accuracy=sum(bool(r['correct']) for r in rs)/len(rs), mean_output_tokens=statistics.mean(r['output_tokens'] for r in rs), median_output_tokens=statistics.median(r['output_tokens'] for r in rs), hit_cap=sum(bool(r.get('hit_token_cap')) for r in rs), invalid=sum(r.get('prediction') is None for r in rs)))
    paired = []
    for s in SCHEDULES[1:]:
        for uncapped in [False, True]:
            pairs = [(b, keyed[(pid,s)]) for (pid,t), b in keyed.items() if t == 'TTTT' and (pid,s) in keyed]
            if uncapped:
                pairs = [(b,v) for b,v in pairs if not b.get('hit_token_cap') and not v.get('hit_token_cap')]
            if not pairs:
                continue
            paired.append(dict(schedule=s, subset='uncapped_pairs' if uncapped else 'all_pairs', n=len(pairs), delta_accuracy=statistics.mean(int(v['correct'])-int(b['correct']) for b,v in pairs), mean_delta_output_tokens=statistics.mean(v['output_tokens']-b['output_tokens'] for b,v in pairs), correct_to_wrong=sum(b['correct'] and not v['correct'] for b,v in pairs), wrong_to_correct=sum(not b['correct'] and v['correct'] for b,v in pairs)))
    return summary, paired

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out_dir', type=Path, required=True)
    a=p.parse_args()
    rows=[json.loads(x) for x in a.input.read_text().splitlines() if x.strip()]
    summary, paired=analyze(rows)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(a.out_dir/'summary.csv', summary)
    write_csv(a.out_dir/'paired.csv', paired)
    for r in summary + paired:
        print(r)

if __name__ == '__main__':
    main()
