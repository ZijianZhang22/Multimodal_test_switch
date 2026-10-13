"""Reuse the original four-hop toy generator, retaining five matched carriers."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

SCHEDULES = ['TTTT', 'TVVV', 'VTVV', 'VVTV', 'VVVT']

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', type=Path, default=Path('toy_cot_100/data'))
    p.add_argument('--n_problems', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.n_problems < 1:
        p.error('n_problems must be positive')
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'generate_switching_dataset.py'), '--out_dir', str(a.out_dir), '--n_problems', str(a.n_problems), '--seed', str(a.seed), '--schedule_set', 'all16'], check=True)
    path = a.out_dir / 'examples.jsonl'
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    rows = [x for x in rows if x['schedule'] in SCHEDULES]
    assert len(rows) == 5 * a.n_problems
    path.write_text(''.join(json.dumps(x) + '\n' for x in rows))
    (a.out_dir / 'config.json').write_text(json.dumps({'n_problems': a.n_problems, 'seed': a.seed, 'schedules': SCHEDULES}, indent=2))
    print(f'Retained {len(rows)} examples: {SCHEDULES}')

if __name__ == '__main__':
    main()
