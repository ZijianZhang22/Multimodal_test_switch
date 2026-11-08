import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,default=Path('real_work'));p.add_argument('--pattern',default='results_budget_*');p.add_argument('--name',default='summary.csv');a=p.parse_args()
rows=[]
for file in sorted(a.work.glob(a.pattern+'/report.json')):
    r=json.loads(file.read_text())
    for method,directions in r['summary'].items():
        for direction,metrics in directions.items():
            for metric,values in metrics.items():
                rows.append(dict(budget=r['config']['budget'],method=method,direction=direction,metric=metric,
                                 mean_percent=100*values['mean'],std_percent=None if values['std'] is None else 100*values['std']))
if not rows: raise RuntimeError('No completed reports')
with (a.work/a.name).open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('Summary:',a.work/a.name)
