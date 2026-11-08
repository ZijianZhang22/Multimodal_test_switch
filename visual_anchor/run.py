import argparse
import json
from pathlib import Path
import numpy as np
from core import PCA, procrustes, fit, retrieval, paired_bootstrap, linear_cka

METHODS = ['supervised','text','visual','bank_text','extra_text','gaussian','broken_pairs']

def validate(d):
    train = ['l_fit','e_fit','i_fit','l_pair','e_pair','i_bridge','e_bridge','l_pool','e_pool','i_bank','e_bank','e_extra']
    test = ['l_test','e_test','l_img_test','i_test']
    for key in train+test:
        x = d[key]
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError('Invalid features: '+key)
        ids = d[key+'_ids']
        if ids.ndim != 1 or len(ids) != len(x) or ids.dtype.kind not in 'US':
            raise ValueError('Use one string content/group ID per row: '+key)
    for a,b in [('l_pair','e_pair'),('i_bridge','e_bridge'),('i_bank','e_bank')]:
        if not np.array_equal(d[a+'_ids'], d[b+'_ids']):
            raise ValueError('Paired rows must share IDs: '+a+', '+b)
    train_ids = set().union(*(set(d[k+'_ids']) for k in train))
    test_ids = set().union(*(set(d[k+'_ids']) for k in test))
    if train_ids & test_ids:
        raise ValueError('Train/test group-ID overlap detected')
    if set(d['e_pool_ids']) & set(d['e_bank_ids']):
        raise ValueError('e_pool/e_bank overlap undermines matched-bank control')
    if set(d['e_extra_ids']) & set().union(*(set(d[k+'_ids']) for k in train if k!='e_extra')):
        raise ValueError('e_extra must be additional, disjoint training data')
    for space in 'lei':
        dims = {d[k].shape[1] for k in train+test if k.startswith(space+'_')}
        if len(dims)!=1:
            raise ValueError('Feature widths inconsistent for '+space)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True); p.add_argument('--out',default='results')
    p.add_argument('--dim',type=int,default=32); p.add_argument('--budget',type=int,default=50)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--iterations',type=int,default=20); p.add_argument('--batch',type=int,default=256)
    p.add_argument('--whiten',type=float,default=0); p.add_argument('--epsilon',type=float,default=.08)
    p.add_argument('--tau',type=float,default=None); p.add_argument('--no-cycle',action='store_true')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    a=p.parse_args()
    if a.budget<1 or a.iterations<1 or a.batch<1 or a.dim<1:
        p.error('budget, iterations, batch and dim must be positive')
    d=dict(np.load(a.data,allow_pickle=False)); validate(d)
    if a.budget>len(d['l_pair']): p.error('budget exceeds available anchors')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    transforms={s:PCA(a.dim,a.whiten).fit(d[s+'_fit']) for s in 'lei'}
    z={k:transforms[k[0]].transform(v) for k,v in d.items() if not k.endswith('_ids')}
    v=procrustes(z['i_bridge'],z['e_bridge'])
    image_bank=z['i_bank']@v
    results=[]; hits={}
    for seed in a.seeds:
        idx=np.random.default_rng(seed).choice(len(z['l_pair']),a.budget,replace=False)
        for method in a.methods:
            w,trace=fit(z['l_pair'][idx],z['e_pair'][idx],z['l_pool'],z['e_pool'],image_bank,z['e_bank'],
                method=method,seed=seed,iterations=a.iterations,batch=a.batch,
                epsilon=a.epsilon,tau=a.tau,cycle=not a.no_cycle,extra_e=z['e_extra'])
            metrics={}; per_query={}
            for name,x,y,xi,yi in [
                ('L-E',z['l_test']@w,z['e_test'],'l_test','e_test'),
                ('E-L',z['e_test'],z['l_test']@w,'e_test','l_test'),
                ('L-I',z['l_img_test']@w,z['i_test']@v,'l_img_test','i_test'),
                ('I-L',z['i_test']@v,z['l_img_test']@w,'i_test','l_img_test')]:
                m,h=retrieval(x,y,d[xi+'_ids'],d[yi+'_ids'])
                metrics[name]=m; per_query[name]=h['R@1']
            hits[(seed,method)]=per_query
            results.append({'seed':seed,'method':method,'metrics':metrics,'trace':trace})
            payload={'W':w,'V_IE':v,'anchor_indices':idx}
            for s,t in transforms.items(): payload[s+'_mean']=t.mean; payload[s+'_proj']=t.proj
            np.savez(out/f'{method}_{seed}.npz',**payload)
            print(seed,method,metrics,flush=True)
    comparisons=[]
    for seed in a.seeds:
        if (seed,'visual') not in hits: continue
        for baseline in a.methods:
            if baseline=='visual': continue
            for direction in ['L-E','E-L','L-I','I-L']:
                key={'L-E':'l_test','E-L':'e_test','L-I':'l_img_test','I-L':'i_test'}[direction]
                ids=d[key+'_ids']; unique=np.unique(ids)
                ha=hits[(seed,'visual')][direction]; hb=hits[(seed,baseline)][direction]
                # Cluster captions by their image/group before bootstrapping.
                aa=np.array([ha[ids==i].mean() for i in unique]); bb=np.array([hb[ids==i].mean() for i in unique])
                comparisons.append({'seed':seed,'baseline':baseline,'direction':direction,**paired_bootstrap(aa,bb,seed)})
    summary={}
    for method in a.methods:
        rows=[r for r in results if r['method']==method]
        summary[method]={direction:{k:{'mean':float(np.mean([r['metrics'][direction][k] for r in rows])),
            'std':float(np.std([r['metrics'][direction][k] for r in rows],ddof=1)) if len(rows)>1 else None}
            for k in ['R@1','R@5','R@10']} for direction in ['L-E','E-L','L-I','I-L']}
    report={'config':vars(a),'summary':summary,'runs':results,'paired_bootstrap':comparisons,
            'bridge_train_cka':linear_cka(z['i_bridge'],z['e_bridge'])}
    (out/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__': main()
