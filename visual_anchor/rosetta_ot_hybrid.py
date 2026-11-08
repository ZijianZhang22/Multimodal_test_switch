"""Rosetta-inspired hard-matching refinement of an existing soft-OT map.

This is NOT the official GeoWarP algorithm (no geometric initialization or
GW readout); instead, it tests whether its alternating Hungarian/Procrustes
refinement improves a warm start from our existing soft OT alignment.

All encoders, PCA projections, and the English/image bridge stay frozen.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from core import PCA, fit, polar, procrustes, retrieval
from run import validate


def hard_refine(w0, lpool, epool, lpair, epair, *, seed=0, steps=20,
                batch=256, match_weight=0.25):
    """Warm-start Hungarian assignment / supervised Procrustes iterations.

    Inputs have already been PCA-projected and row-normalized by the existing
    pipeline. The known pairs are kept in every orthogonal update.
    """
    if steps < 0 or batch < 1 or match_weight < 0:
        raise ValueError('steps >= 0, batch >= 1 and match_weight >= 0 required')
    if lpool.shape[1] != epool.shape[1] or w0.shape != (lpool.shape[1], epool.shape[1]):
        raise ValueError('Feature or map dimension mismatch')
    if not len(lpool) or not len(epool) or not len(lpair) or len(lpair) != len(epair):
        raise ValueError('Unpaired pools and row-aligned anchors required')
    rng = np.random.default_rng(seed)
    w = np.asarray(w0, dtype=np.float64).copy()
    anchor_cov = lpair.T @ epair / len(lpair)
    history = []
    for step in range(steps):
        n = min(batch, len(lpool), len(epool))
        lix = rng.choice(len(lpool), n, replace=False)
        eix = rng.choice(len(epool), n, replace=False)
        x, y = lpool[lix], epool[eix]
        # Unpaired text pools: these are PSEUDO matches, not ground truth.
        rows, cols = linear_sum_assignment(-(x @ w @ y.T))
        matched_cov = x[rows].T @ y[cols] / len(rows)
        w = polar(anchor_cov + match_weight * matched_cov)
        history.append({'step': step, 'matched': len(rows),
                        'anchor_mse': float(np.mean(np.sum((lpair @ w - epair)**2, axis=1)))})
    return w, history


def score(w, z, d, v):
    output = {}
    settings = [
        ('L-E', z['l_test'] @ w, z['e_test'], 'l_test', 'e_test'),
        ('E-L', z['e_test'], z['l_test'] @ w, 'e_test', 'l_test'),
        ('L-I', z['l_img_test'] @ w, z['i_test'] @ v, 'l_img_test', 'i_test'),
        ('I-L', z['i_test'] @ v, z['l_img_test'] @ w, 'i_test', 'l_img_test'),
    ]
    for name, x, y, xkey, ykey in settings:
        output[name], _ = retrieval(x, y, d[xkey+'_ids'], d[ykey+'_ids'])
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='real_work/features_stage1.npz')
    p.add_argument('--out', default='real_work/ot_rosetta')
    p.add_argument('--budget', type=int, default=50)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--dim', type=int, default=32)
    p.add_argument('--init-method', choices=['text','visual','bank_text','extra_text'], default='bank_text')
    p.add_argument('--ot-iterations', type=int, default=20)
    p.add_argument('--ot-batch', type=int, default=256)
    p.add_argument('--refine-steps', type=int, default=20)
    p.add_argument('--refine-batch', type=int, default=256)
    p.add_argument('--match-weight', type=float, default=.25)
    a = p.parse_args()
    if min(a.budget,a.dim,a.ot_iterations,a.ot_batch,a.refine_batch) < 1 or a.refine_steps < 0:
        p.error('positive budget/dim/OT steps/batch, nonnegative refinement steps required')
    d = dict(np.load(a.data, allow_pickle=False)); validate(d)
    if a.budget > len(d['l_pair']):
        p.error('budget exceeds available paired anchors')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    transforms = {s: PCA(a.dim).fit(d[s+'_fit']) for s in 'lei'}
    z = {k: transforms[k[0]].transform(val) for k,val in d.items() if not k.endswith('_ids')}
    v = procrustes(z['i_bridge'], z['e_bridge'])
    image_bank = z['i_bank'] @ v
    runs = []
    for seed in a.seeds:
        idx = np.random.default_rng(seed).choice(len(z['l_pair']), a.budget, replace=False)
        lp, ep = z['l_pair'][idx], z['e_pair'][idx]
        opts = dict(method=a.init_method, seed=seed, batch=a.ot_batch,
                    cycle=False, extra_e=z['e_extra'])
        w_ot,_ = fit(lp,ep,z['l_pool'],z['e_pool'],image_bank,z['e_bank'],
                     iterations=a.ot_iterations,**opts)
        # Budget control: same soft OT, with as many extra updates as refinement steps.
        w_more_ot,_ = fit(lp,ep,z['l_pool'],z['e_pool'],image_bank,z['e_bank'],
                          iterations=a.ot_iterations+a.refine_steps,**opts)
        w_hybrid,trace = hard_refine(w_ot,z['l_pool'],z['e_pool'],lp,ep,
                                    seed=seed+8000,steps=a.refine_steps,
                                    batch=a.refine_batch,match_weight=a.match_weight)
        w_sup = procrustes(lp,ep)
        w_sup_refined,_ = hard_refine(w_sup,z['l_pool'],z['e_pool'],lp,ep,
                                     seed=seed+8000,steps=a.refine_steps,
                                     batch=a.refine_batch,match_weight=a.match_weight)
        methods={'supervised':w_sup, 'ot':w_ot, 'ot_more_steps':w_more_ot,
                 'supervised_hard_refine':w_sup_refined, 'ot_hard_refine':w_hybrid}
        for name,w in methods.items():
            metrics = score(w,z,d,v)
            runs.append({'seed':seed,'method':name,'metrics':metrics,
                         **({'trace':trace} if name=='ot_hard_refine' else {})})
            payload = {'W':w,'V_IE':v,'anchor_indices':idx}
            for s,t in transforms.items():
                payload[s+'_mean'] = t.mean
                payload[s+'_proj'] = t.proj
            np.savez(out/f'{name}_{seed}.npz',**payload)
            print('budget',a.budget,'seed',seed,name,metrics,flush=True)
    directions=['L-E','E-L','L-I','I-L']
    summary={name:{direction:{metric:float(np.mean([r['metrics'][direction][metric]
                                                for r in runs if r['method']==name]))
                              for metric in ['R@1','R@5','R@10']}
                   for direction in directions}
             for name in ['supervised','ot','ot_more_steps','supervised_hard_refine','ot_hard_refine']}
    delta={direction:{metric:summary['ot_hard_refine'][direction][metric]-summary['ot'][direction][metric]
                      for metric in ['R@1','R@5','R@10']} for direction in directions}
    report={'config':vars(a),'summary':summary,'hybrid_minus_ot':delta,'runs':runs,
            'note':'Rosetta-inspired warm-start Hungarian/Procrustes refinement; not full GeoWarP.'}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print('Report:',out/'report.json')

if __name__ == '__main__':
    main()
