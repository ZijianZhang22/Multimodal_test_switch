"""Three-space orthogonal/OT experiment. Convention: x_aligned = x @ W."""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp

def unit(x):
    a=np.asarray(x,dtype=np.float64)
    return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)

def orthogonal(cross_cov):
    u,_,vh=np.linalg.svd(cross_cov,full_matrices=False)
    return u@vh

def procrustes(x,y):
    if x.shape!=y.shape or x.ndim!=2 or len(x)<2:
        raise ValueError('Expected matching (n,d) paired arrays, n >= 2')
    return orthogonal(unit(x).T@unit(y)/len(x))

def balanced_sinkhorn(x,y,eps=.09,n_iter=60):
    """Balanced entropic OT on cosine cost, without any paired indices."""
    if eps<=0: raise ValueError('eps must be positive')
    a,b=unit(x),unit(y)
    logk=(a@b.T)/eps
    nx,ny=len(a),len(b)
    u,v=np.zeros(nx),np.zeros(ny)
    for _ in range(n_iter):
        u=-np.log(nx)-logsumexp(logk+v[None,:],axis=1)
        v=-np.log(ny)-logsumexp(logk+u[:,None],axis=0)
    p=np.exp(logk+u[:,None]+v[None,:])
    return p/p.sum()

def ot_cov(x,y,w,rng,batch,eps):
    # Independently sample each collection; their original row IDs are not used.
    ix=rng.choice(len(x),min(batch,len(x)),replace=False)
    iy=rng.choice(len(y),min(batch,len(y)),replace=False)
    a,b=unit(x[ix]),unit(y[iy])
    p=balanced_sinkhorn(a@w,b,eps)
    return a.T@p@b

def norm_cov(c):
    return c/max(float(np.linalg.norm(c,'fro')),1e-12)

@dataclass
class JointMaps:
    low_to_en: np.ndarray
    image_to_en: np.ndarray
    low_to_image: np.ndarray
    history: list
    def cycle_error(self):
        return float(np.linalg.norm(self.low_to_image@self.image_to_en-self.low_to_en,'fro')/
                     np.sqrt(len(self.low_to_en)))

def fit_joint(low,en,img,seed_idx,*,steps=12,batch=256,eps=.09,
              unsup_weight=.35,cycle_weight=.12,image_prior=.75,seed=42,joint=True):
    """Few paired L-E anchors + unpaired L/E, I/E, L/I OT + cycle prior.
    img must FIRST be Rosetta-transformed to English coordinates.
    This joint extension is NOT the Rosetta/MUSE original method.
    """
    low,en,img=(unit(z) for z in (low,en,img))
    if any(z.ndim!=2 for z in (low,en,img)): raise ValueError('Expected 2D arrays')
    d=en.shape[1]
    if low.shape[1]!=d or img.shape[1]!=d: raise ValueError('Dimensions must match')
    if min(map(len,(low,en,img)))<4: raise ValueError('Need >=4 examples per space')
    seed_idx=np.asarray(seed_idx,dtype=int)
    if len(seed_idx)<2 or np.any(seed_idx<0) or np.any(seed_idx>=min(len(low),len(en))):
        raise ValueError('Need >=2 valid bilingual seeds')
    if len(np.unique(seed_idx))!=len(seed_idx): raise ValueError('Repeated seeds')
    seed_cov=norm_cov(low[seed_idx].T@en[seed_idx]/len(seed_idx))
    wle=orthogonal(seed_cov); wie=np.eye(d); wli=wle.copy()
    hist=[]
    if not joint: return JointMaps(wle,wie,wli,hist)
    rng=np.random.default_rng(seed)
    scale=np.sqrt(d)
    for step in range(steps):
        cle=norm_cov(ot_cov(low,en,wle,rng,batch,eps))
        cie=norm_cov(ot_cov(img,en,wie,rng,batch,eps))
        cli=norm_cov(ot_cov(low,img,wli,rng,batch,eps))
        newle=orthogonal(seed_cov+unsup_weight*cle+cycle_weight*(wli@wie)/scale)
        newie=orthogonal(unsup_weight*cie+cycle_weight*(wli.T@newle)/scale+
                         image_prior*np.eye(d)/scale)
        newli=orthogonal(unsup_weight*cli+cycle_weight*(newle@newie.T)/scale)
        wle,wie,wli=newle,newie,newli
        hist.append({'step':step+1,
                     'seed_cosine':float(np.mean(np.sum((low[seed_idx]@wle)*en[seed_idx],axis=1))),
                     'cycle_error':float(np.linalg.norm(wli@wie-wle,'fro')/scale)})
    return JointMaps(wle,wie,wli,hist)

def retrieval_metrics(left,right):
    """Only evaluation uses held-out row correspondences."""
    if len(left)!=len(right): raise ValueError('One caption per image required')
    scores=unit(left)@unit(right).T
    result={}
    for label,m in [('left_to_right',scores),('right_to_left',scores.T)]:
        n=len(m); truth=m[np.arange(n),np.arange(n)]
        ranks=1+np.sum(m>truth[:,None]+1e-12,axis=1)
        result[label]={'R1':float(np.mean(ranks<=1)),
                       'R5':float(np.mean(ranks<=5)),
                       'R10':float(np.mean(ranks<=10)),
                       'MRR':float(np.mean(1/ranks))}
    return result
