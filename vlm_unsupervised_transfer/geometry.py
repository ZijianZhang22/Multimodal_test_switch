"""Train-only unpaired orthogonal geometry, no paired bilingual anchors.

Text+visual uses a THIRD independent image-token bank, not bilingual images.
The image tokens are *decoder-layer image-prefix states*, not frozen CLIP embeddings;
their geometry may be incompatible with text states. This is a hypothesis/test.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


def unit(x):
    x=np.asarray(x,dtype=np.float64)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-9)


def polar(c):
    u,_,vt=np.linalg.svd(c,full_matrices=False)
    return u@vt


def _pca(x,y,z,dim):
    # One basis for every modality/language; avoids different coordinate frames.
    mx,my=x.mean(0),y.mean(0)
    allx=[x-mx,y-my]
    if z is not None:allx.append(z-my)
    joint=np.concatenate(allx)
    if dim<1 or dim>min(joint.shape[0]-1,joint.shape[1]):
        raise ValueError(f'Invalid PCA dim {dim} for {joint.shape}')
    _,_,v=np.linalg.svd(joint,full_matrices=False)
    return mx,my,v[:dim].T


def pseudo_cov(x,y,w,rng,batch):
    n=min(len(x),len(y),batch)
    ix=rng.choice(len(x),n,replace=False)
    iy=rng.choice(len(y),n,replace=False)
    sim=x[ix]@w@y[iy].T
    row,col=linear_sum_assignment(-sim)
    return (x[ix[row]].T@y[iy[col]])/n


def fit_map(x,y,z=None,*,method='text_ot',dim=16,iterations=30,batch=128,visual_weight=.25,seed=42):
    """All samples MUST be image-disjoint across x/y/z (checked by prepare).

    For method `text_visual_ot`, Z is an unpaired bank of VLM decoder image
    token averages. For `text_noise_ot`, a randomized bank matched to the first
    two moments of Z is used as a negative control. Mean-shift learns no rotation.
    """
    if method not in ('mean_shift','text_ot','text_visual_ot','text_noise_ot'):
        raise ValueError(method)
    x,y=np.asarray(x,np.float64),np.asarray(y,np.float64)
    if x.ndim!=2 or y.ndim!=2 or x.shape[1]!=y.shape[1] or min(len(x),len(y))<3:
        raise ValueError('Invalid X,Y features')
    uses_z=method in ('text_visual_ot','text_noise_ot')
    if uses_z:
        if z is None:raise ValueError('A visual pool is required')
        z=np.asarray(z,np.float64)
        if z.ndim!=2 or z.shape[1]!=x.shape[1] or len(z)<3:
            raise ValueError('Invalid Z features')
    if any(not np.isfinite(t).all() for t in (x,y)+((z,) if z is not None else ())):
        raise ValueError('Non-finite features')
    rng=np.random.default_rng(seed)
    mx,my,basis=_pca(x,y,z if uses_z else None,dim)
    xp=unit((x-mx)@basis)
    yp=unit((y-my)@basis)
    w=np.eye(dim,dtype=np.float64)
    if method=='text_noise_ot':
        # Match mean/covariance of visual bank, not sample-specific semantics.
        mean=z.mean(0); u,s,v=np.linalg.svd(z-mean,full_matrices=False)
        noise=rng.normal(size=(len(z),min(len(s),len(z))))
        z=mean+(noise*s[:noise.shape[1]])@v[:noise.shape[1]]/np.sqrt(max(len(z)-1,1))
    zp=unit((z-my)@basis) if uses_z else None
    hist=[]
    if method!='mean_shift':
        if iterations<1 or batch<2 or not 0<=visual_weight<=3:raise ValueError('Invalid training settings')
        for _ in range(iterations):
            cov=pseudo_cov(xp,yp,w,rng,batch)
            if uses_z:cov+=visual_weight*pseudo_cov(xp,zp,w,rng,batch)
            nxt=polar(cov)
            hist.append(float(np.linalg.norm(nxt-w)))
            w=nxt
    # Training-only target neighborhood reference and quantile calibration.
    mapped=unit(xp@w)
    train_sims=mapped@yp.T
    nearest=np.max(train_sims,axis=1)
    lo,hi=np.quantile(nearest,[.15,.85])
    if hi-lo<1e-5:hi=lo+1e-5
    return dict(basis=basis.astype('float32'),mean_source=mx.astype('float32'),
                mean_english=my.astype('float32'),W=w.astype('float32'),
                english_bank=yp.astype('float32'),confidence_low=np.float32(lo),
                confidence_high=np.float32(hi),method=method,history=hist)


def save_map(path,mapping,meta):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload={k:v for k,v in mapping.items() if isinstance(v,np.ndarray)}
    payload.update(confidence_low=np.array(mapping['confidence_low']),
                   confidence_high=np.array(mapping['confidence_high']),
                   metadata=np.array(json.dumps({**meta,'method':mapping['method'],
                                            'history':mapping['history']})))
    tmp=path.with_name(path.stem+'.tmp.npz')
    np.savez_compressed(tmp,**payload);tmp.replace(path)


def load_map(path,required=None):
    with np.load(path,allow_pickle=False) as v:
        metadata=json.loads(str(v['metadata'].item()))
        if required:
            for k,value in required.items():
                if metadata.get(k)!=value:raise ValueError(f'Map {k} mismatched')
        mp={k:v[k].copy() for k in ('basis','mean_source','mean_english','W','english_bank')}
        mp.update(confidence_low=float(v['confidence_low']),confidence_high=float(v['confidence_high']),method=metadata['method'])
    return mp,metadata


class Alignment:
    """Torch tensors are cached per map/device, not reallocated per token."""
    def __init__(self,mapping,device):
        import torch
        self.basis=torch.as_tensor(mapping['basis'],device=device,dtype=torch.float32)
        self.mx=torch.as_tensor(mapping['mean_source'],device=device,dtype=torch.float32)
        self.my=torch.as_tensor(mapping['mean_english'],device=device,dtype=torch.float32)
        self.W=torch.as_tensor(mapping['W'],device=device,dtype=torch.float32)
        self.bank=torch.as_tensor(mapping['english_bank'],device=device,dtype=torch.float32)
        self.lo=float(mapping['confidence_low']);self.hi=float(mapping['confidence_high'])

    def delta_and_confidence(self,h):
        import torch
        z=(h.float()-self.mx)@self.basis
        mapped=z@self.W
        delta=(mapped-z+(self.my-self.mx)@self.basis)@self.basis.T
        pred=torch.nn.functional.normalize(mapped,dim=-1)
        nearest=(pred@self.bank.T).amax(-1,keepdim=True)
        confidence=((nearest-self.lo)/(self.hi-self.lo)).clamp(0,1)
        return delta,confidence

    def visual_gate(self,h,image_h,delta,temperature=10.0):
        import torch
        if image_h is None:return torch.ones((*h.shape[:-1],1),device=h.device)
        before=torch.nn.functional.normalize((h.float()-self.mx)@self.basis,dim=-1)
        after=torch.nn.functional.normalize((h.float()+delta-self.my)@self.basis,dim=-1)
        vision=torch.nn.functional.normalize((image_h.float()-self.my)@self.basis,dim=-1)
        gain=(after*vision).sum(-1,keepdim=True)-(before*vision).sum(-1,keepdim=True)
        return (temperature*gain).sigmoid()

    def apply(self,h,image_h=None,*,strength=.2,mode='direct',max_relative_norm=.15):
        import torch
        if not 0<=strength<=2 or not 0<max_relative_norm<=1:
            raise ValueError('Invalid alpha or cap')
        if mode not in ('direct','confidence','visual','confidence_visual','constant_half'):
            raise ValueError(mode)
        delta,confidence=self.delta_and_confidence(h)
        size=delta.norm(dim=-1,keepdim=True).clamp_min(1e-9)
        cap=h.float().norm(dim=-1,keepdim=True).clamp_min(1e-9)*max_relative_norm
        delta=delta*torch.minimum(torch.ones_like(size),cap/size)
        gate=torch.ones_like(confidence)
        if mode in ('confidence','confidence_visual'):gate*=confidence
        if mode in ('visual','confidence_visual'):gate*=self.visual_gate(h,image_h,delta)
        if mode=='constant_half':gate*=.5
        return (h.float()+strength*gate*delta).to(dtype=h.dtype)