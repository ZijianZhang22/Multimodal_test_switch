"""Synthetic plumbing check ONLY; not evidence for the research hypothesis."""
import numpy as np
r=np.random.default_rng(123); width=24
q={s:np.linalg.qr(r.normal(size=(width,width)))[0] for s in 'lei'}
d={}
def add(key,z,ids):
    d[key]=z@q[key[0]] + .03*r.normal(size=z.shape)
    d[key+'_ids']=np.asarray(ids)
def latent(n):
    return r.normal(size=(n,width))*np.linspace(.3,2,width)
for key in ['l_fit','e_fit','i_fit','l_pool','e_pool','e_extra']:
    n=400; add(key,latent(n),[key+str(i) for i in range(n)])
for keys,n,prefix in [(['l_pair','e_pair'],100,'anchor'),(['i_bridge','e_bridge'],400,'bridge'),
                      (['i_bank','e_bank'],300,'bank'),(['l_test','e_test'],100,'test_text'),
                      (['l_img_test','i_test'],100,'test_image')]:
    z=latent(n); ids=[prefix+str(i) for i in range(n)]
    for key in keys: add(key,z,ids)
np.savez('demo_features.npz',**d)
print('Wrote synthetic demo_features.npz; not real language data.')
