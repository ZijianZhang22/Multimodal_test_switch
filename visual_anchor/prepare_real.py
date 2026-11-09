"""Flickr8K-CN real-data pilot. No synthetic or newly generated captions."""
import argparse, hashlib, json, subprocess, zipfile, shutil, unicodedata
from pathlib import Path
import numpy as np

REV='a19b9aa4a7ad2a42741460d9f47385637ae270cd'
BASE=f'https://raw.githubusercontent.com/li-xirong/flickr8kcn/{REV}/data/'
IMAGE_URL='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip'
FILES=['flickr8kenc.caption.txt','flickr8kzhg.caption.txt','flickr8kzhc.caption.txt','flickr8kzhmtest.captions.txt','flickr8ktrain.txt','flickr8kval.txt','flickr8ktest.txt']

def download(url,path):
    if path.exists(): return
    temp=path.with_suffix(path.suffix+'.part')
    subprocess.run(['curl','--fail','--location','--retry','5','--retry-delay','3','--connect-timeout','30','--continue-at','-','--output',str(temp),url],check=True)
    temp.replace(path)

def parse_captions(text):
    result={}
    for line in text.splitlines():
        if not line.strip(): continue
        key,caption=line.strip().split(maxsplit=1)
        parts=key.split('#'); image=Path(parts[0]).stem; index=int(parts[-1])
        if (image,index) in result: raise ValueError('Duplicate caption key')
        result[image,index]=caption.strip()
    return result

def norm(s):
    return ''.join(unicodedata.normalize('NFKC',s).lower().split())

def atomic_npz(path,**arrays):
    temp=path.with_suffix('.tmp.npz'); np.savez(temp,**arrays);temp.replace(path)

def groups(train,seed):
    ids=np.array(sorted(train)); np.random.default_rng(seed).shuffle(ids)
    if len(ids)<4500: raise ValueError(f'Need >=4500 clean train images, have {len(ids)}')
    return dict(anchor=ids[:1000],lpool=ids[1000:2000],bridge=ids[2000:3500],bank=ids[3500:4000],extra=ids[4000:4500])

POOL_SOURCES = {'l_fit': ('l_pair', 'l_pool'),
                'e_fit': ('e_pair', 'e_bridge', 'e_bank'),
                'i_fit': ('i_bridge', 'i_bank'),
                # Bank captions must not enter the ordinary English OT pool.
                'e_pool': ('e_bridge',)}

def assemble_pools(data):
    for dest, sources in POOL_SOURCES.items():
        data[dest] = np.concatenate([data[s] for s in sources])
        data[dest+'_ids'] = np.concatenate([data[s+'_ids'] for s in sources])
    if set(data['e_pool_ids']) & set(data['e_bank_ids']):
        raise ValueError('e_pool and e_bank must be disjoint in matched-bank experiments')
    return data

ENCODERS = {
    # L must stay a monolingual encoder in every setting: a multilingual or multilingual-CLIP model has already seen
    # large parallel data and would make the few-pair experiment meaningless.
    'default': {'l': 'shibing624/text2vec-base-chinese', 'e': 'sentence-transformers/all-MiniLM-L6-v2',
                'i': 'facebook/dinov2-small'},
    'clip': {'l': 'shibing624/text2vec-base-chinese', 'e': None, 'i': None},  # E and I = one CLIP checkpoint
}

def model_names_for(encoders, clip_model):
    names = dict(ENCODERS[encoders])
    if encoders == 'clip':
        names['e'] = names['i'] = clip_model
    return names

def resolve_revisions(lock, model_names, lookup):
    """Pin model revisions per work dir; refuse to reuse a lock written for different encoders."""
    if lock.exists():
        saved = json.loads(lock.read_text())
        if 'models' in saved:
            if saved['models'] != model_names:
                raise ValueError(f'{lock} was written for {saved["models"]}; use a separate --work for {model_names}')
            return saved['revisions']
        if set(saved) == {'l', 'e', 'i'} and model_names == ENCODERS['default']:
            return saved  # legacy lock of the default encoders
        raise ValueError(f'{lock} predates encoder selection; use a separate --work for {model_names}')
    revisions = {k: lookup(v) for k, v in model_names.items()}
    lock.write_text(json.dumps({'models': model_names, 'revisions': revisions}, indent=2))
    return revisions

def bridge_diagnostics(data, encoders, dim=32):
    """Held-out English->image retrieval through each bridge variant (test split only, never used for selection)."""
    from core import PCA, procrustes, retrieval
    out = {}
    def score(e_test, i_test):
        return retrieval(e_test, i_test, data['e_test_ids'], data['i_test_ids'])[0]
    pe = PCA(dim).fit(data['e_fit']); pi = PCA(dim).fit(data['i_fit'])
    v = procrustes(pi.transform(data['i_bridge']), pe.transform(data['e_bridge']))
    out['separate_pca_procrustes'] = score(pe.transform(data['e_test']), pi.transform(data['i_test']) @ v)
    if encoders == 'clip':
        out['clip_native'] = score(data['e_test'], data['i_test'])  # CLIP raw vector CSLS diagnostic
        from core import unit
        order = np.argsort(-(unit(data['e_test']) @ unit(data['i_test']).T), axis=1)
        eids, iids = data['e_test_ids'], data['i_test_ids']
        out['clip_native_cosine'] = {
            f'R@{k}': float((iids[order[:, :k]] == eids[:, None]).any(1).mean())
            for k in (1, 5, 10)
        }
        joint = PCA(dim).fit(np.concatenate((data['e_fit'], data['i_fit']), axis=0))
        out['joint_pca_identity'] = score(joint.transform(data['e_test']),
                                          joint.transform(data['i_test']))
        se, si = PCA.fit_shared(dim, [data['e_fit'], data['i_fit']])  # same projection run_rosetta3 --shared-ei-pca uses
        v = procrustes(si.transform(data['i_bridge']), se.transform(data['e_bridge']))
        out['shared_pca_procrustes'] = score(se.transform(data['e_test']), si.transform(data['i_test']) @ v)
        out['shared_pca_identity'] = score(se.transform(data['e_test']), si.transform(data['i_test']))
        out['shared_pca_map_distance_from_identity'] = float(np.linalg.norm(v - np.eye(dim)))
    # Top-level keys keep the old format (the bridge run.py uses with default flags).
    return {**out['separate_pca_procrustes'], 'variants': out}

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,default=Path('real_work'))
    p.add_argument('--encoders',choices=sorted(ENCODERS),default='default',
                   help='clip: E and I from one CLIP checkpoint (use a separate --work)')
    p.add_argument('--clip-model',default='openai/clip-vit-base-patch32')
    p.add_argument('--diagnostic-dim',type=int,default=32,help='dimension matching downstream Rosetta run')
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--test-limit',type=int,default=1000)
    p.add_argument('--device',default='cuda');p.add_argument('--split-seed',type=int,default=42)
    a=p.parse_args(); a.work.mkdir(parents=True,exist_ok=True)
    if a.diagnostic_dim < 1: p.error('--diagnostic-dim must be positive')
    raw=a.work/'raw';raw.mkdir(exist_ok=True)
    for f in FILES: download(BASE+f,raw/f)
    en,mt,zh,human=[parse_captions((raw/f).read_text()) for f in FILES[:4]]
    ids=lambda split:[Path(x.strip()).stem for x in (raw/f'flickr8k{split}.txt').read_text().splitlines() if x.strip()]
    train,test,val=ids('train'),ids('test'),ids('val')
    if set(train)&(set(test)|set(val)) or set(test)&set(val): raise ValueError('Source split overlap')
    # Filter against the entire held-out corpus, not only the chosen test subset.
    held=set(test)|set(val)
    held_text={norm(v) for source in [en,mt,zh,human] for (im,_),v in source.items() if im in held}
    train=[im for im in train if all((im,0) in source for source in [en,mt,zh])
           and not any(norm(source[im,0]) in held_text for source in [en,mt,zh])]
    archive=raw/'Flickr8k_Dataset.zip';download(IMAGE_URL,archive)
    images=raw/'images';images.mkdir(exist_ok=True)
    needed=set(train)|held
    # Only extract known JPEG basenames; no archive paths are trusted.
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name=Path(info.filename).name
            if Path(name).stem in needed and name.lower().endswith('.jpg'):
                dest=images/name
                if not dest.exists():
                    with z.open(info) as src, dest.open('wb') as dst: shutil.copyfileobj(src,dst)
    def file(im): return images/(im+'.jpg')
    missing=[im for im in needed if not file(im).exists()]
    if missing: raise FileNotFoundError(f'Missing {len(missing)} images, e.g. {missing[:3]}')
    def digest(im): return hashlib.sha256(file(im).read_bytes()).hexdigest()
    held_hash={digest(im) for im in held}
    clean=[];seen=set()
    for im in train:
        h=digest(im)
        if h not in held_hash and h not in seen: clean.append(im);seen.add(h)
    g=groups(clean,a.split_seed)
    test=[im for im in sorted(test) if (im,0) in en and (im,0) in human and (im,0) in zh][:a.test_limit]
    if len(test)!=a.test_limit or a.test_limit<10: raise ValueError('Invalid or unavailable test limit')
    from huggingface_hub import model_info
    import torch
    if a.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; use --device cpu explicitly')
    model_names=model_names_for(a.encoders,a.clip_model)
    revisions=resolve_revisions(a.work/'model_revisions.json',model_names,lambda name:model_info(name).sha)
    manifest={'source_revision':REV,'image_url':IMAGE_URL,'models':model_names,'revisions':revisions,
              'groups':{k:v.tolist() for k,v in g.items()},'test':test,'test_caption_index':0,
              'pool_protocol':'stage1_disjoint_e_pool_matched_bank_text',
              'protocol':'Chinese low-supervision pilot; MT anchors; human translated sentence test; native image captions',
              'audit':'image ID + exact image SHA256 + normalized caption exclusion; no near-duplicate or pretraining audit'}
    if a.encoders!='default': manifest['encoders']=a.encoders  # default manifest unchanged: existing caches stay valid
    manifest_text=json.dumps(manifest,ensure_ascii=False,sort_keys=True)
    fingerprint=hashlib.sha256(manifest_text.encode()).hexdigest()[:16]
    cache=a.work/('cache_'+fingerprint);cache.mkdir(exist_ok=True)
    (cache/'manifest.json').write_text(manifest_text)
    index={};texts={'l':[], 'e':[]};imgids=[]
    def add_text(key,space,source,items):
        start=len(texts[space]);texts[space].extend(source[im,0] for im in items)
        index[key]=(space,np.arange(start,len(texts[space])),np.array(items))
    add_text('l_pair','l',mt,g['anchor']);add_text('e_pair','e',en,g['anchor'])
    add_text('l_pool','l',zh,g['lpool'])
    for key,group in [('e_bridge','bridge'),('e_bank','bank'),('e_extra','extra')]: add_text(key,'e',en,g[group])
    add_text('l_test','l',human,test);add_text('e_test','e',en,test);add_text('l_img_test','l',zh,test)
    for key,items in [('i_bridge',g['bridge']),('i_bank',g['bank']),('i_test',test)]:
        start=len(imgids);imgids.extend(items);index[key]=('i',np.arange(start,len(imgids)),np.array(items))
    def cached_batches(space,items,encode):
        chunks=[]
        for start in range(0,len(items),a.batch_size):
            end=min(start+a.batch_size,len(items));target=cache/f'{space}_{start}_{end}.npy'
            if target.exists(): x=np.load(target)
            else:
                x=np.asarray(encode(items[start:end]),dtype=np.float32)
                temp=target.with_suffix('.tmp.npy');np.save(temp,x);temp.replace(target)
            if x.ndim!=2 or len(x)!=end-start or not np.isfinite(x).all(): raise ValueError('Invalid cached embeddings')
            chunks.append(x)
            print(f'{space}: {end}/{len(items)}',flush=True)
        return np.concatenate(chunks)
    features={}
    from sentence_transformers import SentenceTransformer
    clip=None
    if a.encoders=='clip':
        import clip_encoders
        clip=clip_encoders.load_clip(model_names['e'],revisions['e'],a.device)
        features['e']=cached_batches('e',texts['e'],lambda batch:clip_encoders.encode_texts(clip[0],clip[1],batch,a.device))
    for space in ['l','e']:
        if space in features: continue
        model=SentenceTransformer(model_names[space],revision=revisions[space],device=a.device,trust_remote_code=False)
        model.max_seq_length=128
        features[space]=cached_batches(space,texts[space],lambda batch:model.encode(batch,batch_size=len(batch),show_progress_bar=False,convert_to_numpy=True,normalize_embeddings=False))
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    from transformers import AutoImageProcessor,AutoModel
    from PIL import Image
    def load_pics(items):
        pics=[]
        for im in items:
            with Image.open(file(im)) as pic: pics.append(pic.convert('RGB'))
        return pics
    if clip is not None:
        def encode_images(items):
            return clip_encoders.encode_images(clip[0],clip[2],load_pics(items),a.device)
    else:
        proc=AutoImageProcessor.from_pretrained(model_names['i'],revision=revisions['i'])
        model=AutoModel.from_pretrained(model_names['i'],revision=revisions['i']).to(a.device).eval()
        @torch.inference_mode()
        def encode_images(items):
            inputs=proc(images=load_pics(items),return_tensors='pt').to(a.device)
            return model(**inputs).last_hidden_state[:,0].float().cpu().numpy()
    features['i']=cached_batches('i',imgids,encode_images)
    data={}
    for key,(space,rows,identity) in index.items(): data[key]=features[space][rows];data[key+'_ids']=identity
    # Do not expose Lpool image/English counterparts. Extra English remains disjoint.
    assemble_pools(data)
    from run import validate
    validate(data)
    output=a.work/'features.npz';atomic_npz(output,**data)
    (a.work/'manifest.json').write_text(manifest_text)
    # Independent held-out English/image bridge diagnostic, never used for selection.
    bridge=bridge_diagnostics(data,a.encoders,dim=a.diagnostic_dim)
    bridge['diagnostic_dim']=a.diagnostic_dim
    (a.work/'bridge_test.json').write_text(json.dumps(bridge,indent=2))
    print('Real features ready:',output,flush=True)
    for name,metrics in bridge['variants'].items(): print(f'  English->image bridge [{name}]:',metrics,flush=True)

if __name__=='__main__': main()
