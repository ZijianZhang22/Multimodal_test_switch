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

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,default=Path('real_work'))
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--test-limit',type=int,default=1000)
    p.add_argument('--device',default='cuda');p.add_argument('--split-seed',type=int,default=42)
    a=p.parse_args(); a.work.mkdir(parents=True,exist_ok=True)
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
    model_names={'l':'shibing624/text2vec-base-chinese','e':'sentence-transformers/all-MiniLM-L6-v2','i':'facebook/dinov2-small'}
    lock=a.work/'model_revisions.json'
    if lock.exists(): revisions=json.loads(lock.read_text())
    else:
        revisions={k:model_info(v).sha for k,v in model_names.items()};lock.write_text(json.dumps(revisions,indent=2))
    manifest={'source_revision':REV,'image_url':IMAGE_URL,'models':model_names,'revisions':revisions,
              'groups':{k:v.tolist() for k,v in g.items()},'test':test,'test_caption_index':0,
              'protocol':'Chinese low-supervision pilot; MT anchors; human translated sentence test; native image captions',
              'audit':'image ID + exact image SHA256 + normalized caption exclusion; no near-duplicate or pretraining audit'}
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
    for space in ['l','e']:
        model=SentenceTransformer(model_names[space],revision=revisions[space],device=a.device,trust_remote_code=False)
        model.max_seq_length=128
        features[space]=cached_batches(space,texts[space],lambda batch:model.encode(batch,batch_size=len(batch),show_progress_bar=False,convert_to_numpy=True,normalize_embeddings=False))
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    from transformers import AutoImageProcessor,AutoModel
    from PIL import Image
    proc=AutoImageProcessor.from_pretrained(model_names['i'],revision=revisions['i'])
    model=AutoModel.from_pretrained(model_names['i'],revision=revisions['i']).to(a.device).eval()
    @torch.inference_mode()
    def encode_images(items):
        pics=[]
        for im in items:
            with Image.open(file(im)) as pic: pics.append(pic.convert('RGB'))
        inputs=proc(images=pics,return_tensors='pt').to(a.device)
        return model(**inputs).last_hidden_state[:,0].float().cpu().numpy()
    features['i']=cached_batches('i',imgids,encode_images)
    data={}
    for key,(space,rows,identity) in index.items(): data[key]=features[space][rows];data[key+'_ids']=identity
    # Do not expose Lpool image/English counterparts. Extra English remains disjoint.
    for dest,sources in [('l_fit',['l_pair','l_pool']),('e_fit',['e_pair','e_bridge','e_bank']),('i_fit',['i_bridge','i_bank']),('e_pool',['e_bridge','e_bank'])]:
        data[dest]=np.concatenate([data[s] for s in sources]);data[dest+'_ids']=np.concatenate([data[s+'_ids'] for s in sources])
    from run import validate
    validate(data)
    output=a.work/'features.npz';atomic_npz(output,**data)
    (a.work/'manifest.json').write_text(manifest_text)
    # Independent held-out English/image bridge diagnostic, never used for selection.
    from core import PCA,procrustes,retrieval
    pe=PCA(32).fit(data['e_fit']);pi=PCA(32).fit(data['i_fit'])
    v=procrustes(pi.transform(data['i_bridge']),pe.transform(data['e_bridge']))
    bridge,_=retrieval(pe.transform(data['e_test']),pi.transform(data['i_test'])@v,data['e_test_ids'],data['i_test_ids'])
    (a.work/'bridge_test.json').write_text(json.dumps(bridge,indent=2))
    print('Real features ready:',output,'Held-out English->image bridge:',bridge,flush=True)

if __name__=='__main__': main()
