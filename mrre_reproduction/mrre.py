"""Independent MRRE-style implementation; not the authors' unpublished code."""
import argparse, contextlib, hashlib, json, random, re
from pathlib import Path


def read(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in rows))


def prepare(a):
    data=json.loads(Path(a.source).read_text())
    groups={};skipped=[]
    for x in data:
        gid=str(x.get('problem_index',x.get('pid',x.get('sample_index'))))
        if gid=='None': raise ValueError('Missing group ID')
        target=x.get('zh_question')
        if not target:
            skipped.append(gid);continue
        choices=x.get('choices')
        suffix=''
        if choices and 'Choices:' not in x['question']:
            suffix='\n'+'\n'.join(f'{chr(65+i)}: {s}' for i,s in enumerate(choices))
        row=dict(id=str(x.get('sample_index',x.get('pid'))),group=gid,
                 en=x['question']+suffix,zh=target+suffix,image=x.get('image'),
                 answer=str(x['answer']),choices=choices,
                 question_type=x.get('question_type',''),answer_type=x.get('answer_type',''))
        groups.setdefault(gid,[]).append(row)
    print('Skipped missing translations:',len(skipped),flush=True)
    ids=sorted(groups);random.Random(a.seed).shuffle(ids)
    if len(ids)<a.calibration+a.validation+a.test: raise ValueError('Insufficient groups')
    start=0
    for split,n in [('calibration',a.calibration),('validation',a.validation),('test',a.test)]:
        # One variant per group for a small pilot; variants never cross splits.
        rows=[groups[k][0] for k in ids[start:start+n]];start+=n
        write(Path(a.out)/(split+'.jsonl'),rows)
    Path(a.out,'manifest.json').write_text(json.dumps(dict(seed=a.seed,source_sha256=hashlib.sha256(Path(a.source).read_bytes()).hexdigest(),policy='one first variant per shuffled problem group'),indent=2))


def get_layers(model):
    for path in ['model.language_model.layers','model.layers','language_model.model.layers']:
        obj=model
        try:
            for name in path.split('.'): obj=getattr(obj,name)
            if len(obj): return obj
        except (AttributeError,TypeError): pass
    raise ValueError('Unsupported decoder path; inspect model and adapt get_layers()')


def steer_tensor(h,v,alpha):
    import torch
    y=h.clone();old=h[:,-1,:].float();delta=v.to(device=h.device,dtype=torch.float32)
    new=old+alpha*delta
    new=new*(old.norm(dim=-1,keepdim=True)/new.norm(dim=-1,keepdim=True).clamp_min(1e-12))
    y[:,-1,:]=new.to(h.dtype)
    return y


@contextlib.contextmanager
def intervention(layers,spec):
    handles=[]
    try:
        for idx,v,alpha in spec:
            def hook(module,inputs,output,v=v,alpha=alpha):
                if isinstance(output,tuple): return (steer_tensor(output[0],v,alpha),)+output[1:]
                return steer_tensor(output,v,alpha)
            handles.append(layers[idx].register_forward_hook(hook))
        yield
    finally:
        for handle in handles: handle.remove()


class Runner:
    def __init__(self,a):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM,AutoProcessor,Qwen2_5_VLForConditionalGeneration
        self.torch=torch;self.a=a
        kw=dict(torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,device_map='auto',attn_implementation='eager')
        if a.kind=='vlm':
            self.processor=AutoProcessor.from_pretrained(a.model,min_pixels=256*28*28,max_pixels=a.max_pixels)
            self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model,**kw)
            self.tokenizer=self.processor.tokenizer
        else:
            self.processor=None;self.tokenizer=AutoTokenizer.from_pretrained(a.model)
            self.model=AutoModelForCausalLM.from_pretrained(a.model,**kw)
        self.model.eval();self.layers=get_layers(self.model)
    def inputs(self,row,lang,anchor=False):
        if anchor:
            prompt='请用中文回答上述的问题。' if lang=='zh' else 'Please answer the above question in English.'
        else:
            prefix='使用中文进行思考和回答。请逐步推理，最后写出 Final: 答案。\n' if lang=='zh' else 'Use English to think and answer. Reason step by step and end with Final: answer.\n'
            prompt=prefix+row[lang]
        if self.processor:
            content=[];images=None
            if not anchor and row.get('image'):
                from PIL import Image
                path=Path(self.a.image_root)/row['image']
                images=[Image.open(path).convert('RGB')];content.append(dict(type='image'))
            content.append(dict(type='text',text=prompt))
            text=self.processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
            batch=self.processor(text=[text],images=images,return_tensors='pt')
        else:
            if not anchor and row.get('image'): raise ValueError('Text-only model cannot evaluate image rows')
            text=self.tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
            batch=self.tokenizer(text,return_tensors='pt')
        return batch.to(self.model.device)
    def generate(self,batch):
        with self.torch.inference_mode():
            seq=self.model.generate(**batch,max_new_tokens=self.a.max_new_tokens,do_sample=False,use_cache=True)
        tail=seq[0,batch['input_ids'].shape[1]:]
        return seq,self.tokenizer.decode(tail,skip_special_tokens=True),len(tail),len(tail)>=self.a.max_new_tokens
    def capture(self,batch,seq=None):
        # Re-forward completed sequence, retaining original visual inputs.
        if seq is not None:
            batch=dict(batch);batch['input_ids']=seq
            batch['attention_mask']=self.torch.ones_like(seq)
        states={};handles=[]
        for i,layer in enumerate(self.layers):
            def hook(module,args,out,i=i):
                h=out[0] if isinstance(out,tuple) else out
                states[i]=h[0,-1].detach().float().cpu()
            handles.append(layer.register_forward_hook(hook))
        try:
            with self.torch.inference_mode(): self.model(**batch,use_cache=False)
        finally:
            for h in handles:h.remove()
        return self.torch.stack([states[i] for i in range(len(self.layers))])


def extract(a):
    r=Runner(a);rows=read(a.data);diffs=[];traces=[]
    for row in rows:
        states={};ok=True
        for lang in ['en','zh']:
            batch=r.inputs(row,lang);seq,text,n,truncated=r.generate(batch)
            traces.append(dict(id=row['id'],language=lang,response=text,tokens=n,truncated=truncated))
            if truncated:ok=False
            # Last non-EOS generated token, explicitly documented replication choice.
            while seq.shape[1]>batch['input_ids'].shape[1] and seq[0,-1].item() in {r.tokenizer.eos_token_id,r.tokenizer.pad_token_id}:
                seq=seq[:,:-1]
            if seq.shape[1]==batch['input_ids'].shape[1]:ok=False
            states[lang]=r.capture(batch,seq)
        if ok:diffs.append(states['en']-states['zh'])
        write(str(a.out)+'.traces.jsonl',traces)
        print('calibration',row['id'],'valid pairs',len(diffs),flush=True)
    if not diffs:raise ValueError('No untruncated calibration pairs; increase token limit')
    anchor=r.capture(r.inputs({},'zh',True))-r.capture(r.inputs({},'en',True))
    payload=dict(reason=r.torch.stack(diffs).mean(0),anchor=anchor,model=a.model,kind=a.kind,
                 groups=[x['group'] for x in rows],n_valid=len(diffs),convention='zero-based decoder block outputs; raw difference; last non-EOS response token',max_pixels=a.max_pixels)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);r.torch.save(payload,a.out)


def score(text,row):
    # Conservative deterministic scoring, not the paper's external judge.
    m=re.findall(r'Final\s*[:：]\s*([^\n]+)',text,re.I)
    if not m:return None
    answer=m[-1].strip().strip('$').strip().rstrip('。.')
    gold=row['answer'].strip().rstrip('。.')
    if row.get('choices') and gold in row['choices']:gold=chr(65+row['choices'].index(gold))
    if re.fullmatch('[A-H]',gold):
        match=re.match(r'^\(?([A-H])\)?(?:[.、:：\s]|$)',answer)
        return bool(match and match[1]==gold)
    try:
        from decimal import Decimal
        return Decimal(answer.replace(',',''))==Decimal(gold.replace(',',''))
    except Exception:return answer.casefold()==gold.casefold()


def evaluate(a):
    r=Runner(a);v=r.torch.load(a.vectors,map_location='cpu',weights_only=True)
    if v['model']!=a.model or v['kind']!=a.kind:raise ValueError('Vectors belong to another model')
    rows=read(a.data)
    if set(v['groups']) & {x['group'] for x in rows}:raise ValueError('Calibration/evaluation group leakage')
    if not 0<=a.reason_layer<a.anchor_layer<len(r.layers):raise ValueError('Invalid layer indices')
    if a.max_pixels!=v['max_pixels']:raise ValueError('Use same image resolution as calibration')
    reason=v['reason'][a.reason_layer];anchor=v['anchor'][a.anchor_layer]
    gen=r.torch.Generator().manual_seed(a.seed)
    rnd=r.torch.randn(reason.shape,generator=gen);rnd=rnd/rnd.norm()*reason.norm()
    configs={'baseline':[], 'reason':[(a.reason_layer,reason,a.alpha_reason)],
             'anchor':[(a.anchor_layer,anchor,a.alpha_anchor)],
             'both':[(a.reason_layer,reason,a.alpha_reason),(a.anchor_layer,anchor,a.alpha_anchor)],
             'random':[(a.reason_layer,rnd,a.alpha_reason)]}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('Use a new output filename; results will not be overwritten')
    with out.open('w') as f:
        for row in rows:
            for name in a.conditions.split(','):
                with intervention(r.layers,configs[name]):
                    _,text,n,trunc=r.generate(r.inputs(row,a.language))
                result=dict(id=row['id'],group=row['group'],condition=name,language=a.language,response=text,
                            tokens=n,truncated=trunc,correct=score(text,row),gold=row['answer'])
                f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
                print(row['id'],name,result['correct'],n,flush=True)
    Path(str(out)+'.config.json').write_text(json.dumps(vars(a),indent=2))
    summarize(a)


def summarize(a):
    from collections import defaultdict
    groups=defaultdict(list)
    for x in read(a.out):groups[x['condition']].append(x)
    for name,rows in groups.items():
        print(json.dumps(dict(condition=name,n=len(rows),accuracy=sum(x['correct'] is True for x in rows)/len(rows),unparsed=sum(x['correct'] is None for x in rows),truncated=sum(x['truncated'] for x in rows),mean_output_tokens=sum(x['tokens'] for x in rows)/len(rows))))


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--source',required=True);q.add_argument('--out',required=True)
    for k,n in [('calibration',100),('validation',30),('test',100),('seed',42)]:q.add_argument('--'+k,type=int,default=n)
    for cmd in ['extract','evaluate']:
        q=sub.add_parser(cmd);q.add_argument('--model',default='Qwen/Qwen2.5-VL-7B-Instruct');q.add_argument('--kind',choices=['vlm','llm'],default='vlm')
        q.add_argument('--data',required=True);q.add_argument('--out',required=True);q.add_argument('--image-root',default='.')
        q.add_argument('--max-new-tokens',type=int,default=1024);q.add_argument('--max-pixels',type=int,default=1280*28*28)
        if cmd=='evaluate':
            q.add_argument('--vectors',required=True);q.add_argument('--reason-layer',type=int,default=22);q.add_argument('--anchor-layer',type=int,default=24)
            q.add_argument('--alpha-reason',type=float,default=1.0);q.add_argument('--alpha-anchor',type=float,default=.75)
            q.add_argument('--seed',type=int,default=42);q.add_argument('--language',choices=['en','zh'],default='zh')
            q.add_argument('--conditions',default='baseline,reason,anchor,both,random')
    q=sub.add_parser('summarize');q.add_argument('--out',required=True)
    a=p.parse_args();globals()[{'prepare':'prepare','extract':'extract','evaluate':'evaluate','summarize':'summarize'}[a.command]](a)
if __name__=='__main__':main()
