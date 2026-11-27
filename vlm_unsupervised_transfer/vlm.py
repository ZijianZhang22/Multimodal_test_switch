"""Minimal PaliGemma2 layer interception for prompt-prefill only.

Model and processor are pinned by the launcher to transformers==4.51.3.
Only the *last text prompt token* is transformed. Image-prefix averages are
captured from the same layer for the visual steering gate; no test labels used.
"""
from __future__ import annotations
import re
import torch
from PIL import Image

MODEL='google/paligemma2-3b-mix-448'
VQA_PREFIX='answer en Answer the multiple choice question by only responding the letter of the correct answer. '
CAPTION_PREFIX='answer en Consider the following image description: '
VISUAL_PREFIX='answer en Describe the image.'


def load_model(model_id=MODEL):
    from transformers import PaliGemmaProcessor,PaliGemmaForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError('PaliGemma2 experiment requires CUDA; use python -m unittest for CPU tests')
    proc=PaliGemmaProcessor.from_pretrained(model_id)
    model=PaliGemmaForConditionalGeneration.from_pretrained(model_id,torch_dtype=torch.bfloat16,
                                                             attn_implementation='sdpa').eval().to('cuda')
    for param in model.parameters():param.requires_grad_(False)
    return proc,model


def layers(model):
    m=model.language_model.model.layers
    if not m or not hasattr(m[0],'register_forward_hook'):
        raise RuntimeError('Unknown decoder layout, expected language_model.model.layers')
    return m


def inputs(processor,row,prompt):
    with Image.open(row['image']) as f:
        img=f.convert('RGB')
        tensors=processor(images=img,text=prompt,return_tensors='pt')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in tensors.items()}


def positions(batch,image_token_index):
    ids=batch['input_ids']
    if ids.shape[0]!=1:raise ValueError('Only batch 1 supported (unambiguous prefill)')
    attention=batch.get('attention_mask',torch.ones_like(ids)).bool()[0]
    image=ids[0].eq(image_token_index)&attention
    text=~image&attention
    if 'token_type_ids' in batch:
        text &= batch['token_type_ids'][0].eq(0)
    valid=text.nonzero(as_tuple=False).view(-1)
    if not valid.numel() or not image.any():
        raise ValueError('Expected a text prompt token and image-prefix tokens')
    return int(valid[-1].item()),image


def preserve_output(output,patch):
    if isinstance(output,tuple):
        return (patch,)+output[1:]
    return patch


def captured_states(model,batch,candidate_layers,need_visual=False):
    last,mask=positions(batch,model.config.image_token_index)
    handles=[];cache={}
    decoder=layers(model)
    for i in candidate_layers:
        if i<0 or i>=len(decoder):raise ValueError(f'Layer {i} outside [0,{len(decoder)})')
        def attach(layer):
            def hook(mod,args,output):
                states=output[0] if isinstance(output,tuple) else output
                cache[layer]={'text':states[0,last].detach().float().cpu().numpy()}
                if need_visual:
                    cache[layer]['visual']=states[0,mask].mean(0).detach().float().cpu().numpy()
            return hook
        handles.append(decoder[i].register_forward_hook(attach(i)))
    try:
        with torch.inference_mode():model(**batch,use_cache=False,return_dict=True)
    finally:
        for handle in handles:handle.remove()
    if set(cache)!=set(candidate_layers):raise RuntimeError('Missing decoder hook activations')
    return cache


def generate(model,processor,row,prompt,*,layer=None,operator=None,max_new_tokens=8):
    batch=inputs(processor,row,prompt)
    last,mask=positions(batch,model.config.image_token_index)
    handle=None
    if operator is not None:
        if layer is None or not 0<=layer<len(layers(model)):raise ValueError('Invalid hook layer')
        def hook(mod,args,output):
            h=output[0] if isinstance(output,tuple) else output
            # Generation decode uses len=1, whereas prefill has a long image-prefix.
            if h.shape[1] <= last or h.shape[1] != batch['input_ids'].shape[1]:
                return output
            raw=h[0,last].detach()
            vision=h[0,mask].mean(0).detach()
            # Hooks never mutate module parameters, only one prefill hidden token.
            changed=operator(raw,vision)
            if changed.shape!=raw.shape or not torch.isfinite(changed).all():
                raise ValueError('Invalid intervention shape/non-finite output')
            copy=h.clone()
            copy[0,last]=changed
            return preserve_output(output,copy)
        handle=layers(model)[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            result=model.generate(**batch,do_sample=False,max_new_tokens=max_new_tokens,use_cache=True)
        # Some encoder-decoder/generation paths return prompt+completion, others completion.
        original_len=batch['input_ids'].shape[1]
        generated=result[0,original_len:] if result.shape[1]>original_len else result[0]
        return processor.tokenizer.decode(generated,skip_special_tokens=True).strip()
    finally:
        if handle is not None:handle.remove()


def question(row,language):
    return VQA_PREFIX+row['questions'][language]


def caption(row):
    if row['language']=='visual':return VISUAL_PREFIX
    return CAPTION_PREFIX+row['caption']


def letter(text):
    s=str(text).strip()
    for pat in (r'^\(?([A-Fa-f])\)?(?=\W|$)',
                r'^(?:answer|option|choice|jibu|答案|选项)\s*[:：]?\s*\(?([A-Fa-f])\)?(?=\W|$)'):
        m=re.search(pat,s,re.I)
        if m:return m.group(1).upper()
    return ''


def oracle_operator(teacher,strength,cap):
    teacher=torch.as_tensor(teacher,dtype=torch.float32,device='cuda')
    def op(h,vision):
        delta=teacher-h.float()
        d=delta.norm().clamp_min(1e-9)
        delta=delta*min(1.0, float(h.float().norm().item()*cap/d.item()))
        return (h.float()+strength*delta).to(h.dtype)
    return op
