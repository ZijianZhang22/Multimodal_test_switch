"""Offline CPU unit tests; no pretrained model downloads."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from protocol import prepare,load_splits,parse_letter,decoder_hook,prompt_last_index
from rosetta_math import fit_mapping,correction,save_mapping,load_mapping
from lora import target_module_names,alignment_loss
from summarize import contrast

class ProtocolTests(unittest.TestCase):
    def test_split_and_prevent_test_in_train(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            rows=[]
            for task in ('count','depth'):
                for i in range(30):
                    f=root/f'{task}{i}.png'; Image.new('RGB',(3,3),color=(i, 2 if task=='count' else 3, 5)).save(f)
                    rows.append(dict(id=f'{task}{i}',task=task,split='val' if i<15 else 'test',
                         image=str(f),answer='B',questions={v:f'Q {v}' for v in ('en','zh','sw')}))
            src=root/'visual_eval.jsonl';src.write_text('\n'.join(json.dumps(r) for r in rows))
            prepare(src,root/'out',0.2,42)
            result=load_splits(root/'out')
            self.assertEqual(len(result['test']),30)
            self.assertEqual(len(result['val']),6)
            self.assertEqual(len(result['train']),24)
    def test_parse(self):
        self.assertEqual(parse_letter('(C).'), 'C')
        self.assertEqual(parse_letter('答案: B'), 'B')
        self.assertEqual(parse_letter('Unclear'), '')
    def test_hidden_hook_preserves_tuple(self):
        h=torch.ones(1,4,8)
        out=decoder_hook(None,None,(h,torch.tensor(3)),lambda x:x*2)
        self.assertTrue(torch.all(out[0]==2))
        self.assertEqual(out[1],torch.tensor(3))
    def test_prompt_selection_without_suffix(self):
        batch={'input_ids':torch.tensor([[2,3,4,5,6]]),
               'attention_mask':torch.ones(1,5,dtype=torch.long),
               'token_type_ids':torch.tensor([[0,0,0,1,1]])}
        self.assertEqual(prompt_last_index(batch),2)

class AlignmentTests(unittest.TestCase):
    def test_few_pair_procrustes_and_restore(self):
        rng=np.random.default_rng(1);d=6
        true=np.linalg.qr(rng.normal(size=(d,d)))[0]
        src=rng.normal(size=(30,d))
        tgt=src@true
        u=rng.normal(size=(20,d));v=u@true
        m=fit_mapping(src,tgt,u,v,dim=d,method='supervised')
        x=(src-m['mean_src'])@m['basis']; y=(tgt-m['mean_tgt'])@m['basis']
        self.assertLess(np.mean(np.linalg.norm(x@m['W']-y,axis=1)),1e-4)
        self.assertLess(np.linalg.norm(m['W'].T@m['W']-np.eye(d)),1e-4)
        with tempfile.TemporaryDirectory() as d0:
            f=Path(d0)/'mapping.npz'
            save_mapping(f,m,dict(model_id='dummy',data_sha='abc',language='zh',layer=2))
            loaded,_=load_mapping(f,model_id='dummy',data_sha='abc',language='zh',layer=2)
            self.assertEqual(loaded['W'].shape,(d,d))
            with self.assertRaisesRegex(ValueError,'language'):
                load_mapping(f,model_id='dummy',data_sha='abc',language='sw',layer=2)
    def test_unpaired_rosrtta_and_norm_cap(self):
        rng=np.random.default_rng(4);a=rng.normal(size=(30,8));b=rng.normal(size=(30,8))
        m=fit_mapping(a,a+0.01,b,b,dim=6,method='rosetta',iterations=2)
        x=torch.tensor(a[:2],dtype=torch.float32)
        y=correction(x,m,strength=.5,max_relative_norm=.1)
        self.assertEqual(y.shape,x.shape)
        self.assertTrue(torch.all((x-y).norm(dim=-1)<=.051*x.norm(dim=-1)))
    def test_lora_grad_and_modules(self):
        a=torch.randn(2,8,requires_grad=True);b=torch.randn(2,8)
        loss=alignment_loss(a,b);loss.backward()
        self.assertGreater(a.grad.abs().sum().item(),0)
        self.assertEqual(len(target_module_names([8,13])),4)

class StatisticTests(unittest.TestCase):
    def test_pair_accuracy(self):
        def row(k,ok): return dict(id=k,answer='A',task='count',correct=str(ok),valid='1')
        b={str(i):row(str(i),0) for i in range(12)}
        a={str(i):row(str(i),1) for i in range(12)}
        v=contrast(b,a,200)
        self.assertEqual(v['delta_pp'],100)
        self.assertLess(v['mc_nemar_exact_p'],.05)

if __name__ == '__main__':unittest.main()