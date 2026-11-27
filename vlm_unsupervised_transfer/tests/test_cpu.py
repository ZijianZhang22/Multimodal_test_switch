"""CPU tests: no Hugging Face weights, network, or images required."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from geometry import fit_map,load_map,save_map,Alignment
from prepare_data import disjoint_groups,read_jsonl,split_ids,parse_captions
from summarize import paired_stats
from vlm import letter,positions,preserve_output

class ExperimentalTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(7)
        self.x=self.rng.normal(size=(48,20))
        self.y=self.rng.normal(size=(45,20))
        self.z=self.rng.normal(size=(43,20))

    def test_a_only_filters_b_methods(self):
        from experiment import selected_for_arm
        selected={'A/mean_shift':{'method':'mean_shift'},
                  'A/text_ot':{'method':'text_ot'},
                  'B/text_ot/confidence':{'method':'text_ot'}}
        self.assertEqual(set(selected_for_arm(selected,'a')), {'A/mean_shift','A/text_ot'})
        self.assertEqual(set(selected_for_arm(selected,'all')), set(selected))
        with self.assertRaises(ValueError): selected_for_arm(selected,'invalid')

    def test_image_disjoint(self):
        d=disjoint_groups(map(str,range(100)),20,25,30,9)
        self.assertFalse(set(d['source'])&set(d['english']))
        self.assertFalse(set(d['source'])&set(d['visual']))
        self.assertFalse(set(d['english'])&set(d['visual']))

    def test_pool_rejects_short(self):
        with self.assertRaises(ValueError):disjoint_groups(['a','b'],2,2,2,9)

    def test_all_methods_orthogonal(self):
        for name in ('mean_shift','text_ot','text_visual_ot','text_noise_ot'):
            with self.subTest(method=name):
                m=fit_map(self.x,self.y,self.z,method=name,dim=6,iterations=5,batch=16)
                np.testing.assert_allclose(m['W'].T@m['W'],np.eye(6),atol=2e-5)
                self.assertTrue(np.isfinite(m['english_bank']).all())

    def test_visual_required(self):
        with self.assertRaises(ValueError):fit_map(self.x,self.y,method='text_visual_ot')

    def test_zero_supervision_in_api(self):
        # No bilingual pair indices, bilingual labels, supervised covariance or aligned IDs.
        import inspect
        self.assertNotIn('paired',inspect.signature(fit_map).parameters)

    def test_checkpoint_refuses_wrong_data(self):
        mp=fit_map(self.x,self.y,method='text_ot',dim=5,iterations=3,batch=16)
        with tempfile.TemporaryDirectory() as temp:
            dest=Path(temp)/'w.npz'
            save_map(dest,mp,dict(language='zh',layer=8,pool_sha='xyz'))
            restored,metadata=load_map(dest,dict(language='zh',layer=8,pool_sha='xyz'))
            np.testing.assert_allclose(restored['W'],mp['W'])
            with self.assertRaises(ValueError):load_map(dest,dict(language='sw'))

    def test_gating_not_alias(self):
        mp=fit_map(self.x,self.y,self.z,method='text_visual_ot',dim=6,iterations=3,batch=16)
        align=Alignment(mp,'cpu')
        h=torch.tensor(self.x[0],dtype=torch.float32)
        im=torch.tensor(self.z[0],dtype=torch.float32)
        direct=align.apply(h,im,strength=.5,mode='direct')
        half=align.apply(h,im,strength=.5,mode='constant_half')
        self.assertGreater(torch.norm(direct-h).item(),0)
        torch.testing.assert_close(half-h,(direct-h)*.5,atol=2e-6,rtol=2e-5)
        conf=align.apply(h,im,strength=.5,mode='confidence')
        self.assertLessEqual(torch.norm(conf-h).item(),torch.norm(direct-h).item()+1e-6)

    def test_visual_gate_dependent_on_image(self):
        mp=fit_map(self.x,self.y,self.z,method='text_visual_ot',dim=6,iterations=3,batch=16)
        a=Alignment(mp,'cpu')
        h=torch.tensor(self.x[0],dtype=torch.float32)
        delta,_=a.delta_and_confidence(h)
        gate1=a.visual_gate(h,torch.tensor(self.z[0],dtype=torch.float32),delta)
        gate2=a.visual_gate(h,torch.tensor(self.z[1],dtype=torch.float32),delta)
        self.assertTrue(torch.isfinite(gate1).all())
        self.assertTrue(0<gate1.item()<1)
        self.assertNotAlmostEqual(gate1.item(),gate2.item(),places=5)

    def test_prefill_image_mask(self):
        batch={'input_ids':torch.tensor([[99,99,7,8,9]]),
               'attention_mask':torch.ones((1,5),dtype=torch.long)}
        last,mask=positions(batch,99)
        self.assertEqual(last,4)
        self.assertEqual(int(mask.sum()),2)
        with self.assertRaises(ValueError):positions(batch,100)

    def test_tuple_hook_outputs(self):
        hidden=torch.randn((1,3,4)); output=(hidden,torch.ones(2))
        clone=hidden.clone()+1
        ret=preserve_output(output,clone)
        torch.testing.assert_close(ret[0],clone)
        self.assertIs(ret[1],output[1])

    def test_parsers(self):
        self.assertEqual(letter('B'), 'B')
        self.assertEqual(letter('答案: (C)'), 'C')
        self.assertEqual(letter('Maybe D'), '')

    def test_paired_report(self):
        b=[dict(id=str(i),correct=1 if i<5 else 0) for i in range(10)]
        t=[dict(id=str(i),correct=1 if i<7 else 0) for i in range(10)]
        v=paired_stats(b,t,iters=500)
        self.assertAlmostEqual(v['delta_pp'],20)
        self.assertEqual(v['fixed'],2)

if __name__=='__main__':unittest.main()
