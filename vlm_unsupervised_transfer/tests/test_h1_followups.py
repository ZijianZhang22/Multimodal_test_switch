"""CPU regression tests for frozen H1 follow-up diagnostics."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from h1_followups import (ALPHAS, external_records, full_pair, mapped_pair,
                          partial_pair, pixel_digest, retrieval, shift_metric, unit)


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.w=np.array([[0.,-1.],[1.,0.]])
        self.mp={'W':self.w,'basis':np.eye(2),'mean_source':np.zeros(2),
                 'mean_english':np.zeros(2)}
        self.zh=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
        self.en=self.zh@self.w

    def test_forward_map_versus_transpose(self):
        a,b=mapped_pair(self.zh,self.en,self.mp,'W')
        self.assertAlmostEqual(retrieval(a,b)[0]['recall1_zh_en'],1.)
        x,y=mapped_pair(self.zh,self.en,self.mp,'WT')
        self.assertLess(retrieval(x,y)[0]['recall1_zh_en'],1.)

    def test_partial_endpoints(self):
        x0,y0=partial_pair(self.zh,self.en,self.mp,0.)
        xc,yc=unit(self.zh),unit(self.en)
        self.assertTrue(np.allclose(x0,xc))
        self.assertTrue(np.allclose(y0,yc))
        x1,y1=partial_pair(self.zh,self.en,self.mp,1.)
        xw,yw=mapped_pair(self.zh,self.en,self.mp,'W')
        self.assertTrue(np.allclose(x1,xw))
        self.assertTrue(np.allclose(y1,yw))
        self.assertEqual(ALPHAS,(0., .1, .25, .5, .75, 1.))

    def test_full_and_pca_mapped_agree_for_identity_basis(self):
        a,b=full_pair(self.zh,self.en,self.mp,'mapped')
        x,y=mapped_pair(self.zh,self.en,self.mp,'W')
        self.assertTrue(np.allclose(a,x))
        self.assertTrue(np.allclose(b,y))

    def test_mmd_self_and_shift(self):
        rng=np.random.default_rng(5)
        a=rng.normal(size=(32,6))
        b=a.copy()
        same=shift_metric(a,b,seed=4,permutations=19)
        self.assertAlmostEqual(same['mmd2'],0.,places=10)
        shift=shift_metric(a,np.ones((32,6)),seed=4,permutations=19)
        self.assertGreater(shift['mmd2'],same['mmd2'])

    def test_external_repeated_question_stems(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            records=[]
            for i in range(20):
                im=root/f'{i}.png'
                Image.new('RGB',(4,4),(i,2,3)).save(im)
                records.append({'id':f'p{i:03d}','image':im.name,
                                'questions':{'zh':f'这个物体是什么 {i}？',
                                             'en':f'What is this object {i}?'}})
            records[1]['questions']['en']=records[0]['questions']['en']
            fp=root/'data.jsonl'
            fp.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in records),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Repeated question stem'):
                external_records(fp,20,42,set())

    def test_full_space_projection_preserves_orthogonal_complement(self):
        b=np.array([[1.,0.],[0.,1.],[0.,0.]])
        mp=dict(W=self.w,basis=b,mean_source=np.zeros(3),mean_english=np.zeros(3))
        a=np.array([[1.,0.,9.],[0.,1.,8.],[-1.,0.,7.],[0.,-1.,6.]])
        e=np.array([[0.,-1.,9.],[1.,0.,8.],[0.,1.,7.],[-1.,0.,6.]])
        x,y=full_pair(a,e,mp,'mapped')
        expected=np.column_stack([a[:,:2]@self.w,a[:,2]])
        self.assertTrue(np.allclose(x,unit(expected)))
        self.assertTrue(np.allclose(y,unit(e)))

    def test_external_ids_and_duplicate_images(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            Image.new('RGB',(4,4),(0,0,0)).save(root/'a.png')
            Image.new('RGB',(4,4),(0,0,0)).save(root/'b.png')
            sha=pixel_digest(root/'a.png')
            self.assertEqual(sha,pixel_digest(root/'b.png'))
            rows=[dict(id=f's{i:02d}',image='a.png' if i==0 else 'b.png',
                       questions={'zh':'你好？','en':'Hello?'}) for i in range(20)]
            dataset=root/'rows.jsonl'
            dataset.write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError,'Repeated'):
                external_records(dataset,20,42,set())
            rows[0]['id']='../bad'
            dataset.write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError,'safe filename'):
                external_records(dataset,20,42,set())


if __name__=='__main__':unittest.main()