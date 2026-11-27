from __future__ import annotations
import json
import pathlib
import sys
import tempfile
import unittest
import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from bridge_core import (PREFIX, bank_captions, build_prompt, english_prompt,
                         extract_question_for_embedding, image_cluster_bootstrap,
                         map_vectors, parse_flickr_captions, random_rankings, rankings)


class BridgeTests(unittest.TestCase):
    def test_question_extraction_removes_label_choices_and_keeps_hint(self):
        stem = ('请根据图片回答以下选择题。只输出大写字母。\n提示：比较手里的东西\n'
                '男人的手里是什么？\n(A) Apple\n(B) Banana')
        self.assertEqual(extract_question_for_embedding(stem),
                         '提示：比较手里的东西 男人的手里是什么？')
        self.assertNotIn('Banana', extract_question_for_embedding(stem))

    def test_map_preserves_unit_rotation_and_centering(self):
        theta=np.pi/3
        w=np.array([[np.cos(theta),-np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
        raw=np.array([[1.,2.],[2.,1.]])
        ck={'W':w,'mean_x':np.array([[.5,.5]]), 'mean_y':np.array([[1.,1.]])}
        x,y=map_vectors(raw,raw+1.0,ck)
        np.testing.assert_allclose(x, (raw-.5)/np.linalg.norm(raw-.5,axis=1,keepdims=True)@w, atol=1e-8)
        np.testing.assert_allclose(y, (raw)/np.linalg.norm(raw,axis=1,keepdims=True),atol=1e-8)

    def test_inconsistent_map_fails(self):
        ck={'W':np.ones((2,2)),'mean_x':np.zeros((1,2)),'mean_y':np.zeros((1,2))}
        with self.assertRaisesRegex(ValueError,'orthogonal'):
            map_vectors(np.eye(2),np.eye(2),ck)
        ck['W']=np.eye(3)
        with self.assertRaisesRegex(ValueError,'dimensions'):
            map_vectors(np.eye(2),np.eye(2),ck)

    def test_cosine_rankings(self):
        q=np.array([[1.,0.],[0.,1.]])
        bank=np.array([[1.,0.],[0.,1.],[-1.,0.]])
        ids,sim=rankings(q,bank,2)
        self.assertEqual(ids[:,0].tolist(),[0,1])
        self.assertEqual(sim[0,0],1.)
        with self.assertRaises(ValueError):rankings(q,bank,4)

    def test_random_order_independent(self):
        a=random_rankings(3,15,4,['a','b','c'],42)
        b=random_rankings(3,15,4,['c','a','b'],42)
        np.testing.assert_array_equal(a[0],b[1])
        np.testing.assert_array_equal(a[1],b[2])
        self.assertEqual(len(set(a[0].tolist())),4)

    def test_prompts_exact_baseline_and_language_hint_warning(self):
        zh='请回答下面的问题。\n什么颜色？\n(A) Red\n(B) Blue'
        en='Answer this question.\nWhat color?\n(A) Red\n(B) Blue'
        row={'questions':{'zh':zh,'en':en}}
        self.assertEqual(build_prompt(row,'baseline_zh',[]),PREFIX+zh)
        self.assertEqual(build_prompt(row,'baseline_en',[]),PREFIX+en)
        p=build_prompt(row,'joint',['A dog is running.'])
        self.assertIn('A dog is running.',p)
        self.assertIn('OTHER training images',p)
        self.assertIn('not descriptions of the current image',p)
        self.assertIn(zh,p)
        self.assertNotIn(en,p)
        with self.assertRaises(ValueError):english_prompt(zh,[])

    def test_caption_matches_image_ids_not_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=pathlib.Path(tmp)/'caps.txt'
            src.write_text('xyz.jpg#0 A cat.\nabc.jpg#0 A person.\nxyz.jpg#1 Alternative.\n')
            c=parse_flickr_captions(src)
            self.assertEqual(bank_captions(np.array(['abc','xyz']),c),['A person.','A cat.'])
            with self.assertRaises(ValueError):bank_captions(np.array(['abc','abc']),c)
            with self.assertRaises(ValueError):bank_captions(np.array(['abc','absent']),c)

    def test_image_cluster_bootstrap(self):
        rows=[{'id':'a','image_pixel_sha256':'img1'},
              {'id':'b','image_pixel_sha256':'img1'},
              {'id':'c','image_pixel_sha256':'img2'}]
        before={k:{'correct':v} for k,v in [('a',0),('b',1),('c',0)]}
        after={k:{'correct':v} for k,v in [('a',1),('b',0),('c',1)]}
        s=image_cluster_bootstrap(rows,before,after,repeats=100,seed=7)
        self.assertEqual(s['unique_images'],2)
        self.assertEqual(s['repaired'],2)
        self.assertEqual(s['harmed'],1)
        self.assertAlmostEqual(s['delta_pp'],100/3)

    def test_checkpoints_not_used_as_decoder_states(self):
        # Functional test: the map may only consume 2D projected feature arrays.
        ck={'W':np.eye(2),'mean_x':np.zeros((1,2)),'mean_y':np.zeros((1,2))}
        with self.assertRaises((ValueError,IndexError)):
            map_vectors(np.zeros((1,2,2)),np.zeros((2,2)),ck)

    def test_known_unit_vectors_and_identical_map_rank_consistency(self):
        q=np.eye(3)
        bank=np.eye(3)
        ck={'W':np.eye(3),'mean_x':np.zeros((1,3)),'mean_y':np.zeros((1,3))}
        source,target=map_vectors(q,bank,ck)
        ix,_=rankings(source,target,1)
        self.assertEqual(ix.flatten().tolist(),[0,1,2])


if __name__=='__main__': unittest.main()