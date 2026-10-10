#!/usr/bin/env python3
"""Offline tests: no GPU, model, or network required."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from run_dual_image_q1_control import load_pairs, make_messages, make_prompt
from run_text_q1_control import make_prompt as previous_text_prompt
from score_dual_image_q1_control import score


class DualImageTests(unittest.TestCase):
    def test_prompt_keeps_q2_and_old_text_prompt(self):
        q2='A difficult theorem about finite groups; find the number 27.'
        text=make_prompt(q2,'joint_text')
        visual=make_prompt(q2,'joint_image_q1')
        self.assertEqual(text,previous_text_prompt(q2))
        self.assertIn(q2,visual)
        self.assertIn('SECOND image',visual)
        self.assertNotIn('17 + 23',visual)

    def test_image_counts_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); im_a=p/'idis.png'; im_b=p/'equation.png'
            Image.new('RGB',(128,128),'green').save(im_a)
            Image.new('RGB',(300,100),'white').save(im_b)
            single=make_messages(im_a,im_b,'Prompt','joint_text')[0]['content']
            dual=make_messages(im_a,im_b,'Prompt','joint_image_q1')[0]['content']
            self.assertEqual([x['type'] for x in single],['image','text'])
            self.assertEqual([x['type'] for x in dual],['image','image','text'])
            self.assertEqual(dual[0]['image'].size,(512,512))
            self.assertEqual(dual[1]['image'].size,(300,100))

    def test_manifest_stays_fixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'pairs.jsonl'
            p.write_text(json.dumps({'id':'img1','q2_id':'m-1','q2_question':'Problem 1',
                         'q2_gold':'5','q1_label':'dog','images':{'conflicting':'/image'}})+'\n')
            rows=load_pairs(p,'conflicting')
            self.assertEqual(rows[0]['q2_id'],'m-1')
            self.assertEqual(rows[0]['q2_question'],'Problem 1')

    def test_paired_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def row(cond,pred,rep=0):
                return {'id':'img1','q2_id':'m-1','variant':'conflicting', 'condition':cond,
                        'rep':rep,'q1_gold':'40','q2_gold':'42','q2_benchmark':'gsm8k',
                        'pred':pred,'num_new_tokens':50,'hit_max_token':False}
            raw=[row('joint_text','<q1>40</q1><q2>42</q2>'),
                 row('joint_image_q1','<q1>40</q1><q2>41</q2>')]
            file=root/'predictions.jsonl'
            file.write_text(''.join(json.dumps(r)+'\n' for r in raw))
            output=score(file)
            self.assertEqual(output['stats'][0]['matched_n'],1)
            self.assertEqual(output['stats'][0]['delta_image_minus_text'],-1.0)
            with (root/'candidate_interference_dual_image_q1.csv').open() as f:
                flips=list(csv.DictReader(f))
                self.assertEqual(flips[0]['change'],'text_correct_image_wrong')


if __name__ == '__main__':
    unittest.main()
