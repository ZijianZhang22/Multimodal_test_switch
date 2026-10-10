#!/usr/bin/env python3
"""Offline smoke tests: no GPU, model download or datasets required."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from run_image_text_q1_control import prompt_for, planned_tasks
from score_image_text_q1_control import score, score_one


class TestModalityControl(unittest.TestCase):
    def test_prompts(self):
        question = 'Twelve pencils minus five pencils?'
        im = prompt_for('image_q1_q2', question)
        txt = prompt_for('text_q1_q2', question)
        self.assertIn(question, im)
        self.assertIn(question, txt)
        self.assertNotIn('17 + 23', im)
        self.assertIn('17 + 23', txt)
        self.assertEqual(prompt_for('image_q2_only', question), prompt_for('q2_only', question))

    def test_only_q1_once_per_rep(self):
        rows = [{'id':'a','question':'a','gold':'1','benchmark':'gsm8k'},
                {'id':'b','question':'b','gold':'2','benchmark':'gsm8k'}]
        tasks = planned_tasks(rows,['q2_only','image_q1_only'],3)
        self.assertEqual(len(tasks), 9)

    def test_scoring_and_pairing(self):
        def mk(id,cond,pred):
            return {'q2_id':id,'condition':cond,'rep':0,'q2_benchmark':'gsm8k',
                    'q2_gold':'12','q1_gold':'40','pred':pred,
                    'num_new_tokens':50,'hit_max_token':False}
        rows = [
            mk('a','q2_only','<q2>12</q2>'),
            mk('a','text_q1_q2','<q1>40</q1><q2>12</q2>'),
            mk('a','image_q2_only','<q2>12</q2>'),
            mk('a','image_text_q1_q2','<q1>40</q1><q2>12</q2>'),
            mk('a','image_q1_q2','<q1>40</q1><q2>10</q2>'),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            score(path)
            with (path/'paired_comparisons.csv').open() as file:
                contrasts = {x['comparison']:x for x in csv.DictReader(file)}
            self.assertEqual(float(contrasts['active_visual_q1']['mean_accuracy_delta']),-1)
            self.assertEqual(float(contrasts['image_vs_text_with_same_image']['mean_accuracy_delta']),-1)
            with (path/'candidate_interference.csv').open() as file:
                cases=list(csv.DictReader(file))
            self.assertTrue(cases)
            self.assertFalse(score_one(rows[-1])['q2_correct'])
            self.assertTrue(score_one(rows[-1])['q1_correct'])


if __name__ == '__main__':
    unittest.main()
