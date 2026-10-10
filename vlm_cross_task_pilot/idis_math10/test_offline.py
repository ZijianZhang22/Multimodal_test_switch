#!/usr/bin/env python3
"""CPU-only regression tests; no network downloads or GPU inference."""
import json
import tempfile
import unittest
from pathlib import Path

from prepare import remote_filename,priority
from run import prompt_for,split_token_counts
from score import auto_score,final_tag


class OfflineTest(unittest.TestCase):
    def test_hf_math_augmented_path(self):
        entry={'augmented':{'n4':'library/irrelevant/n4/Geometry/example.png'}}
        x=remote_filename(entry,'irrelevant',4)
        self.assertEqual(x,'Idis-math/visual_distractor/irrelevant/n4/Geometry/example.png')

    def test_prefix_only(self):
        q='In the diagram, determine the angle of ABC.'
        a=prompt_for(q,'original')
        b=prompt_for(q,'easy_first')
        self.assertTrue(b.endswith(a))
        self.assertIn('17 + 23',b)
        self.assertNotIn('17 + 23',a)
        self.assertIn('<math_answer>',a)

    def test_priority(self):
        self.assertLess(priority({'problem_version':'Vision Intensive'}),
                        priority({'problem_version':'Text Dominant'}))

    def test_generated_token_split(self):
        class FakeTokenizer:
            def encode(self, x, add_special_tokens=False):
                self_calls=x
                return [3] if x == '</think>' else []
        self.assertEqual(split_token_counts(FakeTokenizer(),[1,2,3,4,5]),(2,2,True))
        self.assertEqual(split_token_counts(FakeTokenizer(),[1,2,4,5]),(None,None,False))

    def test_answer_scoring(self):
        self.assertEqual(final_tag('<math_answer>42</math_answer>','math_answer'),'42')
        self.assertTrue(auto_score('0.5','1/2')[0])
        self.assertIsNone(auto_score(None,'1')[0])


if __name__=='__main__':
    unittest.main()
