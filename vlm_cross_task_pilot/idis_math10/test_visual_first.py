#!/usr/bin/env python3
"""Fast sanity checks for A/B/C prompt and scoring."""
import unittest
from run import prompt_for
from score_three import final_segment,extracted_answer,warmup_answer


class TestConditionB(unittest.TestCase):
    def test_original_prompt_is_exact_prefix(self):
        q="Find the angle x using the figure."
        a=prompt_for(q,"original")
        b=prompt_for(q,"visual_first")
        c=prompt_for(q,"easy_first")
        self.assertTrue(b.startswith(a))
        self.assertTrue(c.endswith(a))
        self.assertIn("ONLY AFTER Question 2",b)
        self.assertIn("17 + 23",b)

    def test_only_final_answers(self):
        r={"prediction":"<think>Perhaps <math_answer>999</math_answer>.</think>"
                        "<math_answer>71</math_answer><warmup>40</warmup>"}
        self.assertEqual(extracted_answer(r),"71")
        self.assertEqual(warmup_answer(r),"40")
        self.assertEqual(final_segment({"prediction":"no final"}),"")


if __name__=="__main__":
    unittest.main()
