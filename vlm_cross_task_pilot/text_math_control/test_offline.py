import unittest
from prepare import is_text_only,select_rows
from run import prompt_for,split_counts
from score import parts,segments
class ToyTokenizer:
 def encode(self,x,add_special_tokens=False):return [3] if x=="</think>" else list(range(len(x.split())))
class Tests(unittest.TestCase):
 def test_data(self):
  data=[dict(problem=f"Find x{i}",answer="1",level=j,unique_id=f"{j}-{i}") for j in range(1,6) for i in range(15)]
  self.assertEqual(len(select_rows(data,50,1043)),50)
  self.assertFalse(is_text_only("Look at the diagram below"))
 def test_prompt(self):
  a,b,c=[prompt_for("Solve x=1",x) for x in "ABC"]
  self.assertTrue(b.startswith(a));self.assertTrue(c.endswith(a))
 def test_split(self):self.assertEqual(split_counts(ToyTokenizer(),[1,3,5]),(1,1,True))
 def test_segments(self):
  a=dict(condition="A",prediction="<think>reason</think>answer",thinking_tokens=1,hit_max_token=False)
  b=dict(a,condition="B",prediction="<think>visual logic\nNow Question 3: 17+23=40</think>answer")
  c=dict(a,condition="C",prediction="<think>17+23=40\nNow Question 2: solve</think>answer")
  self.assertEqual(segments(a,ToyTokenizer()),(1,"exact"));self.assertIsNotNone(segments(b,ToyTokenizer())[0]);self.assertIsNotNone(segments(c,ToyTokenizer())[0])
if __name__=="__main__":unittest.main()
