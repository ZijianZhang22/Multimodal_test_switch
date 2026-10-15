"""Run: python -m unittest discover -s tests -v"""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import conditions
class ConditionsTest(unittest.TestCase):
    def test_six_schedules(self):
        q={"qid":"q1","images":[{"id":str(i)} for i in range(4)],"support_image_ids":["0","2"]}
        cs=list(conditions(q))
        self.assertEqual([x[0] for x in cs],["VVVV","TTTT","VTTT","TVTT","TTVT","TTTV"])
        for schedule,replaced,kept in cs:
            self.assertEqual(len(replaced),schedule.count("T"))
            if kept: self.assertNotIn(kept,replaced)
    def test_refuse_wrong_image_count(self):
        q={"qid":"q1","images":[{"id":str(i)} for i in range(3)]}
        with self.assertRaises(ValueError): list(conditions(q))
if __name__=="__main__": unittest.main()
