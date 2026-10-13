import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment import schedules,grade,answer_from_output,build_content

class ScheduleTests(unittest.TestCase):
    def test_counts(self):
        self.assertEqual({n:len(schedules(n)) for n in (2,3,4,5)},
                         {2:4,3:8,4:10,5:12})
    def test_aliases(self):
        d=schedules(2)
        self.assertEqual(set(d["TV"]),{"T_at_1","V_at_2"})
        self.assertEqual(set(d["VT"]),{"T_at_2","V_at_1"})
    def test_answer(self):
        self.assertTrue(grade(answer_from_output("Reason\nAnswer[C]"),"C",["(A) no","(B) maybe","(C) yes"]))
        self.assertFalse(grade(answer_from_output("Answer[B]"),"C",["(A) no","(B) maybe","(C) yes"]))
    def test_interleaving(self):
        from PIL import Image
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ps=[]
            for i in range(3):
                p=Path(tmp)/f"{i}.png"
                Image.new("RGB",(16,16)).save(p);ps.append(str(p))
            q={"question":"Compare <image> versus <image> and <image>.","options":[],
               "image_paths":ps}
            parts=build_content(q,"VTV",{ps[1]:"A caption for middle"})
            self.assertEqual(sum(t["type"]=="image" for t in parts),2)
            text=" ".join(t["text"] for t in parts if t["type"]=="text")
            self.assertIn("middle",text)
if __name__=="__main__":
    unittest.main()
