import unittest
from prepare_real import parse_captions,groups,norm
class RealTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_captions('abc.jpg#zhm#0 一个 人\nabc.jpg#enc#1 A person'),{('abc',0):'一个 人',('abc',1):'A person'})
    def test_disjoint_groups(self):
        g=groups([str(i) for i in range(5000)],42)
        flat=[v for values in g.values() for v in values]
        self.assertEqual(len(flat),len(set(flat)))
        self.assertEqual(len(g['anchor']),1000)
    def test_normalization(self):
        self.assertEqual(norm('Ａ B\n'),norm('ab'))
if __name__=='__main__': unittest.main()
