import unittest
from prepare_real import parse_captions,groups,norm,assemble_pools
class RealTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_captions('abc.jpg#zhm#0 一个 人\nabc.jpg#enc#1 A person'),{('abc',0):'一个 人',('abc',1):'A person'})
    def test_disjoint_groups(self):
        g=groups([str(i) for i in range(5000)],42)
        flat=[v for values in g.values() for v in values]
        self.assertEqual(len(flat),len(set(flat)))
        self.assertEqual(len(g['anchor']),1000)
    def test_disjoint_bank_text_pool(self):
        import numpy as np
        data={}
        for key, ids in [('l_pair',['a']),('l_pool',['b']),('e_pair',['a']),
                         ('e_bridge',['bridge']),('e_bank',['bank']),
                         ('i_bridge',['bridge']),('i_bank',['bank'])]:
            data[key]=np.ones((len(ids),2));data[key+'_ids']=np.asarray(ids)
        assemble_pools(data)
        self.assertEqual(data['e_pool_ids'].tolist(),['bridge'])
        self.assertTrue(set(data['e_pool_ids']).isdisjoint(set(data['e_bank_ids'])))
    def test_normalization(self):
        self.assertEqual(norm('Ａ B\n'),norm('ab'))
if __name__=='__main__': unittest.main()
