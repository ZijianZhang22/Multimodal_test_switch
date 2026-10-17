import unittest
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from mrre import steer_tensor,intervention,get_layers,score,read

class Tests(unittest.TestCase):
    def test_norm_and_last_token(self):
        h=torch.randn(2,5,32);v=torch.randn(32)
        y=steer_tensor(h,v,.75)
        self.assertTrue(torch.equal(y[:,:-1],h[:,:-1]))
        self.assertTrue(torch.allclose(y[:,-1].norm(dim=-1),h[:,-1].norm(dim=-1),atol=1e-5))
        self.assertTrue(torch.allclose(steer_tensor(h,v,0),h,atol=1e-6))
    def test_actual_decoder_cache_and_cleanup(self):
        torch.manual_seed(1)
        m=Qwen2ForCausalLM(Qwen2Config(vocab_size=40,hidden_size=32,intermediate_size=64,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2)).eval()
        ids=torch.tensor([[2,3,4]])
        base=m(ids).logits.detach();layers=get_layers(m)
        with intervention(layers,[(1,torch.randn(32),1)]):
            changed=m(ids).logits.detach()
            cache=m(ids,use_cache=True).past_key_values
            nxt=m(torch.tensor([[5]]),past_key_values=cache,use_cache=True).logits
            self.assertTrue(torch.isfinite(nxt).all())
        self.assertFalse(torch.allclose(base[:,-1],changed[:,-1]))
        self.assertTrue(torch.equal(base,m(ids).logits.detach()))
        try:
            with intervention(layers,[(1,torch.randn(32),1)]):raise RuntimeError('test')
        except RuntimeError:pass
        self.assertFalse(layers[1]._forward_hooks)
    def test_splits(self):
        sets=[{x['group'] for x in read('data/'+s+'.jsonl')} for s in ['calibration','validation','test']]
        self.assertEqual([len(x) for x in sets],[100,30,100])
        self.assertTrue(all(not x&y for i,x in enumerate(sets) for y in sets[i+1:]))
    def test_scoring(self):
        self.assertIsNone(score('I think D',{'answer':'D'}))
        self.assertTrue(score('Final: D',{'answer':'D'}))
        self.assertFalse(score('Final: C',{'answer':'D'}))
        self.assertTrue(score('Final: 0.5',{'answer':'0.50'}))
if __name__=='__main__':unittest.main()
