"""Exercise prefill-only forward hooks using a tiny fake PaliGemma-like model."""
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from types import SimpleNamespace
from vlm import inputs,captured_states,generate


class Layer(torch.nn.Module):
    def forward(self,h):
        return (h+1.0,None)


class FakeProcessor:
    def __init__(self):
        self.tokenizer=SimpleNamespace(decode=lambda ids,skip_special_tokens=True:'B')
    def __call__(self,images,text,return_tensors):
        assert return_tensors=='pt'
        return {'input_ids':torch.tensor([[99,99,1,2,3]],dtype=torch.long),
                'attention_mask':torch.ones((1,5),dtype=torch.long)}


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(image_token_index=99)
        self.language_model=SimpleNamespace(model=SimpleNamespace(layers=torch.nn.ModuleList([Layer(),Layer()])))
        self.last_prefill=None
        self.last_decode=None
    def forward(self,input_ids,**kw):
        h=input_ids.float().unsqueeze(-1).repeat(1,1,6)
        for layer in self.language_model.model.layers:
            h=layer(h)[0]
        if input_ids.shape[1]>1:self.last_prefill=h.clone()
        else:self.last_decode=h.clone()
        return SimpleNamespace(logits=torch.empty(1))
    def generate(self,input_ids,**kw):
        self.forward(input_ids)
        self.forward(input_ids[:,-1:]) # simulate cache decode callback
        return torch.cat([input_ids,torch.tensor([[17]])],dim=1)


class HookIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/'fake.jpg'
        Image.new('RGB',(2,2)).save(self.path)
        self.row={'image':str(self.path)}
        self.proc=FakeProcessor();self.model=FakeModel()
    def tearDown(self):self.tmp.cleanup()

    def test_capture_text_and_image(self):
        result=captured_states(self.model,inputs(self.proc,self.row,'caption'),[0,1],need_visual=True)
        self.assertEqual(result[0]['text'].shape,(6,))
        self.assertEqual(result[1]['visual'].shape,(6,))
        self.assertAlmostEqual(float(result[0]['text'][0]),4)
        self.assertAlmostEqual(float(result[0]['visual'][0]),100)

    def test_hook_applies_only_to_prefill_last_text(self):
        calls=[]
        def change(h,vision):
            calls.append(True)
            return h+3
        text=generate(self.model,self.proc,self.row,'question',layer=0,operator=change)
        self.assertEqual(text,'B')
        self.assertEqual(len(calls),1,'Do not modify incremental KV-cache generation')
        # first two image tokens and remaining text positions are untouched
        self.assertAlmostEqual(self.model.last_prefill[0,0,0].item(),101)
        self.assertAlmostEqual(self.model.last_prefill[0,4,0].item(),8)
        self.assertAlmostEqual(self.model.last_decode[0,0,0].item(),5)

if __name__=='__main__':unittest.main()