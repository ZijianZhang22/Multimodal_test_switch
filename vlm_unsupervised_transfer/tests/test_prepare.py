"""Real file-level audit tests on tiny, self-contained Flickr-like samples."""
from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from types import SimpleNamespace
from prepare_data import prepare,read_jsonl,validate_pools
from prepare_eval import split_eval

class DataIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.raw=self.root/'raw';(self.raw/'images').mkdir(parents=True)
        names=[f'im{i:03d}' for i in range(25)]
        for i,name in enumerate(names):
            # distinct JPEG content; different pixels
            Image.new('RGB',(8,8),(i*8,i*7,i*6)).save(self.raw/'images'/(name+'.jpg'))
        for kind,lang in [('enc','EN'),('zhg','ZH')]:
            text=''.join(f'{name}.jpg#0 {lang} caption {i}\n' for i,name in enumerate(names))
            (self.raw/f'flickr8k{kind}.caption.txt').write_text(text)
        (self.raw/'flickr8ktrain.txt').write_text('\n'.join(x+'.jpg' for x in names[:22]))
        (self.raw/'flickr8kval.txt').write_text(names[22]+'.jpg\n')
        (self.raw/'flickr8ktest.txt').write_text(names[23]+'.jpg\n')
        self.out=self.root/'pools'
    def tearDown(self):self.tmp.cleanup()

    def test_file_protocol(self):
        args=SimpleNamespace(raw=self.raw,out=self.out,language='zh',swahili_csv=None,
                             download=False,n_source=5,n_english=6,n_visual=7,seed=9,image_archive=None)
        prepare(args)
        rows=validate_pools(self.out)
        self.assertEqual([len(rows[k]) for k in ('source','english','visual')],[5,6,7])
        self.assertTrue(all(x['language']=='zh' for x in rows['source']))
        # Even without changing image IDs, an edited caption invalidates the provenance.
        src=self.out/'source.jsonl'
        src.write_text(src.read_text().replace('ZH caption','CHANGED'))
        with self.assertRaises(ValueError):validate_pools(self.out)

    def test_local_eval_leakage_removal(self):
        x=self.raw/'images'/'im000.jpg'
        y=self.raw/'images'/'im001.jpg'
        rows=[dict(id='a',task='count',split='val',image=str(x)),
              dict(id='b',task='relation',split='test',image=str(x)),
              dict(id='c',task='distance',split='val',image=str(y))]
        splits=split_eval(rows)
        self.assertEqual([r['id'] for r in splits['val']],['c'])
        self.assertEqual([r['id'] for r in splits['test']],['b'])

if __name__=='__main__':unittest.main()