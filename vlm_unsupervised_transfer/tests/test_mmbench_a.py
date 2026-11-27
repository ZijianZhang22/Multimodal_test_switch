"""CPU-only tests: do NOT download models or MMBench or use CUDA."""
import base64
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_mmbench import resolve_base64, pixels_signature, row_question, prepare_pairs,split_image_groups, check_existing_dataset
from run_mmbench_a import choose, score, paired_stats, eval_tag, read_prediction_file, eval_one, check_saved_record


def make_pic(rgb=(40,30,70)):
    im=Image.new('RGB',(4,5),rgb);out=io.BytesIO();im.save(out,format='PNG');return base64.b64encode(out.getvalue()).decode()


def item(idx,image,answer='B',z=False):
    return {'index':str(idx),'image':image,'answer':answer,'question':('哪个物品更大？' if z else 'Which object is bigger?'),
            'A':'左' if z else 'Left','B':'右' if z else 'Right','C':'','D':''}

class DatasetTests(unittest.TestCase):
    def test_image_references(self):
        image=make_pic()
        rows={'1':item(1,image),'2':item(2,'1'),'3':item(3,'2')}
        self.assertEqual(resolve_base64('3',rows),base64.b64decode(image))

    def test_cycle_rejected(self):
        rows={'1':item(1,'2'),'2':item(2,'1')}
        with self.assertRaises(ValueError):resolve_base64('1',rows)

    def test_prompt_en_cn_and_allowed(self):
        a,ans,opts=row_question(item(2,make_pic(),z=True),'zh')
        self.assertEqual((ans,opts),('B',('A','B')))
        self.assertIn('只能',a) if False else self.assertIn('只输出',a)
        self.assertNotIn('(C)',a)

    def test_official_hint_is_included(self):
        en=item(1,make_pic())
        cn=item(1,make_pic(),z=True)
        en['hint']='Only the pictured objects count.'
        cn['hint']='只统计图中物品。'
        self.assertIn('Hint: Only the pictured objects count.',row_question(en,'en')[0])
        self.assertIn('提示：只统计图中物品。',row_question(cn,'zh')[0])
        self.assertNotIn('Hint:',row_question(item(1,make_pic()),'en')[0])

    def test_pair_pixels_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            en={str(i):item(i,make_pic((i,i,0))) for i in range(6)}
            zh={str(i):item(i,make_pic((i,i,0)),z=True) for i in range(6)}
            pairs=prepare_pairs(en,zh,Path(tmp)/'images')
            tune,test=split_image_groups(pairs,2,42)
            self.assertEqual(len(tune)+len(test),6)
            self.assertFalse({x['image_pixel_sha256'] for x in tune}&{x['image_pixel_sha256'] for x in test})
            self.assertEqual(set(p['answer'] for p in pairs),{'B'})

    def test_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            e={'1':item(1,make_pic())}
            z={'1':item(1,make_pic(),answer='A',z=True)}
            with self.assertRaisesRegex(ValueError,'answers'):
                prepare_pairs(e,z,Path(tmp)/'images')

    def test_image_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            e={'1':item(1,make_pic())}
            z={'1':item(1,make_pic((0,0,0)),z=True)}
            with self.assertRaisesRegex(ValueError,'images differ'):
                prepare_pairs(e,z,Path(tmp)/'images')

    def test_image_grouped_split(self):
        data=[{'id':str(i),'image_pixel_sha256':str(i//3)} for i in range(12)]
        tune,test=split_image_groups(data,4,2)
        self.assertFalse({x['image_pixel_sha256'] for x in tune}&{x['image_pixel_sha256'] for x in test})

    def test_paired_stats(self):
        rows=[{'id':str(i),'image_pixel_sha256':str(i//2)} for i in range(10)]
        base={r['id']:{'correct':int(i<5),'valid':1} for i,r in enumerate(rows)}
        tr={r['id']:{'correct':int(i<7),'valid':1} for i,r in enumerate(rows)}
        s=paired_stats(rows,base,tr,42,n_boot=20)
        self.assertEqual((s['repaired'],s['harmed'],s['delta_pp']),(2,0,20.0))
        self.assertIsNone(s['exact_mcnemar_p'])

    def test_tune_selection_from_predictions(self):
        rows=[{'id':str(i),'questions':{'zh':'Prompt '+str(i)},'allowed':['A','B'],
               'answer':'A','image_pixel_sha256':'hash_'+str(i)} for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            for m in ('text_visual_ot','text_ot'):
                for alpha in (0.3,0.6):
                    tag=eval_tag(m,8,alpha,'tune')
                    values=[1,0,1] if alpha==0.6 else [0,0,1]
                    with (Path(tmp)/(tag+'.jsonl')).open('w') as f:
                        for row,val in zip(rows,values):
                            import hashlib
                            pred='A' if val else 'B'
                            record={'id':row['id'],'tag':tag,'answer':row['answer'],
                              'allowed':row['allowed'],'img_sha256':row['image_pixel_sha256'],
                              'prompt_sha256':hashlib.sha256(('answer en Answer the multiple choice question by only responding the letter of the correct answer. '+row['questions']['zh']).encode()).hexdigest(),
                              'prediction':pred,'parsed':pred,'valid':1,'correct':val}
                            f.write(json.dumps(record)+'\n')
            selection,_=choose(rows,Path(tmp),[8],[.3,.6])
            self.assertEqual(selection['text_visual_ot']['alpha'],.6)
            self.assertEqual(selection['text_ot']['alpha'],.6)

    def test_checkpoint_and_invalid_answer(self):
        from types import SimpleNamespace
        rows=[{'id':'x1','questions':{'zh':'Prompt 1'},'answer':'A',
               'allowed':['A','B'],'image_pixel_sha256':'a', 'image':'unused'}]
        with tempfile.TemporaryDirectory() as tmp:
            with patch('run_mmbench_a.generate',return_value='C') as fake:
                one=eval_one(None,None,rows,Path(tmp),'tune_baseline_zh',
                             {'max_tokens':8},lang='zh')
                self.assertEqual((one['x1']['valid'],one['x1']['correct']),(0,0))
                two=eval_one(None,None,rows,Path(tmp),'tune_baseline_zh',
                             {'max_tokens':8},lang='zh')
                self.assertEqual(one,two)
                self.assertEqual(fake.call_count,1)
            p=Path(tmp)/'tune_baseline_zh.jsonl'
            with p.open('ab') as f:f.write(b'{"incomplete":')
            three=eval_one(None,None,rows,Path(tmp),'tune_baseline_zh',
                           {'max_tokens':8},lang='zh')
            self.assertEqual(one,three)

    def test_resume_rejects_false_correct_or_changed_answer(self):
        rows=[{'id':'z','questions':{'zh':'Q'},'answer':'A','allowed':['A','B'],
               'image_pixel_sha256':'xyz'}]
        with tempfile.TemporaryDirectory() as tmp:
            with patch('run_mmbench_a.generate',return_value='B'):
                out=eval_one(None,None,rows,Path(tmp),'tune_baseline_zh',{'max_tokens':8})
            saved=out['z'];self.assertEqual(saved['correct'],0)
            saved['correct']=1
            path=Path(tmp)/'tune_baseline_zh.jsonl'
            path.write_text(json.dumps(saved)+'\n')
            with self.assertRaisesRegex(ValueError,'corrupt'):
                read_prediction_file(path,rows,'tune_baseline_zh')
            with self.assertRaisesRegex(ValueError,'corrupt'):
                eval_one(None,None,rows,Path(tmp),'tune_baseline_zh',{'max_tokens':8})

    def test_complete_tune_then_heldout_test_without_gpu(self):
        from types import SimpleNamespace
        from run_mmbench_a import tune, test, store_config
        import hashlib
        def mkrow(i):
            q='Question '+str(i)
            return {'id':str(i),'questions':{'zh':q,'en':'English '+q},
                    'answer':'B','allowed':['A','B'],
                    'image_pixel_sha256':'hash_'+str(i),'category':'toy'}
        datasets={'tune':[mkrow(i) for i in range(4)],
                  'test':[mkrow(i) for i in range(4,8)]}
        config={'layer_grid':[8], 'strength_grid':[0.3], 'seed':42}
        settings={'model':'stub', 'max_tokens':8,'max_relative_norm':.15}
        mappings={(m,8):{} for m in ('text_visual_ot','text_ot')}
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(out=Path(tmp))
            store_config(args.out,config)
            with patch('run_mmbench_a.load_model',return_value=(None,None)), \
                 patch('run_mmbench_a.Alignment') as fake_align, \
                 patch('run_mmbench_a.generate',side_effect=lambda *a,**kw: 'B' if kw.get('operator') else 'A'):
                tune(args,settings,datasets,mappings,config)
                test(args,settings,datasets,mappings,config)
            result=json.loads((args.out/'test_results.json').read_text())
            self.assertEqual(result['baseline_zh']['correct'],0)
            self.assertEqual(result['arms']['text_visual_ot']['compared_correct'],4)
            self.assertEqual(result['arms']['text_ot']['compared_correct'],4)
            self.assertEqual(result['dataset_counts'],{'tune':4,'test':4})
            self.assertTrue((args.out/'comparison.csv').is_file())

    def test_tune_incomplete_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (Path(tmp)/'p.jsonl').open('w') as f:f.write(json.dumps({'id':'0','tag':'p'})+'\n')
            with self.assertRaises(ValueError):read_prediction_file(Path(tmp)/'p.jsonl',[{'id':'0'},{'id':'1'}],'p')

if __name__=='__main__':unittest.main()
