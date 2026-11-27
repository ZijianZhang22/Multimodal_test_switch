"""CPU data and resume checks for the larger A-only downstream evaluation."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_data import write_jsonl,read_jsonl,digest
from prepare_extended_cvbench import (pixel_sha,answer_letter,fixed_option,
                                       choose_extra,build,build_questions)
from run_visual_large import make_record,group_stats,existing_records,load_protocol,run_predictions
from geometry import save_map


class FakeTranslator:
    def __init__(self,model_name,batch,revision=None):pass
    def translate(self,texts):return ['汉语：'+s for s in texts]


def photo(seed):
    return Image.new('RGB',(7,7),(seed,seed+1,seed+2))


class ExtendedTests(unittest.TestCase):
    def test_options_preserve_numbers_and_direction(self):
        self.assertEqual(fixed_option('LEFT'),'左边')
        self.assertEqual(fixed_option('right.'),'右边')
        self.assertEqual(fixed_option('12'),'12')
        self.assertIsNone(fixed_option('orange house'))
        self.assertEqual(answer_letter('4',['0','1','2','3','4']),'E')
        self.assertEqual(answer_letter('B',['x','y','z']),'B')

    def test_image_encoding_independence(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.png';photo(50).save(p)
            self.assertEqual(pixel_sha(photo(50)),pixel_sha(p))
            self.assertNotEqual(pixel_sha(photo(51)),pixel_sha(p))

    def fake_record(self,pic,id_,split,task='count'):
        return dict(id=id_,image=str(pic),task=task,answer='A',split=split,
                    question='How many dogs?',choices=['one','two'],
                    questions={'en':'How many dogs?\n(A) one\n(B) two',
                               'zh':'几只狗？\n(A) 一\n(B) 二'})

    def test_select_skips_val_and_test_and_image_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            valpic=root/'val.png';testpic=root/'test.png'
            photo(20).save(valpic);photo(21).save(testpic)
            val=[{'id':'cv_1','image_pixel_sha256':pixel_sha(valpic)}]
            test=[{'id':'cv_2','image_pixel_sha256':pixel_sha(testpic)}]
            dataset=[dict(idx=3,task='count',choices=['A','B'],answer='A',question='Test',image=photo(20)),
                     dict(idx=4,task='relation',choices=['left','right'],answer='A',question='Where?',image=photo(30)),
                     dict(idx=5,task='depth',choices=['front','back'],answer='A',question='Which?',image=photo(31))]
            chosen,skip=choose_extra(dataset,val,test,2,42)
            self.assertEqual({r['id'] for r in chosen},{'cv_4','cv_5'})
            self.assertGreaterEqual(skip['pixel_duplicate'],1)

    def test_build_keeps_curated_and_labels_mt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            p1,p2=root/'v.png',root/'t.png'
            photo(1).save(p1);photo(2).save(p2)
            val=root/'val.jsonl';test=root/'test.jsonl'
            write_jsonl(val,[self.fake_record(p1,'cv_1','val')])
            write_jsonl(test,[self.fake_record(p2,'cv_2','test')])
            raw=[dict(idx=3,task='count',question='How many boxes?',choices=['1','2'],
                      answer='B',image=photo(3),source='x'),
                 dict(idx=4,task='relation',question='Which side?',choices=['left','right'],
                      answer='A',image=photo(4),source='x'),
                 dict(idx=5,task='depth',question='Ignore me',choices=['front','back'],
                      answer='A',image=photo(1),source='x')]
            with mock.patch('prepare_extended_cvbench.Translator',FakeTranslator), mock.patch('huggingface_hub.model_info') as mi:
                mi.return_value.sha='a'*40
                info=build(val,test,root/'extended',3,seed=42,dataset=raw)
            self.assertEqual(info['curated_count'],1)
            self.assertEqual(info['machine_translated_count'],2)
            result=read_jsonl(root/'extended'/'test_extended.jsonl')
            self.assertEqual(result[0]['questions']['zh'],'几只狗？\n(A) 一\n(B) 二')
            new=[r for r in result if r['evaluation_group']=='new_mt_zh']
            self.assertEqual(len(new),2)
            self.assertTrue(any('汉语：' in r['questions']['zh'] for r in new))
            for row in new:
                self.assertIn('(A)',row['questions']['zh'])
                self.assertIn('(B)',row['questions']['zh'])
                self.assertEqual(row['answer'], 'B' if row['id']=='cv_3' else 'A')
            # Safe restart and fixed-original validation/test hash checks.
            second=build(val,test,root/'extended',3,seed=42,dataset=[])
            self.assertEqual(second,info)
            with self.assertRaises(ValueError):build(val,test,root/'extended',4,dataset=[])

    def test_checkpoint_recovers_only_trailing_partial_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'pred.jsonl'
            row={'id':'cv_11','task':'count','answer':'A',
                 'evaluation_group':'new_mt_zh','questions':{'zh':'中文问题'},
                 'image_pixel_sha256':'abc'}
            good=make_record(row,'baseline_zh','A')
            path.write_bytes((json.dumps(good,ensure_ascii=False)+'\n'+
                              '{"id": "cv_12", "tag":').encode('utf-8'))
            loaded=existing_records(path,[row],'baseline_zh')
            self.assertEqual(loaded,{'cv_11':good})
            self.assertEqual(path.read_text().count('\n'),1)
            # Never hide a corrupt record that has already been newline-terminated.
            path.write_text(json.dumps(good,ensure_ascii=False)+'\n'+
                            '{"id": "cv_12", "tag":}\n',encoding='utf-8')
            with self.assertRaises(json.JSONDecodeError):
                existing_records(path,[row],'baseline_zh')

    def test_cluster_dependence_disables_naive_mcnemar(self):
        rows=[dict(id=f'cv_{i}',image_pixel_sha256='same',task='count')
              for i in range(3)]
        baseline={r['id']:dict(correct=0,parsed='B') for r in rows}
        visual={r['id']:dict(correct=1,parsed='A') for r in rows}
        report=group_stats(rows,baseline,visual,bootstrap=50)
        self.assertEqual(report['unique_images'],1)
        self.assertIsNone(report['exact_mcnemar_p'])
        self.assertGreater(report['image_cluster_signflip_p'],0.9)

    def test_two_arm_predictions_and_restart_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[dict(id='cv_12',task='depth',answer='A',evaluation_group='new_mt_zh',
                       questions={'zh':'中文？'},image_pixel_sha256='im1',image='not-read.png')]
            calls=[]
            def fake_generate(model,proc,row,prompt,*,layer=None,operator=None,max_new_tokens=8):
                calls.append(layer)
                return 'B' if layer is None else 'A'
            settings=dict(max_relative_norm=.15,max_tokens=8)
            config=dict(layer=8,strength=.6)
            with mock.patch('run_visual_large.Alignment') as fake_align, mock.patch('run_visual_large.generate',fake_generate):
                values=run_predictions(None,None,rows,{},config,settings,root)
                self.assertEqual(calls,[None,8])
                self.assertEqual(values['baseline_zh']['cv_12']['correct'],0)
                self.assertEqual(values['text_visual_ot']['cv_12']['correct'],1)
                calls.clear()
                _=run_predictions(None,None,rows,{},config,settings,root)
                self.assertEqual(calls,[])

    def test_resume_and_paired_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            row={'id':'cv_11','task':'count','answer':'A','evaluation_group':'new_mt_zh',
                 'questions':{'zh':'中文问题'},'image_pixel_sha256':'abc'}
            b=make_record(row,'baseline_zh','B')
            v=make_record(row,'text_visual_ot','A')
            file=root/'b.jsonl'
            file.write_text(json.dumps(b,ensure_ascii=False)+'\n',encoding='utf-8')
            self.assertEqual(existing_records(file,[row],'baseline_zh'),{'cv_11':b})
            self.assertEqual(group_stats([row],{'cv_11':b},{'cv_11':v},bootstrap=10)['repaired'],1)
            corrupted=dict(b);corrupted['answer']='B'
            file.write_text(json.dumps(corrupted)+'\n',encoding='utf-8')
            with self.assertRaises(ValueError):existing_records(file,[row],'baseline_zh')

if __name__=='__main__':unittest.main()
