import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import normalize, partition, parse_answer, make_prompt
from report import summarize


class ProtocolTests(unittest.TestCase):
    def test_v11_zero_based_and_filter(self):
        source = dict(lang='en', prompt_type=1, qa_id=1, food_id=2, question='Which dish?',
                      options=['one', 'two', 'three', 'four', 'five'], mcq_answer_index=0, image_url='https://example/a')
        row = normalize(source)
        self.assertEqual(row['answer'], 'A')
        self.assertEqual(normalize(dict(source, mcq_answer_index=4))['answer'], 'E')
        self.assertIsNone(normalize(dict(source, lang='zh')))
        self.assertIsNone(normalize(dict(source, prompt_type=4)))
        for bad in [-1, 5, True, '1']:
            with self.assertRaises(ValueError):
                normalize(dict(source, mcq_answer_index=bad))
        self.assertIn('E. five', make_prompt(row))

    def test_no_food_or_image_overlap(self):
        rows = [dict(id=f'{dish}-{im}', food_id=str(dish), image_url=f'https://example/{dish}/{im}')
                for dish in range(10) for im in range(3)]
        rows += [dict(rows[0], id='duplicate-question')]
        split = partition(rows, 7, 9, 42)
        val = [r for r in split if r['split'] == 'val']
        test = [r for r in split if r['split'] == 'test']
        self.assertEqual((len(val), len(test)), (7, 9))
        for field in ['food_id', 'image_url']:
            self.assertFalse({r[field] for r in val} & {r[field] for r in test})
        self.assertEqual(split, partition(list(reversed(rows)), 7, 9, 42))
        with self.assertRaises(ValueError):
            partition(rows, 100, 100, 42)

    def test_ambiguous_image_rejected(self):
        rows = [dict(id=str(i), food_id=str(i), image_url='same') for i in range(3)]
        with self.assertRaises(ValueError):
            partition(rows, 1, 1, 42)

    def test_parser_never_guesses_prose(self):
        options = ['dumplings', 'soup', 'bread', 'rice', 'salad']
        for text in ['A', '(a)', 'Answer: A.', 'dumplings']:
            self.assertEqual(parse_answer(text, options), 'A')
        for text in ['A or B', 'A bowl of soup', 'I think A', '', 'F', '1', 'bread and rice']:
            self.assertEqual(parse_answer(text, options), '')
        self.assertEqual(parse_answer('soup', ['soup'] * 5), '')

    def test_paired_reporting_includes_invalid(self):
        rows = []
        for cond, correctness in [('baseline', [1, 0]), ('food', [0, 1])]:
            for i, correct in enumerate(correctness):
                rows.append(dict(id=str(i), food_id=str(i), condition=cond,
                    layer=-1 if cond == 'baseline' else 8, alpha=0. if cond == 'baseline' else .1,
                    random_seed=-1, correct=correct, valid=correct))
        report = summarize(rows, ['0', '1'])
        food = next(r for r in report if r['condition'] == 'food')
        self.assertEqual((food['accuracy'], food['valid_rate'], food['delta']), (.5, .5, 0.))
        self.assertEqual((food['gained'], food['lost']), (1, 1))
        with self.assertRaises(ValueError):
            summarize(rows[:-1], ['0', '1'])

    def test_custom_anchor_integrity(self):
        anchors = json.loads((Path(__file__).resolve().parents[1] / 'concepts.json').read_text())
        self.assertEqual(len(anchors), 20)
        for row in anchors:
            self.assertEqual(row['sentence'].count(row['target']), 1)


if __name__ == '__main__':
    unittest.main()
