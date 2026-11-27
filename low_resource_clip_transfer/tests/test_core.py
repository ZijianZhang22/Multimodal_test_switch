import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from transfer_core import (CIFAR10_EN, CIFAR10_ZH, RECORD_BYTES, class_prototypes,
                  classify, decode_test_records, make_prompts, paired_bootstrap,
                  project_images, project_map, select_balanced_indices, unit)


class TestCLIPTransfer(unittest.TestCase):
    def test_class_labels(self):
        self.assertEqual(len(CIFAR10_EN), 10)
        self.assertEqual(CIFAR10_EN[0], 'airplane')
        self.assertEqual(CIFAR10_ZH[9], '卡车')

    def test_cifar_binary_layout(self):
        one = np.r_[np.array([7], dtype=np.uint8), np.arange(3072, dtype=np.uint16).astype('uint8')]
        data = bytes(one) * 10000
        imgs, labels = decode_test_records(data)
        self.assertEqual(imgs.shape, (10000, 32, 32, 3))
        self.assertEqual(labels[0], 7)
        self.assertEqual(int(imgs[0,0,0,0]), 0)
        self.assertEqual(int(imgs[0,0,0,1]), 0)
        self.assertEqual(int(imgs[0,0,1,0]), 1)

    def test_cifar_rejects_truncated(self):
        with self.assertRaises(ValueError):
            decode_test_records(b'\0' * RECORD_BYTES)

    def test_stratified_selection(self):
        labels = np.tile(np.arange(10), 1000)
        a = select_balanced_indices(labels, 500, 42)
        b = select_balanced_indices(labels, 500, 42)
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.array_equal(np.bincount(labels[a]), np.full(10, 50)))
        self.assertEqual(len(select_balanced_indices(labels, 0, 17)), 10000)

    def test_small_stratified(self):
        labels = np.tile(np.arange(10), 2)
        got = select_balanced_indices(labels, 11, 4)
        self.assertEqual(np.bincount(labels[got]).tolist(), [2] + [1] * 9)

    def test_reject_small(self):
        with self.assertRaises(ValueError):
            select_balanced_indices(np.tile(np.arange(10), 10), 9, 7)

    def test_prompts_and_prototypes(self):
        p = make_prompts(CIFAR10_ZH, ['a {label}', 'b {label}'])
        self.assertEqual(len(p), 20)
        self.assertIn('飞机', p[0])
        raw = np.tile(np.eye(10), (1,2)).reshape(20,10)
        self.assertEqual(class_prototypes(raw, 2).shape, (10, 10))

    def test_map_orthogonal_and_source_centering(self):
        x = np.array([[2., 0.], [0., 2.]])
        chk = {'W': np.array([[0., 1.], [-1., 0.]]),
               'mean_x': np.zeros((1,2)), 'mean_y': np.zeros((1,2))}
        got = project_map(x, chk)
        self.assertTrue(np.allclose(got[0], [0., 1.]))
        self.assertTrue(np.allclose(got[1], [-1., 0.]))

    def test_no_wrong_image_mean(self):
        class PCA:
            def transform(self, x):
                return unit(x)
        x = np.array([[1., 0.], [0., 1.]])
        result = project_images(x, PCA(), np.array([[.1, .1]]))
        self.assertTrue(np.allclose(np.linalg.norm(result, axis=1), 1.))
        with self.assertRaises(ValueError):
            project_images(x, PCA(), np.zeros((1,3)))

    def test_prediction(self):
        proto = np.eye(10)
        p, scores = classify(proto, proto)
        self.assertTrue(np.array_equal(p, np.arange(10)))
        self.assertEqual(scores.shape, (10,10))

    def test_paired_bootstrap_and_repairs(self):
        labels = np.repeat(np.arange(10), 10)
        base = labels.copy()
        alt = labels.copy()
        alt[0] = 9
        stat = paired_bootstrap(labels, base, alt, n_boot=20, seed=3)
        self.assertAlmostEqual(stat['delta_pp'], -1)
        self.assertEqual(stat['harms'], 1)
        self.assertEqual(stat['repairs'], 0)

    def test_map_rejects_bad_shapes(self):
        chk={'W':np.ones((2,2)), 'mean_x':np.zeros((1,2)),'mean_y':np.zeros((1,2))}
        with self.assertRaises(ValueError):
            project_map(np.eye(2), chk)


if __name__ == '__main__':
    unittest.main()