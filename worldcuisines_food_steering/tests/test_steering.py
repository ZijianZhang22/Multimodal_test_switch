"""Runs on CPU when torch is installed; no downloaded weights needed."""
import importlib.util
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'torch not installed')
class HookTests(unittest.TestCase):
    def setUp(self):
        import torch
        from types import SimpleNamespace
        self.torch = torch
        self.block = torch.nn.Identity()
        self.model = SimpleNamespace(model=SimpleNamespace(layers=[self.block]))
        self.hidden = torch.zeros(1, 3, 4)
        self.mask = torch.tensor([[True, False, True]])

    def test_mask_prefill_only_and_removal(self):
        from steering import intervene
        torch = self.torch
        with intervene(self.model, 0, torch.ones(4), .5, self.mask):
            expected = self.mask.unsqueeze(-1).expand(1, 3, 4) * .5
            torch.testing.assert_close(self.block(self.hidden), expected)
            torch.testing.assert_close(self.block(torch.zeros(1, 1, 4)), torch.zeros(1, 1, 4))
        self.assertFalse(self.block._forward_hooks)
        torch.testing.assert_close(self.block(self.hidden), self.hidden)

    def test_zero_and_exception_cleanup(self):
        from steering import intervene
        with intervene(self.model, -1, None, 0, self.mask):
            self.torch.testing.assert_close(self.block(self.hidden), self.hidden)
        with self.assertRaises(ValueError):
            with intervene(self.model, 0, self.torch.ones(9), 1, self.mask):
                self.block(self.hidden)
        self.assertFalse(self.block._forward_hooks)

    def test_masks_and_random_norm(self):
        from steering import token_mask, random_control
        torch = self.torch
        ids = torch.tensor([[99, 2, 10, 0]])
        self.assertEqual(token_mask(ids, 99, [0, 2], 'both').tolist(), [[True, False, True, False]])
        self.assertEqual(token_mask(ids, 99, [0, 2], 'image').sum().item(), 1)
        self.assertEqual(token_mask(ids, 99, [0, 2], 'text').sum().item(), 1)
        v = torch.arange(4, dtype=torch.float32)
        torch.testing.assert_close(random_control(v, 0).norm(), v.norm())
        torch.testing.assert_close(random_control(v, 0), random_control(v, 0))


if __name__ == '__main__':
    unittest.main()
