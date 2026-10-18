#!/usr/bin/env python3
"""CPU-only tests of language steering hooks (torch required)."""
import types
import torch
from combo_core import capture_last, lang_steering, steer_last

class Block(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = torch.nn.Linear(d, d)
    def forward(self, x):
        return (x + self.lin(x),)

def toy(d=8, n=3):
    m = types.SimpleNamespace()
    m.language_model = types.SimpleNamespace(
        model=types.SimpleNamespace(layers=torch.nn.ModuleList(Block(d) for _ in range(n))))
    def run(x):
        for b in m.language_model.model.layers:
            x = b(x)[0]
        return x
    return m, run

def main():
    torch.manual_seed(0)
    h = torch.randn(2, 5, 8)
    v = torch.randn(8)
    y = steer_last(h, v, 1.5)
    assert torch.allclose(y[:, :-1], h[:, :-1])
    assert torch.allclose(y[:, -1].norm(dim=-1), h[:, -1].norm(dim=-1), atol=1e-5)
    assert not torch.allclose(y[:, -1], h[:, -1])
    m, run = toy()
    x = torch.randn(1, 4, 8)
    ref = run(x)
    with lang_steering(m, [(1, v, 1.0)]):
        out = run(x)
    assert torch.allclose(out[:, :-1], ref[:, :-1])
    assert not torch.allclose(out[:, -1], ref[:, -1])
    assert torch.allclose(run(x), ref)
    with capture_last(m, [0, 2]) as st:
        run(x)
    assert set(st) == {0, 2} and st[0].shape == (8,)
    try:
        with lang_steering(m, [(0, v, 1.)]):
            raise RuntimeError("intentional")
    except RuntimeError:
        pass
    assert torch.allclose(run(x), ref)
    print("PASS combo hook tests")

if __name__ == "__main__":
    main()
