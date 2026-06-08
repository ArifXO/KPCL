"""
Stage 5 tests: knot-pattern code S(x) and its Jaccard similarity, on a SYNTHETIC
head with a known grid (grid_range (-1,1), G=5, p=3 -> n_basis=8, 4 active/edge).

Verifies: identical inputs -> w=1; disjoint intervals -> w=0; exactly 4*O*I ones;
S and w carry NO gradient; MLP head raises NotImplementedError.
"""
from __future__ import annotations

import math

import pytest
import torch

from src.models.heads import KANHead, MLPHead
from src.models.knots import jaccard_weight, knot_code, smoothed_weight

IN, OUT, G, P = 2, 3, 5, 3
N_BASIS = G + P                 # 8
ONES_PER_SAMPLE = 4 * OUT * IN  # p+1=4 active basis per edge, O*I edges -> 24


def _head():
    return KANHead(IN, OUT, grid_size=G, spline_order=P,
                   grid_range=(-1.0, 1.0), use_layer_norm=False)


# grid_range (-1,1), G=5: interval midpoints -> bucket c; -0.8 -> c=3 {0,1,2,3},
# 0.8 -> c=7 {4,5,6,7} (disjoint 4-sets on every feature).

def test_identical_inputs_weight_one():
    head = _head()
    h = torch.tensor([[0.1, -0.3], [0.1, -0.3]])  # two identical rows
    w = jaccard_weight(knot_code(head, h))
    assert torch.allclose(w, torch.ones(2, 2), atol=1e-6)


def test_disjoint_intervals_weight_zero():
    head = _head()
    h = torch.tensor([[-0.8, -0.8], [0.8, 0.8]])  # c=3 vs c=7 on every edge
    w = jaccard_weight(knot_code(head, h))
    assert float(w[0, 1]) == pytest.approx(0.0, abs=1e-6)
    assert float(w[1, 0]) == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(torch.diag(w), torch.ones(2))  # self-similarity = 1


def test_code_has_exactly_4OI_ones():
    head = _head()
    h = torch.empty(5, IN).uniform_(-0.95, 0.95)
    S = knot_code(head, h)
    assert S.shape == (5, OUT * IN * N_BASIS)
    assert torch.all(S.sum(dim=1) == ONES_PER_SAMPLE)
    assert torch.all((S == 0) | (S == 1))  # binary


def test_S_and_w_carry_no_gradient():
    head = _head()
    h = torch.randn(4, IN, requires_grad=True)
    S = knot_code(head, h)
    w = jaccard_weight(S)
    assert not S.requires_grad and S.grad is None
    assert not w.requires_grad and w.grad is None


def test_weights_in_unit_interval():
    head = _head()
    h = torch.empty(8, IN).uniform_(-1.5, 1.5)  # includes out-of-grid (clamped)
    w = jaccard_weight(knot_code(head, h))
    assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0


def test_mlp_head_raises():
    mlp = MLPHead(IN, OUT)
    with pytest.raises(NotImplementedError):
        knot_code(mlp, torch.randn(4, IN))


# --- smoothed variant ---

def test_smoothed_identical_one_and_far_smaller():
    head = _head()
    h = torch.tensor([[-0.8, -0.8], [-0.8, -0.8], [0.8, 0.8]])  # rows 0,1 same; 2 far
    w = smoothed_weight(head, h, beta=1.0)
    assert torch.allclose(torch.diag(w), torch.ones(3))
    assert float(w[0, 1]) == pytest.approx(1.0, abs=1e-6)        # same intervals
    assert float(w[0, 2]) < float(w[0, 1])                        # far -> smaller
    # |c=3 - c=7| = 4 per feature, 2 features -> L1=8 -> exp(-8)
    assert float(w[0, 2]) == pytest.approx(math.exp(-8.0), abs=1e-5)


def test_smoothed_no_gradient_and_mlp_raises():
    head = _head()
    w = smoothed_weight(head, torch.randn(4, IN), beta=0.5)
    assert not w.requires_grad
    with pytest.raises(NotImplementedError):
        smoothed_weight(MLPHead(IN, OUT), torch.randn(4, IN), beta=0.5)
