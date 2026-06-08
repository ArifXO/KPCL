"""Stage 9 control: MLP+cosine-FN weight loss (reuses the KPCL loss machinery)."""
from __future__ import annotations

import math

import torch

from src.losses.cosfn import cosfn_loss, cosine_fn_weight

D, N = 8, 4


def test_cosine_weight_in_unit_interval_and_detached():
    z1, z2 = torch.randn(N, D, requires_grad=True), torch.randn(N, D, requires_grad=True)
    w = cosine_fn_weight(z1, z2)
    assert w.shape == (2 * N, 2 * N)
    assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0
    assert not w.requires_grad                      # detached: never on the gradient path


def test_cosfn_loss_finite_and_grad_via_z():
    z1 = torch.randn(N, D, requires_grad=True)
    z2 = torch.randn(N, D, requires_grad=True)
    out = cosfn_loss(z1, z2, temperature=0.2, gamma=2.0)
    assert math.isfinite(float(out["loss"]))
    out["loss"].backward()
    assert z1.grad is not None and float(z1.grad.abs().sum()) > 0
