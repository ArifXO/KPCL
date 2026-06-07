"""
Stage 4 tests: Debiased Contrastive Loss (R3, R7, and tau_plus=0 == InfoNCE).

R3 on synthetic [B=4, D=8]:
  - aligned positives -> loss ~ 0
  - no-signal (all equal) -> finite > 0 (== log(2N-1))
  - inject hard negative -> loss increases
Plus the defining property: tau_plus=0 recovers InfoNCE exactly.
"""
from __future__ import annotations

import math

import pytest
import torch

from src.losses.dcl import dcl_loss
from src.losses.infonce import info_nce_loss

D = 8
N = 4


def _orthogonal():
    e = torch.eye(N, D)
    return e.clone(), e.clone()


def test_aligned_loss_near_zero():
    z1, z2 = _orthogonal()
    out = dcl_loss(z1, z2, temperature=0.1, tau_plus=0.1)
    assert float(out["loss"]) < 1e-2, float(out["loss"])


def test_no_signal_finite_positive():
    z = torch.ones(N, D)
    loss = float(dcl_loss(z.clone(), z.clone(), temperature=0.5, tau_plus=0.0)["loss"])
    assert math.isfinite(loss) and loss > 0
    assert abs(loss - math.log(2 * N - 1)) < 1e-4  # all logits equal -> log(2N-1)


def test_inject_false_negative_increases_loss():
    z1, z2 = _orthogonal()
    base = float(dcl_loss(z1, z2, temperature=0.1, tau_plus=0.1)["loss"])
    z1b, z2b = z1.clone(), z2.clone()
    z1b[1] = z1b[0]
    z2b[1] = z2b[0]
    inj = float(dcl_loss(z1b, z2b, temperature=0.1, tau_plus=0.1)["loss"])
    assert inj > base, (base, inj)


def test_tau_plus_zero_equals_infonce():
    torch.manual_seed(0)
    z1, z2 = torch.randn(N, D), torch.randn(N, D)
    dcl = dcl_loss(z1, z2, temperature=0.5, tau_plus=0.0)["loss"]
    nce = info_nce_loss(z1, z2, temperature=0.5)["loss"]
    assert torch.allclose(dcl, nce, atol=1e-5), (float(dcl), float(nce))


def test_dict_keys_present():
    z1, z2 = _orthogonal()
    out = dcl_loss(z1, z2, temperature=0.2, tau_plus=0.1)
    for k in ("loss", "dcl_component", "tau_plus", "neg_correction_mean"):
        assert k in out and torch.is_tensor(out[k]), k


def test_neg_term_clamped_to_floor():
    """Large tau_plus drives the bracket negative -> g must clamp at exp(-1/t), not go <0."""
    z1, z2 = _orthogonal()
    out = dcl_loss(z1, z2, temperature=0.1, tau_plus=0.9)
    assert float(out["neg_correction_mean"]) >= math.exp(-1.0 / 0.1) - 1e-6
    assert math.isfinite(float(out["loss"]))


def test_raises_on_invalid_tau_plus():
    z1, z2 = _orthogonal()
    with pytest.raises(ValueError, match="tau_plus"):
        dcl_loss(z1, z2, temperature=0.2, tau_plus=1.0)


def test_gradient_flows():
    z1 = torch.randn(N, D, requires_grad=True)
    z2 = torch.randn(N, D, requires_grad=True)
    dcl_loss(z1, z2, temperature=0.2, tau_plus=0.1)["loss"].backward()
    assert z1.grad is not None and float(z1.grad.abs().sum()) > 0
