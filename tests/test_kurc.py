"""
Stage 8 tests: KURC (knot-occupancy entropy regularizer).

q is the SOFT, differentiable occupancy (user-confirmed): H(q) carries gradient so the
regularizer actually spreads knot usage. Synthetic soft activations B (M, I, G+p).
"""
from __future__ import annotations

import math

import torch

from src.losses.infonce import info_nce_loss
from src.losses.kpcl import kpcl_loss
from src.losses.kurc import kurc_loss, soft_occupancy

D, N = 8, 4
I, GP, M = 4, 8, 6          # I edges, G+p basis, M samples


def _z():
    e = torch.eye(N, D)
    return e.clone(), e.clone()


def _collapsed():
    B = torch.zeros(M, I, GP)
    B[:, :, 0] = 1.0           # all mass on basis 0 -> q on I cells -> H = log(I)
    return B


def _uniform():
    return torch.full((M, I, GP), 1.0 / GP)   # q uniform over I*GP cells -> H = log(I*GP)


def test_collapsed_low_entropy_larger_penalty():
    z1, z2 = _z()
    out_c = kurc_loss(z1, z2, _collapsed(), temperature=0.2, lambda_occ=0.5)
    out_u = kurc_loss(z1, z2, _uniform(), temperature=0.2, lambda_occ=0.5)
    assert float(out_c["kurc_entropy"]) < float(out_u["kurc_entropy"])
    assert float(out_c["loss"]) > float(out_u["loss"])           # lower H -> larger penalty
    assert abs(float(out_c["kurc_entropy"]) - math.log(I)) < 1e-4
    assert abs(float(out_u["kurc_entropy"]) - math.log(I * GP)) < 1e-4


def test_lambda_zero_recovers_infonce_exactly():
    torch.manual_seed(0)
    z1, z2 = torch.randn(N, D), torch.randn(N, D)
    kurc = kurc_loss(z1, z2, _uniform(), temperature=0.3, lambda_occ=0.0)["loss"]
    nce = info_nce_loss(z1, z2, temperature=0.3)["loss"]
    assert torch.allclose(kurc, nce, atol=1e-6)


def test_dict_keys_present():
    z1, z2 = _z()
    out = kurc_loss(z1, z2, _uniform(), temperature=0.2, lambda_occ=0.1)
    for k in ("loss", "info_nce_component", "kurc_entropy",
              "occupancy_min", "occupancy_max", "temperature"):
        assert k in out and torch.is_tensor(out[k]), k


def test_soft_occupancy_is_differentiable():
    """The chosen design: H(q) carries gradient -> the regularizer is real, not a no-op."""
    B = torch.rand(M, I, GP, requires_grad=True)
    q = soft_occupancy(B)
    H = -(q * q.clamp_min(1e-12).log()).sum()
    H.backward()
    assert B.grad is not None and float(B.grad.abs().sum()) > 0


def test_regularizer_active_propagates_grad_to_B():
    z1 = torch.randn(N, D, requires_grad=True)
    z2 = torch.randn(N, D, requires_grad=True)
    B = torch.rand(M, I, GP, requires_grad=True)
    kurc_loss(z1, z2, B, temperature=0.2, lambda_occ=0.5)["loss"].backward()
    assert float(B.grad.abs().sum()) > 0          # entropy term drives knot usage
    assert z1.grad is not None                     # base InfoNCE drives z


def test_compose_under_kpcl():
    z1, z2 = _z()
    w = torch.zeros(2 * N, 2 * N)
    kpcl = kpcl_loss(z1, z2, w, temperature=0.2, gamma=2.0)
    out = kurc_loss(z1, z2, _uniform(), temperature=0.2, lambda_occ=0.1, base_dict=kpcl)
    H = float(out["kurc_entropy"])
    expected = float(kpcl["loss"]) - 0.1 * H
    assert abs(float(out["loss"]) - expected) < 1e-5


def test_nan_safe_with_dead_cells():
    z1, z2 = _z()
    out = kurc_loss(z1, z2, _collapsed(), temperature=0.01, lambda_occ=1.0)  # zeros in q
    assert math.isfinite(float(out["loss"]))
    assert float(out["occupancy_min"]) >= 0.0
