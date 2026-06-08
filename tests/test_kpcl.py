"""
Stage 7 tests: KPCL (FNC-InfoNCE). R3 behaviour, EXACT InfoNCE recovery, gradient
isolation (w_ik detached -> grad reaches encoder via z ONLY), and NaN safety.
Synthetic [B=4, D=8] (2N=8).
"""
from __future__ import annotations

import math

import pytest
import torch

from src.losses.infonce import info_nce_loss
from src.losses.kpcl import kpcl_loss
from src.models.heads import KANHead
from src.models.encoder import KANEncoder
from src.models.knots import jaccard_weight, knot_code_compact

D, N = 8, 4
TWO_N = 2 * N


def _orthogonal():
    e = torch.eye(N, D)
    return e.clone(), e.clone()


def _zero_w():
    return torch.zeros(TWO_N, TWO_N)


# --------------------------------------------------------------------------- R3

def test_positive_only_aligned_loss_near_zero():
    z1, z2 = _orthogonal()
    out = kpcl_loss(z1, z2, _zero_w(), temperature=0.1, gamma=2.0)
    assert float(out["loss"]) < 1e-3


def test_negative_only_no_signal_finite_positive():
    z = torch.ones(N, D)
    out = kpcl_loss(z.clone(), z.clone(), _zero_w(), temperature=0.5, gamma=2.0)
    loss = float(out["loss"])
    assert math.isfinite(loss) and loss > 0
    assert abs(loss - math.log(2 * N - 1)) < 1e-4   # w=0 -> InfoNCE -> log(2N-1)


def test_fn_cancellation_lowers_loss_vs_unweighted():
    """A high w on a hard negative pair cancels it -> KPCL loss < InfoNCE loss."""
    z1, z2 = _orthogonal()
    z1[1] = z1[0]               # sample 1 collides with sample 0 (a false negative)
    z2[1] = z2[0]
    base = float(kpcl_loss(z1, z2, _zero_w(), temperature=0.1, gamma=2.0)["loss"])
    w = torch.zeros(TWO_N, TWO_N)
    # mark the 0<->1 collisions (and their views) as semantic neighbours
    for a in (0, 4):
        for b in (1, 5):
            w[a, b] = w[b, a] = 1.0
    canc = float(kpcl_loss(z1, z2, w, temperature=0.1, gamma=2.0)["loss"])
    assert canc < base, (base, canc)


# ------------------------------------------------- exact InfoNCE recovery (R3)

def test_gamma_zero_recovers_infonce_exactly():
    torch.manual_seed(0)
    z1, z2 = torch.randn(N, D), torch.randn(N, D)
    w = torch.rand(TWO_N, TWO_N)          # arbitrary weights; gamma=0 -> all (1-w)^0=1
    kpcl = kpcl_loss(z1, z2, w, temperature=0.5, gamma=0.0)["loss"]
    nce = info_nce_loss(z1, z2, temperature=0.5)["loss"]
    assert torch.allclose(kpcl, nce, atol=1e-6), (float(kpcl), float(nce))


def test_zero_weights_recovers_infonce_exactly():
    torch.manual_seed(1)
    z1, z2 = torch.randn(N, D), torch.randn(N, D)
    kpcl = kpcl_loss(z1, z2, _zero_w(), temperature=0.2, gamma=3.0)["loss"]
    nce = info_nce_loss(z1, z2, temperature=0.2)["loss"]
    assert torch.allclose(kpcl, nce, atol=1e-6)


# ------------------------------------------------------------- R7 dict keys

def test_dict_keys_present():
    z1, z2 = _orthogonal()
    out = kpcl_loss(z1, z2, _zero_w(), temperature=0.2, gamma=2.0)
    for k in ("loss", "info_nce_component", "kpcl_fn_weight_mean", "fn_weight_entropy",
              "temperature", "pos_sim_mean", "neg_sim_mean"):
        assert k in out and torch.is_tensor(out[k]), k


# ------------------------------------------------- gradient isolation (R5)

def test_gradient_reaches_encoder_via_z_only():
    torch.manual_seed(0)
    enc = KANEncoder(D, 16, 2, grid_size=5, spline_order=3)
    head = KANHead(16, 6, grid_size=5, spline_order=3)
    v1 = torch.randn(N, D, requires_grad=True)
    v2 = torch.randn(N, D, requires_grad=True)
    z1 = head(enc(v1))
    z2 = head(enc(v2))
    S = torch.cat([knot_code_compact(enc, v1), knot_code_compact(enc, v2)], dim=0)
    w = jaccard_weight(S)
    assert not S.requires_grad and not w.requires_grad     # structural readout, no grad
    out = kpcl_loss(z1, z2, w, temperature=0.2, gamma=2.0)
    out["loss"].backward()
    # gradient reaches encoder spline coefficients (the only path: through z)
    g = enc.layers[0].coefficients.grad
    assert g is not None and float(g.abs().sum()) > 0
    assert w.grad is None and S.grad is None               # no grad to w_ik or S


def test_loss_does_not_backprop_into_w():
    z1 = torch.randn(N, D, requires_grad=True)
    z2 = torch.randn(N, D, requires_grad=True)
    w = torch.rand(TWO_N, TWO_N, requires_grad=True)       # even if caller passes grad w
    out = kpcl_loss(z1, z2, w, temperature=0.2, gamma=2.0)
    out["loss"].backward()
    assert w.grad is None                                  # detached inside (R5/pitfall 6)
    assert z1.grad is not None


# ------------------------------------------------------------- NaN safety

def test_nan_safe_under_saturated_similarity():
    z = torch.ones(N, D)                                   # all identical -> cos=1 saturated
    w = torch.full((TWO_N, TWO_N), 0.99)                   # near-1 weights -> (1-w)^g tiny
    out = kpcl_loss(z.clone(), z.clone(), w, temperature=0.01, gamma=4.0)
    assert math.isfinite(float(out["loss"]))


def test_all_weights_one_is_finite():
    """w=1 on all negatives -> every negative term drops -> loss is finite (-> 0)."""
    z1, z2 = _orthogonal()
    out = kpcl_loss(z1, z2, torch.ones(TWO_N, TWO_N), temperature=0.1, gamma=2.0)
    assert math.isfinite(float(out["loss"]))
