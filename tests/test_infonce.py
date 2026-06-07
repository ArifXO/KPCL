"""
Stage 3 tests: InfoNCE (R3 mask behaviour), dict contract (R7), geometry sanity.

R3 on synthetic [B=4, D=8] (N=4 samples, two views -> 2N=8 embeddings):
  - positive-only / aligned batch -> loss ~ 0
  - negative-only / no-signal batch -> finite, > 0  (== log(2N-1))
  - injected false negative -> loss increases (the InfoNCE pathology KPCL fixes)
"""
from __future__ import annotations

import math

import pytest
import torch

from src.losses.infonce import info_nce_loss
from src.metrics.geometry import alignment, effective_rank, uniformity

D = 8
N = 4


def _orthogonal_views():
    """4 orthogonal unit directions in D=8; both views identical (perfectly aligned)."""
    e = torch.eye(N, D)  # rows e_0..e_3 in R^8
    return e.clone(), e.clone()


# --------------------------------------------------------------------------- R3

def test_positive_only_aligned_loss_near_zero():
    z1, z2 = _orthogonal_views()
    out = info_nce_loss(z1, z2, temperature=0.1)
    assert float(out["loss"]) < 1e-3, f"aligned loss not ~0: {float(out['loss'])}"


def test_negative_only_no_signal_finite_positive():
    """All embeddings identical: positives no more similar than negatives."""
    z = torch.ones(N, D)
    out = info_nce_loss(z.clone(), z.clone(), temperature=0.5)
    loss = float(out["loss"])
    assert math.isfinite(loss) and loss > 0
    # closed form: every logit equal -> loss = log(2N-1)
    assert abs(loss - math.log(2 * N - 1)) < 1e-4


def test_injected_false_negative_increases_loss():
    """Make samples 0 and 1 identical -> they are mutual false negatives -> loss up."""
    z1, z2 = _orthogonal_views()
    base = float(info_nce_loss(z1, z2, temperature=0.1)["loss"])
    z1_fn, z2_fn = z1.clone(), z2.clone()
    z1_fn[1] = z1_fn[0]  # sample 1 collides with sample 0 (false negative)
    z2_fn[1] = z2_fn[0]
    fn = float(info_nce_loss(z1_fn, z2_fn, temperature=0.1)["loss"])
    assert fn > base, f"injected FN did not increase loss: base={base}, fn={fn}"


# --------------------------------------------------------------------------- R7

def test_dict_keys_present():
    z1, z2 = _orthogonal_views()
    out = info_nce_loss(z1, z2, temperature=0.2)
    for k in ("loss", "info_nce_component", "temperature", "pos_sim_mean", "neg_sim_mean"):
        assert k in out and torch.is_tensor(out[k]), k


def test_pos_sim_higher_than_neg_when_aligned():
    z1, z2 = _orthogonal_views()
    out = info_nce_loss(z1, z2, temperature=0.2)
    assert float(out["pos_sim_mean"]) > float(out["neg_sim_mean"])


def test_raises_on_singleton_batch():
    with pytest.raises(ValueError, match="N>=2"):
        info_nce_loss(torch.randn(1, D), torch.randn(1, D), temperature=0.2)


def test_gradient_flows_through_loss():
    z1 = torch.randn(N, D, requires_grad=True)
    z2 = torch.randn(N, D, requires_grad=True)
    info_nce_loss(z1, z2, temperature=0.2)["loss"].backward()
    assert z1.grad is not None and float(z1.grad.abs().sum()) > 0


# ----------------------------------------------------------------------- geometry

def test_alignment_zero_for_identical_positive():
    z = torch.randn(10, D)
    assert float(alignment(z, z)) < 1e-6


def test_uniformity_collapsed_worse_than_spread():
    collapsed = torch.ones(16, D)
    spread = torch.randn(16, D)
    # collapsed -> all distances 0 -> uniformity = log(1) = 0 (worst/highest)
    assert float(uniformity(spread)) < float(uniformity(collapsed))


def test_effective_rank_collapse_vs_full():
    rank1 = torch.randn(32, 1) @ torch.randn(1, D)  # rank-1 matrix
    full = torch.randn(32, D)
    assert float(effective_rank(rank1)) < float(effective_rank(full))
