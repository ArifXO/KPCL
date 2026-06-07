"""
Stage 4 tests: multi-label SupCon (R3 mask behaviour, R7 keys).

R3 on synthetic [B=4, D=8]: 2 classes via labels (share >=1 label = positive).
  - aligned positives -> loss ~ 0
  - no positive signal (random embeddings) -> finite, > 0  (and > aligned)
  - inject a hard cross-class collision -> loss increases
"""
from __future__ import annotations

import math

import pytest
import torch

from src.losses.supcon import supcon_loss

D = 8
# 4 samples: 0,1 in class A; 2,3 in class B (multi-label one-hot here)
LABELS = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
DIR = torch.tensor([[1.0] + [0.0] * (D - 1), [0.0, 1.0] + [0.0] * (D - 2)])  # e0, e1


def _aligned():
    """Class A -> e0, class B -> e1; identical views (within-class sim=1, cross=0)."""
    z = torch.stack([DIR[0], DIR[0], DIR[1], DIR[1]])  # (4, D)
    return z.clone(), z.clone()


def test_aligned_single_positive_near_zero():
    """Unique labels -> each anchor's only positive is its own view; aligned -> loss ~0.

    (The L^sup_out form has a floor of log|P(i)| per anchor, so ~0 requires |P|=1.)
    """
    labels = torch.eye(4)  # 4 unique single labels
    z = torch.stack([torch.eye(4, D)[i] for i in range(4)])  # distinct unit dirs
    out = supcon_loss(z.clone(), z.clone(), labels, temperature=0.1)
    assert float(out["loss"]) < 1e-2, float(out["loss"])


def test_aligned_multilabel_floor_is_log_num_positives():
    """2-class aligned config: 3 equally-aligned positives/anchor -> loss == log(3)."""
    z1, z2 = _aligned()
    out = supcon_loss(z1, z2, LABELS, temperature=0.1)
    assert abs(float(out["loss"]) - math.log(3.0)) < 1e-3, float(out["loss"])


def test_no_signal_finite_positive_and_larger():
    z1, z2 = _aligned()
    base = float(supcon_loss(z1, z2, LABELS, temperature=0.1)["loss"])
    torch.manual_seed(0)
    r1, r2 = torch.randn(4, D), torch.randn(4, D)
    rnd = float(supcon_loss(r1, r2, LABELS, temperature=0.1)["loss"])
    assert math.isfinite(rnd) and rnd > 0
    assert rnd > base


def test_inject_hard_cross_class_increases_loss():
    z1, z2 = _aligned()
    base = float(supcon_loss(z1, z2, LABELS, temperature=0.1)["loss"])
    z1b, z2b = z1.clone(), z2.clone()
    z1b[2] = z1b[0]   # class-B sample 2 collides with class-A direction (hard negative)
    z2b[2] = z2b[0]
    inj = float(supcon_loss(z1b, z2b, LABELS, temperature=0.1)["loss"])
    assert inj > base, (base, inj)


def test_dict_keys_present():
    z1, z2 = _aligned()
    out = supcon_loss(z1, z2, LABELS, temperature=0.2)
    for k in ("loss", "supcon_component", "temperature", "n_positives_mean"):
        assert k in out and torch.is_tensor(out[k]), k


def test_n_positives_mean_value():
    """Each anchor (2N=8) has 3 same-class positives (2 other same-class x views, minus self)."""
    z1, z2 = _aligned()
    out = supcon_loss(z1, z2, LABELS, temperature=0.2)
    assert abs(float(out["n_positives_mean"]) - 3.0) < 1e-6


def test_raises_when_no_positives():
    """All-distinct single labels AND label-less rows -> no shared labels -> raise."""
    z = torch.randn(2, D)
    labels = torch.zeros(2, 2)  # both label-less -> no positives anywhere
    with pytest.raises(ValueError, match="no positive pairs"):
        supcon_loss(z, z, labels, temperature=0.2)


def test_gradient_flows():
    z1 = torch.randn(4, D, requires_grad=True)
    z2 = torch.randn(4, D, requires_grad=True)
    supcon_loss(z1, z2, LABELS, temperature=0.2)["loss"].backward()
    assert z1.grad is not None and float(z1.grad.abs().sum()) > 0
