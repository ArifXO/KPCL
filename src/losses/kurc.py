"""
KURC — Knot-Uniformity Regularized Contrastive learning.

Adds an entropy BONUS on the knot-occupancy histogram to prevent spline collapse / spline
death (Explainer §5): if most samples activate the same few knots, S(x) is uninformative
and effective rank collapses.

  L_KURC = L_base - lambda_occ * H(q),   H(q) = -sum_c q_c log q_c

q_c is the SOFT occupancy of spline cell c: the batch-mean of the partition-of-unity
B-spline activations B_c(h) over the head's edges (Explainer §6.4 — S(x) is read at the
head). q is DIFFERENTIABLE (the gradient path), so maximizing H actually spreads knot usage;
the hard discrete code S(x) used by KPCL remains detached. q_c clamped at 1e-12 before log
(R9, pitfall 5). lambda_occ=0 recovers the base loss exactly.

base is InfoNCE by default; pass a KPCL loss dict (under_kpcl) to regularize KPCL instead.
Returns dict (R7): loss, info_nce_component, kurc_entropy, occupancy_min, occupancy_max,
temperature.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor

from src.losses.infonce import info_nce_loss


def soft_occupancy(B: Tensor) -> Tensor:
    """Differentiable occupancy distribution q over spline cells from soft activations.

    B: (N, I, G+p) partition-of-unity B-spline activations (sum_c B[n,i,c] = 1). Returns q
    over the I*(G+p) cells, summing to 1. NOT detached — this is KURC's gradient path.
    """
    occ = B.mean(dim=0).reshape(-1)                      # (I*(G+p),)
    return occ / occ.sum().clamp_min(1e-12)


def kurc_loss(z1: Tensor, z2: Tensor, B: Tensor, temperature: float, lambda_occ: float,
              base_dict: Optional[Dict] = None) -> Dict[str, Tensor]:
    """KURC loss. B: (N, I, G+p) soft head activations. base_dict: optional KPCL dict."""
    base = base_dict if base_dict is not None else info_nce_loss(z1, z2, temperature)
    q = soft_occupancy(B)
    entropy = -(q * q.clamp_min(1e-12).log()).sum()      # H(q), differentiable
    loss = base["loss"] - lambda_occ * entropy
    if torch.isnan(loss) or torch.isinf(loss):
        raise ValueError(
            f"NaN/Inf in KURC: lambda_occ={lambda_occ}, H={float(entropy):.4f}, "
            f"q_max={float(q.max()):.4e}, q_min={float(q.min()):.4e}"
        )
    return {
        "loss": loss,
        "info_nce_component": base["info_nce_component"],
        "kurc_entropy": entropy.detach(),
        "occupancy_min": q.min().detach(),
        "occupancy_max": q.max().detach(),
        "temperature": torch.tensor(float(temperature), device=z1.device),
    }
