"""
Debiased Contrastive Loss (Chuang et al., NeurIPS 2020).

Self-supervised: the positive is the augmented view; the other 2N-2 samples are
negatives. Random negatives are actually positive with prior probability tau_plus,
which biases the InfoNCE denominator. DCL corrects it with the debiased estimator of
the negative term (per anchor, M = #negatives = 2N-2):

  g = max( (1/(1-tau_plus)) * (mean_neg_exp - tau_plus * pos_exp),  exp(-1/t) )
  L = -log( pos_exp / (pos_exp + M * g) )

The clamp floor exp(-1/t) is the theoretical minimum of the negative term (since
cosine >= -1 => exp(cos/t) >= exp(-1/t)). tau_plus=0 recovers InfoNCE EXACTLY.

Returns dict (R7): loss, dcl_component, tau_plus, neg_correction_mean (mean g).
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def _normalize(z: Tensor, eps: float = 1e-12) -> Tensor:
    return z / z.norm(dim=-1, keepdim=True).clamp_min(eps)


def dcl_loss(z1: Tensor, z2: Tensor, temperature: float, tau_plus: float) -> Dict[str, Tensor]:
    """DCL over the stacked 2N batch. temperature, tau_plus from cfg.loss."""
    N = z1.shape[0]
    if N < 2:
        raise ValueError(f"DCL needs batch N>=2, got N={N}")
    if not (0.0 <= tau_plus < 1.0):
        raise ValueError(f"tau_plus must be in [0, 1), got {tau_plus}")

    t = temperature
    z = torch.cat([_normalize(z1), _normalize(z2)], dim=0)   # (2N, D)
    two_n = 2 * N
    exp_sim = torch.exp((z @ z.t()) / t)                     # (2N, 2N)

    rows = torch.arange(two_n, device=z.device)
    pos_idx = torch.cat([torch.arange(N, two_n), torch.arange(0, N)]).to(z.device)
    pos_exp = exp_sim[rows, pos_idx]                         # (2N,)

    self_mask = torch.eye(two_n, dtype=torch.bool, device=z.device)
    pos_mask = torch.zeros_like(self_mask)
    pos_mask[rows, pos_idx] = True
    neg_mask = ~(self_mask | pos_mask)                       # (2N, 2N)
    M = neg_mask.sum(dim=1).float()                          # = 2N-2
    mean_neg = (exp_sim * neg_mask).sum(dim=1) / M           # (2N,)

    floor = float(torch.exp(torch.tensor(-1.0 / t)))         # theoretical negative-term floor
    g = (mean_neg - tau_plus * pos_exp) / (1.0 - tau_plus)
    g = g.clamp_min(floor)                                   # debiased negative term (R9 clamp)

    loss = (-torch.log(pos_exp / (pos_exp + M * g))).mean()
    if torch.isnan(loss) or torch.isinf(loss):
        raise ValueError(
            f"NaN/Inf in DCL: tau_plus={tau_plus}, t={t}, N={N}, "
            f"min_g={float(g.min()):.4e}, min_pos={float(pos_exp.min()):.4e}"
        )

    return {
        "loss": loss,
        "dcl_component": loss.detach(),
        "tau_plus": torch.tensor(float(tau_plus), device=z.device),
        "neg_correction_mean": g.mean().detach(),
    }
