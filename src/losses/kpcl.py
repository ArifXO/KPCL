"""
KPCL — False-Negative-Cancellation InfoNCE (the core method; build only after H1 passed).

Standard SimCLR InfoNCE pushes every non-positive sample away from the anchor. But some
of those "negatives" are true semantic neighbours (false negatives). KPCL down-weights a
negative k by (1 - w_ik)^gamma, where w_ik is the DETACHED knot-Jaccard similarity from
src/models/knots.py (the MLP-impossible structural signal validated by SPEC H1):

  L_KPCL(i) = -log [ exp(z_i·z_i+ / tau) /
               ( exp(z_i·z_i+ / tau) + sum_k (1 - w_ik)^gamma * exp(z_i·z_k / tau) ) ]

gamma is a FIXED Hydra hyperparameter (NOT learned). gamma=0 (or w=0 everywhere) recovers
InfoNCE EXACTLY. w_ik is detached: the ONLY gradient path is through z (CLAUDE.md R5 /
pitfall 6). Computed with log-sum-exp for NaN safety under saturated similarities.

Returns dict (R7): loss, info_nce_component, kpcl_fn_weight_mean, fn_weight_entropy,
temperature, pos_sim_mean, neg_sim_mean.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

_NEG_INF = float("-inf")


def _normalize(z: Tensor, eps: float = 1e-12) -> Tensor:
    return z / z.norm(dim=-1, keepdim=True).clamp_min(eps)


def kpcl_loss(z1: Tensor, z2: Tensor, w: Tensor, temperature: float, gamma: float) -> Dict[str, Tensor]:
    """FNC-InfoNCE. w: (2N, 2N) DETACHED knot-Jaccard weights aligned with [z1; z2]."""
    if z1.shape != z2.shape:
        raise ValueError(f"view shape mismatch: {tuple(z1.shape)} vs {tuple(z2.shape)}")
    N = z1.shape[0]
    if N < 2:
        raise ValueError(f"KPCL needs batch N>=2, got N={N}")
    two_n = 2 * N
    if w.shape != (two_n, two_n):
        raise ValueError(f"w must be ({two_n},{two_n}), got {tuple(w.shape)}")

    w = w.detach()                                       # w_ik NEVER on the gradient path
    z = torch.cat([_normalize(z1), _normalize(z2)], dim=0)
    cos = z @ z.t()
    logits = cos / temperature                           # (2N, 2N)

    rows = torch.arange(two_n, device=z.device)
    pos_idx = torch.cat([torch.arange(N, two_n), torch.arange(0, N)]).to(z.device)
    self_mask = torch.eye(two_n, dtype=torch.bool, device=z.device)
    pos_mask = torch.zeros_like(self_mask)
    pos_mask[rows, pos_idx] = True
    neg_mask = ~(self_mask | pos_mask)

    pos_logit = logits[rows, pos_idx]                    # (2N,) unweighted positive term
    # negative terms: logit + log((1-w)^gamma); (1-w)^gamma=0 -> log=-inf -> term drops
    log_wgt = (1.0 - w).clamp(0.0, 1.0).pow(gamma).log()
    T = torch.where(neg_mask, logits + log_wgt, torch.full_like(logits, _NEG_INF))
    T[rows, pos_idx] = pos_logit                         # positive in the denominator, unweighted
    loss = (torch.logsumexp(T, dim=1) - pos_logit).mean()
    if torch.isnan(loss) or torch.isinf(loss):
        raise ValueError(
            f"NaN/Inf in KPCL: tau={temperature}, gamma={gamma}, N={N}, "
            f"max|cos|={float(cos.abs().max()):.4f}, max_w={float(w.max()):.4f}"
        )

    with torch.no_grad():
        base = logits.masked_fill(self_mask, _NEG_INF)
        info_nce_component = (torch.logsumexp(base, dim=1) - pos_logit).mean()
        wgt = (1.0 - w).clamp(0.0, 1.0).pow(gamma).masked_fill(~neg_mask, 0.0)
        p = wgt / wgt.sum(dim=1, keepdim=True).clamp_min(1e-12)
        fn_weight_entropy = -(p * p.clamp_min(1e-12).log()).sum(dim=1).mean()

    return {
        "loss": loss,
        "info_nce_component": info_nce_component,
        "kpcl_fn_weight_mean": w[neg_mask].mean(),
        "fn_weight_entropy": fn_weight_entropy,
        "temperature": torch.tensor(float(temperature), device=z.device),
        "pos_sim_mean": cos[pos_mask].mean(),
        "neg_sim_mean": cos[neg_mask].mean(),
    }
