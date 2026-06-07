"""
Multi-label Supervised Contrastive loss (Khosla et al. 2020), L^sup_out form.

POSITIVE CONVENTION (multi-label): two samples are positives iff they share >=1
label (Y_i · Y_j > 0). The two augmented views of one sample share all labels, so
the SimCLR augmentation-positive is subsumed. Samples with an all-zero label vector
have no positives and are skipped (R5: documented; raises only if NO anchor has any
positive). Operates on the stacked 2N batch of two views.

Returns dict (R7): loss, supcon_component, temperature, n_positives_mean.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def _normalize(z: Tensor, eps: float = 1e-12) -> Tensor:
    return z / z.norm(dim=-1, keepdim=True).clamp_min(eps)


def supcon_loss(z1: Tensor, z2: Tensor, labels: Tensor, temperature: float) -> Dict[str, Tensor]:
    """Multi-label SupCon. labels: (N, L) binary; temperature from cfg.loss.temperature."""
    N = z1.shape[0]
    if N < 2:
        raise ValueError(f"SupCon needs batch N>=2, got N={N}")

    z = torch.cat([_normalize(z1), _normalize(z2)], dim=0)   # (2N, D)
    Y = torch.cat([labels, labels], dim=0).float()           # (2N, L)
    two_n = 2 * N
    sim = (z @ z.t()) / temperature                          # (2N, 2N)
    self_mask = torch.eye(two_n, dtype=torch.bool, device=z.device)

    # log p(a | i) with the denominator over all a != i (log-softmax, stable)
    sim_denom = sim.masked_fill(self_mask, float("-inf"))
    log_prob = sim - torch.logsumexp(sim_denom, dim=1, keepdim=True)

    pos = (Y @ Y.t() > 0) & ~self_mask                       # share >=1 label, exclude self
    pos_counts = pos.sum(dim=1)                              # (2N,)
    valid = pos_counts > 0
    if not bool(valid.any()):
        raise ValueError("SupCon: no positive pairs in batch (no two samples share a label)")

    mean_log_prob_pos = (pos.float() * log_prob).sum(dim=1)[valid] / pos_counts[valid].float()
    loss = -mean_log_prob_pos.mean()
    if torch.isnan(loss) or torch.isinf(loss):
        raise ValueError(
            f"NaN/Inf in SupCon: tau={temperature}, N={N}, "
            f"valid_anchors={int(valid.sum())}/{two_n}"
        )

    return {
        "loss": loss,
        "supcon_component": loss.detach(),
        "temperature": torch.tensor(float(temperature), device=z.device),
        "n_positives_mean": pos_counts.float().mean(),
    }
