"""
SimCLR InfoNCE (NT-Xent) over a 2N batch of two augmented views.

Given view embeddings z1, z2 (each (N, D)), the positive of view-1 sample i is
view-2 sample i (and vice versa); the other 2N-2 samples are negatives. Embeddings
are L2-normalised with a clamped denominator (1e-12, pitfall 1 / R9).

Returns dict (R7): loss, info_nce_component, temperature, pos_sim_mean, neg_sim_mean.
For plain InfoNCE info_nce_component == loss; KPCL/KURC add terms on top of it.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor


def _normalize(z: Tensor, eps: float = 1e-12) -> Tensor:
    return z / z.norm(dim=-1, keepdim=True).clamp_min(eps)


def info_nce_loss(z1: Tensor, z2: Tensor, temperature: float) -> Dict[str, Tensor]:
    """NT-Xent loss over the stacked 2N batch. temperature from cfg.loss.temperature."""
    if z1.shape != z2.shape:
        raise ValueError(f"view shape mismatch: {tuple(z1.shape)} vs {tuple(z2.shape)}")
    N = z1.shape[0]
    if N < 2:
        raise ValueError(f"InfoNCE needs batch N>=2 for negatives, got N={N}")

    z = torch.cat([_normalize(z1), _normalize(z2)], dim=0)  # (2N, D)
    cos = z @ z.t()                                          # (2N, 2N) cosine
    logits = cos / temperature
    two_n = 2 * N
    diag = torch.eye(two_n, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(diag, float("-inf"))        # drop self-similarity
    targets = torch.cat([torch.arange(N, two_n), torch.arange(0, N)]).to(z.device)
    loss = F.cross_entropy(logits, targets)
    if torch.isnan(loss) or torch.isinf(loss):
        raise ValueError(
            f"NaN/Inf in InfoNCE loss: tau={temperature}, N={N}, "
            f"max|cos|={float(cos.abs().max()):.4f}"
        )

    with torch.no_grad():
        pos_mask = torch.zeros_like(diag)
        idx = torch.arange(N, device=z.device)
        pos_mask[idx, idx + N] = True
        pos_mask[idx + N, idx] = True
        neg_mask = ~(diag | pos_mask)
        pos_sim_mean = cos[pos_mask].mean()
        neg_sim_mean = cos[neg_mask].mean()

    return {
        "loss": loss,
        "info_nce_component": loss.detach(),
        "temperature": torch.tensor(float(temperature), device=z.device),
        "pos_sim_mean": pos_sim_mean,
        "neg_sim_mean": neg_sim_mean,
    }
