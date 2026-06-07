"""
Representation-geometry metrics (Wang & Isola 2020; Roy & Vetterli 2007).

alignment(z1, z2)   — mean squared distance between L2-normalised positive pairs
                      (lower = views of the same sample map closer).
uniformity(z, t)    — log E_{i!=j} exp(-t * ||z_i - z_j||^2) over normalised z
                      (more negative = more uniformly spread on the sphere).
effective_rank(z)   — exp(entropy of normalised singular values) of z
                      (collapse / spline-death detector; see Explainer §5).

All accept torch Tensors and return scalar Tensors.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _normalize(z: Tensor, eps: float = 1e-12) -> Tensor:
    return z / z.norm(dim=-1, keepdim=True).clamp_min(eps)


def alignment(z1: Tensor, z2: Tensor) -> Tensor:
    z1, z2 = _normalize(z1), _normalize(z2)
    return (z1 - z2).pow(2).sum(dim=1).mean()


def uniformity(z: Tensor, t: float = 2.0) -> Tensor:
    z = _normalize(z)
    sq_pdist = torch.pdist(z).pow(2)  # (N*(N-1)/2,)
    return torch.log(torch.exp(-t * sq_pdist).mean())


def effective_rank(z: Tensor, eps: float = 1e-12) -> Tensor:
    s = torch.linalg.svdvals(z)
    s = s[s > eps]
    if s.numel() == 0:
        return torch.tensor(1.0)
    p = s / s.sum()
    entropy = -(p * (p + eps).log()).sum()
    return torch.exp(entropy)
