"""
Two-view stochastic augmentation for tabular contrastive learning.

Each view independently applies, with Hydra-configured strengths (cfg.aug):
  - feature masking: zero a random fraction (mask_frac) of features per sample
  - Gaussian noise:  add N(0, noise_std^2)
  - mixup:           convex-combine each row with a random batch partner,
                     lambda ~ Beta(mixup_alpha, mixup_alpha); skipped if alpha<=0

two_views(x, cfg) -> (v1, v2), each same shape/dtype as x. An optional
torch.Generator drives masking/noise/permutation for reproducibility; the mixup
lambda is drawn from the global RNG (Beta has no generator kwarg) and is therefore
reproducible at the run level via torch.manual_seed.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _augment_once(
    x: Tensor, mask_frac: float, noise_std: float, mixup_alpha: float,
    g: torch.Generator | None,
) -> Tensor:
    out = x
    if mask_frac > 0:
        mask = torch.rand(x.shape, generator=g, device=x.device) < mask_frac
        out = out.masked_fill(mask, 0.0)
    if noise_std > 0:
        out = out + noise_std * torch.randn(x.shape, generator=g, device=x.device)
    if mixup_alpha > 0:
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample(
            (x.shape[0], 1)
        ).to(x.device)
        perm = torch.randperm(x.shape[0], generator=g, device=x.device)
        out = lam * out + (1.0 - lam) * out[perm]
    return out.to(torch.float32)


def two_views(x: Tensor, cfg, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    """Return two independently-augmented views of x. cfg exposes mask_frac,
    noise_std, mixup_alpha."""
    a = _augment_once(x, cfg.mask_frac, cfg.noise_std, cfg.mixup_alpha, generator)
    b = _augment_once(x, cfg.mask_frac, cfg.noise_std, cfg.mixup_alpha, generator)
    return a, b
