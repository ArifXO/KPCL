"""
MLP + cosine-FN-weight control (the decisive baseline, Explainer §7).

KPCL's false-negative-cancellation loss, but the FN weight w_ik comes from cos(z) — the
best false-negative signal an embedding (and thus an MLP) can produce — instead of the
knot-Jaccard structural signal. At matched parameters and matched loss, the only difference
between KPCL and this control is the SOURCE of the FN signal: discrete knot pattern vs the
continuous embedding. If KPCL beats this, the knot structure provides usable information the
embedding does not.

w_cos is DETACHED (a weight, never on the gradient path) — same invariant as KPCL.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from src.losses.kpcl import _normalize, kpcl_loss


def cosine_fn_weight(z1: Tensor, z2: Tensor) -> Tensor:
    """Detached soft cosine FN weight in [0,1] over the stacked 2N batch."""
    z = torch.cat([_normalize(z1), _normalize(z2)], dim=0)
    return (z @ z.t()).clamp(0.0, 1.0).detach()


def cosfn_loss(z1: Tensor, z2: Tensor, temperature: float, gamma: float) -> Dict[str, Tensor]:
    """FN-cancellation loss with the cosine-derived weight (control for KPCL)."""
    return kpcl_loss(z1, z2, cosine_fn_weight(z1, z2), temperature, gamma)
