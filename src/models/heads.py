"""
KAN and MLP projection heads — parameter-matched twins.

KANHead: a single KANLayer in_dim -> out_dim (default 64).
MLPHead: a single Linear in_dim -> out_dim, the twin.

The final contrastive embedding is L2-normalised by the caller via l2_normalize()
(clamp norm at 1e-12 before dividing — numerical-stability pitfall 1).
MLPHead.knot_indices raises NotImplementedError (no spline grid, R9).
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from src.models.spline_kan import KANLayer


def l2_normalize(z: Tensor, eps: float = 1e-12) -> Tensor:
    """Row-wise L2 normalisation with clamped denominator (pitfall 1)."""
    return z / z.norm(dim=-1, keepdim=True).clamp_min(eps)


class KANHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 64,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-2.0, 2.0),
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        # Parameter-free pre-norm keeps the head's spline inputs in grid_range
        # (the head consumes encoder outputs, which otherwise saturate). 0 params.
        self.norm = nn.LayerNorm(in_dim, elementwise_affine=False) if use_layer_norm else None
        self.layer = KANLayer(in_dim, out_dim, grid_size, spline_order, grid_range)

    def forward(self, x: Tensor) -> Tensor:
        if self.norm is not None:
            x = self.norm(x)
        return self.layer(x)

    @property
    def coefficients(self) -> Tensor:
        return self.layer.coefficients

    def knot_indices(self, h: Tensor) -> Tensor:
        return self.layer.knot_indices(h)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 64, use_layer_norm: bool = False) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, elementwise_affine=False) if use_layer_norm else None
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        if self.norm is not None:
            x = self.norm(x)
        return self.layer(x)

    def knot_indices(self, h: Tensor) -> Tensor:
        raise NotImplementedError(
            "MLPHead has no spline grid; knot codes are undefined (R9)."
        )

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
