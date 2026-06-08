"""
KAN and MLP encoders — parameter-matched twins (R1).

KANEncoder: L stacked KANLayers (default L=2), input_dim -> hidden -> hidden.
MLPEncoder: L stacked Linear+SiLU layers, the parameter-matched twin.

knot_indices(x): canonical knot code S(x) = the FIRST layer's active basis
indices on the raw input (detached). KPCL/KURC read this. MLPEncoder.knot_indices
raises NotImplementedError — an MLP has no spline grid (R9), never returns zeros.

Numerical stability: KANs require weight_decay >= 1e-4 (pitfall 2); the optimizer
applies it via cfg.model.weight_decay. Depth kept shallow, L=2 (pitfall 3).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.spline_kan import KANLayer


def _count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class KANEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-2.0, 2.0),
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList(
            KANLayer(dims[i], dims[i + 1], grid_size, spline_order, grid_range)
            for i in range(n_layers)
        )
        # Parameter-free LayerNorm between layers keeps each deeper layer's spline
        # inputs inside grid_range (else the spline path saturates -> KAN collapses
        # onto its SiLU base path; Explainer §2). 0 params -> R1 parity preserved.
        # Layer-0 (raw standardised features) and knot_indices(x)=S(x) are untouched.
        self.norms = (
            nn.ModuleList(nn.LayerNorm(hidden_dim, elementwise_affine=False)
                          for _ in range(n_layers - 1))
            if use_layer_norm else None
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.norms is not None and i < len(self.layers) - 1:
                x = self.norms[i](x)
        return x

    @torch.no_grad()
    def knot_indices(self, x: Tensor) -> Tensor:
        """Detached S(x): first-layer active basis indices, (batch, I, p+1)."""
        return self.layers[0].knot_indices(x)

    def param_count(self) -> int:
        return _count(self)


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 2,
                 use_layer_norm: bool = False) -> None:
        super().__init__()
        # use_layer_norm mirrors the KAN's inter-layer LayerNorm so an MLP+LN twin can be
        # compared fairly (parameter-free -> param parity preserved). Default off.
        dims = [input_dim] + [hidden_dim] * n_layers
        layers: list[nn.Module] = []
        for i in range(n_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.SiLU())
            if use_layer_norm and i < n_layers - 1:
                layers.append(nn.LayerNorm(dims[i + 1], elementwise_affine=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def knot_indices(self, x: Tensor) -> Tensor:
        raise NotImplementedError(
            "MLPEncoder has no spline grid; knot codes are undefined (R9). "
            "Use KANEncoder for KPCL/KURC."
        )

    def param_count(self) -> int:
        return _count(self)
