"""
KANLayer — cubic B-spline edges with a SiLU residual base path.

Each edge (o, i) carries a univariate function:
    f_{o,i}(h_i) = w_b[o,i] * SiLU(h_i) + sum_m W[o,i,m] * B_m(h_i)
{B_m} are order-p (p=3, cubic) B-spline basis functions over a uniform grid of
G intervals on [a, b], extended by p knots on each side (Cox-de Boor). There are
G + p basis functions per edge; the trainable coefficient tensor W
(`.coefficients`) has shape (O, I, G+p). The base path weight w_b != 0.

CRITICAL READOUT — knot_indices(h)  (KPCL/KURC depend on this):
  Returns, for input h_i and feature i, the exactly p+1 = 4 basis indices whose
  support contains h_i (the active "knot cell"). The active set depends ONLY on
  (i, h_i), NOT on the output o — the basis functions are shared across every
  output edge that consumes input i. Shape (batch, I, p+1); to obtain a literal
  per-edge (o,i) code, broadcast across o (identical). For points inside [a,b]
  these are exactly the nonzero spline columns; for out-of-range points the
  bucket is clamped so the readout stays a stable 4-index signature. The result
  is a detached LongTensor — discrete, non-differentiable, never on the gradient
  path (numerical-stability pitfall 6). S(x) := encoder layer-0 knot_indices.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class KANLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-2.0, 2.0),
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = (float(grid_range[0]), float(grid_range[1]))

        G, p = grid_size, spline_order
        a, b = self.grid_range
        step = (b - a) / G
        # extended knot vector: indices -p .. G+p  ->  G + 2p + 1 points
        grid = torch.arange(-p, G + p + 1, dtype=torch.float32) * step + a
        self.register_buffer("grid", grid.expand(in_features, -1).contiguous())

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, G + p))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.1)

    @property
    def coefficients(self) -> Tensor:
        """Trainable spline coefficients W, shape (O, I, G+p)."""
        return self.spline_weight

    def b_splines(self, x: Tensor) -> Tensor:
        """Cox-de Boor recursion. x: (batch, I) -> (batch, I, G+p)."""
        grid = self.grid  # (I, G+2p+1)
        x = x.unsqueeze(-1)  # (batch, I, 1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            # uniform grid => denominators are k*step (constant, never zero)
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
            right = (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:-k])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        if torch.isnan(bases).any():  # R9: no silent fallback
            raise ValueError(
                f"NaN in B-spline eval: x=[{float(x.min()):.4f},{float(x.max()):.4f}], "
                f"grid_range={self.grid_range}, G={self.grid_size}, p={self.spline_order}"
            )
        return bases.contiguous()

    def forward(self, x: Tensor) -> Tensor:
        base = F.linear(F.silu(x), self.base_weight)  # (batch, O)
        spline = torch.einsum("bik,oik->bo", self.b_splines(x), self.spline_weight)
        out = base + spline
        if torch.isnan(out).any():  # R9
            raise ValueError(
                f"NaN in KANLayer forward: in={self.in_features}, out={self.out_features}"
            )
        return out

    @torch.no_grad()
    def knot_indices(self, h: Tensor) -> Tensor:
        """Active basis indices, detached (batch, I, p+1). See module docstring."""
        p, G = self.spline_order, self.grid_size
        # bucket c: index of last grid knot <= h_i (consistent with degree-0 indicator)
        c = (h.unsqueeze(-1) >= self.grid.unsqueeze(0)).sum(dim=-1) - 1  # (batch, I)
        c = c.clamp(min=p, max=G + p - 1)
        offsets = torch.arange(-p, 1, device=h.device)  # [-p, ..., 0] -> p+1 indices
        return (c.unsqueeze(-1) + offsets).long()  # (batch, I, p+1)
