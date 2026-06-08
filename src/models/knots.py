"""
Knot-pattern code S(x) and its similarity — the core, MLP-impossible signal.

S(x) records, for each spline edge (o, i), WHICH grid interval the input h_i fell
into — i.e. which of the (G+p) B-spline basis functions are active (exactly p+1=4 for
cubic). It is a QUANTILE of the input: it depends only on the interval index, NOT on
activation magnitude. That is precisely why an MLP cannot reproduce it (an MLP has no
spline grid, so no interval structure) and why it is information distinct from the
continuous embedding z. S is a DETACHED structural readout — never on the gradient path
(CLAUDE.md pitfall 6); gradients flow only through z.

NOTE (forward consistency): knot_code uses module.knot_indices(h) on the h you pass.
KANHead.forward applies a parameter-free pre-norm before its spline, so a head code is
forward-consistent only if you pass the normalised input. The canonical KPCL S(x) is the
ENCODER layer-0 code (no pre-norm), which is forward-consistent by construction — pass a
KANEncoder to use it.

Public API:
  knot_code(module, h)            -> (B, O*I*(G+p)) detached binary S, exactly 4*O*I ones.
  jaccard_weight(S)               -> (B, B) w_ik = |S_i∩S_k|/|S_i∪S_k| in [0,1], detached.
  smoothed_weight(module, h, beta)-> (B, B) exp(-beta * L1(kappa_i-kappa_k)), detached.
An MLP module (no spline grid) raises NotImplementedError (R9).
"""
from __future__ import annotations

import torch
from torch import Tensor


def _spline_layer(module):
    """Return the KANLayer whose grid defines knot_indices for this module."""
    if hasattr(module, "layer"):
        return module.layer            # KANHead
    if hasattr(module, "layers"):
        return module.layers[0]        # KANEncoder -> S(x) is the layer-0 code
    raise TypeError(f"{type(module).__name__} exposes no spline layer for knot_code.")


@torch.no_grad()
def knot_code(module, h: Tensor) -> Tensor:
    """Binary structural code S of shape (B, O*I*(G+p)); exactly 4*O*I active bits.

    Calls module.knot_indices(h) first, so an MLP module raises NotImplementedError (R9).
    """
    idx = module.knot_indices(h)               # (B, I, p+1); raises for MLP modules
    layer = _spline_layer(module)
    o, i = layer.out_features, layer.in_features
    n_basis = layer.grid_size + layer.spline_order   # G + p
    b = h.shape[0]

    code = torch.zeros(b, i, n_basis, dtype=torch.float32, device=h.device)
    code.scatter_(2, idx, 1.0)                 # 4 active bits per (sample, input feature)
    # every output edge (o,i) shares input i's code -> replicate across O
    S = code.unsqueeze(1).expand(b, o, i, n_basis).reshape(b, o * i * n_basis)
    return S.detach()


@torch.no_grad()
def jaccard_weight(S: Tensor) -> Tensor:
    """w_ik = |S_i ∩ S_k| / |S_i ∪ S_k| in [0,1] for binary S (B, M).

    Diagonal w_ii = 1 (a sample's code is identical to itself). Invariant to the O-fold
    replication in knot_code. Raises if any weight leaves [0,1] (R9, pitfall 4).
    """
    inter = S @ S.t()                          # (B, B) shared active bits
    sizes = S.sum(dim=1)                       # |S_i|
    union = sizes[:, None] + sizes[None, :] - inter
    w = inter / union.clamp_min(1.0)           # empty-code pairs -> 0 (no 0/0)
    if float(w.min()) < -1e-6 or float(w.max()) > 1.0 + 1e-6:
        raise ValueError(
            f"Jaccard weight out of [0,1]: min={float(w.min()):.4f}, max={float(w.max()):.4f}"
        )
    return w.clamp_(0.0, 1.0)


@torch.no_grad()
def smoothed_weight(module, h: Tensor, beta: float) -> Tensor:
    """Smoothed variant: w_ik = exp(-beta * L1(kappa_i - kappa_k)) over interval indices.

    kappa_i is the single active interval index per edge (the bucket c = max active basis
    index). w in (0,1], w_ii = 1, detached. MLP module raises NotImplementedError (R9).
    Selected over jaccard_weight by a Hydra flag in the KPCL loss config.
    """
    idx = module.knot_indices(h)               # (B, I, p+1); raises for MLP
    kappa = idx[..., -1].float()               # (B, I) interval index c per feature
    dist = torch.cdist(kappa, kappa, p=1.0)    # (B, B) L1 over interval indices
    return torch.exp(-beta * dist)
