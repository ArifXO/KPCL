"""
Model assembly for contrastive training.

ContrastiveModel = encoder + projection head. forward(x) returns the L2-normalised
projection z (for the contrastive loss); represent(x) returns the encoder output
(for the linear probe); knot_indices(x) delegates to the encoder (KPCL/KURC readout).

build_model(cfg, input_dim) constructs the KAN or MLP twin. For the MLP with
hidden_dim=null, the width is auto-matched (R1) to the canonical KAN reference
(kan.yaml: hidden 128, embed 64, L=2, G=5, p=3) via utils.parity.
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from src.models.encoder import KANEncoder, MLPEncoder
from src.models.heads import KANHead, MLPHead, l2_normalize
from utils.parity import parity_for_input_dim


class ContrastiveModel(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        return l2_normalize(self.head(self.encoder(x)))

    def represent(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def knot_indices(self, x: Tensor) -> Tensor:
        return self.encoder.knot_indices(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg, input_dim: int) -> ContrastiveModel:
    m = cfg.model
    if m.type == "kan":
        gr = tuple(m.grid_range)
        enc = KANEncoder(input_dim, m.hidden_dim, m.n_enc_layers,
                         m.grid_size, m.spline_order, gr, m.use_layer_norm)
        head = KANHead(m.hidden_dim, m.embed_dim, m.grid_size, m.spline_order, gr,
                       m.use_layer_norm)
    elif m.type == "mlp":
        hidden = m.hidden_dim
        if hidden is None:  # R1 auto-match to canonical KAN reference
            hidden = parity_for_input_dim(input_dim, out_dim=m.embed_dim)["mlp_hidden"]
        use_ln = bool(getattr(m, "use_layer_norm", False))
        enc = MLPEncoder(input_dim, hidden, m.n_enc_layers, use_ln)
        head = MLPHead(hidden, m.embed_dim, use_ln)
    else:
        raise ValueError(f"Unknown model type '{m.type}'. Valid: ['kan', 'mlp'].")
    return ContrastiveModel(enc, head)
