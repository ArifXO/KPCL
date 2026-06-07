"""
Contract — KAN Uniformity Regularization with Cluster entropy (KURC).

KURC adds an entropy regularizer over soft cluster occupancy q_c to encourage
uniform representation of the embedding space. It is a fallback method (H3 gate)
tested independently of KPCL (R2).

Numerical stability (rule 5):
  q_c are batch occupancy fractions; clamp q_c >= 1e-12 before log.

Returns dict[str, Tensor] (R7) — keys:
  loss         — scalar KURC objective
  kurc_entropy — entropy term H(q) (diagnostic)
  lambda_occ   — echo of cfg.loss.lambda_occ
  temperature  — echo of cfg.loss.temperature

H3 gate target:
  KURC >= InfoNCE on macro-AUROC AND uniformity improvement >= 0.1 nats.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def kurc_loss(z: Tensor, labels: Tensor, cfg) -> Dict[str, Tensor]:
    """Compute KURC loss.

    Args:
        z:      (B, D) L2-normalised embeddings.
        labels: (B, L) binary multi-label matrix (float32).
        cfg:    loss config; must expose cfg.loss.temperature, cfg.loss.lambda_occ.

    Returns:
        dict with keys: loss, kurc_entropy, lambda_occ, temperature.

    Raises:
        NotImplementedError: stub — implement in Stage 4.
    """
    raise NotImplementedError("kurc_loss — implement in Stage 4")
