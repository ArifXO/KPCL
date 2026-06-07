"""
Contract — Knot-code Positive Contrastive Learning (KPCL) loss.

Extends InfoNCE by up-weighting false-negative pairs using Jaccard similarity
of B-spline knot codes S(x_i) and S(x_k).

Gradient isolation invariant (CLAUDE.md, numerical stability rule 6):
  w_ik = jaccard(S(x_i), S(x_k))  — DETACHED; never on gradient path.
  S(x)                             — DETACHED output of extract_knot_codes().
  Gradients flow only through z.

Correctness contract (R3):
  w_ik = 0 for ALL pairs  →  kpcl_loss == info_nce_loss  (assert allclose in tests)
  inject true-positive pair marked negative with w_ik > 0  →  loss decreases

Returns dict[str, Tensor] (R7) — keys:
  loss               — scalar KPCL objective
  info_nce_component — InfoNCE term (detached diagnostic)
  kpcl_fn_weight_mean — mean w_ik over FN-candidate pairs (diagnostic)
  temperature        — echo of cfg.loss.temperature
  pos_sim_mean       — mean cosine sim over positive pairs
  neg_sim_mean       — mean cosine sim over negative pairs

Jaccard validity (numerical stability rule 4):
  w_ik ∈ [0, 1] by construction; assert before use; raise if violated.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def kpcl_loss(
    z: Tensor,
    labels: Tensor,
    knot_codes: Tensor,
    cfg,
) -> Dict[str, Tensor]:
    """Compute KPCL loss.

    Args:
        z:           (B, D) L2-normalised embeddings.
        labels:      (B, L) binary multi-label matrix (float32).
        knot_codes:  (B, K) DETACHED knot codes S(x) from KANEncoder.
        cfg:         loss config; must expose cfg.loss.temperature, cfg.loss.gamma.

    Returns:
        dict with keys: loss, info_nce_component, kpcl_fn_weight_mean,
                        temperature, pos_sim_mean, neg_sim_mean.

    Raises:
        ValueError:          if w_ik ∉ [0, 1] (rule 4).
        NotImplementedError: stub — implement in Stage 3.
    """
    raise NotImplementedError("kpcl_loss — implement in Stage 3")
