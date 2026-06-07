"""
Contract — evaluation metrics for multi-label contrastive representations.

Public API:
  macro_auroc(y_true, y_score) -> float
      Macro-averaged AUROC over all labels; skip labels with only one class.

  fn_ranking_auc(z, labels, knot_codes) -> float
      FN-ranking AUC for H1 gate: measures whether knot-Jaccard w_ik ranks
      true positives (mislabelled as negatives) above true negatives.
      Target: AUC >= 0.55 absolute AND >= AUC_cos(z) + 0.05 (H1).

Both functions accept numpy arrays; torch Tensors are accepted and converted.
"""
from __future__ import annotations

import numpy as np


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Macro-averaged AUROC over all labels.

    Args:
        y_true:  (N, L) binary ground-truth labels.
        y_score: (N, L) predicted scores (higher = more positive).

    Returns:
        float in [0, 1].

    Raises:
        NotImplementedError: stub — implement in Stage 2.
    """
    raise NotImplementedError("macro_auroc — implement in Stage 2")


def fn_ranking_auc(
    z: np.ndarray,
    labels: np.ndarray,
    knot_codes: np.ndarray,
) -> float:
    """Knot-Jaccard FN-ranking AUC for H1 gate.

    Measures whether w_ik (Jaccard over knot codes) ranks true-positive pairs
    that are mislabelled as negatives above true-negative pairs.

    Args:
        z:           (N, D) L2-normalised embeddings (numpy or Tensor).
        labels:      (N, L) binary multi-label matrix.
        knot_codes:  (N, K) knot-code matrix from KANEncoder (detached).

    Returns:
        float AUC in [0, 1].

    Raises:
        NotImplementedError: stub — implement in Stage 3 (H1 probe).
    """
    raise NotImplementedError("fn_ranking_auc — implement in Stage 3")
