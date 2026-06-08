"""
False-negative ranking AUC — the SPEC H1 measurement.

Question: does a scorer rank TRUE SEMANTIC NEIGHBOURS (pairs sharing many labels) above
DISJOINT-LABEL pairs? In SimCLR these true neighbours are false negatives. We compare:
  (i)   cos(z_i, z_k)          — the continuous embedding baseline
  (ii)  knot-Jaccard w_ik      — our discrete structural signal (MLP-impossible)
  (iii) smoothed knot weight   — exp(-beta * L1(interval indices))  (optional)

Pair definition (on a labelled val set, upper triangle i<k):
  positive = Jaccard(Y_i, Y_k) >= 0.5   (true semantic neighbour)
  negative = disjoint labels (Y_i · Y_k == 0)
  pairs with 0 < Jaccard < 0.5 are ambiguous and excluded.
AUC = P(score(positive) > score(negative)); higher = better FN ranking.

All scorers are DETACHED structural/geometric readouts; no gradients here.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.models.knots import jaccard_weight, knot_code_compact, smoothed_weight


def _upper_tri(mask: torch.Tensor) -> torch.Tensor:
    return torch.triu(mask, diagonal=1)


def _label_pair_masks(Y: torch.Tensor):
    """positive = label-Jaccard >= 0.5, negative = disjoint labels; upper triangle."""
    inter = Y @ Y.t()
    sizes = Y.sum(dim=1)
    union = sizes[:, None] + sizes[None, :] - inter
    jac = inter / union.clamp_min(1.0)
    pos = _upper_tri(jac >= 0.5)
    neg = _upper_tri(inter == 0)
    return pos, neg


def _pair_auc(score: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> float:
    """ROC-AUC of score over positive (label=1) vs negative (label=0) pairs."""
    s_pos = score[pos]
    s_neg = score[neg]
    y = np.concatenate([np.ones(s_pos.numel()), np.zeros(s_neg.numel())])
    s = torch.cat([s_pos, s_neg]).float().cpu().numpy()
    return float(roc_auc_score(y, s))


@torch.no_grad()
def fn_ranking_aucs(model, X_val, Y_val, device, beta: float = 0.1,
                    max_val: int = 3000, seed: int = 0) -> dict:
    """Compute FN-ranking AUC for cos(z), knot-Jaccard, and smoothed knot weight.

    model: ContrastiveModel (encoder+head). X_val/Y_val: numpy val features/labels.
    Caps the val set to max_val rows (deterministic per seed) to bound the O(N^2) pairs.
    """
    n = len(X_val)
    if n > max_val:
        idx = np.random.default_rng(seed).choice(n, max_val, replace=False)
        X_val, Y_val = X_val[idx], Y_val[idx]
    x = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    Y = torch.as_tensor(Y_val, dtype=torch.float32, device=device)

    pos, neg = _label_pair_masks(Y)
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"FN-ranking needs both pair types: n_pos={n_pos}, n_neg={n_neg} "
            f"(val rows={len(X_val)}). Cannot form the gate metric."
        )

    z = model(x)                                       # normalised projection
    cos = z @ z.t()
    w_knot = jaccard_weight(knot_code_compact(model.encoder, x))   # canonical S(x)
    w_smooth = smoothed_weight(model.encoder, x, beta)

    return {
        "auc_cos": _pair_auc(cos, pos, neg),
        "auc_knot": _pair_auc(w_knot, pos, neg),
        "auc_smoothed": _pair_auc(w_smooth, pos, neg),
        "n_pos": n_pos, "n_neg": n_neg,
    }


def paired_bootstrap_p(auc_a, auc_b, n_boot: int = 10000, seed: int = 0) -> float:
    """One-sided paired bootstrap over seeds: p = P(mean(a-b) <= 0).

    a, b are per-seed AUC arrays (paired: same seed/encoder). Tests whether scorer a
    (knot) beats scorer b (cos) significantly across seeds.
    """
    a, b = np.asarray(auc_a, float), np.asarray(auc_b, float)
    delta = a - b
    rng = np.random.default_rng(seed)
    k = len(delta)
    boot_means = delta[rng.integers(0, k, size=(n_boot, k))].mean(axis=1)
    return float((boot_means <= 0).mean())
