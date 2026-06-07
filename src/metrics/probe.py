"""
Linear-probe evaluation: freeze the encoder, fit a linear classifier on its
representation (encoder output, PRE projection head — Explainer §6/§7), report
macro-AUROC and macro mAP.

macro-AUROC / mAP skip labels that are degenerate in the test split (single class,
or no positives) — these are undefined, not zero (R9: no silent fudging).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score


@torch.no_grad()
def extract_repr(model, X: np.ndarray, device, batch_size: int = 512) -> torch.Tensor:
    """Frozen encoder representation for probing."""
    model.eval()
    xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    out = [model.represent(xt[i:i + batch_size]) for i in range(0, len(xt), batch_size)]
    return torch.cat(out, dim=0)


def _macro_skip_degenerate(y_true: np.ndarray, y_score: np.ndarray, fn) -> float:
    vals = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if col.min() == col.max():  # single class -> AUROC/AP undefined
            continue
        vals.append(fn(col, y_score[:, j]))
    if not vals:
        raise ValueError("No evaluable labels in test split (all degenerate).")
    return float(np.mean(vals))


def linear_probe(model, data, device, epochs: int = 150, lr: float = 1e-2) -> Dict[str, float]:
    """Fit a linear probe on frozen train reprs; report macro-AUROC + mAP on test."""
    Xtr = extract_repr(model, data.X_train, device)
    Xte = extract_repr(model, data.X_test, device)
    ytr = torch.as_tensor(data.y_train, dtype=torch.float32, device=device)

    probe = nn.Linear(Xtr.shape[1], ytr.shape[1]).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        bce(probe(Xtr), ytr).backward()
        opt.step()

    with torch.no_grad():
        scores = torch.sigmoid(probe(Xte)).cpu().numpy()
    y_test = data.y_test
    return {
        "macro_auroc": _macro_skip_degenerate(y_test, scores, roc_auc_score),
        "mAP": _macro_skip_degenerate(y_test, scores, average_precision_score),
    }
