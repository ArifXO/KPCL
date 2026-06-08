"""
Contract: top-level API for the data pipeline. Ties together datasets, splits,
and prevalence into a single DataSplit for use by the training loop.

Scaler fit on TRAIN ONLY (R5). Prevalence computed from train (Dataset Scope rule).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from src.data.datasets import load_raw
from src.data.prevalence import compute_prevalence
from src.data.splits import make_splits


class DataSplit(NamedTuple):
    X_train: np.ndarray    # float32, (N_train, F), scaled
    y_train: np.ndarray    # float32, (N_train, L), binary
    X_val: np.ndarray      # float32, (N_val, F), scaled
    y_val: np.ndarray      # float32, (N_val, L), binary
    X_test: np.ndarray     # float32, (N_test, F), scaled
    y_test: np.ndarray     # float32, (N_test, L), binary
    prevalence: np.ndarray  # float32, (L,), per-label positive fraction from train


def load_dataset(cfg) -> DataSplit:
    """Load cfg.data.name, split, scale, compute prevalence; return DataSplit.

    Args:
        cfg: resolved Hydra config; must expose cfg.data.{name, val_frac, test_frac}
             and cfg.seed, cfg.experiment.batch_size.

    Returns:
        DataSplit with all splits scaled and prevalence from train.

    Raises:
        ValueError: unknown dataset name or disjointness violation.
        RuntimeError: download failure.
    """
    X, y = load_raw(cfg.data.name)
    splits = make_splits(
        X, y,
        val_frac=cfg.data.val_frac,
        test_frac=cfg.data.test_frac,
        seed=cfg.seed,
    )
    stats = compute_prevalence(splits["y_train"], batch_size=cfg.experiment.batch_size)
    return DataSplit(
        X_train=splits["X_train"],
        y_train=splits["y_train"],
        X_val=splits["X_val"],
        y_val=splits["y_val"],
        X_test=splits["X_test"],
        y_test=splits["y_test"],
        prevalence=stats["per_label_rate"],
    )
