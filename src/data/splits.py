"""
Contract: iterative stratified multi-label train/val/test split with disjointness
guarantee and scaler fit on train only.

No patient IDs exist in these benchmarks (yeast, scene, emotions, mediamill, bibtex).
Row-level disjointness is therefore the finest available granularity; this deviation
from "patient-level" splitting is documented here per R4.

Split procedure (two-pass):
  pass 1 — full dataset → train_val + test  (test_frac held out)
  pass 2 — train_val   → train + val        (val_frac / (1 - test_frac) held out)

Returns a dict with X/y splits (float32), original row index arrays, and the fitted scaler.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from skmultilearn.model_selection import IterativeStratification


def assert_disjoint(
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
) -> None:
    """Raise ValueError if any two index arrays share elements (R4/R5).

    Args:
        idx_train, idx_val, idx_test: integer index arrays.

    Raises:
        ValueError with details of overlapping sets.
    """
    st, sv, se = set(idx_train.tolist()), set(idx_val.tolist()), set(idx_test.tolist())
    problems: list[str] = []
    if ov := st & sv:
        problems.append(f"train∩val: {len(ov)} rows")
    if ov := st & se:
        problems.append(f"train∩test: {len(ov)} rows")
    if ov := sv & se:
        problems.append(f"val∩test: {len(ov)} rows")
    if problems:
        raise ValueError(f"Split index overlap detected — {'; '.join(problems)}")


def _stratified_indices(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run IterativeStratification and return (train_idx, test_idx).

    sample_distribution_per_fold=[test_size, 1-test_size] places test_size
    fraction in fold-0 (the test fold yielded second by split()), and
    1-test_size fraction in fold-1 (the train fold yielded first).
    """
    np.random.seed(seed)  # IterativeStratification lacks random_state kwarg
    y_sp = sp.csr_matrix(y) if not sp.issparse(y) else y
    strat = IterativeStratification(
        n_splits=2,
        order=1,
        sample_distribution_per_fold=[test_size, 1.0 - test_size],
    )
    train_idx, test_idx = next(strat.split(X, y_sp))
    return train_idx.astype(np.intp), test_idx.astype(np.intp)


def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float,
    test_frac: float,
    seed: int = 42,
) -> dict:
    """Produce stratified train/val/test splits with scaler fit on train only.

    Args:
        X:        float32 feature matrix (n, d).
        y:        float32 binary label matrix (n, L).
        val_frac: desired fraction of total data for validation.
        test_frac: desired fraction of total data for test.
        seed:     numpy random seed for reproducibility.

    Returns:
        dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
                        idx_train, idx_val, idx_test, scaler.

    Raises:
        ValueError: if disjointness check fails (R4/R5 guard).
    """
    # Pass 1: carve out test
    tv_idx, te_idx = _stratified_indices(X, y, test_size=test_frac, seed=seed)
    X_tv, y_tv = X[tv_idx], y[tv_idx]

    # Pass 2: carve val from train_val (adjust fraction to be relative to train_val)
    adj_val = val_frac / (1.0 - test_frac)
    tr_local, va_local = _stratified_indices(X_tv, y_tv, test_size=adj_val, seed=seed + 1)

    # Map local indices back to original row indices
    idx_train = tv_idx[tr_local]
    idx_val = tv_idx[va_local]
    idx_test = te_idx

    assert_disjoint(idx_train, idx_val, idx_test)  # R4/R5

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]

    # Fit scaler on train only; apply to val and test (R5, no leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return {
        "X_train": X_train,
        "y_train": y_train.astype(np.float32),
        "X_val": X_val,
        "y_val": y_val.astype(np.float32),
        "X_test": X_test,
        "y_test": y_test.astype(np.float32),
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "scaler": scaler,
    }
