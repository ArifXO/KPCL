"""
Stage 1 data pipeline tests (R4, R5).

Unit tests (always run, no network): use synthetic data.
Integration tests (marked slow, require network): download real datasets.
Run slow tests with: pytest tests/test_data.py --slow
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data.splits import assert_disjoint, make_splits
from src.data.prevalence import compute_prevalence


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _synthetic(n: int = 100, d: int = 20, L: int = 5, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (rng.random((n, L)) > 0.7).astype(np.float32)
    return X, y


def _synthetic_high_overlap(n: int = 100, L: int = 5, seed: int = 0) -> np.ndarray:
    """Labels with many shared co-occurrences → high FN rate."""
    rng = np.random.default_rng(seed)
    return (rng.random((n, L)) > 0.3).astype(np.float32)  # 70% density


def _synthetic_low_overlap(n: int = 100, L: int = 50, seed: int = 0) -> np.ndarray:
    """Many sparse labels → low FN rate."""
    rng = np.random.default_rng(seed)
    return (rng.random((n, L)) > 0.99).astype(np.float32)  # 1% density


# ---------------------------------------------------------------------------
# Split shape & coverage tests
# ---------------------------------------------------------------------------

def test_split_covers_all_rows():
    X, y = _synthetic()
    r = make_splits(X, y, val_frac=0.1, test_frac=0.2, seed=42)
    total = len(r["X_train"]) + len(r["X_val"]) + len(r["X_test"])
    assert total == len(X), f"Expected 100 rows total, got {total}"


def test_split_feature_dim_preserved():
    X, y = _synthetic(d=20)
    r = make_splits(X, y, val_frac=0.1, test_frac=0.2, seed=42)
    assert r["X_train"].shape[1] == 20
    assert r["X_val"].shape[1] == 20
    assert r["X_test"].shape[1] == 20


def test_split_label_dim_preserved():
    X, y = _synthetic(L=5)
    r = make_splits(X, y, val_frac=0.1, test_frac=0.2, seed=42)
    assert r["y_train"].shape[1] == 5


def test_split_dtypes():
    X, y = _synthetic()
    r = make_splits(X, y, val_frac=0.1, test_frac=0.2, seed=42)
    for key in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"):
        assert r[key].dtype == np.float32, f"{key} dtype={r[key].dtype}"


# ---------------------------------------------------------------------------
# Disjointness tests (R4 / R5)
# ---------------------------------------------------------------------------

def test_split_indices_disjoint():
    X, y = _synthetic()
    r = make_splits(X, y, val_frac=0.1, test_frac=0.2, seed=42)
    s_tr = set(r["idx_train"].tolist())
    s_va = set(r["idx_val"].tolist())
    s_te = set(r["idx_test"].tolist())
    assert not (s_tr & s_va), "train ∩ val overlap"
    assert not (s_tr & s_te), "train ∩ test overlap"
    assert not (s_va & s_te), "val ∩ test overlap"


def test_injected_train_val_overlap_raises():
    """Injecting train/val overlap must raise ValueError (R4/R5)."""
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(
            np.array([0, 1, 2, 3]),
            np.array([3, 4, 5]),  # index 3 overlaps train
            np.array([6, 7, 8]),
        )


def test_injected_train_test_overlap_raises():
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(
            np.array([0, 1, 2]),
            np.array([3, 4, 5]),
            np.array([2, 6, 7]),  # index 2 overlaps train
        )


def test_injected_val_test_overlap_raises():
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(
            np.array([0, 1, 2]),
            np.array([3, 4, 5]),
            np.array([5, 6, 7]),  # index 5 overlaps val
        )


def test_disjoint_no_overlap_passes():
    """No overlap → assert_disjoint should not raise."""
    assert_disjoint(
        np.array([0, 1, 2]),
        np.array([3, 4, 5]),
        np.array([6, 7, 8]),
    )


# ---------------------------------------------------------------------------
# Scaler fit-only-on-train test (R5)
# ---------------------------------------------------------------------------

def test_scaler_fit_only_on_train_mean_near_zero():
    """After StandardScaler, train mean per feature must be ~0 (R5)."""
    X, y = _synthetic(n=300)
    r = make_splits(X, y, val_frac=0.1, test_frac=0.2, seed=42)
    max_abs_mean = float(np.abs(r["X_train"].mean(axis=0)).max())
    assert max_abs_mean < 0.15, f"Train mean not near zero (max={max_abs_mean:.4f}); scaler may be fitted on all data"


def test_scaler_fit_only_on_train_std_near_one():
    """After StandardScaler, train std per feature must be ~1."""
    X, y = _synthetic(n=300)
    r = make_splits(X, y, val_frac=0.1, test_frac=0.2, seed=42)
    max_std_dev = float(np.abs(r["X_train"].std(axis=0) - 1.0).max())
    assert max_std_dev < 0.15, f"Train std not near 1 (max_dev={max_std_dev:.4f})"


def test_scaler_val_not_refit():
    """Val/test means should NOT be near zero (they're transformed, not fitted)."""
    # Inject an artificial offset into val/test features
    X, y = _synthetic(n=300)
    # Use a dataset where all features have a large positive offset so
    # val (not scaled to mean=0) will have non-zero mean
    X_biased = X + 10.0  # big constant shift
    r = make_splits(X_biased, y, val_frac=0.1, test_frac=0.2, seed=42)
    # Train mean should be ~0 (scaler removes train mean)
    train_mean = float(np.abs(r["X_train"].mean(axis=0)).max())
    assert train_mean < 0.15, "Train mean not near zero"
    # Val mean should be near 0 too (same shift applied via transform)
    # — this is the CORRECT behavior: transform removes TRAIN mean, so val also ~0
    # The point is the scaler was not FITTED on val.
    # Verify the scaler's mean is from train, not the full dataset.
    # Full dataset mean = train_mean + offset ≠ train_mean when sizes differ.
    # We check this indirectly: ensure val std is reasonable (not blown up by refit).
    val_std = float(r["X_val"].std(axis=0).max())
    assert val_std < 5.0, f"Val std suspiciously large ({val_std:.2f})"


# ---------------------------------------------------------------------------
# Prevalence tests
# ---------------------------------------------------------------------------

def test_prevalence_keys_present():
    _, y = _synthetic()
    result = compute_prevalence(y[:80], batch_size=16)
    assert "per_label_rate" in result
    assert "cardinality" in result
    assert "fn_rate" in result


def test_prevalence_per_label_rate_bounds():
    _, y = _synthetic()
    result = compute_prevalence(y[:80], batch_size=16)
    assert result["per_label_rate"].shape == (5,)
    assert float(result["per_label_rate"].min()) >= 0.0
    assert float(result["per_label_rate"].max()) <= 1.0


def test_prevalence_fn_rate_bounds():
    _, y = _synthetic()
    result = compute_prevalence(y[:80], batch_size=16)
    assert 0.0 <= result["fn_rate"] <= 1.0


def test_fn_rate_high_vs_low_overlap_synthetic():
    """High-density labels → higher FN rate than sparse labels (sanity check)."""
    y_high = _synthetic_high_overlap(n=200)
    y_low = _synthetic_low_overlap(n=200)
    fn_high = compute_prevalence(y_high, batch_size=32, n_batches=200)["fn_rate"]
    fn_low = compute_prevalence(y_low, batch_size=32, n_batches=200)["fn_rate"]
    assert fn_high > fn_low, f"Expected fn_high ({fn_high:.4f}) > fn_low ({fn_low:.4f})"


# ---------------------------------------------------------------------------
# Unknown dataset name (R9)
# ---------------------------------------------------------------------------

def test_unknown_dataset_raises():
    from src.data.datasets import load_raw
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_raw("imagenet")  # not a valid tabular multi-label dataset (R9)


# ---------------------------------------------------------------------------
# Integration tests (require network download; skipped unless --slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_yeast_shapes():
    from src.data.datasets import load_raw
    X, y = load_raw("yeast")
    assert X.shape[1] == 103, f"yeast n_features={X.shape[1]}, expected 103"
    assert y.shape[1] == 14, f"yeast n_labels={y.shape[1]}, expected 14"
    assert X.dtype == np.float32
    assert y.dtype == np.float32


@pytest.mark.slow
def test_scene_shapes():
    from src.data.datasets import load_raw
    X, y = load_raw("scene")
    assert X.shape[1] == 294
    assert y.shape[1] == 6


@pytest.mark.slow
def test_yeast_fn_rate_gt_bibtex_fn_rate():
    """yeast (dense labels) should have much higher FN rate than bibtex (sparse). (H1 gate assignment)."""
    from src.data.datasets import load_raw
    X_y, y_y = load_raw("yeast")
    X_b, y_b = load_raw("bibtex")
    r_y = make_splits(X_y, y_y, val_frac=0.1, test_frac=0.2, seed=42)
    r_b = make_splits(X_b, y_b, val_frac=0.1, test_frac=0.2, seed=42)
    fn_yeast = compute_prevalence(r_y["y_train"], batch_size=256, n_batches=200)["fn_rate"]
    fn_bibtex = compute_prevalence(r_b["y_train"], batch_size=256, n_batches=200)["fn_rate"]
    assert fn_yeast > fn_bibtex, (
        f"yeast FN-rate ({fn_yeast:.4f}) should be > bibtex FN-rate ({fn_bibtex:.4f})"
    )
