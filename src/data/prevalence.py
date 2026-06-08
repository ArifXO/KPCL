"""
Contract: per-dataset multi-label prevalence statistics and empirical FN-rate.

Computed quantities (all from TRAIN split only — never from test or a paper):
  (a) per_label_rate  — per-label positive fraction, shape (L,)
  (b) cardinality     — mean number of labels per sample (label cardinality)
  (c) fn_rate         — empirical fraction of random in-batch pairs that share ≥1 label;
                        at the configured batch size B, this is the rate at which naive
                        random negatives are actually false negatives

Outputs:
  prevalence_table()  — formatted console table
  save_csv()          — runs/results/prevalence/prevalence.csv
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np


def compute_prevalence(
    y_train: np.ndarray,
    batch_size: int,
    n_batches: int = 500,
    seed: int = 42,
) -> dict:
    """Compute prevalence statistics from training labels.

    FN rate = fraction of all pairwise combinations in a random batch where
    both samples share ≥1 label. Averaged over n_batches random batches.

    Args:
        y_train:    float32 binary label matrix (n, L).
        batch_size: B; capped at n if n < B.
        n_batches:  number of random batches to sample.
        seed:       random seed.

    Returns:
        dict: per_label_rate (ndarray L,), cardinality (float), fn_rate (float).
    """
    n = len(y_train)
    B = min(batch_size, n)
    rng = np.random.default_rng(seed)

    fn_rates: list[float] = []
    total_pairs = B * (B - 1) / 2
    for _ in range(n_batches):
        idx = rng.choice(n, size=B, replace=(B > n))
        y_batch = y_train[idx]  # (B, L)
        # co[i,j] = True if sample i and j share ≥1 label
        co = (y_batch @ y_batch.T) > 0  # (B, B)
        shared = float(np.triu(co, k=1).sum())
        fn_rates.append(shared / total_pairs)

    return {
        "per_label_rate": y_train.mean(axis=0).astype(np.float32),
        "cardinality": float(y_train.sum(axis=1).mean()),
        "fn_rate": float(np.mean(fn_rates)),
    }


def prevalence_table(rows: list[dict]) -> str:
    """Format a list of per-dataset prevalence dicts as a console table."""
    header = f"{'Dataset':<12} {'n_train':>8} {'n_feat':>7} {'n_lab':>6} "
    header += f"{'card':>6} {'lbl_rate_mean':>13} {'FN_rate@B':>10} {'arm':>5}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in rows:
        arm = "high-FN" if r["fn_rate"] >= 0.15 else "low-FN"
        lines.append(
            f"{r['name']:<12} {r['n_train']:>8} {r['n_features']:>7} "
            f"{r['n_labels']:>6} {r['cardinality']:>6.2f} "
            f"{r['per_label_rate'].mean():>13.4f} "
            f"{r['fn_rate']:>10.4f} {arm:>7}"
        )
    lines.append(sep)
    return "\n".join(lines)


def save_csv(rows: list[dict], path: str | Path = "runs/results/prevalence/prevalence.csv") -> None:
    """Save prevalence rows to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "n_train", "n_features", "n_labels", "cardinality",
              "per_label_rate_mean", "fn_rate", "batch_size"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "name": r["name"],
                "n_train": r["n_train"],
                "n_features": r["n_features"],
                "n_labels": r["n_labels"],
                "cardinality": f"{r['cardinality']:.4f}",
                "per_label_rate_mean": f"{r['per_label_rate'].mean():.4f}",
                "fn_rate": f"{r['fn_rate']:.4f}",
                "batch_size": r["batch_size"],
            })
    print(f"Saved: {path}")


def run_all_datasets(batch_size: int = 256, seed: int = 42) -> list[dict]:
    """Load all 5 datasets, compute prevalence, print table, save CSV.

    Intended to be called from the command line or a notebook.
    Downloads datasets on first run (scikit-multilearn caches to ~/scikit_ml_learn_data/).
    """
    from src.data.datasets import VALID_DATASETS, load_raw
    from src.data.splits import make_splits

    rows: list[dict] = []
    for name in VALID_DATASETS:
        print(f"Loading {name}...", flush=True)
        X, y = load_raw(name)
        splits = make_splits(X, y, val_frac=0.1, test_frac=0.2, seed=seed)
        stats = compute_prevalence(splits["y_train"], batch_size=batch_size, seed=seed)
        rows.append({
            "name": name,
            "n_train": len(splits["y_train"]),
            "n_features": X.shape[1],
            "n_labels": y.shape[1],
            **stats,
            "batch_size": batch_size,
        })

    print(prevalence_table(rows))
    save_csv(rows)
    return rows
