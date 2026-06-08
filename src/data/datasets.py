"""
Contract: factory that loads tabular multi-label datasets.

Sources (scikit-multilearn server is unavailable; using direct academic mirrors):
  - yeast, scene, bibtex, mediamill: LibSVM multi-label collection (NTU, CSIE)
  - emotions: OpenML dataset id=40589 (not on LibSVM)

Data is cached to ~/.kpcl_data/ after first download.
Returns dense float32 X (n, d) and binary float32 Y (n, L).
Unknown name → ValueError listing valid names (R9).
"""
from __future__ import annotations

import bz2
import hashlib
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np

VALID_DATASETS: tuple[str, ...] = ("yeast", "scene", "emotions", "mediamill", "bibtex")

_LIBSVM_BASE = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/"

class _DatasetSpec(NamedTuple):
    n_features: int
    n_labels: int
    source: str  # "libsvm_single" | "libsvm_traintest" | "openml"
    urls: tuple[str, ...]  # one or two urls
    openml_id: int | None = None


_SPECS: dict[str, _DatasetSpec] = {
    "yeast": _DatasetSpec(103, 14, "libsvm_traintest",
        (_LIBSVM_BASE + "yeast_train.svm.bz2", _LIBSVM_BASE + "yeast_test.svm.bz2")),
    "scene": _DatasetSpec(294, 6, "libsvm_traintest",
        (_LIBSVM_BASE + "scene_train.bz2", _LIBSVM_BASE + "scene_test.bz2")),
    "emotions": _DatasetSpec(72, 6, "openml",
        (), openml_id=40589),
    "bibtex": _DatasetSpec(1836, 159, "libsvm_single",
        (_LIBSVM_BASE + "bibtex.bz2",)),
    "mediamill": _DatasetSpec(120, 101, "libsvm_traintest",
        (_LIBSVM_BASE + "mediamill/train-exp1.svm.bz2",
         _LIBSVM_BASE + "mediamill/test-exp1.svm.bz2")),
}

_CACHE_DIR = Path.home() / ".kpcl_data"


def _cache_path(url: str) -> Path:
    _CACHE_DIR.mkdir(exist_ok=True)
    fname = hashlib.md5(url.encode()).hexdigest() + "_" + Path(url).name
    return _CACHE_DIR / fname


def _fetch_bz2(url: str) -> str:
    """Download a bz2-compressed text file; cache locally. Returns decoded text."""
    cache = _cache_path(url)
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    import requests  # noqa: PLC0415
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    text = bz2.decompress(r.content).decode("utf-8")
    cache.write_text(text, encoding="utf-8")
    return text


def _parse_libsvm_ml(text: str, n_features: int, n_labels: int) -> tuple[np.ndarray, np.ndarray]:
    """Parse multi-label LibSVM format → dense float32 (X, y).

    Format per line: label_idx1,label_idx2,... feat_idx1:val ...
    Labels: 0-indexed integers. Features: 1-indexed sparse.
    Lines with an empty label field (no positive labels) start with
    whitespace, making the first token look like 'idx:val' — detected
    by the presence of ':' in parts[0].
    """
    lines = [l for l in text.split("\n") if l.strip()]
    X = np.zeros((len(lines), n_features), dtype=np.float32)
    y = np.zeros((len(lines), n_labels), dtype=np.float32)
    for i, line in enumerate(lines):
        parts = line.split()
        if not parts:
            continue
        # If first token has ':', the label field is absent (0 positive labels)
        if ":" in parts[0]:
            feat_parts = parts
        else:
            for raw in parts[0].split(","):
                if raw:
                    y[i, int(raw)] = 1.0
            feat_parts = parts[1:]
        for feat in feat_parts:
            if ":" in feat:
                idx_str, val_str = feat.split(":", 1)
                X[i, int(idx_str) - 1] = float(val_str)
    return X, y


def _load_openml_emotions(spec: _DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    """Load emotions via sklearn fetch_openml (OpenML id=40589)."""
    from sklearn.datasets import fetch_openml  # noqa: PLC0415
    bunch = fetch_openml(data_id=spec.openml_id, as_frame=True, parser="auto")
    X = bunch.data.to_numpy(dtype=np.float32)
    # Target is a DataFrame with categories 'FALSE'/'TRUE' or similar binary
    y_df = bunch.target
    y = (y_df.to_numpy() == "TRUE").astype(np.float32)
    return X, y


def load_raw(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the full (unsplit) dataset; return (X, y) as float32 numpy arrays.

    Args:
        name: one of VALID_DATASETS.

    Returns:
        X: float32, shape (n, d).
        y: float32 binary, shape (n, L).

    Raises:
        ValueError: unknown dataset name.
        RuntimeError: download or parse failure.
    """
    if name not in VALID_DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Valid names: {list(VALID_DATASETS)}"
        )
    spec = _SPECS[name]
    try:
        if spec.source == "openml":
            return _load_openml_emotions(spec)

        if spec.source == "libsvm_single":
            text = _fetch_bz2(spec.urls[0])
            return _parse_libsvm_ml(text, spec.n_features, spec.n_labels)

        # libsvm_traintest — concatenate train and test
        text_tr = _fetch_bz2(spec.urls[0])
        text_te = _fetch_bz2(spec.urls[1])
        X_tr, y_tr = _parse_libsvm_ml(text_tr, spec.n_features, spec.n_labels)
        X_te, y_te = _parse_libsvm_ml(text_te, spec.n_features, spec.n_labels)
        return np.vstack([X_tr, X_te]), np.vstack([y_tr, y_te])

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load dataset '{name}': {exc}"
        ) from exc
