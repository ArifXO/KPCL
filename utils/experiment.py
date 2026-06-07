"""
Shared experiment utilities used by the runnable drivers in scripts/.

Holds the seed/config/subsample helpers and the single train+probe+geometry run
(one_run) shared by spec_h0, regate_h0, baseline_yeast, and diag_gridrange. Keeping
this in utils/ (not in any one script) avoids cross-script imports.
"""
from __future__ import annotations

import os
import random
import time

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from src.data.augment import two_views
from src.metrics.geometry import alignment, effective_rank, uniformity
from src.metrics.probe import linear_probe
from src.training.loop import train_contrastive

SEEDS = [42, 1337, 2024]
DATASETS = ["yeast", "scene", "emotions", "mediamill", "bibtex"]


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def standard_cfg(data: str, model: str, seed: int, loss: str = "infonce",
                 experiment: str = "spec_h0", extra: list[str] | None = None):
    """Compose the standard Hydra config for an experiment run."""
    overrides = [f"data={data}", f"model={model}", f"loss={loss}",
                 f"experiment={experiment}", "aug=default", f"seed={seed}"]
    if extra:
        overrides += list(extra)
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("configs")):
        return compose(config_name="config", overrides=overrides)


def subsample(data, k: int, seed: int):
    """Cap the training rows to k (spec-tier speed); unchanged if already <= k."""
    n = len(data.X_train)
    if n <= k:
        return data
    idx = np.random.default_rng(seed).choice(n, k, replace=False)
    return data._replace(X_train=data.X_train[idx], y_train=data.y_train[idx])


def one_run(cfg, data, device) -> dict:
    """Train (loss = cfg.loss.type), linear-probe, and measure geometry on test."""
    set_seed(cfg.seed)
    t0 = time.time()
    model, hist = train_contrastive(cfg, data, device)
    probe = linear_probe(model, data, device, cfg.experiment.probe_epochs, cfg.experiment.probe_lr)
    with torch.no_grad():
        xte = torch.as_tensor(data.X_test, dtype=torch.float32, device=device)
        z = model(xte)
        v1, v2 = two_views(xte[:512], cfg.aug)
        geom = {"alignment": float(alignment(model(v1), model(v2))),
                "uniformity": float(uniformity(z)),
                "effective_rank": float(effective_rank(z))}
    return {"macro_auroc": probe["macro_auroc"], "mAP": probe["mAP"], **geom,
            "params": model.param_count(), "final_loss": hist[-1],
            "secs": round(time.time() - t0, 1)}
