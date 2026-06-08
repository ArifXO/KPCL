"""
Shared experiment utilities used by the runnable drivers in scripts/.

Holds the seed/config/subsample helpers and the single train+probe+geometry run
(one_run) shared by spec_h0, regate_h0, baseline_yeast, and diag_gridrange. Keeping
this in utils/ (not in any one script) avoids cross-script imports.
"""
from __future__ import annotations

import json
import os
import random
import time
import uuid
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.data.augment import two_views
from src.metrics.geometry import alignment, effective_rank, uniformity
from src.metrics.probe import linear_probe
from scripts.training.loop import train_contrastive

SEEDS = [42, 1337, 2024]
DATASETS = ["yeast", "scene", "emotions", "mediamill", "bibtex"]


def pick_device() -> str:
    """Select the compute device: CUDA GPU when available, else CPU (with a loud note).

    Experiments are intended to run on the GPU; a CPU fallback is surfaced explicitly
    rather than silently (R9), so a missing/un-built CUDA is never hidden.
    """
    if torch.cuda.is_available():
        dev = f"cuda  ({torch.cuda.get_device_name(0)})"
        print(f"[device] running on {dev}", flush=True)
        return "cuda"
    print("[device] WARNING: CUDA not available — falling back to CPU", flush=True)
    return "cpu"


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


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


def run_with_model(cfg, data, device):
    """Train (loss = cfg.loss.type), probe, measure geometry. Returns (metrics, model)."""
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
    metrics = {"macro_auroc": probe["macro_auroc"], "mAP": probe["mAP"], **geom,
               "params": model.param_count(), "final_loss": hist[-1],
               "secs": round(time.time() - t0, 1)}
    return metrics, model


def one_run(cfg, data, device) -> dict:
    """Train + probe + geometry; return the metrics dict only."""
    return run_with_model(cfg, data, device)[0]


def save_run_artifacts(cfg, model, metrics: dict) -> str:
    """Write the R8 artifact set for one run; return its run_name (timestamp+UUID)."""
    run_name = time.strftime("%Y%m%d-%H%M%S") + "_" + uuid.uuid4().hex[:8]
    ck = Path("runs/checkpoints") / run_name
    ck.mkdir(parents=True, exist_ok=True)
    resolved = deepcopy(cfg)
    resolved.run_name = run_name
    OmegaConf.save(resolved, ck / "config.yaml")
    torch.save(model.state_dict(), ck / "model.pt")
    (ck / "metrics.json").write_text(
        json.dumps({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in metrics.items()}, indent=2), encoding="utf-8")
    (ck / "param_count.txt").write_text(f"{model.param_count()} trainable params\n", encoding="utf-8")
    (ck / "git_info.txt").write_text("no git repo yet (pending git init)\n", encoding="utf-8")
    return run_name
