"""
Contrastive training loop (SimCLR InfoNCE).

train_contrastive(cfg, data, device) builds the model, trains it with two-view
InfoNCE, and returns (model, history of per-epoch mean loss). Seeds are set by the
caller for reproducibility. KAN runs enforce weight_decay >= 1e-4 (pitfall 2).
Hyperparameters come from cfg (R6): epochs, batch_size, max_steps_per_epoch,
cfg.model.lr/weight_decay, cfg.loss.temperature, cfg.aug.*.
"""
from __future__ import annotations

import torch

from src.data.augment import two_views
from src.losses.infonce import info_nce_loss
from src.training.build import build_model


def train_contrastive(cfg, data, device) -> tuple:
    model = build_model(cfg, data.X_train.shape[1]).to(device)
    if cfg.model.type == "kan" and cfg.model.weight_decay < 1e-4:
        raise ValueError(
            f"KAN requires weight_decay >= 1e-4 (partition-of-unity nullspace, "
            f"pitfall 2); got {cfg.model.weight_decay}"
        )
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg.model.lr, weight_decay=cfg.model.weight_decay
    )
    X = torch.as_tensor(data.X_train, dtype=torch.float32, device=device)
    n = X.shape[0]
    bs = cfg.experiment.batch_size
    cap = cfg.experiment.max_steps_per_epoch

    history: list[float] = []
    model.train()
    for _ in range(cfg.experiment.epochs):
        perm = torch.randperm(n, device=device)
        losses: list[float] = []
        for step, start in enumerate(range(0, n, bs)):
            if cap and step >= cap:
                break
            idx = perm[start:start + bs]
            if idx.numel() < 2:  # NT-Xent needs >=2 samples for negatives
                continue
            v1, v2 = two_views(X[idx], cfg.aug)
            out = info_nce_loss(model(v1), model(v2), cfg.loss.temperature)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            losses.append(out["loss"].detach().item())
        history.append(sum(losses) / max(len(losses), 1))
    return model, history
