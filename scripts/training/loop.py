"""
Contrastive training loop (two-view; loss selected by cfg.loss.type).

train_contrastive(cfg, data, device) builds the model, trains it, and returns
(model, history of per-epoch mean loss). Dispatches to infonce / supcon / dcl —
SupCon additionally consumes the batch labels. Seeds are set by the caller. KAN runs
enforce weight_decay >= 1e-4 (pitfall 2). Hyperparameters come from cfg (R6).
"""
from __future__ import annotations

import torch

from src.data.augment import two_views
from src.losses.cosfn import cosfn_loss
from src.losses.dcl import dcl_loss
from src.losses.infonce import info_nce_loss
from src.losses.kpcl import kpcl_loss
from src.losses.kurc import kurc_loss
from src.losses.supcon import supcon_loss
from src.models.heads import l2_normalize
from src.models.knots import jaccard_weight, knot_code_compact
from scripts.training.build import build_model


def _kpcl(cfg, model, v1, v2, z1, z2):
    S = torch.cat([knot_code_compact(model.encoder, v1),   # detached encoder layer-0 S(x)
                   knot_code_compact(model.encoder, v2)], dim=0)
    return kpcl_loss(z1, z2, jaccard_weight(S), cfg.loss.temperature, cfg.loss.gamma)


def _compute_loss(cfg, model, v1, v2, labels):
    t = cfg.loss.type
    if t == "kurc":
        # KURC reads soft occupancy at the head, so compute the encoder output explicitly
        h1, h2 = model.encoder(v1), model.encoder(v2)
        z1, z2 = l2_normalize(model.head(h1)), l2_normalize(model.head(h2))
        hn = torch.cat([h1, h2], dim=0)
        if model.head.norm is not None:        # the head's spline sees the pre-normed input
            hn = model.head.norm(hn)
        B = model.head.layer.b_splines(hn)     # (2N, I, G+p) DIFFERENTIABLE partition of unity
        base = _kpcl(cfg, model, v1, v2, z1, z2) if cfg.loss.get("under_kpcl", False) else None
        return kurc_loss(z1, z2, B, cfg.loss.temperature, cfg.loss.lambda_occ, base_dict=base)

    z1, z2 = model(v1), model(v2)
    if t == "infonce":
        return info_nce_loss(z1, z2, cfg.loss.temperature)
    if t == "supcon":
        return supcon_loss(z1, z2, labels, cfg.loss.temperature)
    if t == "dcl":
        return dcl_loss(z1, z2, cfg.loss.temperature, cfg.loss.tau_plus)
    if t == "cosfn":
        return cosfn_loss(z1, z2, cfg.loss.temperature, cfg.loss.gamma)
    if t == "kpcl":
        return _kpcl(cfg, model, v1, v2, z1, z2)
    raise ValueError(f"Unknown loss type '{t}'. Valid: infonce/supcon/dcl/cosfn/kpcl/kurc.")


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
    Y = torch.as_tensor(data.y_train, dtype=torch.float32, device=device)
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
            out = _compute_loss(cfg, model, v1, v2, Y[idx])
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            losses.append(out["loss"].detach().item())
        history.append(sum(losses) / max(len(losses), 1))
    return model, history
