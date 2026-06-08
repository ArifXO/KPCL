"""
DIAGNOSTIC (H0 fairness check, not tuning): does the KAN's layer-2 spline path
saturate because its inputs leave grid_range, collapsing the KAN onto its SiLU
base path? If so on the datasets where KAN loses H0, that is an implementation
handicap to fix, not a falsified premise.

Per dataset (1 seed, short KAN train): measure at layer-1 input (standardised
features) and layer-2 input (= layer-1 output):
  - out_frac   : fraction of activations with |h| > grid bound (2.0)
  - coverage   : mean partition-of-unity sum_c B_c(h)  (~1 in-range, ->0 saturated)
  - layer-2 spline-vs-base magnitude ratio ||spline|| / (||spline||+||base||)

Saves runs/results/spec_h0/gridrange_diag.csv. Run: python -m scripts.smoke.diag_gridrange
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.loader import load_dataset
from utils.experiment import DATASETS, set_seed, standard_cfg, subsample
from scripts.training.loop import train_contrastive

OUT = Path("runs/results/spec_h0")


def _layer_stats(layer, h: torch.Tensor) -> tuple[float, float]:
    bound = max(abs(b) for b in layer.grid_range)
    out_frac = float((h.abs() > bound).float().mean())
    coverage = float(layer.b_splines(h).sum(dim=-1).mean())  # partition of unity
    return out_frac, coverage


def run() -> None:
    rows = []
    for ds in DATASETS:
        cfg = standard_cfg(ds, "kan", 42, extra=["experiment.epochs=10"])
        set_seed(42)
        data = subsample(load_dataset(cfg), cfg.experiment.subsample_train, 42)
        model, _ = train_contrastive(cfg, data, "cpu")
        enc = model.encoder
        with torch.no_grad():
            x = torch.as_tensor(data.X_train[:2000], dtype=torch.float32)
            l1, l2 = enc.layers[0], enc.layers[1]
            of1, cov1 = _layer_stats(l1, x)
            h1 = l1(x)
            if enc.norms is not None:        # actual layer-2 input (post inter-layer norm)
                h1 = enc.norms[0](h1)
            of2, cov2 = _layer_stats(l2, h1)
            base = F.linear(F.silu(h1), l2.base_weight)
            spline = torch.einsum("bik,oik->bo", l2.b_splines(h1), l2.spline_weight)
            ratio = float(spline.norm() / (spline.norm() + base.norm()))
        rows.append({"dataset": ds, "l1_out_frac": round(of1, 4), "l1_coverage": round(cov1, 4),
                     "l2_out_frac": round(of2, 4), "l2_coverage": round(cov2, 4),
                     "l2_spline_frac": round(ratio, 4)})
        print(f"{ds:10s} L1[out>2={of1:.3f} cov={cov1:.3f}]  "
              f"L2[out={of2:.3f} cov={cov2:.3f} spline_share={ratio:.3f}]", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "gridrange_diag.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {OUT / 'gridrange_diag.csv'}")


if __name__ == "__main__":
    run()
