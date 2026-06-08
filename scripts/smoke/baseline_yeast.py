"""
Stage 4 baseline table: 1-seed yeast, KAN encoder — InfoNCE vs SupCon vs DCL.

Same data split/subsample and same KAN init (seed 42) across losses; linear-probe
macro-AUROC + mAP. Writes runs/results/baseline_yeast/baseline.csv.

This is a baseline reference table (R2: baselines before KPCL), NOT a gate. SupCon
positive convention: two samples are positives iff they share >=1 label.

Run: python -m scripts.smoke.baseline_yeast
"""
from __future__ import annotations

import csv
from pathlib import Path

from src.data.loader import load_dataset
from utils.experiment import one_run, pick_device, set_seed, standard_cfg, subsample

LOSSES = ["infonce", "supcon", "dcl"]
OUT = Path("runs/results/baseline_yeast")


def run() -> None:
    device = pick_device()
    base = standard_cfg("yeast", "kan", 42)
    set_seed(42)
    data = subsample(load_dataset(base), base.experiment.subsample_train, 42)

    rows = []
    for loss in LOSSES:
        cfg = standard_cfg("yeast", "kan", 42, loss=loss)
        r = one_run(cfg, data, device)
        rows.append({"loss": loss, "macro_auroc": round(r["macro_auroc"], 4),
                     "mAP": round(r["mAP"], 4), "final_loss": round(r["final_loss"], 4),
                     "params": r["params"], "secs": r["secs"]})
        print(f"{loss:8s} AUROC={r['macro_auroc']:.4f} mAP={r['mAP']:.4f} "
              f"loss={r['final_loss']:.3f} {r['secs']}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "baseline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {OUT / 'baseline.csv'}")
    best = max(rows, key=lambda r: r["macro_auroc"])["loss"]
    print(f"Best macro-AUROC on yeast (1 seed, KAN): {best}")


if __name__ == "__main__":
    run()
