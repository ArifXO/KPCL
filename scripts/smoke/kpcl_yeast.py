"""
Stage 7 baseline table: 1-seed yeast, KAN encoder — KPCL (gamma in {1,2,4}) vs
InfoNCE / SupCon / DCL. Same data split and KAN init (seed 42) across methods;
linear-probe macro-AUROC + mAP. Writes runs/results/kpcl_yeast/kpcl.csv.

CAVEAT (R1/R5): SPEC H1 showed yeast carries essentially NO knot signal (FN-ranking
AUC_knot ~ AUC_cos ~ 0.53); the validated signal is on mediamill (many labels). So KPCL
is expected to ~match InfoNCE here, not beat it — yeast is a sanity check, and the real
KPCL test is the H2 sweep (mediamill/scene/emotions). Reported as-is.

Run: python -m scripts.smoke.kpcl_yeast
"""
from __future__ import annotations

import csv
from pathlib import Path

from src.data.loader import load_dataset
from utils.experiment import one_run, pick_device, set_seed, standard_cfg, subsample

# (loss, gamma) — gamma None for non-KPCL
METHODS = [("infonce", None), ("supcon", None), ("dcl", None),
           ("kpcl", 1.0), ("kpcl", 2.0), ("kpcl", 4.0)]
OUT = Path("runs/results/kpcl_yeast")


def run() -> None:
    device = pick_device()
    base = standard_cfg("yeast", "kan", 42)
    set_seed(42)
    data = subsample(load_dataset(base), base.experiment.subsample_train, 42)

    rows = []
    for loss, gamma in METHODS:
        extra = [f"loss.gamma={gamma}"] if gamma is not None else None
        cfg = standard_cfg("yeast", "kan", 42, loss=loss, extra=extra)
        r = one_run(cfg, data, device)
        label = f"kpcl_g{int(gamma)}" if loss == "kpcl" else loss
        rows.append({"method": label, "gamma": gamma if gamma is not None else "",
                     "macro_auroc": round(r["macro_auroc"], 4), "mAP": round(r["mAP"], 4),
                     "final_loss": round(r["final_loss"], 4), "secs": r["secs"]})
        print(f"{label:10s} AUROC={r['macro_auroc']:.4f} mAP={r['mAP']:.4f} {r['secs']}s",
              flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "kpcl.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {OUT / 'kpcl.csv'}")

    dcl = next(r for r in rows if r["method"] == "dcl")["macro_auroc"]
    kpcls = [r for r in rows if r["method"].startswith("kpcl")]
    best = max(kpcls, key=lambda r: r["macro_auroc"])
    print(f"DCL macro-AUROC = {dcl:.4f}")
    print(f"Best KPCL = {best['method']} ({best['macro_auroc']:.4f}) | "
          f"margin over DCL = {best['macro_auroc'] - dcl:+.4f}")


if __name__ == "__main__":
    run()
