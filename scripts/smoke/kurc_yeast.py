"""
Stage 8 table: 1-seed yeast, KAN encoder — KURC vs InfoNCE. Reports macro-AUROC AND
uniformity (Wang-Isola; more negative = more uniform). Same split and seed-42 init.
Writes runs/results/kurc_yeast/kurc.csv.

H3 framing: KURC >= InfoNCE AUROC AND uniformity improved by >= 0.1 nats (more negative).
KURC's entropy regularizer is independent of the FN signal, so it can help on yeast even
though KPCL cannot (yeast has no knot-FN signal per H1). Reported as-is.

Run: python -m scripts.smoke.kurc_yeast
"""
from __future__ import annotations

import csv
from pathlib import Path

from src.data.loader import load_dataset
from utils.experiment import one_run, pick_device, set_seed, standard_cfg, subsample

METHODS = ["infonce", "kurc"]
OUT = Path("runs/results/kurc_yeast")


def run() -> None:
    device = pick_device()
    base = standard_cfg("yeast", "kan", 42)
    set_seed(42)
    data = subsample(load_dataset(base), base.experiment.subsample_train, 42)

    rows = []
    for loss in METHODS:
        cfg = standard_cfg("yeast", "kan", 42, loss=loss)
        r = one_run(cfg, data, device)
        rows.append({"method": loss, "macro_auroc": round(r["macro_auroc"], 4),
                     "mAP": round(r["mAP"], 4), "uniformity": round(r["uniformity"], 4),
                     "effective_rank": round(r["effective_rank"], 2), "secs": r["secs"]})
        print(f"{loss:8s} AUROC={r['macro_auroc']:.4f} uniformity={r['uniformity']:.4f} "
              f"eff_rank={r['effective_rank']:.1f} {r['secs']}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "kurc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {OUT / 'kurc.csv'}")

    nce = next(r for r in rows if r["method"] == "infonce")
    kurc = next(r for r in rows if r["method"] == "kurc")
    d_auroc = kurc["macro_auroc"] - nce["macro_auroc"]
    d_uni = kurc["uniformity"] - nce["uniformity"]   # more negative = more uniform
    print(f"KURC - InfoNCE:  dAUROC={d_auroc:+.4f}  d_uniformity={d_uni:+.4f} nats "
          f"({'more uniform' if d_uni < 0 else 'less uniform'})")


if __name__ == "__main__":
    run()
