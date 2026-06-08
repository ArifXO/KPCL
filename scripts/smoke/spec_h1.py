"""
SPEC EXPERIMENT H1 (decision-grade, binding) — does the knot-Jaccard weight rank false
negatives better than cos(z)? This decides KPCL vs KURC.

Protocol: train KAN+InfoNCE on yeast and mediamill to epoch>=100 (GPU), 5 seeds; on each
seed's val set compute FN-ranking AUC for cos(z), knot-Jaccard, and smoothed knot weight.
Paired bootstrap (knot vs cos) over the 5 seeds. Saves runs/results/spec_h1/h1.csv.

PASS iff on >=1 of {yeast, mediamill}:
  mean AUC_knot >= 0.55  AND  mean AUC_knot >= mean AUC_cos + 0.05  AND  bootstrap p < 0.05.
FAIL -> KPCL is falsified; proceed with KURC as the primary method. Not tuned to pass.

Run: python -m scripts.smoke.spec_h1
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.data.loader import load_dataset
from src.metrics.fn_ranking import fn_ranking_aucs, paired_bootstrap_p
from scripts.training.loop import train_contrastive
from utils.experiment import pick_device, set_seed, standard_cfg, subsample

DATASETS = ["yeast", "mediamill"]
SEEDS = [42, 1337, 2024, 7, 9001]
MAX_VAL = 5000
OUT = Path("runs/results/spec_h1")


def run() -> None:
    device = pick_device()
    rows: list[dict] = []
    for ds in DATASETS:
        for seed in SEEDS:
            cfg = standard_cfg(ds, "kan", seed, loss="infonce", experiment="spec_h1")
            set_seed(seed)
            data = subsample(load_dataset(cfg), cfg.experiment.subsample_train, seed)
            model, hist = train_contrastive(cfg, data, device)
            a = fn_ranking_aucs(model, data.X_val, data.y_val, device,
                                max_val=MAX_VAL, seed=seed)
            rows.append({"dataset": ds, "seed": seed, **a, "final_loss": round(hist[-1], 3)})
            print(f"{ds:10s} seed={seed} cos={a['auc_cos']:.4f} knot={a['auc_knot']:.4f} "
                  f"smooth={a['auc_smoothed']:.4f} (n_pos={a['n_pos']}, n_neg={a['n_neg']})",
                  flush=True)
    _write_csv(rows)
    _verdict(rows)


def _write_csv(rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "seed", "auc_cos", "auc_knot", "auc_smoothed",
              "n_pos", "n_neg", "final_loss"]
    with open(OUT / "h1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {OUT / 'h1.csv'}")


def _col(rows, ds, key):
    return np.array([r[key] for r in rows if r["dataset"] == ds], float)


def _verdict(rows: list[dict]) -> None:
    lines = ["# H1 Verdict - knot-Jaccard vs cos(z) FN-ranking (KPCL go/no-go)\n",
             f"Seeds: {SEEDS}. Per-dataset mean+/-std over 5 seeds; paired bootstrap "
             "(knot vs cos, resampling seeds).\n",
             "| dataset | AUC_cos | AUC_knot | AUC_smooth | knot-cos | boot p | >=0.55 | "
             "+0.05 | p<0.05 | PASS |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    any_pass = False
    for ds in DATASETS:
        cos, knot = _col(rows, ds, "auc_cos"), _col(rows, ds, "auc_knot")
        smooth = _col(rows, ds, "auc_smoothed")
        km, cm = float(knot.mean()), float(cos.mean())
        p = paired_bootstrap_p(knot, cos, n_boot=10000, seed=0)
        c_abs, c_margin, c_sig = km >= 0.55, km >= cm + 0.05, p < 0.05
        ds_pass = c_abs and c_margin and c_sig
        any_pass = any_pass or ds_pass
        lines.append(
            f"| {ds} | {cm:.4f}+/-{cos.std():.4f} | {km:.4f}+/-{knot.std():.4f} | "
            f"{smooth.mean():.4f} | {km - cm:+.4f} | {p:.4f} | "
            f"{'Y' if c_abs else 'N'} | {'Y' if c_margin else 'N'} | "
            f"{'Y' if c_sig else 'N'} | {'**YES**' if ds_pass else 'no'} |")
    lines += [
        f"\n## VERDICT: H1 {'PASS' if any_pass else 'FAIL'} "
        "(need all 3 conditions on >=1 dataset)\n",
        ("KPCL signal CONFIRMED - the knot-Jaccard weight ranks false negatives above "
         "disjoint pairs, beyond cos(z), significantly. Proceed to build KPCL (Stage 7)."
         if any_pass else
         "KPCL FALSIFIED — the knot pattern does NOT rank false negatives better than the "
         "cos(z) embedding by the pre-registered margin/significance. Per the binding gate, "
         "we DO NOT build KPCL; we proceed with KURC as the primary method (its uniformity "
         "/ spline-collapse benefit is independent of the FN signal). Reported as-is, not "
         "tuned to pass."),
    ]
    (OUT / "h1_verdict.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    run()
