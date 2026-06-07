"""
H0 RE-GATE with the grid-range-fixed KAN (use_layer_norm), against two MLP controls:
  - mlp     : plain param-matched MLP twin
  - mlp_ln  : MLP twin WITH matched parameter-free LayerNorm (controls for the fact that
              LayerNorm is itself a performance booster — isolates the KAN's contribution)

Three arms x 5 datasets x 3 seeds = 45 runs. Same data split/subsample per (dataset,seed)
across arms (fair). Writes runs/results/spec_h0/h0_regate.csv and h0_regate_verdict.md.

Two verdicts, both pre-registered threshold >=3/5:
  (A) KAN >= MLP       (original H0 question, fixed KAN)
  (B) KAN >= MLP+LN    (decisive control: KAN edge beyond normalisation)

Reports as-is, not massaged. Run: python -m scripts.regate_h0
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.data.loader import load_dataset
from utils.experiment import DATASETS, SEEDS, one_run, set_seed, standard_cfg, subsample

MODELS = ["kan", "mlp", "mlp_ln"]
OUT = Path("runs/results/spec_h0")


def run() -> None:
    rows: list[dict] = []
    for ds in DATASETS:
        for seed in SEEDS:
            base = standard_cfg(ds, "kan", seed)
            set_seed(seed)
            data = subsample(load_dataset(base), base.experiment.subsample_train, seed)
            for model in MODELS:
                cfg = standard_cfg(ds, model, seed)
                r = one_run(cfg, data, "cpu")
                rows.append({"dataset": ds, "model": model, "seed": seed, **r})
                print(f"{ds:10s} {model:6s} seed={seed} AUROC={r['macro_auroc']:.4f} "
                      f"mAP={r['mAP']:.4f} params={r['params']} {r['secs']}s", flush=True)
    _write_csv(rows)
    _verdict(rows)


def _write_csv(rows: list[dict]) -> None:
    fields = ["dataset", "model", "seed", "macro_auroc", "mAP", "alignment",
              "uniformity", "effective_rank", "params", "final_loss", "secs"]
    with open(OUT / "h0_regate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {OUT / 'h0_regate.csv'}")


def _mean(rows, ds, mdl) -> float:
    return float(np.mean([r["macro_auroc"] for r in rows
                          if r["dataset"] == ds and r["model"] == mdl]))


def _std(rows, ds, mdl) -> float:
    return float(np.std([r["macro_auroc"] for r in rows
                         if r["dataset"] == ds and r["model"] == mdl]))


def _verdict(rows: list[dict]) -> None:
    lines = ["# H0 RE-GATE - grid-range-fixed KAN (use_layer_norm) vs MLP and MLP+LN\n",
             f"Seeds: {SEEDS}. Metric: macro-AUROC (mean+/-std), linear probe. "
             "Param-matched; LayerNorm is parameter-free.\n",
             "| dataset | KAN+LN | MLP | MLP+LN | KAN-MLP | KAN-(MLP+LN) |",
             "|---|---|---|---|---|---|"]
    wins_a = wins_b = 0
    for ds in DATASETS:
        k = _mean(rows, ds, "kan")
        m = _mean(rows, ds, "mlp")
        ml = _mean(rows, ds, "mlp_ln")
        wins_a += k >= m
        wins_b += k >= ml
        lines.append(f"| {ds} | {k:.4f}+/-{_std(rows,ds,'kan'):.4f} | {m:.4f} | {ml:.4f} | "
                     f"{k - m:+.4f} | {k - ml:+.4f} |")
    pa, pb = wins_a >= 3, wins_b >= 3
    lines += [
        f"\n**(A) KAN >= MLP on {wins_a}/5** -> H0(A) {'PASS' if pa else 'FAIL'}",
        f"**(B) KAN >= MLP+LN on {wins_b}/5** -> H0(B) {'PASS' if pb else 'FAIL'}\n",
        "## Interpretation\n",
        ("- (A) PASS means the fixed KAN matches/beats a plain MLP." if pa else
         "- (A) FAIL: even fixed, the KAN does not beat a plain MLP on >=3/5."),
        ("- (B) PASS means the KAN's edge survives giving the MLP the SAME normalisation "
         "-- evidence the advantage is KAN-structural, not just LayerNorm." if pb else
         "- (B) FAIL: once the MLP gets matched LayerNorm, the KAN's edge disappears -- "
         "the gain was largely from normalisation, not the spline/knot structure."),
        "\nReported as-is, not massaged. The recorded original H0 (no-norm KAN) stands "
        "separately in h0_verdict.md.",
    ]
    (OUT / "h0_regate_verdict.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    run()
