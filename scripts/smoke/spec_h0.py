"""
SPEC EXPERIMENT H0 (decision-grade, NOT smoke).

Trains KAN-encoder+InfoNCE vs param-matched MLP-encoder+InfoNCE on all 5 datasets,
short but real, >=3 seeds; probes both (frozen linear probe, macro-AUROC + mAP) and
records geometry. Writes runs/results/spec_h0/h0.csv and h0_verdict.md.

H0 premise: KAN >= MLP macro-AUROC on >=3/5 datasets (mean over seeds). FAIL -> the
KAN premise is broken; STOP, do not build KPCL. Results are reported, not massaged.

Run:  python -m scripts.smoke.spec_h0
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.data.loader import load_dataset
from utils.experiment import DATASETS, SEEDS, one_run, set_seed, standard_cfg, subsample

MODELS = ["kan", "mlp"]
OUT = Path("runs/results/spec_h0")


def run() -> None:
    device = "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            base = standard_cfg(dataset, "kan", seed)
            set_seed(seed)
            data = subsample(load_dataset(base), base.experiment.subsample_train, seed)
            for model in MODELS:
                cfg = standard_cfg(dataset, model, seed)
                r = one_run(cfg, data, device)
                rows.append({"dataset": dataset, "model": model, "seed": seed, **r})
                print(f"{dataset:10s} {model:3s} seed={seed} "
                      f"AUROC={r['macro_auroc']:.4f} mAP={r['mAP']:.4f} "
                      f"params={r['params']} loss={r['final_loss']:.3f} {r['secs']}s", flush=True)
    _write_csv(rows)
    _verdict(rows)


def _write_csv(rows: list[dict]) -> None:
    fields = ["dataset", "model", "seed", "macro_auroc", "mAP", "alignment",
              "uniformity", "effective_rank", "params", "final_loss", "secs"]
    with open(OUT / "h0.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {OUT / 'h0.csv'}")


def _verdict(rows: list[dict]) -> None:
    def stats(ds, mdl):
        v = [r["macro_auroc"] for r in rows if r["dataset"] == ds and r["model"] == mdl]
        return float(np.mean(v)), float(np.std(v))
    lines = ["# H0 Verdict - KAN-InfoNCE vs MLP-InfoNCE (param-matched)\n",
             f"Seeds: {SEEDS}. Metric: macro-AUROC (mean+/-std over seeds), linear probe.\n",
             "| dataset | KAN AUROC | MLP AUROC | delta (KAN-MLP) | KAN>=MLP |",
             "|---|---|---|---|---|"]
    wins = 0
    for ds in DATASETS:
        km, ks = stats(ds, "kan")
        mm, ms = stats(ds, "mlp")
        win = km >= mm
        wins += win
        lines.append(f"| {ds} | {km:.4f}+/-{ks:.4f} | {mm:.4f}+/-{ms:.4f} | "
                     f"{km - mm:+.4f} | {'yes' if win else 'no'} |")
    passed = wins >= 3
    lines += [
        f"\n**KAN >= MLP on {wins}/5 datasets.**",
        f"\n## VERDICT: H0 {'PASS' if passed else 'FAIL'} (threshold >=3/5)\n",
        ("Premise holds - KAN encoder matches/beats the param-matched MLP under plain "
         "InfoNCE. Proceed to Stage 4 (SupCon/DCL baselines), then Stage 5/6 toward KPCL."
         if passed else
         "Premise BROKEN - the KAN does not match the param-matched MLP under plain "
         "InfoNCE on >=3/5 datasets. STOP: do not build KPCL. Reconsider the KAN premise "
         "(grid range, depth, augmentation strength, or training length) before any "
         "further build. Results reported as-is, not massaged."),
    ]
    (OUT / "h0_verdict.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    run()
