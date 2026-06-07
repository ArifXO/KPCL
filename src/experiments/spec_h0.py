"""
SPEC EXPERIMENT H0 (decision-grade, NOT smoke).

Trains KAN-encoder+InfoNCE vs param-matched MLP-encoder+InfoNCE on all 5 datasets,
short but real, >=3 seeds; probes both (frozen linear probe, macro-AUROC + mAP) and
records geometry. Writes runs/results/spec_h0/h0.csv and h0_verdict.md.

H0 premise: KAN >= MLP macro-AUROC on >=3/5 datasets (mean over seeds). FAIL -> the
KAN premise is broken; STOP, do not build KPCL. Results are reported, not massaged.

Run:  python -m src.experiments.spec_h0
"""
from __future__ import annotations

import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from src.data.augment import two_views
from src.data.loader import load_dataset
from src.metrics.geometry import alignment, effective_rank, uniformity
from src.metrics.probe import linear_probe
from src.training.loop import train_contrastive

SEEDS = [42, 1337, 2024]
DATASETS = ["yeast", "scene", "emotions", "mediamill", "bibtex"]
MODELS = ["kan", "mlp"]
OUT = Path("runs/results/spec_h0")


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _cfg(dataset: str, model: str, seed: int):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("configs")):
        return compose(config_name="config", overrides=[
            f"data={dataset}", f"model={model}", "loss=infonce",
            "experiment=spec_h0", "aug=default", f"seed={seed}"])


def _subsample(data, k: int, seed: int):
    n = len(data.X_train)
    if n <= k:
        return data
    idx = np.random.default_rng(seed).choice(n, k, replace=False)
    return data._replace(X_train=data.X_train[idx], y_train=data.y_train[idx])


def _one_run(cfg, data, device) -> dict:
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
    return {"macro_auroc": probe["macro_auroc"], "mAP": probe["mAP"], **geom,
            "params": model.param_count(), "final_loss": hist[-1],
            "secs": round(time.time() - t0, 1)}


def run() -> None:
    device = "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            base = _cfg(dataset, "kan", seed)
            set_seed(seed)
            data = _subsample(load_dataset(base), base.experiment.subsample_train, seed)
            for model in MODELS:
                cfg = _cfg(dataset, model, seed)
                r = _one_run(cfg, data, device)
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
