"""
Stage 9 — full H2/H3 sweep (GPU). 8 methods x 5 datasets x 5 seeds, full R8 artifacts.

Encoders (per task): MLP+LayerNorm for the InfoNCE/SupCon/DCL baselines and the
cosine-FN control; KAN for KPCL/KURC/KPCL+KURC. kan_infonce is added as the same-encoder
reference so H2/H3 can be read cleanly (vs the conflated MLP baseline too).

Aggregates to runs/results/sweep/sweep_tables.csv; evaluates H2/H3 -> verdict.md.

H2 (KPCL = kan_kpcl): beats InfoNCE (mlp) by >=1.5 AUROC and DCL (mlp) by >=0.5, mean over
{yeast,scene,emotions,mediamill}, >=3/4 individually significant (paired bootstrap p<0.05).
Also report KPCL vs the MLP+cosine-FN control and vs kan_infonce.
H3 (KURC = kan_kurc): >= InfoNCE AUROC and uniformity improved by >=0.1 nats.
Reported as-is, not massaged. Run: python -m scripts.smoke.sweep_h2
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.data.loader import load_dataset
from src.metrics.fn_ranking import paired_bootstrap_p
from utils.experiment import (pick_device, run_with_model, save_run_artifacts,
                              set_seed, standard_cfg, subsample)

H2_DATASETS = ["yeast", "scene", "emotions", "mediamill"]
REPORTED = ["bibtex"]                       # low-FN stress test; reported, not in H2/H3
ALL_DATASETS = H2_DATASETS + REPORTED
SEEDS = [42, 1337, 2024, 7, 9001]
# (method name, model config, loss config)
METHODS = [
    ("mlp_infonce", "mlp_ln", "infonce"),
    ("mlp_supcon", "mlp_ln", "supcon"),
    ("mlp_dcl", "mlp_ln", "dcl"),
    ("mlp_cosfn", "mlp_ln", "cosfn"),
    ("kan_infonce", "kan", "infonce"),      # same-encoder reference for KPCL/KURC
    ("kan_kpcl", "kan", "kpcl"),
    ("kan_kurc", "kan", "kurc"),
    ("kan_kpclkurc", "kan", "kpcl_kurc"),
]
OUT = Path("runs/results/sweep")


def run() -> None:
    device = pick_device()
    rows: list[dict] = []
    for ds in ALL_DATASETS:
        for seed in SEEDS:
            base = standard_cfg(ds, "kan", seed, experiment="sweep_h2")
            set_seed(seed)
            data = subsample(load_dataset(base), base.experiment.subsample_train, seed)
            for name, model, loss in METHODS:
                cfg = standard_cfg(ds, model, seed, loss=loss, experiment="sweep_h2")
                metrics, m = run_with_model(cfg, data, device)
                rid = save_run_artifacts(cfg, m, metrics)
                rows.append({"dataset": ds, "method": name, "seed": seed,
                             "run_name": rid, **metrics})
                print(f"{ds:10s} {name:13s} s={seed} AUROC={metrics['macro_auroc']:.4f} "
                      f"uni={metrics['uniformity']:.3f} {metrics['secs']}s", flush=True)
    _write_tables(rows)
    _verdict(rows)


def _col(rows, ds, method, key):
    return np.array([r[key] for r in rows if r["dataset"] == ds and r["method"] == method], float)


def _write_tables(rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    methods = [m[0] for m in METHODS]
    fields = ["dataset", "method", "macro_auroc_mean", "macro_auroc_std", "mAP_mean",
              "mAP_std", "uniformity_mean", "uniformity_std", "eff_rank_mean", "n_seeds"]
    with open(OUT / "sweep_tables.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ds in ALL_DATASETS:
            for mth in methods:
                a = _col(rows, ds, mth, "macro_auroc")
                if a.size == 0:
                    continue
                w.writerow({
                    "dataset": ds, "method": mth,
                    "macro_auroc_mean": round(a.mean(), 4), "macro_auroc_std": round(a.std(), 4),
                    "mAP_mean": round(_col(rows, ds, mth, "mAP").mean(), 4),
                    "mAP_std": round(_col(rows, ds, mth, "mAP").std(), 4),
                    "uniformity_mean": round(_col(rows, ds, mth, "uniformity").mean(), 4),
                    "uniformity_std": round(_col(rows, ds, mth, "uniformity").std(), 4),
                    "eff_rank_mean": round(_col(rows, ds, mth, "effective_rank").mean(), 2),
                    "n_seeds": a.size})
    print(f"\nSaved {OUT / 'sweep_tables.csv'}")


def _verdict(rows: list[dict]) -> None:
    L = ["# H2 / H3 Sweep Verdict\n",
         f"Seeds {SEEDS}. KPCL=kan_kpcl, KURC=kan_kurc; baselines = MLP+LayerNorm. "
         "kan_infonce = same-encoder reference. Paired bootstrap over 5 seeds. Not massaged.\n"]

    # ---- H2 ----
    L.append("## H2 — KPCL vs InfoNCE(+1.5) and DCL(+0.5), mean over 4 datasets, >=3/4 significant\n")
    L.append("| dataset | KPCL | MLP-InfoNCE | MLP-DCL | MLP-cosFN | KAN-InfoNCE | KPCL−InfoNCE | p(vs InfoNCE) | KPCL−DCL | KPCL−cosFN |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    d_inf, d_dcl, n_sig = [], [], 0
    for ds in H2_DATASETS:
        kp = _col(rows, ds, "kan_kpcl", "macro_auroc")
        inf = _col(rows, ds, "mlp_infonce", "macro_auroc")
        dcl = _col(rows, ds, "mlp_dcl", "macro_auroc")
        cos = _col(rows, ds, "mlp_cosfn", "macro_auroc")
        kinf = _col(rows, ds, "kan_infonce", "macro_auroc")
        di, dd, dc = kp.mean() - inf.mean(), kp.mean() - dcl.mean(), kp.mean() - cos.mean()
        p = paired_bootstrap_p(kp, inf)
        sig = p < 0.05 and di > 0
        n_sig += sig
        d_inf.append(di); d_dcl.append(dd)
        L.append(f"| {ds} | {kp.mean():.4f} | {inf.mean():.4f} | {dcl.mean():.4f} | {cos.mean():.4f} | "
                 f"{kinf.mean():.4f} | {di:+.4f}{' *' if sig else ''} | {p:.3f} | {dd:+.4f} | {dc:+.4f} |")
    mi, md = float(np.mean(d_inf)), float(np.mean(d_dcl))
    h2 = (mi >= 0.015) and (md >= 0.005) and (n_sig >= 3)
    L += [f"\nMean KPCL−InfoNCE = **{mi:+.4f}** (need ≥+0.015); KPCL−DCL = **{md:+.4f}** "
          f"(need ≥+0.005); significant on **{n_sig}/4** (need ≥3).",
          f"\n## VERDICT: H2 {'PASS' if h2 else 'FAIL'}\n"]

    # ---- H3 ----
    L.append("## H3 — KURC ≥ InfoNCE AUROC and uniformity improved ≥0.1 nats\n")
    L.append("| dataset | KURC AUROC | InfoNCE AUROC | ΔAUROC | KURC uni | InfoNCE uni | Δuni (nats) |")
    L.append("|---|---|---|---|---|---|---|")
    da, du = [], []
    for ds in H2_DATASETS:
        ku = _col(rows, ds, "kan_kurc", "macro_auroc"); iu = _col(rows, ds, "mlp_infonce", "macro_auroc")
        kun = _col(rows, ds, "kan_kurc", "uniformity"); iun = _col(rows, ds, "mlp_infonce", "uniformity")
        da.append(ku.mean() - iu.mean()); du.append(kun.mean() - iun.mean())
        L.append(f"| {ds} | {ku.mean():.4f} | {iu.mean():.4f} | {ku.mean()-iu.mean():+.4f} | "
                 f"{kun.mean():.4f} | {iun.mean():.4f} | {kun.mean()-iun.mean():+.4f} |")
    mda, mdu = float(np.mean(da)), float(np.mean(du))
    h3 = (mda >= 0.0) and (mdu <= -0.1)            # uniformity more negative = improved
    L += [f"\nMean ΔAUROC = **{mda:+.4f}** (need ≥0); mean Δuniformity = **{mdu:+.4f}** nats "
          "(need ≤−0.1, i.e. more uniform).",
          f"\n## VERDICT: H3 {'PASS' if h3 else 'FAIL'}\n",
          "_Note: KPCL/KURC use the KAN encoder, baselines the MLP — 'vs InfoNCE' conflates "
          "encoder + loss; kan_infonce and KPCL−cosFN isolate the loss/signal. bibtex is in "
          "sweep_tables.csv (reported-only)._"]
    (OUT / "verdict.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    run()
