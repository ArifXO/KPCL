# BUG_claude_code.md — Claude Code Bug Response Log

Managed by: **Claude Code** agent.
Protocol: read BUG_codex.md first; for each issue record what was broken,
what was fixed, or why deferred (with rationale). Cite the source review.
Newest entries appended at the bottom under a dated `## YYYY-MM-DD` heading.

---

<!-- No entries yet — project scaffolded on 2026-06-07. -->

## 2026-06-08 — GATE H0 verdict (Stage 3 spec experiment)

**Result: H0 FAIL — KAN ≥ MLP macro-AUROC on only 2/5 datasets (threshold ≥3/5).**
Artifacts: `runs/results/spec_h0/h0.csv`, `runs/results/spec_h0/h0_verdict.md`,
`runs/results/spec_h0/run.log`. Param-matched (≤0.2%), 3 seeds [42,1337,2024],
spec_h0 config (25 epochs, B=256, subsample≤4000), linear probe macro-AUROC.

| dataset | KAN | MLP | Δ (KAN−MLP) | winner |
|---|---|---|---|---|
| yeast | 0.6813 | 0.7242 | −0.0429 | MLP (clear) |
| scene | 0.9338 | 0.9402 | −0.0064 | MLP (tie, < seed std) |
| emotions | 0.8253 | 0.8275 | −0.0022 | MLP (tie, < seed std) |
| mediamill | 0.7095 | 0.5973 | +0.1122 | KAN (decisive) |
| bibtex | 0.8811 | 0.7839 | +0.0972 | KAN (decisive) |

Per CLAUDE.md gate: **STOPPED — did not build KPCL.** Results reported as-is, not massaged.

Honest diagnostics (from saved data, no extra runs):
- KAN wins decisively on the two high-dim / many-label sets (mediamill 101 labels,
  bibtex 159 labels); loses on low-label yeast; statistically ties scene/emotions.
- KAN's InfoNCE pretext loss is consistently HIGHER than MLP's (e.g. bibtex 2.42 vs
  1.88) yet KAN probes BETTER there — pretext-loss ≠ probe quality.
- Open fairness question (NOT yet investigated, would be a bug-fix not a tune):
  layer-2 KAN spline inputs may exceed grid_range (−2,2) → spline path saturates →
  KAN partly reduced to its SiLU base path. If true on yeast/scene/emotions this is a
  legitimate implementation handicap to fix before treating the premise as dead.

Note: no git repo yet → no commit hash recorded (playbook asks for one). Pending `git init`.

No Codex review (BUG_codex.md) to respond to as of this date.

### Grid-range saturation diagnostic (follow-up, `diag_gridrange.py`)

Measured layer-2 spline health (1 seed, 10-epoch KAN). `runs/results/spec_h0/gridrange_diag.csv`.

| dataset | L2 out-of-grid | L2 coverage Σ_c B_c | L2 spline share | H0 result |
|---|---|---|---|---|
| yeast | 3% | 0.998 | 0.685 | KAN loses — **healthy** |
| emotions | 2% | 0.999 | 0.744 | KAN ties — **healthy** |
| scene | 28% | 0.901 | 0.473 | KAN ties — partial sat |
| mediamill | 14% | 0.972 | 0.561 | KAN wins |
| bibtex | 99% | 0.016 | 0.027 | KAN wins — **spline dead, base-path only** |

Findings (honest):
1. Saturation does NOT explain the H0 losses: yeast & emotions layer-2 splines are
   perfectly healthy (cov≈0.998, spline_share≈0.7) and the KAN still fails to beat MLP.
   The premise weakness there is real, not a grid artifact.
2. Saturation IS a real defect on scene (28% out) and catastrophic on bibtex (99% out,
   spline path contributes 2.7% → layer-2 is effectively an MLP/SiLU layer).
3. Irony: bibtex's KAN "win" is driven by the base path, NOT the spline structure KPCL
   would exploit — so it is weak evidence for the KAN premise.
4. A principled inter-layer-normalisation fix (keep layer-2 inputs in grid range, per
   Explainer §2 / pitfall) is warranted on its own merits and could flip scene (tie +
   saturated) toward 3/5, but cannot help the already-healthy yeast/emotions. Even a
   perfect fix is NOT guaranteed to pass H0. Decision on the fix deferred to the user.

### FIX APPLIED — parameter-free inter-layer + pre-head LayerNorm (user: "fix, don't re-gate yet")

Added `nn.LayerNorm(..., elementwise_affine=False)` between KAN encoder layers and
before the KAN head (`use_layer_norm: true`, kan.yaml). 0 params → R1 parity intact;
layer-0 and `knot_indices(x)`=S(x) untouched. Re-ran `diag_gridrange` (splines now alive):

| dataset | L2 out-of-grid | L2 coverage | L2 spline share |
|---|---|---|---|
| yeast | 3%→4% | 0.998→0.999 | 0.69→0.72 |
| scene | 28%→4% | 0.901→0.999 | 0.47→0.74 |
| emotions | 2%→5% | 0.999→0.998 | 0.74→0.73 |
| mediamill | 14%→4% | 0.972→0.996 | 0.56→0.71 |
| bibtex | 99%→4% | 0.016→0.988 | 0.03→0.76 |

All layer-2 splines now healthy (≈layer-1 profile). 49 tests green; param parity unchanged.

**H0 verdict NOT changed.** The recorded H0 FAIL (2/5) stands for the as-tested
(no-LayerNorm) architecture; the fixed KAN has NOT been re-gated (user deferred that
decision). NOTE for any future H0 re-run: the fix materially changes the KAN (esp.
bibtex), so prior h0.csv is stale for the fixed arch; also decide whether the MLP twin
should get matched LayerNorm (it is a known performance booster) for a fair comparison.

### H0 RE-GATE — fixed KAN vs MLP and MLP+LN (user requested both arms)

45 runs (3 arms × 5 × 3 seeds). `runs/results/spec_h0/h0_regate.csv` + `h0_regate_verdict.md`.

| dataset | KAN+LN | MLP | MLP+LN | KAN−MLP | KAN−(MLP+LN) |
|---|---|---|---|---|---|
| yeast | 0.6956 | 0.7242 | 0.7157 | −0.0285 | −0.0201 |
| scene | 0.9413 | 0.9402 | 0.9469 | +0.0012 | −0.0056 |
| emotions | 0.8316 | 0.8275 | 0.8198 | +0.0041 | +0.0118 |
| mediamill | 0.7156 | 0.5973 | 0.6886 | +0.1183 | +0.0270 |
| bibtex | 0.8722 | 0.7839 | 0.8226 | +0.0883 | +0.0495 |

- **(A) KAN ≥ MLP on 4/5 → H0(A) PASS** (was 2/5 before the grid fix; loses only yeast).
- **(B) KAN ≥ MLP+LN on 3/5 → H0(B) PASS** (decisive control; loses yeast & scene).

Reading: the grid-range fix was a real handicap removal (2/5 → 4/5 vs plain MLP). The
KAN's edge **survives the LayerNorm control** on emotions/mediamill/bibtex, so it is not
merely a normalisation artifact — but it is **structural and robust only on the
many-label sets** (mediamill 101, bibtex 159: KAN−(MLP+LN) = +0.027, +0.050). scene's
variant-A win was a normalisation effect (flips once MLP gets LN); yeast the KAN loses
outright. **H0 premise now holds** → unblocks Stage 4/KPCL, with the honest caveat that
the KAN advantage concentrates on high-label-cardinality data. Not massaged.

## 2026-06-08 — GATE H1 verdict (Stage 6 spec experiment) — BINDING, decides KPCL vs KURC

**Result: H1 PASS.** Knot-Jaccard FN-ranking AUC vs cos(z), KAN+InfoNCE @100 epochs, 5
seeds [42,1337,2024,7,9001], on yeast + mediamill val sets. Now on GPU (RTX 3060).
Artifacts: `runs/results/spec_h1/{h1.csv,h1_verdict.md,run.log}`.

| dataset | AUC_cos | AUC_knot | knot−cos | boot p | ≥0.55 | +0.05 | p<0.05 | PASS |
|---|---|---|---|---|---|---|---|---|
| yeast | 0.5305±0.0035 | 0.5340±0.0052 | +0.0035 | 0.137 | N | N | N | no |
| mediamill | 0.5132±0.0052 | 0.6419±0.0015 | +0.1286 | 0.000 | Y | Y | Y | **YES** |

Pre-registered gate (PASS iff all 3 on ≥1 dataset) → **mediamill clears all three** decisively
and consistently (per-seed knot 0.639–0.644 vs cos 0.506–0.522; n_pos ≈ 1.1M pairs/seed).
yeast shows no signal (knot≈cos, both <0.55) — same pattern as H0: the KAN/knot advantage
concentrates on high-label-cardinality data (mediamill 101 labels vs yeast 14).

**→ KPCL is NOT falsified. Proceed to build KPCL (Stage 7).** Reported as-is, not tuned to pass.

Notes:
- knot-Jaccard and the smoothed knot weight gave identical AUC every seed. Not a bug: in the
  in-grid regime, knot-Jaccard = (4F − L1)/(4F + L1) is a monotone transform of the L1 over
  interval indices the smoothed weight uses, so they are rank-equivalent here.
- Efficiency: used `knot_code_compact` (O-collapsed code) for w_ik — the full O-replicated
  knot_code would be ~2 GB for mediamill's val set; Jaccard is O-invariant so identical result.
- H0 encoders (25-epoch) were not persisted, so H1 retrained KAN+InfoNCE at 100 epochs per the
  protocol's "epoch ≥ 100" — documented in Deviations (see D6.1).
