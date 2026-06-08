# KPCL / KURC — Results & Hypothesis Report

A living, analytical record of every experiment and **what it says about the thesis
hypotheses**. Rewritten/extended after each run. All numbers are read directly from
`runs/results/.../*.csv` — reported as-is, not massaged.

Last updated: **2026-06-08**, through Stage 9 (**full H2/H3 sweep — both gates FAIL**). Compute:
**RTX 3060 (cu128)** from H1 onward; the prevalence/parity tables and H0 ran on CPU. No git repo.

---

## 1. The thesis, in one paragraph

A KAN encoder built on cubic B-splines produces, for each input, a discrete **knot-pattern
code `S(x)`** — *which spline interval each feature falls into*. The claim is that the Jaccard
similarity of these codes, `w_ik`, detects **false negatives** (samples that share labels but
are treated as negatives in contrastive learning) **better than the embedding `cos(z)` can**,
because `S(x)` is a quantile structure an MLP cannot produce. **KPCL** uses `w_ik` to cancel
those false negatives; **KURC** is a safer fallback that regularizes knot-occupancy entropy to
prevent spline collapse. Four pre-registered gates decide whether the story holds.

| Gate | Hypothesis | Status | One-line reading |
|---|---|---|---|
| **H0** | KAN-InfoNCE ≥ MLP-InfoNCE on ≥3/5 (param-matched) | ✅ **PASS** (4/5; 3/5 vs MLP+LN) | Premise holds, but **cardinality-dependent** |
| **H1** | knot-Jaccard ranks FNs better than `cos(z)` on ≥1 of {yeast, mediamill} | ✅ **PASS** (mediamill, decisive) | **The MLP-impossible signal is real** — on many-label data |
| **H2** | KPCL beats InfoNCE +1.5 & DCL +0.5 macro-AUROC (≥3/4 sig) | ❌ **FAIL** | KPCL-the-loss adds **~0**; loses to MLP-DCL |
| **H3** | KURC ≥ InfoNCE AUROC & uniformity +0.1 nats | ❌ **FAIL** | Regularizer right-direction but **~8× too weak** at λ=0.1 |

**Bottom line (final for the proposed methods):** the *premise* (H0: KANs match MLPs) and the
*novel signal* (H1: the knot code ranks false negatives, +0.13 AUC over `cos(z)` on mediamill)
both **hold**. But **neither proposed method clears its gate (H2, H3 both FAIL)**. The decisive
same-encoder reference shows **the KPCL false-negative-cancellation loss adds nothing downstream**
(`kan_kpcl − kan_infonce` ≈ −0.001 to −0.016 on every dataset); the only real value is the **KAN
encoder** on high-cardinality data — and even that is **beaten by a plain MLP + DCL** on the
showcase datasets. So: the knot signal is *real but not downstream-useful as a loss*, and KURC is
a *valid-but-weak* regularizer. Honest, somewhat negative result for the methods; positive for the
encoder premise and the existence of the signal.
The open risk is whether that signal **translates into a broad downstream win (H2)**, because
every result so far says the KAN/knot advantage **lives on high-label-cardinality datasets and
is absent on low-cardinality ones**.

---

## 2. Dataset characterization — sets up everything (Stage 1)

`runs/results/prevalence/prevalence.csv`. FN-rate@256 = fraction of random in-batch pairs that
share ≥1 label (how often a naive "negative" is actually a false negative).

| dataset | n_train | n_feat | **n_labels** | cardinality | FN-rate@256 | arm |
|---|---|---|---|---|---|---|
| yeast | 1675 | 103 | 14 | 4.27 | **0.795** | high-FN |
| scene | 1681 | 294 | 6 | 1.08 | 0.194 | high-FN |
| emotions | 406 | 72 | 6 | 1.90 | 0.485 | high-FN |
| mediamill | 30734 | 120 | **101** | 4.40 | 0.716 | high-FN |
| bibtex | 5133 | 1836 | **159** | 2.41 | 0.058 | low-FN |

**Why this matters / first non-obvious finding:** the FN-*rate* (how many false negatives exist)
turns out **not** to predict whether the knot code can *detect* them. yeast has the **highest**
FN-rate (0.795) yet **zero** knot signal (H1); mediamill (0.716) has a **strong** signal. The
discriminating variable is **label cardinality** (n_labels: yeast 14 vs mediamill 101), not FN
prevalence. Keep this in mind — it recurs in every gate.

---

## 3. Fair-comparison guarantee — R1 parameter parity (Stage 2)

`param_count.txt`. MLP hidden width auto-matched per dataset; every pair within **≤0.17%**.

| dataset | KAN params | MLP params | ratio |
|---|---|---|---|
| yeast | 339,840 | 339,256 | 0.17% |
| scene | 559,872 | 560,564 | 0.12% |
| emotions | 304,128 | 304,439 | 0.10% |
| mediamill | 359,424 | 359,864 | 0.12% |
| bibtex | 2,336,256 | 2,335,663 | 0.03% |

Every KAN-vs-MLP claim below is budget-fair (a KAN win is not a capacity win). The LayerNorm
added during the H0 fix is parameter-free, so parity is unchanged.

---

## 4. GATE H0 — does a KAN even match a param-matched MLP? (premise)

**Hypothesis H0:** under plain InfoNCE, a KAN encoder ≥ a parameter-matched MLP encoder on
≥3/5 datasets (macro-AUROC, linear probe). If H0 fails, the KAN premise is dead.

### 4.1 First attempt — **FAIL (2/5)**
3 seeds, 25 epochs, train≤4000. `runs/results/spec_h0/h0_verdict.md`.

| dataset | KAN | MLP | Δ (KAN−MLP) |
|---|---|---|---|
| yeast | 0.6813 | 0.7242 | −0.0429 |
| scene | 0.9338 | 0.9402 | −0.0064 |
| emotions | 0.8253 | 0.8275 | −0.0022 |
| mediamill | 0.7095 | 0.5973 | **+0.1122** |
| bibtex | 0.8811 | 0.7839 | **+0.0972** |

KAN won only mediamill + bibtex → 2/5 → **FAIL**. But the *pattern* (KAN wins exactly the two
many-label sets) was suspicious enough to check for a bug before declaring the premise dead.

### 4.2 Diagnostic — the KAN was partly crippled
`runs/results/spec_h0/gridrange_diag.csv`. The KAN's layer-2 spline inputs were leaving the
grid range, so the spline path saturated and the KAN collapsed onto its plain SiLU base path
(i.e., behaved like an MLP). Coverage = partition-of-unity sum (≈1 healthy, →0 dead):

| dataset | L2 out-of-grid | L2 coverage | L2 spline share | (before → after fix) |
|---|---|---|---|---|
| yeast | 3% → 4% | 0.998 → 0.999 | 0.69 → 0.72 | healthy already |
| emotions | 2% → 5% | 0.999 → 0.998 | 0.74 → 0.73 | healthy already |
| scene | **28% → 4%** | 0.901 → 0.999 | 0.47 → 0.74 | partly saturated |
| mediamill | 14% → 4% | 0.972 → 0.996 | 0.56 → 0.71 | partly saturated |
| bibtex | **99% → 4%** | **0.016 → 0.988** | **0.03 → 0.76** | **fully dead** |

**Key insight:** the failure was **not** uniformly a bug. yeast and emotions were already
healthy and *still* lost — that loss is genuine. But bibtex's "win" was an illusion (its spline
path contributed 3% — the KAN was an MLP-in-disguise there), and scene/mediamill were partly
handicapped. Fix = parameter-free inter-layer + pre-head `LayerNorm` to keep spline inputs in
range (motivated by Explainer §2; applied uniformly; **not** tuned to flip a result).

### 4.3 Re-gate with the fixed KAN + a decisive control — **PASS**
Three arms: KAN+LN, plain MLP, and **MLP+LN** (the MLP given the *same* normalization, to test
whether the KAN's edge is real structure or just a normalization freebie).
`runs/results/spec_h0/h0_regate_verdict.md`.

| dataset | KAN+LN | MLP | MLP+LN | KAN−MLP | **KAN−(MLP+LN)** |
|---|---|---|---|---|---|
| yeast | 0.6956 | 0.7242 | 0.7157 | −0.0285 | −0.0201 |
| scene | 0.9413 | 0.9402 | 0.9469 | +0.0012 | −0.0056 |
| emotions | 0.8316 | 0.8275 | 0.8198 | +0.0041 | **+0.0118** |
| mediamill | 0.7156 | 0.5973 | 0.6886 | +0.1183 | **+0.0270** |
| bibtex | 0.8722 | 0.7839 | 0.8226 | +0.0883 | **+0.0495** |

- **(A) KAN ≥ MLP on 4/5 → PASS** (was 2/5 — the grid fix was the whole difference).
- **(B) KAN ≥ MLP+LN on 3/5 → PASS** (the harder, decisive control).

**Does this validate H0? Yes — and the control makes it a strong validation.** The KAN's edge
**survives giving the MLP the same LayerNorm** on emotions, mediamill, and bibtex, so it is
**structural**, not a normalization artifact. Two honest caveats: (i) scene's win disappears
once the MLP gets LN (it was a norm effect); (ii) yeast loses outright in both arms. The
advantage is **concentrated on high-label-cardinality data** — the same thread as §2.

---

## 5. Baseline losses — sanity table (Stage 4)

1-seed yeast, KAN, `runs/results/baseline_yeast/baseline.csv`. Establishes the comparators
before KPCL (rule R2: baselines green and probed first).

| loss | macro-AUROC | mAP |
|---|---|---|
| InfoNCE | 0.6923 | 0.4843 |
| DCL | 0.6784 | 0.4656 |
| SupCon | 0.6779 | 0.4615 |

On yeast, plain InfoNCE is already best; DCL/SupCon don't help. (Yeast is low-cardinality, so
this is unsurprising in hindsight.)

---

## 6. GATE H1 — the decisive test: is the knot signal real? (KPCL go/no-go)

**Hypothesis H1:** on ≥1 of {yeast, mediamill}, the knot-Jaccard `w_ik` ranks true semantic
neighbours (label-Jaccard ≥ 0.5 — the false negatives) above disjoint-label pairs **better than
`cos(z)`**, by AUC ≥ 0.55 *and* ≥ `cos` + 0.05, paired-bootstrap p < 0.05 over 5 seeds. This is
**the experiment the whole thesis rests on** — it directly measures the novel claim.

KAN+InfoNCE @ **100 epochs**, 5 seeds. `runs/results/spec_h1/h1.csv`.

| dataset | AUC_cos | AUC_knot | knot − cos | bootstrap p | ≥0.55? | +0.05? | p<0.05? | **PASS** |
|---|---|---|---|---|---|---|---|---|
| yeast | 0.5305 ± 0.0035 | 0.5340 ± 0.0052 | +0.0035 | 0.137 | ✗ | ✗ | ✗ | no |
| mediamill | 0.5132 ± 0.0052 | **0.6419 ± 0.0015** | **+0.1286** | **0.000** | ✓ | ✓ | ✓ | **YES** |

Per-seed mediamill knot AUC: 0.644 / 0.639 / 0.643 / 0.641 / 0.643 (tight), on ~1.1M positive
pairs/seed — this is not noise.

**Does this validate the hypothesis? YES, decisively, on mediamill — this is the headline
positive result of the project so far.** The knot code ranks false negatives at **0.64 AUC
while `cos(z)` sits at chance (0.51)** — a **+0.13** gap that the continuous embedding simply
cannot produce. That is direct evidence for the central, "MLP-impossible" claim: the discrete
spline-interval structure carries label-relevant information the embedding geometry does not.

**But read the yeast column honestly:** on yeast the knot code is **at chance** (0.534 vs
0.531) — *no* signal, despite yeast having the **highest** FN-rate (0.795). So the signal is
**not** a function of how many false negatives exist; it is a function of **label cardinality /
structure** (mediamill 101 labels vs yeast 14). The gate only requires ≥1 dataset, so H1
**passes** — but the concentration is the key scientific finding and the central risk going
forward.

---

## 7. KPCL — the method, 1-seed yeast (Stage 7)

`runs/results/kpcl_yeast/kpcl.csv`. γ (FN-cancellation exponent) swept {1,2,4}.

| method | macro-AUROC |
|---|---|
| InfoNCE | 0.6923 |
| **kpcl_g2** | **0.6921** |
| kpcl_g1 | 0.6918 |
| kpcl_g4 | 0.6896 |
| DCL | 0.6784 |
| SupCon | 0.6779 |

Best-KPCL margin over DCL = **+0.0137**. **Interpretation — exactly what H1 predicted:** yeast
has no knot signal, so `w_ik` is uninformative, and KPCL correctly **falls back to InfoNCE
behaviour** (0.6921 ≈ 0.6923) — it neither helps nor hurts. γ=2 is the default (γ=4
over-cancels slightly when the weights are noise). The +0.0137 over DCL is real but driven by
DCL being weak on yeast, **not** by FN cancellation. **This is a sanity check, not evidence for
KPCL.** The genuine KPCL test is H2 on the many-label datasets.

---

## 8. KURC — the fallback regularizer, 1-seed yeast (Stage 8)

`runs/results/kurc_yeast/kurc.csv`, λ_occ = 0.1. (Design note: `q` is the **soft, differentiable**
knot-occupancy — a detached `q` would make the entropy a no-op; this was a deliberate decision.)

| method | macro-AUROC | uniformity (↓ = more uniform) | eff-rank |
|---|---|---|---|
| InfoNCE | 0.6923 | −3.5468 | 49.5 |
| KURC | 0.6917 | **−3.5614** | **50.4** |

ΔAUROC = −0.0006 (equal), **Δuniformity = −0.0146 nats** (more uniform), **eff-rank +0.9** (less
spline collapse). **Interpretation:** the regularizer is genuinely **active** — it moves both
the geometry (uniformity) and the anti-collapse metric (effective rank) in the intended
direction without hurting accuracy. But at λ=0.1 the effect is **well below H3's +0.1-nat
target**. This is expected on a low-cardinality set and at a conservative λ; the H3 verdict is
the sweep's job, likely with a larger λ_occ. **Not** tuned on yeast to chase H3.

---

## 9. GATES H2 / H3 — the full sweep (Stage 9): **both FAIL**

8 methods × 5 datasets × 5 seeds, 50 epochs, GPU. Baselines = **MLP+LayerNorm** (architecture-
matched to the KAN's LN); KPCL/KURC = KAN; `kan_infonce` added as the **same-encoder reference**.
`runs/results/sweep/{sweep_tables.csv, verdict.md}` + full R8 per-run artifacts.

### 9.1 H2 — KPCL vs the baselines (macro-AUROC, mean over 5 seeds)

| dataset | **kan_kpcl** | mlp_infonce | mlp_dcl | mlp_supcon | mlp_cosfn | kan_infonce |
|---|---|---|---|---|---|---|
| yeast | 0.7016 | 0.7124 | 0.7109 | 0.6989 | 0.7083 | 0.7032 |
| scene | 0.9399 | 0.9486 | 0.9424 | 0.9543 | 0.9484 | 0.9415 |
| emotions | 0.8277 | 0.8162 | 0.8139 | 0.8075 | 0.8220 | 0.8282 |
| mediamill | 0.7040 | 0.6255 | **0.7354** | **0.7323** | 0.6322 | 0.7057 |

- KPCL − MLP-InfoNCE: mean **+0.0176** (clears +0.015) — but only emotions(+0.012*) and
  mediamill(+0.079*) are significant; yeast(−0.011) and scene(−0.009) are losses → **2/4 sig**.
- KPCL − MLP-DCL: mean **−0.0073** (needs +0.005) — **DCL wins** (mediamill DCL 0.735 ≫ KPCL 0.704).
- **VERDICT: H2 FAIL** (fails the DCL margin and the ≥3/4-significant clause).

### 9.2 Why — the decisive same-encoder decomposition

The "KPCL vs MLP-InfoNCE" comparison conflates **encoder** with **loss**. Splitting them with
`kan_infonce`:

| effect | comparison | yeast | scene | emotions | mediamill | reading |
|---|---|---|---|---|---|---|
| **KPCL loss** | kan_kpcl − kan_infonce | −0.0016 | −0.0016 | −0.0005 | −0.0017 | **adds ~0 everywhere** |
| **KAN encoder** | kan_infonce − mlp_infonce | −0.009 | −0.007 | +0.012 | **+0.080** | the only real effect |

So the entire mediamill "KPCL win" (+0.079 vs MLP-InfoNCE) is the **KAN encoder** (+0.080); the
**FN-cancellation loss contributes −0.0017**. **The H1 knot signal does not translate into a
downstream KPCL gain.** Worse: where KPCL was supposed to win (mediamill, bibtex), a plain
**MLP+DCL beats it outright** (mediamill 0.735 vs 0.704; bibtex 0.895 vs 0.864). The decisive
control confirms the same story: KPCL − cosFN on mediamill is +0.072, but `kan_infonce − cosfn`
is already +0.074 — i.e. encoder, not signal.

> **Likely (untested) reason:** in H1 the knot AUC (0.64) was measured on *clean* val inputs, but
> during KPCL training `w_ik` is computed on *augmented* inputs (feature-mask + noise perturb the
> spline intervals), so the training-time FN signal is much noisier — and a noisy weight cancels
> true negatives about as often as false ones. A future fix (compute `w_ik` on un-augmented
> inputs, or weaker augmentation) is worth trying but was **not** applied here.

### 9.3 H3 — KURC (macro-AUROC + uniformity)

KURC ≥ InfoNCE on AUROC (mean ΔAUROC +0.016 vs MLP-InfoNCE; ~0 vs same-encoder kan_infonce), but
**uniformity is not improved by 0.1 nats**. At matched encoder (kurc − kan_infonce): Δuniformity
≈ **−0.013 nats** (right direction, ~8× short of −0.1) and Δeff-rank **+0.4…+1.4** (anti-collapse
works, weakly). **VERDICT: H3 FAIL** — magnitude, not direction; would need a much larger λ_occ
than the 0.1 used (untested; raising λ to pass would be tuning-to-the-gate).

### 9.4 bibtex (reported-only, low-FN stress test)
MLP-DCL 0.895 > kan_infonce 0.879 > kan_kurc 0.873 > kan_kpcl 0.864 > MLP-InfoNCE 0.754. KPCL is
*below* plain KAN-InfoNCE here — consistent with "no FN signal on low-FN data."

---

## 10. Extra findings worth recording

- **The methods don't pay off; the encoder (partly) does.** The single biggest result of the
  sweep: KPCL-the-loss and KURC-the-regularizer add ≈0 at matched encoder. The thesis's headline
  mechanisms do not improve downstream performance at these settings. What *does* help is the KAN
  **encoder** on high-cardinality data — and even that loses to MLP+DCL on mediamill/bibtex.
- **MLP+DCL is the surprise winner on many-label data** (mediamill 0.735, bibtex 0.895). The
  debiased *self-supervised* baseline beats the full KAN+KPCL method where KPCL was meant to shine.
- **Cardinality is the master variable.** The KAN/knot advantage scales with `n_labels`: none on
  yeast(14)/scene(6)/emotions(6), strong on mediamill(101)/bibtex(159).
- **FN-rate ≠ FN-detectability.** yeast has the most false negatives yet they are undetectable
  by the knot code; mediamill's are highly detectable. The thesis should not claim "high-FN
  datasets benefit" — it should claim "high-label-cardinality datasets benefit."
- **bibtex's H0 win is base-path, not spline.** Before the fix, bibtex's spline path contributed
  3%; the KAN was effectively an MLP there and still won. So bibtex is weak *evidence for the
  KAN-spline premise* even though it passes H0 — don't lean on it in the KAN argument.
- **knot-Jaccard ≡ smoothed knot weight (rank-equivalent).** In H1 both gave identical AUC every
  seed. Not a bug: in the in-grid regime, Jaccard = (4F − L1)/(4F + L1) is a monotone transform
  of the L1 the smoothed weight uses, so they rank pairs identically. Either readout works.
- **The grid-range fix was the pivot of the whole project.** It turned H0 from FAIL(2/5) to
  PASS(4/5) by un-crippling the KAN's splines — and it was a genuine defect fix (Explainer §2),
  verified by the diagnostic, not a result-driven tweak. Without it, there would be no project.
- **Two methodological flags for the write-up:** the original no-norm H0 FAIL stands on record
  (`h0_verdict.md`) and should be disclosed; and the H1 encoders were retrained at 100 epochs
  (H0's 25-epoch models weren't persisted). Both are logged in `Deviations_claude_code.md`.

---

*Provenance: all tables above are computed by the drivers in `scripts/smoke/` and saved under
`runs/results/`. Full deviation log in `Deviations_claude_code.md`; gate verdicts also in
`BUG_claude_code.md`.*
