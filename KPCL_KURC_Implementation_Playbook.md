# KPCL & KURC — Claude Code Implementation Playbook (From Scratch)

**Project:** Knot-Pattern Contrastive Learning (KPCL) + Knot-Uniformity Regularized Contrastive learning (KURC), on tabular multi-label data, with a cubic B-spline KAN encoder + head.

**Governing principle: SPEC EXPERIMENT FIRST.** No expensive run, no full sweep, no thesis claim is made before the cheap pre-registered gates (H0 → H1) pass. The whole point is to *know before committing*. If H1 fails, we pivot to KURC before spending compute on KPCL headline runs.

**Hardware:** single RTX 4070 Ti (16GB). **Stack:** Python 3.11 venv, PyTorch, scikit-multilearn, Hydra, pytest.

---

## How to use this playbook with Claude Code

Each stage gives you: a **Claude Code prompt** (full context + the rule it must satisfy + the verification that closes it), a **Codex prompt** (terse diff-style), and a **model tier**. Run stages in order. Do not skip a gate. After each stage, Claude Code reads `CLAUDE.md` + `BUG_codex.md` + `BUG_claude_code.md` first, states a plan, then codes.

**Model tiers** (from your escalation rules): **Opus** for losses, split logic, knot-extraction, NaN-debug; **Sonnet** for eval modules, configs, orchestration; **Haiku** for renames/TODOs.

---

## Repository layout (exact)

```
kpcl/
├── CLAUDE.md                      # the rules file below — paste verbatim
├── BUG_codex.md                   # cross-agent channel (Codex writes)
├── BUG_claude_code.md             # cross-agent channel (Claude Code writes)
├── TODO.md
├── pyproject.toml                 # venv deps
├── configs/                       # Hydra
│   ├── config.yaml
│   ├── data/{yeast,scene,emotions,mediamill,bibtex}.yaml
│   ├── model/{kan_encoder,kan_head,mlp_encoder,mlp_head}.yaml
│   ├── loss/{infonce,supcon,dcl,kpcl,kurc}.yaml
│   └── experiment/{smoke,spec_h0,spec_h1,sweep_h2}.yaml
├── src/
│   ├── data/                      # loaders, splits, augment, prevalence
│   │   ├── datasets.py
│   │   ├── splits.py
│   │   ├── augment.py
│   │   └── prevalence.py
│   ├── models/
│   │   ├── spline_kan.py          # cubic B-spline KAN layer (+ knot readout)
│   │   ├── mlp.py
│   │   ├── encoder.py
│   │   ├── heads.py
│   │   └── knots.py               # S(x) extraction + Jaccard
│   ├── losses/
│   │   ├── infonce.py
│   │   ├── supcon.py
│   │   ├── dcl.py
│   │   ├── kpcl.py
│   │   └── kurc.py
│   ├── metrics/
│   │   ├── probe.py               # linear probe AUROC / mAP
│   │   ├── geometry.py            # alignment, uniformity, eff_rank
│   │   └── fn_ranking.py          # H1 gate: FN-ranking AUC
│   └── train.py
├── tests/
└── runs/
    ├── checkpoints/<run_name>/    # model.pt, config.yaml, param_count.txt, git_info.txt, metrics.json
    └── results/<run_name>/        # *.csv (probe, loss_components, gate verdicts)
```

**Your runs/ convention (locked):** `runs/checkpoints/<run_name>/` logs the checkpoints + run metadata; `runs/results/<run_name>/` holds the CSV files for that run. Every run computes `<run_name>` from timestamp+UUID, never user input.

---

## The CLAUDE.md file (paste this verbatim into the repo root)

> This is the ported rules file. It keeps your R1–R10 scientific rules, the cross-agent bug channel, and the numerical-stability pitfalls, but is rewritten for KPCL/KURC + tabular (no image/ChestMNIST/ResNet content). Paste everything between the rules below.

```markdown
# CLAUDE.md — KPCL/KURC Scientific Rules & Agent Guidelines

This document defines the inviolable scientific rules and agent configuration for
Claude Code work on the KPCL/KURC thesis. Every module, loss, and dataset handler
must comply. Spec experiments gate everything: no expensive run before cheap gates pass.

## Cross-Agent Bug Communication
Two agents review this codebase — Codex and Claude Code — communicating through:
- `BUG_codex.md` — Codex writes code reviews and bug findings (severity-ranked, with evidence + repro).
- `BUG_claude_code.md` — Claude Code reads BUG_codex.md, then records per issue what was broken and what was fixed (or why deferred, with rationale).
Rules: on "review"/"fix bugs"/"check the other agent", read BOTH files first. Do not
create ad-hoc *.md bug files. Newest entries at the bottom under a dated heading. Cite
the source review in each reply. Deferrals must state a reason.

## Coding rules (bias to caution over speed; use judgment on trivial tasks)

### 1. Think Before Coding
State assumptions explicitly; if uncertain, ask. Present multiple interpretations
rather than silently picking. If a simpler approach exists, say so. If something is
unclear, stop and name it.

### 2. Simplicity First
Minimum code that solves the problem. No speculative features, no abstractions for
single-use code, no unrequested configurability, no error handling for impossible
scenarios. If 200 lines could be 50, rewrite.

### 3. Surgical Changes
Touch only what you must. Don't improve adjacent code or refactor what isn't broken.
Match existing style. Remove only orphans your own changes created.

### 4. Goal-Driven Execution
Turn tasks into verifiable goals ("add validation" → "write tests for invalid inputs,
then make them pass"). State a brief numbered plan with a verify step for each.

### 5. Document Justified Deviations
If literal instruction would be wrong, implement the corrected version and state what
changed and why. Reusable lessons go in this file.
Worked example — gradient isolation: a test asserting "param X got gradient" only
proves the intended path if every OTHER path to X is detached. For KPCL, the FN weight
w_ik and the knot code S(x) MUST be detached; the only gradient path is through z. A
test must assert no gradient reaches S(x) or w_ik.

---

## Dataset Scope
- Tabular multi-label only, via scikit-multilearn: yeast, scene, emotions, mediamill, bibtex.
- Features dense float32; labels binary. Scaler fit on TRAIN ONLY.
- No image/graph/CXR code anywhere.
- Per-label positive prevalence MUST be computed from the loader and reported; never cited from a paper.

---

## 10 Scientific Rules (R1–R10)

### R1: Every KAN result pairs with a parameter-matched MLP baseline (±15%)
Print param counts as config comments and to param_count.txt. No cross-budget
comparison without explicit "parameter ablation" framing. A KAN win at 2× params is a
capacity win, not a KAN win.

### R2: Do NOT implement a combined method before its baselines + tests pass
InfoNCE (+ MLP and KAN) must be green and probed before KPCL begins. KURC's regularizer
must be tested before being switched on under KPCL. If a downstream stage fails you must
be able to isolate whether the bug is yours or in a dependency.

### R3: All contrastive masks/weights have unit tests (pos / neg / FN cases)
Every loss producing a mask or weighting tests, on synthetic [B=4, D=8]:
- positive-only batch → loss near zero
- negative-only batch → finite, > 0
- inject a true positive marked as negative → loss changes in the expected direction
For KPCL specifically: w_ik=0 for all pairs recovers InfoNCE EXACTLY (assert allclose).

### R4: Splits are disjoint at the available granularity
These benchmarks have no patient IDs → use scikit-multilearn iterative (stratified)
multi-label split at ROW level. Assert train/val/test index sets are disjoint; raise
ValueError on overlap. Document the no-patient-ID deviation in a comment.

### R5: No data leakage
After splitting: assert disjoint index sets, raise on overlap. Standardizer fit on train
only, applied to val/test. Document split sizes.

### R6: Experiments config-driven (Hydra). No hardcoded hyperparams
Everything via YAML: lr, batch size, dims, grid size, spline order, temperature, gamma,
lambda_occ, epochs, seeds. Use cfg.model.grid_size, not grid=5. Configs committed.

### R7: Every loss returns dict[str, Tensor] with named components
Never a bare scalar. Keys include: loss, plus named parts (e.g. info_nce_component,
kpcl_fn_weight_mean, kurc_entropy, temperature, pos_sim_mean, neg_sim_mean). Tests verify
all keys present.

### R8: Every run saves an artifact set
runs/checkpoints/<run_name>/: config.yaml (resolved), model.pt, metrics.json,
param_count.txt, git_info.txt. runs/results/<run_name>/: the run's CSVs. run_name from
timestamp+UUID, not user input.

### R9: No silent fallbacks. Raise descriptive errors. No bare except: pass
Errors include context (e.g. f"NaN in loss: tau={tau}, max_w={w.max():.4f}"). For the
KAN, knot extraction on a module with no spline grid (an MLP) MUST raise
NotImplementedError, not silently return zeros. Let errors propagate.

### R10: Modules ≤ 200 lines (non-test). Split if larger
Split into submodules and import in __init__.py. Comments/docstrings count.

---

## Numerical Stability Pitfalls (Critical)
1. L2-normalization: clamp norm at 1e-12 before dividing.
2. KAN partition-of-unity nullspace: weight decay ≥ 1e-4 is REQUIRED for KANs to train.
3. Keep KAN depth shallow (encoder L=2, head L=1): Jacobian rank decays with depth.
4. Jaccard weight w_ik ∈ [0,1] by construction; assert it; raise if outside.
5. KURC entropy: q_c are batch occupancy fractions; clamp q_c at 1e-12 before log.
6. Knot code S(x) and w_ik are DETACHED — never on the gradient path.

---

## Hypothesis Gates (pre-registered; binding; recorded with commit hash)
- H0 (premise): KAN-InfoNCE ≥ MLP-InfoNCE on ≥3/5 datasets (param-matched). FAIL → KAN premise broken, stop.
- H1 (KPCL signal): on yeast & mediamill, knot-Jaccard FN-ranking AUC ≥ 0.55 absolute AND ≥ AUC_cos(z) + 0.05, on ≥1/2 datasets, ≥5 seeds, paired bootstrap p<0.05. FAIL → KPCL falsified, fall back to KURC.
- H2 (KPCL headline): KPCL beats InfoNCE by ≥1.5 macro-AUROC and DCL by ≥0.5, mean over yeast/scene/emotions/mediamill, ≥3/4 individually significant.
- H3 (KURC fallback): KURC ≥ InfoNCE on AUROC and improves uniformity by ≥0.1 nats.

### Smoke runs are not evidence
A smoke run (≤10 steps, B=4) only proves the pipeline executes. Tag smoke run_ids smoke_*
so they are filtered from gate evaluation. Only multi-seed full runs count.

## When to escalate to Opus
Losses (R3/R7 subtle), split logic (R4/R5), knot extraction, NaN/non-convergence debug.
Use Sonnet for probes/metrics/configs/orchestration. Haiku for renames/TODOs.

## Git commit format
[Stage<N>] <what> — <why one sentence>

## Metadata for thesis claims
Cite stage number, commit hash, seeds [42,1337,2024,7,9001], mean±std from results CSVs.
```

---

# THE STAGES

| Stage | Deliverable | Gate before next | Tier |
|---|---|---|---|
| **0** | venv + Hydra scaffold + CLAUDE.md + bug channels | imports clean, pytest collects | Sonnet |
| **1** | Tabular data layer (5 sets, splits, prevalence report) | leakage/disjoint tests green; prevalence CSV printed | Opus |
| **2** | Cubic B-spline KAN layer + encoder/head + MLP twins, param parity | R1 parity printed; forward/grad tests green | Opus |
| **3** | InfoNCE + tabular augs + probe/geometry metrics | R3 tests; **SPEC H0** on KAN vs MLP | Opus |
| **4** | SupCon + DCL baselines | R3 tests; baseline table on yeast | Opus |
| **5** | Knot extraction S(x) + Jaccard w_ik | synthetic-spline unit tests; MLP raises | Opus |
| **6** | **SPEC H1** — FN-ranking AUC of w_ik vs cos(z) | **H1 verdict recorded → go KPCL or pivot KURC** | Opus |
| **7** | KPCL loss (if H1 passed) | R3 + γ=0≡InfoNCE test; 1-seed yeast vs DCL | Opus |
| **8** | KURC loss (always build; fallback or precondition) | R3 + entropy test; 1-seed yeast | Opus |
| **9** | Full sweep 4 datasets × suite × 5 seeds | **H2 / H3 verdict** | Sonnet |
| **10** | Thesis/paper artifacts (tables, figs, pre-reg appendix) | committee-ready | Sonnet |

---

## Stage 0 — venv + scaffold

**Tier: Sonnet.**

**Claude Code prompt:**
```
Build the kpcl repo from scratch (tabular multi-label contrastive learning; KAN encoder).
No image/graph/CXR code anywhere.

1. Create a Python 3.11 venv and pyproject.toml pinning: torch, numpy, scipy,
   scikit-learn, scikit-multilearn, arff (liac-arff), hydra-core, omegaconf, pandas,
   pytest. Provide exact install commands for the venv.
2. Create the directory layout I will paste (configs/, src/{data,models,losses,metrics},
   tests/, runs/checkpoints, runs/results). Add importable stubs with docstrings stating
   each module's contract.
3. configs/config.yaml wiring Hydra defaults across data/model/loss/experiment groups.
4. Paste the CLAUDE.md I provide verbatim at repo root. Create BUG_codex.md,
   BUG_claude_code.md, TODO.md with header stubs.
5. src/train.py stub that resolves a Hydra config and prints it.

Verify: in the venv, `pip install -e .` succeeds; `python -c "import src.train"` works;
`pytest tests/ --collect-only` runs with no import errors; `python src/train.py` prints a
resolved config. State the plan first (R1).
```
**Codex prompt:** `Scaffold kpcl repo: py3.11 venv + pyproject (torch,numpy,scipy,scikit-learn,scikit-multilearn,liac-arff,hydra-core,omegaconf,pandas,pytest). Dir layout (configs/,src/{data,models,losses,metrics},tests/,runs/checkpoints,runs/results) with importable stubs. config.yaml Hydra defaults. Paste CLAUDE.md, create BUG_codex.md/BUG_claude_code.md/TODO.md. train.py prints resolved cfg. Verify pip install -e ., import, pytest --collect-only, train.py.`

**venv commands (for your reference):**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

---

## Stage 1 — Data layer (R4/R5 critical)

**Tier: Opus** (leakage risk).

**Claude Code prompt:**
```
Implement src/data/ for tabular multi-label datasets via scikit-multilearn load_dataset:
yeast, scene, emotions, mediamill, bibtex. Return dense float32 X (n,d) and binary Y (n,L).

1. datasets.py: a factory keyed by cfg.data.name; unknown name → ValueError listing valid
   names (R9). One configs/data/*.yaml per dataset with n_features, n_labels as comments.
2. splits.py: train/val/test via scikit-multilearn iterative_train_test_split (stratified
   multi-label). No patient IDs in these sets, so ROW-level disjointness: assert
   train/val/test index sets are disjoint; raise ValueError on injected overlap (R4/R5).
   Document the no-patient-ID deviation in a comment.
3. StandardScaler fit on TRAIN ONLY, applied to val/test (R5). No leakage.
4. prevalence.py: for each dataset compute (a) per-label positive rate, (b) label
   cardinality, (c) the empirical FALSE-NEGATIVE rate = fraction of in-batch negative
   pairs that share ≥1 label, at the configured batch size. Print a table and save
   runs/results/prevalence/prevalence.csv. This table assigns datasets to high-FN
   (yeast/scene/emotions/mediamill) vs low-FN (bibtex) arms — used by the gates.

Verify: tests/test_data.py asserts shapes, injected-overlap raises, scaler fit only on
train, prevalence computed and yeast FN-rate >> bibtex FN-rate. pytest green. Print the
prevalence table. State assumptions (R1).
```
**Codex prompt:** `src/data/: datasets.py (scikit-multilearn loaders yeast/scene/emotions/mediamill/bibtex, dense float32 X + binary Y, factory raises on unknown, Hydra config each). splits.py (iterative stratified split + ROW disjointness asserts, raise on overlap). train-only StandardScaler. prevalence.py (per-label rate, cardinality, in-batch FN-pair rate -> runs/results/prevalence/prevalence.csv). tests/test_data.py: shapes, overlap raises, no leakage, yeast FN>>bibtex. pytest green.`

> **STOP after Stage 1.** Read the prevalence table. Confirm yeast/scene/emotions/mediamill are meaningfully high-FN and bibtex is low. This empirically grounds H1/H2.

---

## Stage 2 — Cubic B-spline KAN layer + encoder/head + MLP twins (R1)

**Tier: Opus** (spline silent-failure risk).

**Claude Code prompt:**
```
Implement the KAN building blocks and parameter-matched MLP twins.

1. src/models/spline_kan.py: KANLayer with cubic B-spline (order p=3) edges over a grid of
   G intervals (so G+p basis functions per edge), learnable coefficients
   W of shape (O, I, G+p), a residual SiLU base path (w_b != 0), and a configurable grid
   range. CRITICAL CONTRACT: expose a method `knot_indices(h)` returning, for each edge
   (o,i) and input h_i, the set of active basis indices (exactly 4 for cubic) — this is the
   readout KPCL/KURC depend on; document it. Also expose `.coefficients`.
2. src/models/encoder.py: KAN encoder (L=2, hidden 128, grid G=5 from Hydra) and an MLP
   encoder twin. src/models/heads.py: KAN head (1 layer, out 64) and MLP head twin.
3. R1 PARAMETER PARITY: print param counts for KAN vs MLP encoder+head; tune MLP hidden
   dims so totals are within ±15%. Write counts as config comments and to
   param_count.txt (R8). Do not proceed if parity violated.
4. Numerical stability: weight decay hook documented (≥1e-4), guard NaN in spline eval,
   clamp norms (R9, pitfalls). Keep modules ≤200 lines (R10) — split spline math if needed.

Verify: tests/test_kan.py asserts (a) forward shape, (b) knot_indices returns exactly 4
active indices per edge per sample and they match a hand-computed bucketize on a synthetic
grid, (c) gradient flows to coefficients, (d) KAN vs MLP param parity ≤15%, (e) calling
knot_indices on the MLP module raises NotImplementedError. pytest green.
```
**Codex prompt:** `src/models/spline_kan.py: cubic (p=3) B-spline KANLayer, coeffs W(O,I,G+p), residual SiLU base, Hydra grid range; method knot_indices(h)->4 active basis idx per edge; expose .coefficients. encoder.py KAN(L=2,h128,G=5)+MLP twin; heads.py KAN(1,out64)+MLP twin. R1 parity: print KAN vs MLP counts, tune MLP to ±15%, param_count.txt. NaN guards, wd≥1e-4. tests/test_kan.py: fwd shape, knot_indices=4 & matches bucketize, grad-to-coeff, parity, MLP knot_indices raises. pytest green.`

---

## Stage 3 — InfoNCE + augs + metrics + SPEC H0

**Tier: Opus** (loss + the first gate).

**Claude Code prompt:**
```
Implement the contrastive baseline, metrics, and run the H0 spec experiment.

1. src/data/augment.py: two views via feature masking (random subset→0), Gaussian noise,
   mixup; strengths from Hydra.
2. src/losses/infonce.py: SimCLR InfoNCE over the 2N batch. Return dict (R7): loss,
   info_nce_component, temperature, pos_sim_mean, neg_sim_mean. Clamp-min norms (R9).
3. src/metrics/probe.py: freeze encoder, train a linear probe, report macro-AUROC + mAP.
   src/metrics/geometry.py: alignment, uniformity, effective_rank.
4. R3 mask tests in tests/test_infonce.py on synthetic [B=4,D=8]: positive-only→~0,
   negative-only→finite>0, injected FN→changes as expected.

SPEC EXPERIMENT H0 (decision-grade, NOT smoke): train KAN-encoder + InfoNCE and
MLP-encoder + InfoNCE (param-matched) on all 5 datasets, short but real, ≥3 seeds. Probe
both. Save runs/results/spec_h0/h0.csv. EVALUATE H0: KAN ≥ MLP macro-AUROC on ≥3/5
datasets? Write a verdict to runs/results/spec_h0/h0_verdict.md. If H0 FAILS, STOP and
report — the KAN premise is broken; do not build KPCL. Do not massage results.
```
**Codex prompt:** `augment.py (feature-mask, gaussian, mixup, 2 views, Hydra). losses/infonce.py SimCLR, dict(loss,info_nce_component,temperature,pos/neg_sim_mean), clamp norm. metrics/probe.py (AUROC,mAP) + geometry.py (alignment,uniformity,eff_rank). tests/test_infonce.py R3 pos/neg/FN [4,8]. SPEC H0: KAN-InfoNCE vs MLP-InfoNCE (param-matched) all 5 sets ≥3 seeds -> runs/results/spec_h0/h0.csv + verdict. KAN≥MLP on ≥3/5? STOP if fail.`

> **GATE H0.** Record the verdict + commit hash in CLAUDE.md and BUG_claude_code.md. Pass → continue. Fail → the whole KAN premise is broken; reconsider before any further build.

---

## Stage 4 — SupCon + DCL baselines

**Tier: Opus** (losses).

**Claude Code prompt:**
```
Add the comparison baselines. Both return dict (R7); both get R3 tests.

1. src/losses/supcon.py: multi-label SupCon — two samples are positives if they share ≥1
   label (document this convention). Keys: loss, supcon_component, temperature,
   n_positives_mean.
2. src/losses/dcl.py: Debiased Contrastive Loss (Chuang et al. 2020). Debiased denominator
   with class-prior tau_plus (Hydra). Keys: loss, dcl_component, tau_plus,
   neg_correction_mean. Clamp the corrected negative term to its theoretical floor; raise
   on NaN with context (R9).

R3 tests for each. Then a 1-seed baseline table on yeast (KAN encoder): InfoNCE vs SupCon
vs DCL on macro-AUROC + mAP -> runs/results/baseline_yeast/baseline.csv. State the SupCon
multi-label positive convention (R1, R5).
```
**Codex prompt:** `losses/supcon.py (multi-label: share≥1 label=positive; dict loss/supcon_component/temperature/n_positives_mean) + losses/dcl.py (Chuang2020 debiased denom, tau_plus Hydra, clamp neg term, NaN-raise; dict loss/dcl_component/tau_plus/neg_correction_mean). R3 tests each. 1-seed yeast KAN: InfoNCE/SupCon/DCL AUROC+mAP -> runs/results/baseline_yeast/baseline.csv.`

---

## Stage 5 — Knot extraction S(x) + Jaccard (the heart of the method)

**Tier: Opus.**

**Claude Code prompt:**
```
Implement the knot-pattern code and its similarity — the core, MLP-impossible signal.

src/models/knots.py:
1. knot_code(head, h) -> sparse/binary S of shape (B, O*I*(G+p)): for each edge (o,i),
   mark the exactly-4 active basis indices for input h_i (use head.knot_indices). S is a
   DETACHED structural readout (no grad). Document that it depends only on which grid
   interval h_i falls into (a quantile), NOT on activation magnitude — this is why it is
   distinct from z and why an MLP cannot produce it.
2. jaccard_weight(S) -> w of shape (B, B): w_ik = |S_i ∩ S_k| / |S_i ∪ S_k| in [0,1].
   Vectorized via sparse intersater/union or matmul on the binary codes. Assert w in [0,1]
   (R9, pitfall 4). Diagonal handling documented.
3. (Optional, behind a Hydra flag) smoothed variant:
   w_ik = exp(-beta * L1(kappa_i - kappa_k)) over single active-interval indices.
4. If passed an MLP head (no knot_indices) -> raise NotImplementedError (R9).

Tests in tests/test_knots.py on a SYNTHETIC head with known grid:
- two identical inputs -> w = 1
- two inputs in completely disjoint intervals on every edge -> w = 0
- S has exactly 4*O*I ones per sample
- S and w carry NO gradient (assert .requires_grad is False / grad is None)
- MLP head -> raises NotImplementedError
pytest green. State assumptions (R1).
```
**Codex prompt:** `src/models/knots.py: knot_code(head,h)->detached binary S(B,O*I*(G+p)) via head.knot_indices (4 active idx/edge, depends on grid interval not magnitude). jaccard_weight(S)->w(B,B) in[0,1], assert bounds, vectorized. optional smoothed exp(-beta*L1(kappa)) behind Hydra flag. MLP head->NotImplementedError. tests/test_knots.py: identical->w=1, disjoint->w=0, exactly 4*O*I ones, S/w no grad, MLP raises. pytest green.`

---

## Stage 6 — SPEC H1: does the knot pattern actually carry FN signal? (THE decision gate)

**Tier: Opus.**

This is the single most important stage. It tells you, cheaply, whether KPCL can work *before* you build and tune the full loss and sweep. Do not skip it.

**Claude Code prompt:**
```
Run the H1 spec experiment: does the knot-Jaccard weight rank false negatives better than
the cosine-of-z baseline? This decides whether we build KPCL or pivot to KURC.

src/metrics/fn_ranking.py: given a trained KAN encoder+head and a labeled val set, define
positives = pairs with Jaccard(Y_i,Y_k) >= 0.5 (true semantic neighbors) and negatives =
disjoint-label pairs. Compute FN-ranking AUC three ways:
  (i)  cos(z_i, z_k)            — the baseline that beat us last time
  (ii) w_ik = knot-Jaccard      — our signal
  (iii) smoothed knot weight    — optional
Higher score should rank true-neighbor (FN) pairs above disjoint pairs.

SPEC H1 PROTOCOL (decision-grade): take the KAN+InfoNCE encoders from H0 on yeast and
mediamill, at epoch >= 100, 5 seeds. Compute the three FN-ranking AUCs per seed. Paired
bootstrap test (ii) vs (i). Save runs/results/spec_h1/h1.csv with per-seed numbers.

EVALUATE H1 (pre-registered, binding):
  PASS iff on >=1 of {yeast, mediamill}: AUC_knot >= 0.55 absolute
        AND AUC_knot >= AUC_cos + 0.05, paired bootstrap p < 0.05 over 5 seeds.
Write runs/results/spec_h1/h1_verdict.md with the verdict and the exact numbers.
Do NOT tune to pass. Report what the seeds show. If FAIL, state plainly that KPCL is
falsified and we proceed with KURC as the primary method.
```
**Codex prompt:** `src/metrics/fn_ranking.py: positives=Jaccard(Y_i,Y_k)>=0.5, negatives=disjoint labels; FN-ranking AUC for (i)cos(z) (ii)knot-Jaccard w (iii)smoothed. SPEC H1: KAN+InfoNCE from H0 on yeast+mediamill, epoch>=100, 5 seeds, paired bootstrap (ii)vs(i) -> runs/results/spec_h1/h1.csv. PASS iff knot AUC>=0.55 & >=cos+0.05 & p<0.05 on >=1 set. Write h1_verdict.md. No tuning to pass.`

> **GATE H1 — THE PIVOT POINT.** Record verdict + commit hash.
> - **PASS →** build KPCL (Stage 7), KURC as precondition/secondary (Stage 8), full sweep targeting H2.
> - **FAIL →** KPCL is falsified. Skip Stage 7's headline ambition; KURC (Stage 8) becomes the primary method, sweep targets H3, and the paper reframes to representation geometry. This is a planned outcome, not a failure of the project.

---

## Stage 7 — KPCL loss (build only if H1 passed)

**Tier: Opus** (core novelty loss).

**Claude Code prompt:**
```
Implement KPCL. Build only if H1 passed; otherwise skip to KURC.

src/losses/kpcl.py: false-negative-cancellation InfoNCE using the DETACHED knot-Jaccard
weight w_ik from src/models/knots.py:
  L_KPCL(i) = -log [ exp(z_i·z_i+ / tau) /
               ( exp(z_i·z_i+ / tau) + sum_k (1 - w_ik)^gamma * exp(z_i·z_k / tau) ) ]
- gamma is a FIXED Hydra hyperparameter (NOT learned); sweep {1,2,4}.
- w_ik detached: the ONLY gradient path is through z (R5 worked example). Assert no grad
  reaches w_ik or S.
- Return dict (R7): loss, info_nce_component, kpcl_fn_weight_mean, fn_weight_entropy,
  temperature, pos_sim_mean, neg_sim_mean.

Tests in tests/test_kpcl.py:
- R3 pos/neg/FN behavior
- gamma=0 (or all w_ik=0) recovers InfoNCE EXACTLY (torch.allclose vs Stage 3 loss)
- gradient isolation: w_ik.requires_grad is False; gradient reaches encoder via z only
- NaN-safe under saturated similarities

Then 1-seed KAN+KPCL on yeast vs DCL/SupCon/InfoNCE -> runs/results/kpcl_yeast/kpcl.csv.
Report margin over DCL. State the gamma chosen and why (R1, R5).
```
**Codex prompt:** `src/losses/kpcl.py: FNC-InfoNCE, denom negatives scaled by (1-w_ik)^gamma, w_ik detached from knots.py, gamma fixed Hydra {1,2,4}. dict(loss,info_nce_component,kpcl_fn_weight_mean,fn_weight_entropy,temperature,pos/neg_sim_mean). tests/test_kpcl.py: R3; gamma=0≡InfoNCE allclose; w_ik no grad & grad via z only; NaN-safe. 1-seed yeast KPCL vs DCL/SupCon/InfoNCE -> runs/results/kpcl_yeast/kpcl.csv. Report DCL margin.`

---

## Stage 8 — KURC loss (always build: fallback + precondition)

**Tier: Opus** (loss).

**Claude Code prompt:**
```
Implement KURC — the knot-occupancy entropy regularizer. Build regardless of H1 (it is
either the fallback method or a precondition switched on under KPCL).

src/losses/kurc.py:
  L_KURC = L_InfoNCE - lambda_occ * H(q),  H(q) = -sum_c q_c log q_c
where q_c is the empirical fraction of the batch that activates knot c (the knot-occupancy
histogram computed from S(x) over the head edges). Clamp q_c at 1e-12 before log
(R9, pitfall 5). lambda_occ from Hydra. q is detached (structural). Return dict (R7):
loss, info_nce_component, kurc_entropy, occupancy_min, occupancy_max, temperature.
Provide a flag to compose KURC UNDER KPCL (L = L_KPCL - lambda_occ*H(q)).

Tests in tests/test_kurc.py:
- collapsed occupancy (all mass on few knots) -> low entropy -> larger penalty
- uniform occupancy -> high entropy -> small penalty
- lambda_occ=0 recovers InfoNCE exactly
- q carries no gradient

Then 1-seed KAN+KURC on yeast: report AUROC and uniformity vs InfoNCE
-> runs/results/kurc_yeast/kurc.csv.
```
**Codex prompt:** `src/losses/kurc.py: L=InfoNCE - lambda_occ*H(q), q=batch knot-occupancy from S(x), clamp 1e-12 before log, lambda_occ Hydra, q detached. dict(loss,info_nce_component,kurc_entropy,occupancy_min/max,temperature). flag to compose under KPCL. tests/test_kurc.py: collapsed->low H->big penalty, uniform->high H, lambda=0≡InfoNCE, q no grad. 1-seed yeast KURC vs InfoNCE AUROC+uniformity -> runs/results/kurc_yeast/kurc.csv.`

---

## Stage 9 — Full sweep + H2/H3 verdict

**Tier: Sonnet** (orchestration).

**Claude Code prompt:**
```
Run the full experiment matrix and evaluate H2 (if H1 passed) and H3.

Matrix: datasets {yeast, scene, emotions, mediamill} (bibtex as reported-only low-FN
stress test) x methods {InfoNCE, SupCon, DCL, KPCL(if H1 passed), KURC,
KPCL+KURC(if H1 passed)} x seeds {42,1337,2024,7,9001}. KAN encoder for KPCL/KURC;
param-matched MLP encoder for the InfoNCE/SupCon/DCL baselines AND an MLP+cosine-FN-weight
control (the decisive baseline proving the knot signal beats an embedding proxy at matched
params). Use configs/experiment/sweep_h2.yaml (Hydra multirun). Full R8 artifacts per run.

Seed-averaged tables (mean±std) of macro-AUROC, mAP, uniformity -> runs/results/sweep/
sweep_tables.csv. Then evaluate:
  H2 (if H1 passed): KPCL beats InfoNCE by >=1.5 AUROC and DCL by >=0.5, mean over the 4
    datasets, >=3/4 individually significant (paired bootstrap p<0.05). Also report KPCL
    vs MLP+cosine-FN control.
  H3: KURC >= InfoNCE AUROC and uniformity improved by >=0.1 nats.
Write runs/results/sweep/verdict.md. Do not massage; report what the seeds show.
```
**Codex prompt:** `Hydra multirun: {yeast,scene,emotions,mediamill}(+bibtex reported) x {InfoNCE,SupCon,DCL,KPCL?,KURC,KPCL+KURC?} x seeds{42,1337,2024,7,9001}, KAN enc for KPCL/KURC, param-matched MLP for baselines + MLP+cosine-FN control. Full R8 artifacts. Seed-avg AUROC/mAP/uniformity -> runs/results/sweep/sweep_tables.csv. H2: KPCL-InfoNCE>=1.5 & -DCL>=0.5, >=3/4 sig; vs MLP+cosine control. H3: KURC>=InfoNCE & uniformity+0.1. verdict.md, no massaging.`

---

## Stage 10 — Thesis / paper artifacts

**Tier: Sonnet.**

**Claude Code prompt:**
```
Generate artifacts from runs/results/ (only committed CSV numbers; no fabrication):
- LaTeX tables (booktabs, mean±std, bold best, underline second) for the baseline
  comparison and the full sweep.
- Figures: FN-prevalence vs KPCL-over-DCL margin (the controlled-instrument plot);
  KAN-vs-MLP geometry (alignment/uniformity/eff_rank); H1 FN-ranking AUC bar chart
  (knot vs cos(z)).
- A pre-registration appendix dumping H0/H1/H2/H3, their verdicts, and the commit hash for
  every cited result (R8 metadata).
Save to runs/results/artifacts/. Lead the narrative with representation geometry +
multi-label AUROC; FN detection is the mechanism, not the headline.
```
**Codex prompt:** `From runs/results/: LaTeX tables (booktabs, mean±std, bold/underline) baseline + sweep; figs (FN-prevalence vs KPCL-DCL margin; KAN-vs-MLP geometry; H1 knot-vs-cos AUC bars); pre-reg appendix (H0-H3 verdicts + commit hashes). -> runs/results/artifacts/. Only committed numbers.`

---

## Session checklist (Claude Code, every time)

1. Read `CLAUDE.md` + `BUG_codex.md` + `BUG_claude_code.md`.
2. Confirm the previous stage's gate passed (H0 before Stage 4; H1 before Stage 7).
3. State a numbered plan (R1) with a verify step each (R4).
4. Implement; run the stage's tests; write artifacts to the exact `runs/` paths.
5. Commit `[Stage<N>] <what> — <why>`; append fixes to `BUG_claude_code.md` if reviewing.

## The two gates that decide the thesis (do not skip, do not tune to pass)

- **H0 (Stage 3):** KAN must beat MLP under plain InfoNCE, or the premise is dead.
- **H1 (Stage 6):** the knot pattern must out-rank cos(z) on false negatives, or KPCL is dead and KURC takes over.

Both are cheap, both come *before* the expensive sweep, and both are pre-registered. That is the entire risk-management strategy: **know before committing.**
