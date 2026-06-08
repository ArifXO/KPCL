# TODO.md — KPCL/KURC Thesis Work Tracker

Stages gate sequentially (R2). Do not start Stage N+1 before Stage N tests are green.

---

## Stage 0 — Scaffolding (done)
- [x] pyproject.toml + venv setup
- [x] Directory layout + importable stubs
- [x] Hydra config skeleton (configs/)
- [x] BUG_codex.md, BUG_claude_code.md stubs
- [x] `git init` + first commit `[Stage0] scaffold — project skeleton`

## Stage 1 — Data Pipeline (R4, R5)
- [x] Implement `loader.load_dataset` for all 5 datasets (ARFF via liac-arff)
- [x] Iterative stratified multi-label split (scikit-multilearn)
- [x] Assert disjoint train/val/test index sets; raise ValueError on overlap
- [x] StandardScaler fit on train only; applied to val/test
- [x] Report per-label positive prevalence (from loader, never from paper)
- [x] Unit tests: disjoint splits, no-leakage, prevalence shape/range

## Stage 2 — InfoNCE Baselines (R1, R2, R3, R7) [H0 gate]
- [x] KAN building blocks: KANLayer (cubic B-spline + SiLU base), knot_indices readout
- [x] KAN encoder (L=2, hidden 128) + KAN head (out 64)
- [x] MLP encoder/head twins, param parity ≤15% (all 5 datasets ~0.1%) → param_count.txt
- [x] tests/test_kan.py green (forward, knot_indices, grad isolation, parity, MLP raises)
- [x] `info_nce_loss` (SimCLR NT-Xent, 2N) + R3 tests (pos≈0, neg>0, FN↑, R7 keys)
- [x] augment.py (feature-mask/gaussian/mixup, 2 views), probe.py (AUROC/mAP), geometry.py
- [x] Training loop wiring (build encoder+head from cfg; optimizer weight_decay≥1e-4)
- [x] H0 gate RAN: KAN-InfoNCE vs MLP-InfoNCE, 5 datasets × 3 seeds → runs/results/spec_h0/
- [!] **H0 FAIL (2/5)** on as-tested arch. KAN wins mediamill+bibtex; loses yeast; ties scene/emotions.
- [x] Diagnosed grid-range: saturation real on scene(28%)/bibtex(99%), NOT on yeast/emotions (healthy).
- [x] FIXED: parameter-free inter-layer + pre-head LayerNorm (use_layer_norm). Splines now alive
      on all 5 (L2 coverage ~0.99, spline_share ~0.73). 0 params, R1 parity intact, 49 tests green.
- [x] **RE-GATED (fixed KAN, 3 arms): H0 PASS.** (A) KAN≥MLP 4/5; (B) KAN≥MLP+LN 3/5.
      KAN edge is structural+robust on many-label sets (mediamill/bibtex); scene was a
      norm effect; yeast lost. runs/results/spec_h0/h0_regate.{csv,verdict.md}.
      → Premise holds; KPCL line UNBLOCKED.

## Stage 4 — SupCon + DCL baselines — DONE
- [x] supcon.py (multi-label: share≥1 label=positive; keys loss/supcon_component/temperature/n_positives_mean)
- [x] dcl.py (Chuang2020 debiased denom, tau_plus Hydra, clamp g≥exp(−1/t), NaN-raise; keys
      loss/dcl_component/tau_plus/neg_correction_mean). tau_plus=0 ≡ InfoNCE (allclose test).
- [x] R3 tests both (test_supcon.py, test_dcl.py) — 16 green. Loop dispatches by cfg.loss.type.
- [x] 1-seed yeast KAN baseline → runs/results/baseline_yeast/baseline.csv:
      InfoNCE 0.6907 / SupCon 0.6745 / DCL 0.6839 macro-AUROC (yeast = KAN's weakest set).

## Stage 5 — Knot extraction S(x) + Jaccard w_ik — DONE
- [x] src/models/knots.py: knot_code(module,h)->(B,O*I*(G+p)) detached binary (4*O*I ones);
      jaccard_weight(S)->(B,B) in [0,1] (R9 assert, diag=1); smoothed_weight (exp(-beta*L1)).
      MLP module raises NotImplementedError. Works on KANHead AND KANEncoder (canonical S(x)).
- [x] tests/test_knots.py (8): identical→w=1, disjoint→w=0, 4*O*I ones, S&w no-grad, MLP raises.
      73 tests green total.

## Stage 6 — SPEC H1: knot-Jaccard FN-ranking AUC vs cos(z) — DONE → **H1 PASS**
- [x] src/metrics/fn_ranking.py (cos/knot/smoothed AUC, label-pair masks, paired bootstrap);
      knot_code_compact in knots.py; tests/test_fn_ranking.py (5). 78 tests green.
- [x] scripts/smoke/spec_h1.py: KAN+InfoNCE @100ep, 5 seeds, yeast+mediamill → runs/results/spec_h1/
- [x] **H1 PASS:** mediamill knot 0.642 vs cos 0.513 (+0.129, p≈0, ≥0.55) clears all 3 conditions;
      yeast no signal (knot≈cos, <0.55). → KPCL CONFIRMED, build it. (GPU; RTX 3060)

## Stage 7 — KPCL loss (H1 passed → GO) — DONE
- [x] `kpcl_loss(z1,z2,w,tau,gamma)`: FNC-InfoNCE, negatives scaled by (1-w_ik)^gamma,
      w_ik DETACHED; log-sum-exp NaN-safe. Loop dispatches kpcl (builds w from v1,v2 knot codes).
- [x] tests/test_kpcl.py (10): R3 + FN-cancellation, gamma=0 & w=0 ≡ InfoNCE (allclose),
      grad reaches encoder via z ONLY (w.grad/S.grad None), NaN-safe. 88 tests green.
- [x] 1-seed yeast → runs/results/kpcl_yeast/kpcl.csv. KPCL≈InfoNCE (no yeast signal, as H1
      predicted); best kpcl_g2 0.6921 vs DCL 0.6784 (+0.0137). Default gamma=2.
      Real KPCL test = H2 sweep on mediamill/scene/emotions.

## Stage 8 — KURC (knot-occupancy entropy regularizer) — DONE
- [x] `kurc_loss(z1,z2,B,tau,lambda_occ,base_dict)`: L = base - lambda*H(q). q = SOFT
      differentiable occupancy over head spline cells (partition of unity) — user-confirmed
      real regularizer (H(q) carries gradient); hard S(x) stays detached (KPCL). clamp 1e-12.
      under_kpcl flag composes L_KPCL - lambda*H. tests/test_kurc.py (7). 95 tests green.
- [x] 1-seed yeast → runs/results/kurc_yeast/kurc.csv: KURC AUROC 0.6917 ≈ InfoNCE 0.6923
      (Δ-0.0006); uniformity -3.561 vs -3.547 (Δ-0.015 nats, more uniform); eff_rank 50.4 vs
      49.5 (less collapse). Regularizer active but lambda=0.1 mild on yeast (< H3's 0.1-nat).

## Stage 9 — H2/H3 sweep (4 datasets × suite × 5 seeds, GPU) — NEXT
- [ ] H2: KPCL beats InfoNCE +1.5 & DCL +0.5 macro-AUROC (mediamill/scene/emotions focus)
- [ ] H3: KURC ≥ InfoNCE AUROC & uniformity +0.1 nats (may need larger lambda_occ)

## Stage 8+ — KURC (always build) + H2/H3 sweep
- [ ] kurc_loss (entropy reg); H2: KPCL vs InfoNCE(+1.5)/DCL(+0.5); H3: KURC vs InfoNCE

## Stage 4 — KURC Fallback (H3)
- [ ] `kurc_loss` implementation + tests (entropy keys, q_c clamped at 1e-12)
- [ ] H3 gate: AUROC ≥ InfoNCE; uniformity improvement ≥ 0.1 nats
