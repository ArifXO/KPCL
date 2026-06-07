# TODO.md — KPCL/KURC Thesis Work Tracker

Stages gate sequentially (R2). Do not start Stage N+1 before Stage N tests are green.

---

## Stage 0 — Scaffolding (done)
- [x] pyproject.toml + venv setup
- [x] Directory layout + importable stubs
- [x] Hydra config skeleton (configs/)
- [x] BUG_codex.md, BUG_claude_code.md stubs
- [ ] `git init` + first commit `[Stage0] scaffold — project skeleton`

## Stage 1 — Data Pipeline (R4, R5)
- [ ] Implement `loader.load_dataset` for all 5 datasets (ARFF via liac-arff)
- [ ] Iterative stratified multi-label split (scikit-multilearn)
- [ ] Assert disjoint train/val/test index sets; raise ValueError on overlap
- [ ] StandardScaler fit on train only; applied to val/test
- [ ] Report per-label positive prevalence (from loader, never from paper)
- [ ] Unit tests: disjoint splits, no-leakage, prevalence shape/range

## Stage 2 — InfoNCE Baselines (R1, R2, R3, R7) [H0 gate]
- [x] KAN building blocks: KANLayer (cubic B-spline + SiLU base), knot_indices readout
- [x] KAN encoder (L=2, hidden 128) + KAN head (out 64)
- [x] MLP encoder/head twins, param parity ≤15% (all 5 datasets ~0.1%) → param_count.txt
- [x] tests/test_kan.py green (forward, knot_indices, grad isolation, parity, MLP raises)
- [x] `info_nce_loss` (SimCLR NT-Xent, 2N) + R3 tests (pos≈0, neg>0, FN↑, R7 keys)
- [x] augment.py (feature-mask/gaussian/mixup, 2 views), probe.py (AUROC/mAP), geometry.py
- [x] Training loop wiring (build encoder+head from cfg; optimizer weight_decay≥1e-4)
- [x] H0 gate RAN: KAN-InfoNCE vs MLP-InfoNCE, 5 datasets × 3 seeds → runs/results/spec_h0/
- [!] **H0 FAIL (2/5).** KAN wins mediamill+bibtex decisively; loses yeast; ties scene/emotions.
      GATE: STOP — do not build KPCL until premise resolved. See BUG_claude_code.md 2026-06-08.
      Open fairness check before declaring premise dead: layer-2 spline grid-range saturation.

## Stage 3 — KPCL (R3, R7, H1, H2)
- [ ] `kpcl_loss`: knot-Jaccard w_ik (detached), w_ik=0 → InfoNCE exact (R3)
- [ ] Unit tests: w_ik=0 allclose to InfoNCE; FN injection changes loss direction
- [ ] H1 gate: FN-ranking AUC ≥ 0.55 & ≥ AUC_cos(z)+0.05 on ≥1/2 datasets
- [ ] H2 gate: full benchmark vs InfoNCE (+1.5 macro-AUROC) and DCL (+0.5)

## Stage 4 — KURC Fallback (H3)
- [ ] `kurc_loss` implementation + tests (entropy keys, q_c clamped at 1e-12)
- [ ] H3 gate: AUROC ≥ InfoNCE; uniformity improvement ≥ 0.1 nats
