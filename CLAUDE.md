# CLAUDE.md — KPCL/KURC Scientific Rules & Agent Guidelines

This document defines the inviolable scientific rules and agent configuration for
Claude Code work on the KPCL/KURC thesis. Every module, loss, and dataset handler
must comply. Spec experiments gate everything: no expensive run before cheap gates pass.

---

## Reference Documents

Two documents live in the repo root and are authoritative for all design decisions.
Read them before asking why something is built a certain way — they contain the
justification, not just the conclusion.

- `KPCL_KURC_Explainer.md` — Architecture explainer. Covers why a KAN and not an MLP,
  the knot-pattern code S(x) and its Jaccard similarity, the KPCL and KURC loss
  derivations, training algorithm, and hyperparameter rationale. Source of truth for
  *what the method is and why every piece is shaped the way it is*.
  Key sections: §2 (KAN vs MLP), §3 (S(x) code), §4 (KPCL loss), §5 (KURC),
  §6 (full architecture + Algorithm 1), §7 (decisive baseline), §8 (gates).

- `KPCL_KURC_Implementation_Playbook.md` — Stage-by-stage build plan. Covers the exact
  repo layout, every stage's Claude Code prompt, gate conditions (H0–H3), model tiers,
  and session checklist. Source of truth for *what to build next and in what order*.

**Conflict resolution:** when these documents conflict with inline CLAUDE.md rules,
CLAUDE.md wins. Use the documents for context and rationale; use CLAUDE.md for enforcement.

---

## Cross-Agent Bug Communication

Two agents review this codebase — Codex and Claude Code — communicating through:
- `BUG_codex.md` — Codex writes code reviews and bug findings (severity-ranked, with evidence + repro).
- `BUG_claude_code.md` — Claude Code reads BUG_codex.md, then records per issue what was broken and what was fixed (or why deferred, with rationale).

Rules: on "review"/"fix bugs"/"check the other agent", read BOTH files first. Do not
create ad-hoc *.md bug files. Newest entries at the bottom under a dated heading. Cite
the source review in each reply. Deferrals must state a reason.

---

## Coding Rules (bias to caution over speed; use judgment on trivial tasks)

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
See also: `KPCL_KURC_Explainer.md` §6, Algorithm 1, Remark 2 for full gradient hygiene rationale.

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

Full justification for pitfalls 2–6 (partition-of-unity nullspace, rank decay, Jaccard
bounds, entropy clamping, gradient hygiene) is in `KPCL_KURC_Explainer.md` §2–§6.

---

## Hypothesis Gates (pre-registered; binding; recorded with commit hash)

- **H0 (premise):** KAN-InfoNCE ≥ MLP-InfoNCE on ≥3/5 datasets (param-matched).
  FAIL → KAN premise broken, stop.
- **H1 (KPCL signal):** on yeast & mediamill, knot-Jaccard FN-ranking AUC ≥ 0.55
  absolute AND ≥ AUC_cos(z) + 0.05, on ≥1/2 datasets, ≥5 seeds, paired bootstrap p<0.05.
  FAIL → KPCL falsified, fall back to KURC.
- **H2 (KPCL headline):** KPCL beats InfoNCE by ≥1.5 macro-AUROC and DCL by ≥0.5,
  mean over yeast/scene/emotions/mediamill, ≥3/4 individually significant.
- **H3 (KURC fallback):** KURC ≥ InfoNCE on AUROC and improves uniformity by ≥0.1 nats.

For gate rationale and what each failure means for the thesis, see
`KPCL_KURC_Explainer.md` §8. For the exact experiment protocols that evaluate H0 and H1,
see Stage 3 and Stage 6 of `KPCL_KURC_Implementation_Playbook.md`.

### Smoke runs are not evidence
A smoke run (≤10 steps, B=4) only proves the pipeline executes. Tag smoke run_ids smoke_*
so they are filtered from gate evaluation. Only multi-seed full runs count.

---

## When to Escalate to Opus

Losses (R3/R7 subtle), split logic (R4/R5), knot extraction, NaN/non-convergence debug.
Use Sonnet for probes/metrics/configs/orchestration. Haiku for renames/TODOs.

---

## Git Commit Format

```
[Stage<N>] <what> — <why one sentence>
```

---

## Metadata for Thesis Claims

Cite stage number, commit hash, seeds [42,1337,2024,7,9001], mean±std from results CSVs.

---

## Session Checklist (run at the start of every Claude Code session)

1. Read `CLAUDE.md` + `BUG_codex.md` + `BUG_claude_code.md`. If starting a new stage
   or implementing a loss/knot module for the first time, also skim the relevant section
   of `KPCL_KURC_Explainer.md` (§3 for S(x)/Jaccard, §4 for KPCL loss, §5 for KURC,
   §6 for the full architecture) and confirm the current stage in
   `KPCL_KURC_Implementation_Playbook.md`.
2. Confirm the previous stage's gate passed (H0 before Stage 4; H1 before Stage 7).
3. State a numbered plan (R1) with a verify step each (R4).
4. Implement; run the stage's tests; write artifacts to the exact `runs/` paths.
5. Commit `[Stage<N>] <what> — <why>`; append fixes to `BUG_claude_code.md` if reviewing.
6. Update the TODO after every prompt.