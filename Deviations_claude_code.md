# Deviations_claude_code.md — Justified Deviations made by claudecode

Managed by: **Claude code** agent.
Newest entries appended at the bottom under a dated `## YYYY-MM-DD` heading.

---

Per CLAUDE.md Coding Rule 5 ("Document Justified Deviations"): where a literal
instruction would have been wrong, impossible, or strictly worse, I implemented the
corrected version and recorded what changed and why. Each entry cites where the
deviation already lives in code/config/artifacts.

---

## 2026-06-07 — Stage 0 (scaffold)

**D0.1 — Python 3.10 instead of 3.11.**
Instruction: "Create a Python 3.11 venv." Reality: only Python 3.10.5 is installed
(`py -3.11` not found). Did: set `requires-python = ">=3.10,<3.13"` and built `.venv` with
3.10. Why: 3.11 unavailable; no code uses 3.11-only features. Re-create with 3.11 if the
thesis environment requires it. (pyproject.toml)

**D0.2 — `requests` added as an explicit dependency.**
Not requested. scikit-multilearn imports `requests` but does not declare it, so
`pytest --collect-only` failed at import. Added `requests>=2.31`. Why: transitive dependency
the upstream package omits. (pyproject.toml)

---

## 2026-06-07 — Stage 1 (data layer)

**D1.1 — Dataset source: LibSVM + OpenML mirrors instead of `skmultilearn.dataset.load_dataset`.**
Instruction: load the 5 datasets "via scikit-multilearn load_dataset." Reality: the
scikit-multilearn download server (`http://scikit.ml/datasets/data.list`) returns HTTP 403
(dead) for every variant; browser User-Agent and direct file URLs also 403. Did: load
yeast/scene/bibtex/mediamill from the LibSVM multi-label collection (NTU CSIE) and emotions
from OpenML id=40589, parsed to dense float32 X + binary Y, cached to `~/.kpcl_data/`. The
stratified split still uses skmultilearn's `IterativeStratification`. Why: the literal source
is unavailable; datasets, shapes, and the stratifier are identical. (src/data/datasets.py
header; memory project note)

**D1.2 — LibSVM parser handles label-less rows.**
mediamill has rows with an empty label field (0 positive labels), which broke the naive
`label,label feat:val` split. The parser detects a leading `:` token and treats such rows as
zero-label. (src/data/datasets.py `_parse_libsvm_ml`)

---

## 2026-06-07 — Stage 2 (KAN building blocks)

**D2.1 — `knot_indices` returns shape `(B, I, p+1)`, not `(B, O, I, p+1)`.**
Contract: "for each edge (o,i) ... the set of active basis indices." Did: return one code per
input feature `i`. Why: the active basis set is provably **o-independent** — basis functions
are shared across all output edges consuming input `i`; only coefficients differ by `o`.
Per-`(o,i)` would be O identical copies. Broadcast across `o` if a literal per-edge code is
wanted. (src/models/spline_kan.py docstring; tests/test_kan.py)

**D2.2 — MLP twin hidden dim auto-matched per dataset (`hidden_dim: null`).**
Instruction: "tune MLP hidden dims so totals are within ±15%." Did: make MLP width `null` and
solve it per `input_dim` at build time (utils.parity) so R1 parity holds by construction
(≤0.17% on all 5 datasets). Why: required width depends on input_dim, which varies per dataset
(yeast 504, bibtex 849, …). (configs/model/mlp.yaml; utils/parity.py; scripts/training/build.py)

**D2.3 — `param_count.txt` is a repo-level parity table for all 5 datasets.**
R8 specifies a per-run `param_count.txt`; at the building-block stage there is no run yet, so I
wrote a repo-root parity table (KAN vs matched-MLP per dataset). The per-run R8 artifact is
produced by training runs. (param_count.txt)

---

## 2026-06-07/08 — Stage 3 (InfoNCE + augs + SPEC H0)

**D3.1 — `info_nce_loss(z1, z2, temperature)` signature (two views, explicit temperature, no labels).**
Stage-0 stub contract was `info_nce_loss(z, labels, cfg)`. Did: SimCLR NT-Xent over the 2N
batch of two augmented views, explicit `temperature` float, no labels. Why: SimCLR InfoNCE is
self-supervised (positives = augmentation views); an explicit scalar is more testable and the
loop passes `cfg.loss.temperature` (R6 preserved). (src/losses/infonce.py)

**D3.2 — R3 case semantics adapted to SimCLR's structural mask.**
SimCLR's positive mask is fixed/structural, so the three R3 cases were realised as: aligned
positives → loss≈0; all-identical embeddings → `log(2N−1)` (no signal); injected
high-similarity false negative → loss increases. Why: "make the batch all-positive /
all-negative" is not expressible when the mask is structural. (tests/test_infonce.py)

**D3.3 — SPEC H0 subsamples training data and fixes a bounded budget.**
Instruction: "short but real, ≥3 seeds." Did: 3 seeds, 25 epochs, batch 256, **train
subsampled to ≤4000 rows**, `max_steps_per_epoch=40`. Why: CPU-only, 30 runs; keeps it
decision-grade but bounded (~8 min). The subsample is a real departure from full-data training,
logged so H0 is read as a short-tier result; 25 epochs (not fewer) gives the slower-converging
KAN a fair, identical budget. (configs/experiment/spec_h0.yaml; runs/results/spec_h0/)

**D3.4 — Added a parameter-free LayerNorm to the KAN (and an MLP+LN control) after H0.**
NOT in any prompt. H0 first FAILED (2/5). The grid-range diagnostic showed the KAN's layer-2
spline path saturated (bibtex: 99% of activations out of grid_range, spline contributing 3% →
KAN reduced to its SiLU base path). Did: add inter-layer + pre-head
`LayerNorm(elementwise_affine=False)` (`use_layer_norm`, default true) to keep deeper spline
inputs in grid_range, plus an `mlp_ln.yaml` twin so the MLP gets the same normalisation as a
fairness control. Why: a genuine implementation defect (Explainer §2 requires in-grid
activations), parameter-free so R1 parity is intact, applied uniformly, re-gated honestly (not
tuned-to-pass). Re-gate: KAN≥MLP 4/5, KAN≥MLP+LN 3/5 → H0 PASS (edge concentrates on many-label
sets). The original no-norm H0 FAIL stands separately on record. (configs/model/kan.yaml,
mlp_ln.yaml; src/models/encoder.py, heads.py; BUG_claude_code.md 2026-06-08;
runs/results/spec_h0/h0_regate*)

---

## 2026-06-08 — Stage 4 (SupCon + DCL baselines)

**D4.1 — SupCon "positive-only → ~0" tested with a single positive, plus an exact `log(K)` test.**
The multi-label `L^sup_out` form has a per-anchor floor of `log|P(i)|`, so with K equally-aligned
positives the minimum loss is `log(K)`, not 0. Did: test ≈0 with unique labels (one positive
per anchor) and add an exact `loss == log(3)` assertion for the 3-positive aligned case. Why:
literal "~0" is unattainable for multi-positive SupCon-out; the `log(K)` test still verifies the
positive mask groups all same-label samples. (tests/test_supcon.py)

**D4.2 — `neg_correction_mean` defined as the mean debiased negative term g.**
The key name was specified but not its exact value; defined as the mean over anchors of the
clamped debiased estimator `g` (most informative single diagnostic of the DCL correction).
(src/losses/dcl.py)

---

## 2026-06-08 — Repo reorganisation (user-directed)

**D5.1 — `src/train.py` left in `src/`, not moved to scripts/.**
When moving experiment drivers to `scripts/`, I left the Hydra entry `src/train.py` in place.
Why: the selected option specified moving `experiments/*`; `train.py` is the package's config
entry point, not an experiment driver. It would move safely (its `config_path="../configs"`
resolves the same from a sibling top-level dir) — flagged for the user, not done unilaterally.

---

## 2026-06-08 — Stage 5 (knot code + Jaccard)

**D5.2 — `knot_code` also accepts a `KANEncoder`; the head code is not forward-norm-consistent.**
Instruction: `knot_code(head, h)`. Did: a `_spline_layer` helper so `knot_code` works on a
`KANHead` (`.layer`) AND a `KANEncoder` (`.layers[0]`). Why: the canonical KPCL `S(x)` is the
encoder layer-0 code, not the head's. Caveat documented: `KANHead.knot_indices` reads raw `h`
while `KANHead.forward` applies a pre-norm first, so a *head* code is not forward-consistent;
the *encoder* layer-0 code has no pre-norm and is forward-consistent by construction — use the
encoder for KPCL. (src/models/knots.py)

**D5.3 — `knot_code` returns the O-replicated `(B, O*I*(G+p))` per spec despite O-redundancy.**
Followed the spec shape and the `4*O*I`-ones test exactly, though Jaccard is provably invariant
to the O-fold replication (numerator and denominator both scale by O). Noted that a caller may
compute `w` on the compact `(B, I*(G+p))` per-feature code for speed. (src/models/knots.py)
