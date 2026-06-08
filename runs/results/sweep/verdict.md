# H2 / H3 Sweep Verdict

Seeds [42, 1337, 2024, 7, 9001]. KPCL=kan_kpcl, KURC=kan_kurc; baselines = MLP+LayerNorm. kan_infonce = same-encoder reference. Paired bootstrap over 5 seeds. Not massaged.

## H2 — KPCL vs InfoNCE(+1.5) and DCL(+0.5), mean over 4 datasets, >=3/4 significant

| dataset | KPCL | MLP-InfoNCE | MLP-DCL | MLP-cosFN | KAN-InfoNCE | KPCL−InfoNCE | p(vs InfoNCE) | KPCL−DCL | KPCL−cosFN |
|---|---|---|---|---|---|---|---|---|---|
| yeast | 0.7016 | 0.7124 | 0.7109 | 0.7083 | 0.7032 | -0.0108 | 1.000 | -0.0093 | -0.0067 |
| scene | 0.9399 | 0.9486 | 0.9424 | 0.9484 | 0.9415 | -0.0087 | 1.000 | -0.0025 | -0.0085 |
| emotions | 0.8277 | 0.8162 | 0.8139 | 0.8220 | 0.8282 | +0.0115 * | 0.008 | +0.0138 | +0.0057 |
| mediamill | 0.7040 | 0.6255 | 0.7354 | 0.6322 | 0.7057 | +0.0785 * | 0.000 | -0.0314 | +0.0718 |

Mean KPCL−InfoNCE = **+0.0176** (need ≥+0.015); KPCL−DCL = **-0.0073** (need ≥+0.005); significant on **2/4** (need ≥3).

## VERDICT: H2 FAIL

## H3 — KURC ≥ InfoNCE AUROC and uniformity improved ≥0.1 nats

| dataset | KURC AUROC | InfoNCE AUROC | ΔAUROC | KURC uni | InfoNCE uni | Δuni (nats) |
|---|---|---|---|---|---|---|
| yeast | 0.7026 | 0.7124 | -0.0098 | -3.6446 | -3.7919 | +0.1473 |
| scene | 0.9417 | 0.9486 | -0.0069 | -3.6821 | -3.8045 | +0.1224 |
| emotions | 0.8263 | 0.8162 | +0.0101 | -3.3766 | -3.4917 | +0.1152 |
| mediamill | 0.6974 | 0.6255 | +0.0719 | -3.7418 | -3.7068 | -0.0350 |

Mean ΔAUROC = **+0.0163** (need ≥0); mean Δuniformity = **+0.0875** nats (need ≤−0.1, i.e. more uniform).

## VERDICT: H3 FAIL

_Note: KPCL/KURC use the KAN encoder, baselines the MLP — 'vs InfoNCE' conflates encoder + loss; kan_infonce and KPCL−cosFN isolate the loss/signal. bibtex is in sweep_tables.csv (reported-only)._
---

## Clean same-encoder decomposition (the honest reading)

The headline "KPCL vs MLP-InfoNCE" conflates ENCODER (KAN vs MLP) with LOSS (KPCL vs
InfoNCE). The kan_infonce reference separates them.

**KPCL-the-LOSS effect** (kan_kpcl − kan_infonce, same KAN encoder):
yeast −0.0016, scene −0.0016, emotions −0.0005, mediamill −0.0017, bibtex −0.0159.
→ **The KPCL FN-cancellation loss adds ~0 (slightly hurts) everywhere.** The H1 knot signal
does NOT translate into a downstream accuracy gain over plain InfoNCE at matched encoder.

**KAN-ENCODER effect** (kan_infonce − mlp_infonce, plain InfoNCE both):
yeast −0.0092, scene −0.0071, emotions +0.0120, mediamill **+0.0802**, bibtex **+0.1257**.
→ The ENTIRE apparent "KPCL win" on mediamill (+0.0785 vs MLP-InfoNCE) is the KAN encoder
(+0.0802); the loss contributes −0.0017. The KAN encoder helps on high-cardinality data and
hurts on low.

**But the strongest baselines beat KPCL where it should win:** on mediamill, MLP-DCL 0.7354
and MLP-SupCon 0.7323 both beat kan_kpcl 0.7040; on bibtex MLP-DCL 0.8946 ≫ kan_kpcl 0.8635.
So even the KAN-encoder advantage does not make the full KPCL method competitive with a plain
MLP + a debiased/supervised loss.

**KURC-the-REGULARIZER effect** (kan_kurc − kan_infonce, same encoder), λ_occ=0.1:
Δuniformity (nats): yeast −0.019, scene −0.016, emotions −0.004, mediamill −0.011 (mean
≈ −0.013 — right direction, but ~8× below the −0.1 target). Δeff-rank: +0.4…+1.4 consistently
(anti-collapse works, weakly). AUROC unchanged. → H3 fails on magnitude, not direction; would
need a much larger λ_occ.

**Summary:** H0 (KAN premise) and H1 (knot signal exists) hold; but neither proposed METHOD
clears its gate — KPCL's loss adds nothing downstream, and KURC's regularizer is ~8× too weak
at λ=0.1. The KAN *encoder* is the only source of value, and even it is beaten by MLP+DCL on
the high-cardinality sets.

**Possible (untested) reason KPCL fails downstream:** w_ik is computed on AUGMENTED inputs
during training (feature-mask + noise perturb the knot intervals), so the training-time FN
signal is noisier than the clean-input H1 measurement (AUC 0.64). A noisy weight cancels true
negatives about as often as false ones. Future direction, not a fix applied here.
