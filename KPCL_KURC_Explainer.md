# KPCL & KURC
## Knot-Pattern Contrastive Learning and Knot-Uniformity Regularization for KAN-Based Multi-Label Representation Learning

**Thesis Research Explainer — June 2026**

---

## Abstract

This document explains, in rigorous but accessible terms, the architecture and training procedure of two related methods: **KPCL** (Knot-Pattern Contrastive Learning) and its safer sibling **KURC** (Knot-Uniformity Regularized Contrastive learning). Both use a Kolmogorov–Arnold Network (KAN) encoder and projection head built on cubic B-splines.

The central idea is that a KAN computes, for every input, a discrete pattern of active spline knots — a quantity that simply does not exist inside a multilayer perceptron (MLP). KPCL turns this pattern into a deterministic false-negative weight for the contrastive loss; KURC turns it into a regularizer that prevents spline collapse. We explain what each architectural component does, why it is built the way it is, why an MLP cannot replicate the signal, and how the whole system is implemented and falsified. The emphasis throughout is on the architecture and the reasons behind every design choice.

> *A companion to the de-risking report. Every design choice is justified against a specific failure mode of the previous (falsified) approach.*

---

## Contents

1. [The Problem We Are Solving](#1-the-problem-we-are-solving)
2. [The Architectural Foundation: Why a KAN, Not an MLP](#2-the-architectural-foundation-why-a-kan-not-an-mlp)
3. [The Knot-Pattern Code S(x)](#3-the-knot-pattern-code-sx)
4. [Method 1: KPCL — Knot-Pattern Contrastive Learning](#4-method-1-kpcl--knot-pattern-contrastive-learning)
5. [Method 2: KURC — Knot-Uniformity Regularized Contrastive Learning](#5-method-2-kurc--knot-uniformity-regularized-contrastive-learning)
6. [The Full Architecture, End to End](#6-the-full-architecture-end-to-end)
7. [The Decisive Baseline: Giving the MLP Its Best Shot](#7-the-decisive-baseline-giving-the-mlp-its-best-shot)
8. [Falsifiable Hypotheses and Pre-Registered Gates](#8-falsifiable-hypotheses-and-pre-registered-gates)
9. [Staged Implementation Plan](#9-staged-implementation-plan)
10. [Summary: The Idea in One Page](#10-summary-the-idea-in-one-page)

[Key References (arXiv IDs)](#key-references-arxiv-ids)

---

## 1. The Problem We Are Solving

### 1.1 Self-supervised contrastive learning, in one paragraph

We want to learn good representations of data without labels. The contrastive recipe: take a sample **x**, make two augmented views **x′**, **x″**, push their representations together (a positive pair), and push them apart from all other samples in the batch (negatives). The standard objective is InfoNCE:

$$L_{\text{InfoNCE}}(i) = -\log \frac{\exp(z_i \cdot z_i^+ / \tau)}{\exp(z_i \cdot z_i^+ / \tau) + \sum_{k \in \mathcal{N}(i)} \exp(z_i \cdot z_k / \tau)}$$

where $z_i$ is the projected embedding, $z_i^+$ its positive view, $\tau$ a temperature, and $\mathcal{N}(i)$ the set of negatives.

### 1.2 The false-negative problem

Because we have no labels during training, the negatives in $\mathcal{N}(i)$ include samples that actually share a class with $x_i$. Pushing these apart is wrong — they are **false negatives**. In multi-label data the problem is acute: two samples can share three of five labels and still be treated as a hard negative. Correcting for false negatives is the central technical problem of this thesis.

### 1.3 Why the previous attempt failed (and what it teaches us)

The earlier design computed a per-edge activation fingerprint **F** and a disagreement signal $\delta_{ik} = \cos(F_i, F_k) - \cos(z_i, z_k)$, hypothesizing that positive $\delta$ marks a false negative. A pre-registered gate showed $\delta$ ranked false negatives at chance (≈ 0.50 AUC).

> **The lesson that drives every choice below**
>
> **F** and **z** come from the same forward pass: $z_o = \sum_i \varphi_{o,i}(h_i)$ is a linear sum of the very edge activations that **F** flattens. So $\cos(F)$ and $\cos(z)$ are tightly coupled by construction, and their difference carries no class information. Worse, a learned scorer over **F** gave an MLP the same advantage as a KAN, because activation magnitudes are MLP-replicable.
>
> **Conclusion: the signal must not be a magnitude, and it must not be fed through a learned scorer. It must be a discrete, structural property of the KAN, entering the loss deterministically.**

---

## 2. The Architectural Foundation: Why a KAN, Not an MLP

### 2.1 MLP: fixed activations on nodes

An MLP layer computes $y = \sigma(Wx + b)$. The nonlinearity $\sigma$ (ReLU, GELU) is fixed and sits on the nodes. Each edge contributes a single scalar $w_{oi} x_i$. There is one number per edge per sample.

### 2.2 KAN: learnable functions on edges

A KAN layer places a learnable univariate function on every edge. For input $h \in \mathbb{R}^I$ and output $y \in \mathbb{R}^O$:

$$y_o = \sum_{i=1}^{I} \varphi_{o,i}(h_i), \quad \varphi_{o,i}(h) = \sum_{c=1}^{G+p} W_{o,i,c} \, B_c(h)$$

where each $\varphi_{o,i}$ is a cubic B-spline ($p = 3$) with $G$ grid intervals and $B_c$ the B-spline basis functions. The edge weight is no longer a scalar — it is an entire curve evaluated at the input.

### 2.3 The one property that matters: local support

Cubic B-splines have **compact local support**. For any input value $h$, only $p+1 = 4$ consecutive basis functions $B_c(h)$ are non-zero. Which four depends on which grid interval $h$ falls into. This is the structural fact the entire thesis is built on.

> **The signal that does not exist in an MLP**
>
> For each edge $(o, i)$ and input $h_i$, define the active knot index
>
> $$\kappa_{o,i}(x) = \{c : B_c(h_i) \neq 0\} \quad \text{(exactly 4 indices for cubic B-splines)}.$$
>
> This index is a **discrete, categorical descriptor** of which piece of each learned function processed this sample. An MLP has no grid, no knots, and no intervals — there is nothing analogous to extract. This is why the signal is MLP-impossible in a meaningful sense: not merely hard to compute, but *undefined* in the MLP forward pass.

### 2.4 An important honesty caveat about "MLP-impossible"

A theoretical result (arXiv:2410.01803) shows a cubic-B-spline KAN with no base path is exactly representable by a wide ReLU³-MLP. So the *function* a KAN computes is not unique to KANs. What is unique is the discrete intermediate representation $\kappa(x)$: the MLP would have to learn to discretize its activations into bins and would pay a sample-complexity cost to do so, whereas the KAN exposes $\kappa(x)$ exactly and for free. We therefore claim **MLP-impracticality**, demonstrated empirically by a baseline in which the MLP is given its best shot at the same mechanism (Section 7).

---

## 3. The Knot-Pattern Code S(x)

### 3.1 Definition

**Definition 1 (Knot-pattern code).** Let the projection head have edges indexed by $(o, i)$, $o \in [O]$, $i \in [I]$, each a cubic B-spline over a grid of $G$ intervals (so $G + p$ basis functions). For a sample $x$ with head-input $h_i$, define the per-sample binary code

$$S(x) \in \{0, 1\}^{O \cdot I \cdot (G+p)}, \quad S(x)_{o,i,c} = \mathbb{1}[B_c(h_i) \neq 0].$$

By local support, $S(x)$ has exactly $4 \cdot O \cdot I$ ones.

### 3.2 Why this code and not the activations

Three reasons, each tracing to a failure of the previous approach.

1. **Distinct from z.** $S(x)$ depends only on which interval $h_i$ lies in — a quantile of the input, not its magnitude. Two samples can have very different $z$ yet identical $S$ (processed by the same spline pieces), and vice versa. The information in $S$ is not recoverable from $z$ by a linear map, which is exactly what **F** failed to guarantee.

2. **Deterministic.** $S(x)$ is read off the forward pass with a single `bucketize` per edge. No parameters, no learned scorer — so no MLP can also learn it, and there is no scorer to collapse.

3. **Discrete and bounded.** Set overlap of two codes lives in $[0, 1]$ and is numerically trivial, sidestepping the $\log(1-p)$ instabilities and saturation collapse documented for learned FN scorers.

### 3.3 The similarity between two codes

$$w_{ik} = \text{Jaccard}\!\left(\text{supp}\, S(x_i),\, \text{supp}\, S(x_k)\right) = \frac{|S(x_i) \cap S(x_k)|}{|S(x_i) \cup S(x_k)|} \in [0, 1]$$

High $w_{ik}$ means the two samples were routed through largely the same spline pieces on most edges — a strong, label-free hint that they are semantically similar and therefore a likely false negative.

> **Remark 1 (Smoothed relaxation).** Pure Jaccard is discrete and can give a noisy training signal. A continuous fallback weights by knot-index distance: $w_{ik} = \exp(-\beta \|\kappa(x_i) - \kappa(x_k)\|_1)$ with $\beta \approx 0.5$, where $\kappa$ stacks the (single) active interval index per edge. Use this only if Jaccard underperforms at the H1 gate.

---

## 4. Method 1: KPCL — Knot-Pattern Contrastive Learning

### 4.1 The loss

KPCL down-weights likely false negatives in the InfoNCE denominator using the deterministic weight (3):

$$L_{\text{KPCL}}(i) = -\log \frac{\exp(z_i \cdot z_i^+ / \tau)}{\exp(z_i \cdot z_i^+ / \tau) + \sum_k (1 - w_{ik})^\gamma \exp(z_i \cdot z_k / \tau)}$$

with $\gamma \in \{1, 2, 4\}$ a fixed (not learned) sharpness hyperparameter. When $w_{ik} \to 1$ (clear false negative) the negative's contribution vanishes; when $w_{ik} \to 0$ (genuine negative) it is kept at full strength.

### 4.2 Why it is built exactly this way

- **Multiplicative down-weighting, not removal.** A hard threshold (delete negatives with $w > 0.5$) is brittle and non-differentiable in $w$. The factor $(1 - w_{ik})^\gamma$ degrades smoothly, and because $w_{ik}$ is detached (it is a structural readout, not a path we backprop through), it acts as a stable per-pair scalar.

- **The weight comes from the head, not the encoder.** The projection head is where contrastive geometry is shaped and discarded after pre-training, so reading $S$ there keeps the encoder free to learn general features while the FN signal is computed where the contrast actually happens.

- **Strict generalization.** Setting $\gamma = 0$ (or all $w_{ik} = 0$) recovers plain InfoNCE exactly. Every baseline is a special case, which makes ablation clean.

### 4.3 What KPCL is and is not

KPCL is an FNC-style (false-negative-cancellation) method whose novel ingredient is the *source* of the FN signal: a discrete KAN knot pattern rather than an embedding-space cosine or a learned scorer. It does not introduce a new contrastive objective family; it changes what tells the objective which negatives to trust.

---

## 5. Method 2: KURC — Knot-Uniformity Regularized Contrastive Learning

### 5.1 Motivation: spline collapse

KAN B-spline bases obey a partition of unity ($\sum_c B_c(h) \equiv 1$) and are linearly dependent, giving a rank-deficient Hessian; without care, many spline pieces go unused (*spline death*), collapsing effective rank (observed: 117 → 45 previously). If most samples activate the same few knots, $S(x)$ carries little information and KPCL has nothing to work with.

### 5.2 The regularizer

Let $q_c$ be the empirical fraction of a batch that activates knot $c$ (the knot-occupancy histogram). KURC adds an entropy bonus that encourages samples to spread across spline pieces:

$$L_{\text{KURC}} = L_{\text{InfoNCE}} - \lambda_{\text{occ}} \, H(q), \quad H(q) = -\sum_c q_c \log q_c$$

### 5.3 Why KURC exists as a separate method

KURC is the **safe fallback** and also a **precondition for KPCL**.

- **As a precondition:** if knots are well spread, $S(x)$ is informative, so KURC can be switched on underneath KPCL.

- **As a fallback:** even if the knot pattern carries no false-negative signal (KPCL's H1 gate fails), preventing spline collapse is independently valuable for representation quality, giving a publishable result on representation geometry (alignment/uniformity) rather than FN detection.

---

## 6. The Full Architecture, End to End

### 6.1 Component stack

1. **Input.** Tabular multi-label vector $x \in \mathbb{R}^d$ (standardized, scaler fit on train only). Datasets: yeast, scene, emotions, mediamill (high/mid false-negative prevalence); bibtex as a low-prevalence stress test.

2. **Augmentations.** Two views via feature masking, Gaussian noise, and mixup. Tabular analogues of SimCLR's image augmentations.

3. **KAN encoder.** $L = 2$ cubic-B-spline KAN layers, hidden width 128, grid $G = 5$, residual SiLU base path kept ($w_b \neq 0$), weight decay $\geq 10^{-4}$. Maps $x \mapsto h \in \mathbb{R}^{128}$. Depth kept shallow because KAN Jacobian rank decays with depth.

4. **KAN projection head.** 1 cubic-B-spline KAN layer, output 64. This is where $S(x)$ is read. Discarded after pre-training.

5. **Embedding z.** $\ell_2$-normalized head output (norm clamped at $10^{-12}$ for stability).

6. **Loss.** KPCL (Eq. 4) or KURC (Eq. 5), optionally combined (KURC underneath KPCL).

7. **Evaluation.** Freeze encoder, train a linear probe, report macro-AUROC, mAP, alignment, uniformity, effective rank.

### 6.2 Why these specific hyperparameters

- **Grid $G = 5$, cubic ($p = 3$).** Matches the borrowed Khan-GCL architecture; gives $G + p = 8$ basis functions per edge — enough knots for a rich pattern code, few enough to keep $S(x)$ low-dimensional.

- **Shallow depth ($L = 2$ encoder, 1 head).** Directly mitigates the rank-collapse pathology; deeper KANs lose Jacobian rank exponentially.

- **Residual base path $w_b \neq 0$.** Keeps a stable gradient highway and also breaks the exact MLP-equivalence (which requires $w_b = 0$), marginally strengthening the KAN moat.

- **Weight decay $\geq 10^{-4}$.** Necessary for KANs to train at all given the partition-of-unity nullspace.

- **Batch 256–512.** Sufficient for InfoNCE/DCL-class methods on a single 16 GB GPU; small batch is not the bottleneck here.

### 6.3 Training algorithm

**Algorithm 1 — KPCL pre-training step (KURC term optional)**

```
1:  sample minibatch {xⱼ}; build two views x′ⱼ, x″ⱼ
2:  h′ⱼ, h″ⱼ  ←  Encoder(x′ⱼ), Encoder(x″ⱼ)
3:  z′ⱼ, z″ⱼ  ←  ℓ₂(Head(h′ⱼ)), ℓ₂(Head(h″ⱼ))
4:  S′ⱼ, S″ⱼ  ←  active-knot codes from the head forward pass   ▷ bucketize, detached
5:  wᵢₖ        ←  Jaccard(Sᵢ, Sₖ) for all pairs              ▷ deterministic, no grad
6:  L          ←  L_KPCL(z, w) [− λ_occ H(q) if KURC]
7:  backprop through z only; w and S carry no gradient
8:  update encoder + head
```

> **Remark 2 (Gradient hygiene).** $S(x)$ and $w_{ik}$ are detached. They steer which negatives the loss trusts but are never themselves optimized — exactly the property that prevents the scorer-collapse failure mode. The only gradient path is through **z**.

---

## 7. The Decisive Baseline: Giving the MLP Its Best Shot

The claim "MLP cannot do this" must be demonstrated, not asserted. The decisive control is an MLP encoder/head of matched parameter count (±15%) trained with the same FN-cancellation loss, but where the FN weight is computed from the best signal an MLP can produce: $w_{ik}^{\text{MLP}} = \mathbb{1}[\cos(z_i, z_k) > \theta]$ or a soft cosine weight.

> **What the comparison proves**
>
> If KPCL (knot-Jaccard weight, KAN) beats MLP+cosine-weight at matched parameters and matched loss, then the only difference is the source of the FN signal — the discrete knot pattern. That is a clean, attributable demonstration that the KAN-internal structure provides usable information the embedding space does not. This is the experiment that the previous design lacked.

### 7.1 Full baseline suite (all at parameter parity, ≥ 5 seeds)

- **Loss axis** (encoder fixed = KAN): InfoNCE, SupCon (share-≥1-label positives), DCL, hard-negative debiasing, MulSupCon, GloFND.
- **Architecture axis** (loss fixed): MLP encoder + each loss above; MLP + cosine-threshold FN weight (the decisive control).
- **Mechanism ablations** (KAN + KPCL): remove FN weight ($\gamma = 0$); random FN weight; cosine FN weight instead of knot-Jaccard; discrete vs. smoothed weight.
- **Sanity references**: supervised KAN classifier; binary-relevance baseline.

---

## 8. Falsifiable Hypotheses and Pre-Registered Gates

**H0 (premise).** A KAN encoder matches or beats a parameter-matched MLP encoder under plain InfoNCE on ≥ 3/5 datasets. If H0 fails, the KAN premise is broken — stop.

**H1 (KPCL signal gate).** On yeast and mediamill, the knot-Jaccard weight ranks false negatives with AUC ≥ 0.55 absolute and ≥ AUC$_{\cos(z)}$ + 0.05, on ≥ 1/2 datasets, ≥ 5 seeds, paired bootstrap $p < 0.05$. If H1 fails, KPCL is falsified; fall back to KURC.

**H2 (KPCL headline).** KPCL beats InfoNCE by ≥ 1.5 macro-AUROC and DCL by ≥ 0.5, averaged over yeast/scene/emotions/mediamill, ≥ 3/4 individually significant.

**H3 (KURC fallback).** KURC matches or beats InfoNCE on AUROC and improves uniformity by ≥ 0.1 nats — the publishable result even if H1/H2 fail.

> **The honest status**
>
> H1 is untested. The premise that discrete knot patterns carry false-negative signal is plausible but unproven. KURC exists precisely so that a failed H1 still yields a defensible representation-geometry contribution rather than a dead thesis.

---

## 9. Staged Implementation Plan

1. **Data & baselines.** Tabular loaders, train-only scaling, row-level disjoint splits, per-label prevalence report. Implement InfoNCE/SupCon/DCL on MLP and KAN. Gate: H0.

2. **Knot extraction.** ~30 lines: bucketize head inputs to grid intervals, build $S(x)$, compute pairwise Jaccard. Unit-test on a synthetic spline with known active intervals.

3. **KPCL loss.** Plug $w_{ik}$ into the InfoNCE denominator; verify $\gamma = 0$ recovers InfoNCE exactly; mask tests (positive-only, negative-only, injected false negative). Gate: H1.

4. **Full sweep.** 4 datasets × baseline suite × 5 seeds. Gate: H2 (KPCL) or H3 (KURC).

5. **Write-up.** Lead with representation geometry and multi-label AUROC; FN detection as the mechanism, not the headline.

---

## 10. Summary: The Idea in One Page

We build a KAN encoder and head on cubic B-splines. Every input activates exactly four basis functions per edge; which four is a discrete, per-sample knot pattern $S(x)$ that no MLP produces. KPCL reads $S(x)$ from the projection head, measures pattern overlap with Jaccard similarity $w_{ik}$, and uses $(1 - w_{ik})^\gamma$ to down-weight likely false negatives in the InfoNCE denominator — a deterministic, non-learned correction, which is precisely why an MLP cannot launder away the advantage as it did before. KURC prevents the spline collapse that would empty $S(x)$ of information by rewarding knot-occupancy entropy, and doubles as the safe fallback if the knot pattern turns out to carry no false-negative signal. The decisive experiment gives a parameter-matched MLP its best shot at the same mechanism (cosine-based FN weight); if the KAN knot pattern wins, the gain is attributable to KAN-internal structure and nothing else.

Every choice — discreteness, determinism, head placement, shallow depth, residual base, weight decay — is a direct response to a documented failure of the previous approach.

**KPCL is the high-reward hypothesis; KURC is the floor that keeps the thesis alive.**

---

## Key References (arXiv IDs)

| Paper | arXiv ID |
|---|---|
| Khan-GCL | 2505.15103 |
| Contrastive-KAN | 2507.10808 |
| Beyond Linear Bottlenecks | 2507.23436 |
| KAN expressiveness | 2410.01803 |
| KAN (original) | 2404.19756 |
| KAN nullspace | 2505.18131 |
| AC-PKAN | 2505.08687 |
| Spline locality | 2602.02056 |
| DCL | 2007.00224 / 2110.06848 |
| FNC | 2011.11765 |
| Hard-negative debiasing | 2010.04592 |
| SogCLR | 2202.12387 |
| GloFND | 2502.20612 |
| MulSupCon | AAAI 2024 |
