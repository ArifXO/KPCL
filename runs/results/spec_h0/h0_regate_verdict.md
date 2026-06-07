# H0 RE-GATE - grid-range-fixed KAN (use_layer_norm) vs MLP and MLP+LN

Seeds: [42, 1337, 2024]. Metric: macro-AUROC (mean+/-std), linear probe. Param-matched; LayerNorm is parameter-free.

| dataset | KAN+LN | MLP | MLP+LN | KAN-MLP | KAN-(MLP+LN) |
|---|---|---|---|---|---|
| yeast | 0.6956+/-0.0053 | 0.7242 | 0.7157 | -0.0285 | -0.0201 |
| scene | 0.9413+/-0.0004 | 0.9402 | 0.9469 | +0.0012 | -0.0056 |
| emotions | 0.8316+/-0.0124 | 0.8275 | 0.8198 | +0.0041 | +0.0118 |
| mediamill | 0.7156+/-0.0072 | 0.5973 | 0.6886 | +0.1183 | +0.0270 |
| bibtex | 0.8722+/-0.0043 | 0.7839 | 0.8226 | +0.0883 | +0.0495 |

**(A) KAN >= MLP on 4/5** -> H0(A) PASS
**(B) KAN >= MLP+LN on 3/5** -> H0(B) PASS

## Interpretation

- (A) PASS means the fixed KAN matches/beats a plain MLP.
- (B) PASS means the KAN's edge survives giving the MLP the SAME normalisation -- evidence the advantage is KAN-structural, not just LayerNorm.

Reported as-is, not massaged. The recorded original H0 (no-norm KAN) stands separately in h0_verdict.md.