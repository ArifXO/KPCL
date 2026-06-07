# H0 Verdict - KAN-InfoNCE vs MLP-InfoNCE (param-matched)

Seeds: [42, 1337, 2024]. Metric: macro-AUROC (mean+/-std over seeds), linear probe.

| dataset | KAN AUROC | MLP AUROC | delta (KAN-MLP) | KAN>=MLP |
|---|---|---|---|---|
| yeast | 0.6813+/-0.0093 | 0.7242+/-0.0067 | -0.0429 | no |
| scene | 0.9338+/-0.0038 | 0.9402+/-0.0012 | -0.0064 | no |
| emotions | 0.8253+/-0.0075 | 0.8275+/-0.0087 | -0.0022 | no |
| mediamill | 0.7095+/-0.0104 | 0.5973+/-0.0081 | +0.1122 | yes |
| bibtex | 0.8811+/-0.0023 | 0.7839+/-0.0044 | +0.0972 | yes |

**KAN >= MLP on 2/5 datasets.**

## VERDICT: H0 FAIL (threshold >=3/5)

Premise BROKEN - the KAN does not match the param-matched MLP under plain InfoNCE on >=3/5 datasets. STOP: do not build KPCL. Reconsider the KAN premise (grid range, depth, augmentation strength, or training length) before any further build. Results reported as-is, not massaged.