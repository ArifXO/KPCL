# H1 Verdict - knot-Jaccard vs cos(z) FN-ranking (KPCL go/no-go)

Seeds: [42, 1337, 2024, 7, 9001]. Per-dataset mean+/-std over 5 seeds; paired bootstrap (knot vs cos, resampling seeds).

| dataset | AUC_cos | AUC_knot | AUC_smooth | knot-cos | boot p | >=0.55 | +0.05 | p<0.05 | PASS |
|---|---|---|---|---|---|---|---|---|---|
| yeast | 0.5305+/-0.0035 | 0.5340+/-0.0052 | 0.5340 | +0.0035 | 0.1375 | N | N | N | no |
| mediamill | 0.5132+/-0.0052 | 0.6419+/-0.0015 | 0.6419 | +0.1286 | 0.0000 | Y | Y | Y | **YES** |

## VERDICT: H1 PASS (need all 3 conditions on >=1 dataset)

KPCL signal CONFIRMED - the knot-Jaccard weight ranks false negatives above disjoint pairs, beyond cos(z), significantly. Proceed to build KPCL (Stage 7).