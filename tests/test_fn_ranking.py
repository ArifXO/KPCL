"""
Stage 6 tests: correctness of the FN-ranking AUC machinery (the H1 gate depends on it).

Synthetic, deterministic checks of the label-pair masks, the pair AUC, and the paired
bootstrap — independent of any trained model.
"""
from __future__ import annotations

import numpy as np
import torch

from src.metrics.fn_ranking import (_label_pair_masks, _pair_auc,
                                     paired_bootstrap_p)


def test_label_masks_positive_and_negative():
    # rows 0,1 share both labels (J=1 -> positive); row 2 disjoint from 0,1 (negative)
    Y = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    pos, neg = _label_pair_masks(Y)
    assert bool(pos[0, 1]) and not bool(pos[0, 2])
    assert bool(neg[0, 2]) and bool(neg[1, 2]) and not bool(neg[0, 1])
    # upper-triangle only (no self, no double count)
    assert not bool(pos[1, 0]) and pos.sum() == 1


def test_half_jaccard_is_positive_boundary():
    # J = 1/3 (< 0.5) -> neither positive nor negative (ambiguous middle)
    Y = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])  # inter=1, union=3
    pos, neg = _label_pair_masks(Y)
    assert pos.sum() == 0 and neg.sum() == 0


def test_pair_auc_perfect_and_inverted():
    Y = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # (0,1) pos, (0,2)&(1,2) neg
    pos, neg = _label_pair_masks(Y)
    # score that ranks the positive pair highest -> AUC = 1
    perfect = torch.tensor([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])
    assert _pair_auc(perfect, pos, neg) == 1.0
    # inverted score -> AUC = 0
    assert _pair_auc(-perfect, pos, neg) == 0.0


def test_paired_bootstrap_all_positive_is_significant():
    knot = [0.62, 0.60, 0.63, 0.61, 0.64]
    cos = [0.55, 0.54, 0.56, 0.55, 0.57]
    p = paired_bootstrap_p(knot, cos, n_boot=5000, seed=0)
    assert p < 0.05            # every seed knot>cos -> strongly significant


def test_paired_bootstrap_no_difference_not_significant():
    a = [0.60, 0.58, 0.61, 0.59, 0.60]
    p = paired_bootstrap_p(a, a, n_boot=5000, seed=0)
    assert p >= 0.05           # identical -> not significant (p ~ 1)
