"""
Stage 2 KAN building-block tests.

Covers: (a) forward shapes, (b) knot_indices = exactly 4 active indices per edge
per sample matching a hand-computed bucketize AND the true nonzero spline columns,
(c) gradient flows to coefficients (and NOT to the detached knot code),
(d) KAN vs MLP parameter parity <= 15%, (e) MLP knot_indices raises.
"""
from __future__ import annotations

import pytest
import torch

from src.models.encoder import KANEncoder, MLPEncoder
from src.models.heads import KANHead, MLPHead, l2_normalize
from src.models.spline_kan import KANLayer
from src.models.parity import parity_for_input_dim

DATASET_DIMS = {"yeast": 103, "scene": 294, "emotions": 72, "mediamill": 120, "bibtex": 1836}


# ---------------------------------------------------------------------------
# (a) forward shapes
# ---------------------------------------------------------------------------

def test_kan_layer_forward_shape():
    layer = KANLayer(10, 7, grid_size=5, spline_order=3)
    out = layer(torch.randn(8, 10))
    assert out.shape == (8, 7)


def test_kan_encoder_head_forward_shape():
    enc = KANEncoder(input_dim=103, hidden_dim=128, n_layers=2)
    head = KANHead(in_dim=128, out_dim=64)
    z = head(enc(torch.randn(16, 103)))
    assert z.shape == (16, 64)
    zn = l2_normalize(z)
    assert torch.allclose(zn.norm(dim=-1), torch.ones(16), atol=1e-5)


def test_mlp_twin_forward_shape():
    enc = MLPEncoder(input_dim=103, hidden_dim=256, n_layers=2)
    head = MLPHead(in_dim=256, out_dim=64)
    z = head(enc(torch.randn(16, 103)))
    assert z.shape == (16, 64)


# ---------------------------------------------------------------------------
# (b) knot_indices: exactly 4, hand-computed bucketize, true nonzero columns
# ---------------------------------------------------------------------------

def test_knot_indices_hand_computed_bucketize():
    """On a synthetic grid (-1,1), G=5, p=3, interval midpoints have known codes."""
    layer = KANLayer(1, 4, grid_size=5, spline_order=3, grid_range=(-1.0, 1.0))
    # midpoints of the 5 in-range intervals
    h = torch.tensor([[-0.8], [-0.4], [0.0], [0.4], [0.8]])  # (5, 1)
    idx = layer.knot_indices(h)  # (5, 1, 4)
    assert idx.shape == (5, 1, 4)
    expected = torch.tensor([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
    ])  # hand-computed: c=[3,4,5,6,7], indices [c-3..c]
    assert torch.equal(idx.squeeze(1), expected)


def test_knot_indices_contain_nonzero_spline_columns():
    """The actually-nonzero B-spline columns are a subset of the 4-index cell readout.

    Equality holds strictly-interior to a cell (covered by the midpoint test);
    on a knot boundary the cell's edge basis is exactly 0 there, so nonzero is a
    strict subset. The readout always returns exactly p+1=4 cell indices.
    """
    torch.manual_seed(0)
    layer = KANLayer(3, 5, grid_size=5, spline_order=3, grid_range=(-1.0, 1.0))
    h = torch.empty(20, 3).uniform_(-0.95, 0.95)  # strictly in-range
    idx = layer.knot_indices(h)              # (20, 3, 4)
    bspl = layer.b_splines(h)                # (20, 3, 8)
    assert idx.shape[-1] == 4
    nz = bspl > 1e-9
    for b in range(h.shape[0]):
        for d in range(h.shape[1]):
            active = set(torch.nonzero(nz[b, d]).flatten().tolist())
            returned = set(idx[b, d].tolist())
            assert active.issubset(returned), (b, d, returned, active)
            assert len(returned) == 4


def test_knot_indices_always_four_and_in_range_even_out_of_grid():
    layer = KANLayer(4, 5, grid_size=5, spline_order=3, grid_range=(-1.0, 1.0))
    h = torch.empty(32, 4).uniform_(-5.0, 5.0)  # includes out-of-grid -> clamp
    idx = layer.knot_indices(h)
    assert idx.shape[-1] == 4
    # consecutive integers
    assert torch.all(idx[..., 1:] - idx[..., :-1] == 1)
    # valid basis range [0, G+p-1] = [0, 7]
    assert int(idx.min()) >= 0 and int(idx.max()) <= 7


def test_knot_indices_detached_long():
    layer = KANLayer(4, 5)
    idx = layer.knot_indices(torch.randn(6, 4))
    assert idx.dtype == torch.long
    assert not idx.requires_grad


# ---------------------------------------------------------------------------
# (c) gradient flows to coefficients; NOT to the knot code
# ---------------------------------------------------------------------------

def test_gradient_flows_to_coefficients():
    layer = KANLayer(4, 5, grid_size=5, spline_order=3)
    layer(torch.randn(8, 4)).pow(2).sum().backward()
    assert layer.coefficients.grad is not None
    assert float(layer.coefficients.grad.abs().sum()) > 0.0


def test_knot_code_has_no_gradient_path():
    """Gradient isolation (CLAUDE.md): S(x) is detached, never carries grad."""
    layer = KANLayer(4, 5)
    x = torch.randn(8, 4, requires_grad=True)
    idx = layer.knot_indices(x)
    assert not idx.requires_grad
    assert not idx.is_floating_point()  # discrete -> no differentiable path


# ---------------------------------------------------------------------------
# (d) KAN vs MLP parameter parity <= 15%
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,dim", DATASET_DIMS.items())
def test_param_parity_within_15pct(name, dim):
    r = parity_for_input_dim(dim, kan_hidden=128, out_dim=64, n_enc_layers=2,
                             grid_size=5, spline_order=3)
    assert r["within_tol"], f"{name}: ratio={r['ratio']:.2%} KAN={r['kan_total']} MLP={r['mlp_total']}"


def test_parity_counts_match_real_modules():
    """Closed-form counts must equal actual constructed-module param counts."""
    dim = 103
    r = parity_for_input_dim(dim, kan_hidden=128, out_dim=64, n_enc_layers=2)
    kan = KANEncoder(dim, 128, 2).param_count() + KANHead(128, 64).param_count()
    mlp = (MLPEncoder(dim, r["mlp_hidden"], 2).param_count()
           + MLPHead(r["mlp_hidden"], 64).param_count())
    assert kan == r["kan_total"]
    assert mlp == r["mlp_total"]


# ---------------------------------------------------------------------------
# (e) MLP knot_indices raises NotImplementedError (R9)
# ---------------------------------------------------------------------------

def test_mlp_encoder_knot_indices_raises():
    enc = MLPEncoder(input_dim=10, hidden_dim=32, n_layers=2)
    with pytest.raises(NotImplementedError, match="no spline grid"):
        enc.knot_indices(torch.randn(4, 10))


def test_mlp_head_knot_indices_raises():
    head = MLPHead(in_dim=32, out_dim=64)
    with pytest.raises(NotImplementedError, match="no spline grid"):
        head.knot_indices(torch.randn(4, 32))
