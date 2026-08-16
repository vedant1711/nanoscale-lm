"""Hypothesis property tests for the optimizers (spec D2)."""

from __future__ import annotations

import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nanoscale.optim import (
    AdamW,
    Muon,
    cautious_decay_mask,
    muon_update_scale,
    newton_schulz_orthogonalize,
)

_SETTINGS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@_SETTINGS
@given(
    rows=st.integers(min_value=1, max_value=24),
    cols=st.integers(min_value=1, max_value=24),
    steps=st.integers(min_value=1, max_value=8),
)
def test_newton_schulz_output_is_always_finite_and_shape_preserving(
    rows: int, cols: int, steps: int
) -> None:
    m = torch.randn(rows, cols, dtype=torch.float64)
    out = newton_schulz_orthogonalize(m, steps=steps, compute_dtype=torch.float64)
    assert out.shape == m.shape
    assert torch.isfinite(out).all()


@_SETTINGS
@given(
    rows=st.integers(min_value=2, max_value=20),
    cols=st.integers(min_value=2, max_value=20),
)
def test_newton_schulz_always_bounds_the_spectrum(rows: int, cols: int) -> None:
    """Whatever goes in, no singular value comes out above ~1.4."""
    m = torch.randn(rows, cols, dtype=torch.float64) * 1000.0
    out = newton_schulz_orthogonalize(m, steps=5, compute_dtype=torch.float64)
    assert float(torch.linalg.svdvals(out).max()) < 1.5


@_SETTINGS
@given(scale=st.floats(min_value=1e-4, max_value=1e4, allow_nan=False))
def test_newton_schulz_is_scale_invariant_in_direction(scale: float) -> None:
    """Orthogonalization discards magnitude: ``cA`` and ``A`` give the same direction.

    Invariance is checked as a cosine similarity rather than elementwise equality,
    because it is only *approximate*: the ``eps = 1e-7`` floor on the spectral
    normalisation denominator perturbs the result by ``O(eps / ‖A‖)``, which is ~1e-4
    relative at the small end of this scale range. That deviation is a deliberate
    numerical guard, and the direction is what the optimizer actually consumes.
    """
    torch.manual_seed(0)
    m = torch.randn(8, 8, dtype=torch.float64)
    a = newton_schulz_orthogonalize(m, steps=5, compute_dtype=torch.float64)
    b = newton_schulz_orthogonalize(m * scale, steps=5, compute_dtype=torch.float64)
    cosine = float((a * b).sum() / (a.norm() * b.norm()))
    assert cosine > 1.0 - 1e-6, f"direction changed with scale {scale:g}: cos={cosine:.9f}"


@_SETTINGS
@given(
    rows=st.integers(min_value=1, max_value=512),
    cols=st.integers(min_value=1, max_value=512),
)
def test_update_scale_is_always_positive_and_finite(rows: int, cols: int) -> None:
    for mode in ("shape", "rms"):
        value = muon_update_scale(rows, cols, mode)
        assert value > 0.0
        assert value == value  # not NaN


@_SETTINGS
@given(n=st.integers(min_value=1, max_value=64))
def test_cautious_mask_is_always_binary_and_shape_preserving(n: int) -> None:
    p = torch.randn(n)
    u = torch.randn(n)
    mask = cautious_decay_mask(p, u)
    assert mask.shape == p.shape
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


@_SETTINGS
@given(
    lr=st.floats(min_value=1e-5, max_value=1e-1),
    beta1=st.floats(min_value=0.0, max_value=0.95),
    beta2=st.floats(min_value=0.5, max_value=0.999),
    wd=st.floats(min_value=0.0, max_value=0.5),
)
def test_adamw_always_matches_torch(lr: float, beta1: float, beta2: float, wd: float) -> None:
    torch.manual_seed(0)
    ours = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    theirs = ours.detach().clone().requires_grad_(True)
    opt_ours = AdamW([ours], lr=lr, betas=(beta1, beta2), weight_decay=wd)
    opt_theirs = torch.optim.AdamW([theirs], lr=lr, betas=(beta1, beta2), weight_decay=wd)
    for _ in range(20):
        g = torch.randn(6, 3, dtype=torch.float64)
        for p, opt in ((ours, opt_ours), (theirs, opt_theirs)):
            opt.zero_grad()
            p.grad = g.clone()
            opt.step()
    torch.testing.assert_close(ours, theirs, atol=1e-12, rtol=1e-12)


@_SETTINGS
@given(
    rows=st.integers(min_value=1, max_value=16),
    cols=st.integers(min_value=1, max_value=16),
    momentum=st.floats(min_value=0.0, max_value=0.99),
)
def test_muon_never_produces_nans(rows: int, cols: int, momentum: float) -> None:
    p = torch.randn(rows, cols, requires_grad=True)
    opt = Muon([p], lr=0.05, momentum=momentum)
    for _ in range(5):
        opt.zero_grad()
        p.grad = torch.randn_like(p)
        opt.step()
        assert torch.isfinite(p).all()
