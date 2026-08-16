"""Optimizer correctness (spec D1 / Phase 3 acceptance).

Four claims, each tested rather than asserted:

1. Our AdamW equals ``torch.optim.AdamW`` to floating-point tolerance over many steps.
2. Newton–Schulz output is (near-)orthogonal: singular values cluster at 1.
3. Muon beats AdamW on a matrix-structured problem — the regime it exists for.
4. The parameter router assigns every tensor to the group the spec says it should.
"""

from __future__ import annotations

import io
import math
from typing import Any

import pytest
import torch
from torch import nn

from nanoscale.config import ModelConfig, OptimConfig
from nanoscale.model import NanoScaleLM
from nanoscale.optim import (
    NS_COEFFS,
    AdamW,
    CompositeOptimizer,
    Muon,
    build_optimizer,
    cautious_decay_mask,
    cautious_mask_fraction,
    muon_update_scale,
    newton_schulz_orthogonalize,
    split_parameters,
)


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(20260816)


def backward(loss: torch.Tensor) -> None:
    """Call ``Tensor.backward``, which torch ships without type annotations."""
    loss.backward()  # type: ignore[no-untyped-call]


# =====================================================================================
# AdamW == torch.optim.AdamW
# =====================================================================================


def _adamw_parity(
    lr: float,
    betas: tuple[float, float],
    eps: float,
    wd: float,
    steps: int,
) -> float:
    """Run ours and torch's AdamW side by side; return the max absolute divergence."""
    torch.manual_seed(0)
    a = torch.randn(16, 8, dtype=torch.float64)
    b = torch.randn(16, 1, dtype=torch.float64)
    ours = torch.zeros(8, 1, dtype=torch.float64, requires_grad=True)
    theirs = torch.zeros(8, 1, dtype=torch.float64, requires_grad=True)
    opt_ours = AdamW([ours], lr=lr, betas=betas, eps=eps, weight_decay=wd)
    opt_theirs = torch.optim.AdamW([theirs], lr=lr, betas=betas, eps=eps, weight_decay=wd)
    for _ in range(steps):
        for param, opt in ((ours, opt_ours), (theirs, opt_theirs)):
            opt.zero_grad()
            backward(((a @ param - b) ** 2).mean())
            opt.step()
    return float((ours - theirs).abs().max().detach())


@pytest.mark.parametrize(
    ("lr", "betas", "eps", "wd", "steps"),
    [
        (1e-3, (0.9, 0.999), 1e-8, 0.0, 200),
        (1e-2, (0.9, 0.95), 1e-8, 0.1, 200),
        (3e-4, (0.8, 0.99), 1e-6, 0.01, 200),
        # Aggressive settings: exact, but see the Lyapunov note below for the horizon.
        (1e-1, (0.5, 0.5), 1e-10, 0.5, 60),
    ],
)
def test_adamw_matches_torch_on_a_toy_convex_problem(
    lr: float, betas: tuple[float, float], eps: float, wd: float, steps: int
) -> None:
    """Least squares in fp64: any formula difference shows up immediately."""
    assert _adamw_parity(lr, betas, eps, wd, steps) < 1e-12


def test_adamw_parity_divergence_is_chaos_not_a_formula_difference() -> None:
    """Distinguish "our formula is wrong" from "this trajectory is unstable".

    At lr=0.1 with betas=(0.5, 0.5) and wd=0.5, our AdamW and torch's agree to
    ~1e-17 after one step, but the gap grows roughly geometrically -- ~1e-15 by step
    20, ~1e-12 by step 100, ~1e-5 by step 200. That is the signature of a positive
    Lyapunov exponent amplifying last-bit rounding differences between two
    algebraically identical but differently-ordered arithmetic sequences, not of a
    wrong update rule: a genuine formula error would be visible at step one and would
    not start at machine epsilon.
    """
    settings = (1e-1, (0.5, 0.5), 1e-10, 0.5)
    after_1 = _adamw_parity(*settings, steps=1)
    after_20 = _adamw_parity(*settings, steps=20)
    after_200 = _adamw_parity(*settings, steps=200)

    assert after_1 < 1e-15, "a formula difference would be visible on the very first step"
    assert after_1 <= after_20 <= after_200, "divergence should grow, not oscillate"
    assert after_200 > after_20, "and it should keep growing -- this is the chaos signature"


def test_adamw_matches_torch_with_multiple_tensors_and_shapes() -> None:
    torch.manual_seed(1)
    shapes = [(4,), (3, 5), (2, 3, 4)]
    ours = [torch.randn(s, dtype=torch.float64, requires_grad=True) for s in shapes]
    theirs = [p.detach().clone().requires_grad_(True) for p in ours]

    opt_ours = AdamW(ours, lr=5e-3, weight_decay=0.02)
    opt_theirs = torch.optim.AdamW(theirs, lr=5e-3, weight_decay=0.02)

    for step in range(50):
        for params, opt in ((ours, opt_ours), (theirs, opt_theirs)):
            opt.zero_grad()
            loss = torch.stack([((p * (step + 1)) ** 2).sum() for p in params]).sum()
            backward(loss)
            opt.step()

    for a, b in zip(ours, theirs, strict=True):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def test_adamw_decoupled_decay_is_not_scaled_by_the_second_moment() -> None:
    """The 'W' in AdamW: decay must be applied to the parameter, not the gradient."""
    p = torch.tensor([[1.0]], dtype=torch.float64, requires_grad=True)
    opt = AdamW([p], lr=0.1, weight_decay=0.5, betas=(0.0, 0.0), eps=1e-30)
    p.grad = torch.zeros_like(p)  # zero gradient: only decay should move the parameter
    opt.step()
    torch.testing.assert_close(p, torch.tensor([[1.0 - 0.1 * 0.5]], dtype=torch.float64))


def test_adamw_rejects_bad_hyperparameters() -> None:
    p = torch.zeros(2, requires_grad=True)
    with pytest.raises(ValueError, match="lr must be positive"):
        AdamW([p], lr=0.0)
    with pytest.raises(ValueError, match="betas"):
        AdamW([p], betas=(1.0, 0.9))
    with pytest.raises(ValueError, match="eps must be non-negative"):
        AdamW([p], eps=-1e-8)
    with pytest.raises(ValueError, match="weight_decay"):
        AdamW([p], weight_decay=-1.0)


def test_adamw_skips_parameters_without_gradients() -> None:
    a = torch.ones(2, requires_grad=True)
    b = torch.ones(2, requires_grad=True)
    opt = AdamW([a, b], lr=0.1)
    a.grad = torch.ones_like(a)
    opt.step()
    assert not torch.allclose(a, torch.ones(2))
    torch.testing.assert_close(b, torch.ones(2))


def test_adamw_rejects_sparse_gradients() -> None:
    p = torch.zeros(4, requires_grad=True)
    opt = AdamW([p], lr=0.1)
    p.grad = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.0]), (4,))
    with pytest.raises(RuntimeError, match="sparse"):
        opt.step()


def _serialize(state: dict[str, Any]) -> dict[str, Any]:
    """Round-trip through torch.save/load, as a real checkpoint would.

    ``Optimizer.load_state_dict`` does not deep-copy state tensors, so restoring
    directly from a live optimizer's ``state_dict()`` leaves the two sharing momentum
    buffers. Serialising is both the realistic path and the one that actually tests
    restoration rather than aliasing.
    """
    buf = io.BytesIO()
    torch.save(state, buf)
    buf.seek(0)
    loaded: dict[str, Any] = torch.load(buf, weights_only=False)
    return loaded


def test_adamw_state_dict_round_trip() -> None:
    p = torch.randn(3, 3, requires_grad=True)
    opt = AdamW([p], lr=0.1)
    p.grad = torch.randn_like(p)
    opt.step()
    state = _serialize(opt.state_dict())

    q = p.detach().clone().requires_grad_(True)
    opt2 = AdamW([q], lr=0.1)
    opt2.load_state_dict(state)
    grad = torch.randn_like(p)
    p.grad, q.grad = grad, grad.clone()
    opt.step()
    opt2.step()
    torch.testing.assert_close(p, q)


# =====================================================================================
# Newton–Schulz orthogonalization
# =====================================================================================


@pytest.mark.parametrize("shape", [(8, 8), (16, 4), (4, 16), (64, 32), (1, 8), (8, 1)])
def test_newton_schulz_output_has_singular_values_near_one(shape: tuple[int, int]) -> None:
    """The defining property: every singular value is pushed toward 1."""
    m = torch.randn(*shape, dtype=torch.float64)
    out = newton_schulz_orthogonalize(m, steps=5, compute_dtype=torch.float64)
    sigma = torch.linalg.svdvals(out)
    assert sigma.min() > 0.6, f"smallest singular value {float(sigma.min()):.3f} too small"
    assert sigma.max() < 1.4, f"largest singular value {float(sigma.max()):.3f} too large"


def test_newton_schulz_flattens_a_badly_conditioned_spectrum() -> None:
    """The point of the algorithm: kill anisotropy in the momentum matrix."""
    u, _ = torch.linalg.qr(torch.randn(32, 32, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(32, 32, dtype=torch.float64))
    sigma = torch.logspace(0, -4, 32, dtype=torch.float64)  # condition number 1e4
    m = u @ torch.diag(sigma) @ v.T

    before = torch.linalg.svdvals(m)
    after = torch.linalg.svdvals(
        newton_schulz_orthogonalize(m, steps=8, compute_dtype=torch.float64)
    )
    assert float(before.max() / before.min()) > 1e3
    assert float(after.max() / after.min()) < 3.0


def test_newton_schulz_preserves_the_singular_vector_subspaces() -> None:
    """It should produce UV^T: the same subspaces, with the spectrum flattened."""
    m = torch.randn(24, 16, dtype=torch.float64)
    u, _, vh = torch.linalg.svd(m, full_matrices=False)
    target = u @ vh  # the exact orthogonal polar factor

    approx = newton_schulz_orthogonalize(m, steps=5, compute_dtype=torch.float64)
    similarity = float((approx * target).sum() / (approx.norm() * target.norm()))
    assert similarity > 0.97, f"cosine similarity to UV^T is only {similarity:.4f}"

    # The tuned quintic does not converge to exactly UV^T -- it lands near it and stays
    # there. What matters is that it is far closer than the un-orthogonalized input.
    raw = float((m * target).sum() / (m.norm() * target.norm()))
    assert similarity > raw


def test_more_iterations_get_closer_to_orthogonal() -> None:
    m = torch.randn(16, 16, dtype=torch.float64)

    def spread(steps: int) -> float:
        sigma = torch.linalg.svdvals(
            newton_schulz_orthogonalize(m, steps=steps, compute_dtype=torch.float64)
        )
        return float((sigma - 1.0).abs().max())

    assert spread(5) < spread(1)


def test_newton_schulz_is_orientation_invariant() -> None:
    """Transposing the input must transpose the output; the internal flip is bookkeeping."""
    m = torch.randn(32, 8, dtype=torch.float64)
    a = newton_schulz_orthogonalize(m, steps=5, compute_dtype=torch.float64)
    b = newton_schulz_orthogonalize(m.T, steps=5, compute_dtype=torch.float64)
    torch.testing.assert_close(a, b.T, atol=1e-10, rtol=1e-10)


def test_newton_schulz_handles_a_zero_matrix() -> None:
    out = newton_schulz_orthogonalize(torch.zeros(4, 6), steps=5)
    assert torch.isfinite(out).all()


def test_newton_schulz_coefficients_are_the_speedrun_values() -> None:
    assert NS_COEFFS == (3.4445, -4.7750, 2.0315)


def test_newton_schulz_validates_input() -> None:
    with pytest.raises(ValueError, match="2D matrix"):
        newton_schulz_orthogonalize(torch.randn(4))
    with pytest.raises(ValueError, match="non-negative"):
        newton_schulz_orthogonalize(torch.randn(4, 4), steps=-1)


def test_zero_steps_is_just_spectral_normalisation() -> None:
    m = torch.randn(6, 6, dtype=torch.float64)
    out = newton_schulz_orthogonalize(m, steps=0, compute_dtype=torch.float64)
    # The 1e-7 floor on the normalisation denominator sets the achievable tolerance.
    torch.testing.assert_close(out, m / m.norm(), atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize(("rows", "cols"), [(8, 8), (32, 8), (8, 32)])
def test_update_scale_modes(rows: int, cols: int) -> None:
    assert muon_update_scale(rows, cols, "shape") == math.sqrt(max(1.0, rows / cols))
    assert muon_update_scale(rows, cols, "rms") == 0.2 * math.sqrt(max(rows, cols))
    with pytest.raises(ValueError, match="Unknown scale mode"):
        muon_update_scale(rows, cols, "nope")  # type: ignore[arg-type]


# =====================================================================================
# Muon behaviour
# =====================================================================================


def test_muon_rejects_non_2d_parameters() -> None:
    with pytest.raises(ValueError, match="only accepts 2D parameters"):
        Muon([torch.zeros(4, requires_grad=True)])


def test_muon_rejects_bad_hyperparameters() -> None:
    p = torch.zeros(2, 2, requires_grad=True)
    with pytest.raises(ValueError, match="lr must be positive"):
        Muon([p], lr=0.0)
    with pytest.raises(ValueError, match="momentum"):
        Muon([p], momentum=1.0)
    with pytest.raises(ValueError, match="ns_steps"):
        Muon([p], ns_steps=0)


def _deep_linear_run(
    kind: str, lr: float, *, steps: int = 200, dim: int = 48, depth: int = 4, bs: int = 64
) -> float:
    """Train a deep linear network on stochastic minibatches; return the final loss."""
    torch.manual_seed(11)
    layers = [nn.Linear(dim, dim, bias=False) for _ in range(depth)]
    net = nn.Sequential(*layers)
    for layer in layers:
        nn.init.normal_(layer.weight, std=0.5 / dim**0.5)
    torch.manual_seed(5)
    target = torch.randn(dim, dim) / dim**0.5
    params = list(net.parameters())
    opt: torch.optim.Optimizer = (
        Muon(params, lr=lr, momentum=0.95)
        if kind == "muon"
        else AdamW(params, lr=lr, betas=(0.9, 0.95))
    )
    gen = torch.Generator().manual_seed(99)
    loss = torch.zeros(())
    for _ in range(steps):
        x = torch.randn(bs, dim, generator=gen)
        y = x @ target.T
        opt.zero_grad()
        loss = ((net(x) - y) ** 2).mean()
        backward(loss)
        opt.step()
    return float(loss.detach())


def test_muon_beats_adamw_on_a_matrix_structured_problem() -> None:
    """A deep linear network trained on stochastic minibatches, each at its best LR.

    Choosing this problem rather than a single-matrix least squares is deliberate, and
    the negative result is worth recording: on a **convex, single-matrix, full-batch**
    problem AdamW wins comfortably at every learning rate we tried. That is not a bug.
    Adam's diagonal preconditioner is close to optimal on a well-conditioned quadratic,
    and there is no accumulation of a spiky momentum spectrum for orthogonalization to
    fix.

    Muon's advantage appears in the regime it was actually designed for: **stacked**
    weight matrices with **stochastic** gradients, where the momentum matrix becomes
    dominated by a few singular directions and per-coordinate scaling cannot undo that.
    """
    lrs = [0.001, 0.003, 0.01, 0.03, 0.1]
    muon_best = min(_deep_linear_run("muon", lr) for lr in lrs)
    adam_best = min(_deep_linear_run("adamw", lr) for lr in lrs)
    assert muon_best < adam_best, (
        f"muon {muon_best:.3e} did not beat adamw {adam_best:.3e} at their best LRs"
    )


def test_adamw_wins_on_a_convex_single_matrix_problem() -> None:
    """The honest converse, pinned so the claim above stays scoped.

    If a future change made Muon win here too, that would be surprising enough to be
    worth looking at rather than quietly accepting.
    """
    torch.manual_seed(3)
    dim = 32
    a = torch.randn(dim, 256)
    target = torch.randn(dim, dim) / dim**0.5
    b = target @ a

    def run(kind: str, lr: float, steps: int = 120) -> float:
        torch.manual_seed(7)
        w = torch.zeros(dim, dim, requires_grad=True)
        opt: torch.optim.Optimizer = (
            Muon([w], lr=lr, momentum=0.95)
            if kind == "muon"
            else AdamW([w], lr=lr, betas=(0.9, 0.95))
        )
        for _ in range(steps):
            opt.zero_grad()
            backward(((w @ a - b) ** 2).mean())
            opt.step()
        return float(((w @ a - b) ** 2).mean().detach())

    lrs = [0.003, 0.01, 0.03, 0.1]
    assert min(run("adamw", lr) for lr in lrs) < min(run("muon", lr) for lr in lrs)


def test_muon_reduces_a_simple_quadratic() -> None:
    w = torch.randn(8, 8, requires_grad=True)
    opt = Muon([w], lr=0.01)
    start = float((w**2).sum().detach())
    for _ in range(30):
        opt.zero_grad()
        backward((w**2).sum())
        opt.step()
    assert float((w**2).sum().detach()) < start


def test_muon_step_size_is_governed_by_the_scale_not_the_gradient_magnitude() -> None:
    """Orthogonalization discards magnitude: a 1000x larger gradient takes the same step."""
    small = torch.ones(4, 4, requires_grad=True)
    large = torch.ones(4, 4, requires_grad=True)
    opt_s = Muon([small], lr=0.1, momentum=0.0, nesterov=False)
    opt_l = Muon([large], lr=0.1, momentum=0.0, nesterov=False)

    g = torch.randn(4, 4)
    small.grad = g.clone()
    large.grad = g.clone() * 1000.0
    opt_s.step()
    opt_l.step()
    torch.testing.assert_close(small, large, atol=1e-5, rtol=1e-5)


def test_muon_nesterov_flag_changes_the_trajectory() -> None:
    """Uses varying gradients on purpose.

    With a *constant* gradient direction the momentum buffer stays parallel to it, and
    since orthogonalization discards magnitude the two variants produce byte-identical
    updates. The flag only bites when the direction changes between steps.
    """
    torch.manual_seed(2)
    a = torch.randn(8, 8, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    opt_a = Muon([a], lr=0.1, nesterov=True)
    opt_b = Muon([b], lr=0.1, nesterov=False)
    for _ in range(5):
        g = torch.randn(8, 8)
        for p, opt in ((a, opt_a), (b, opt_b)):
            opt.zero_grad()
            p.grad = g.clone()
            opt.step()
    assert not torch.allclose(a, b)


def test_muon_state_dict_round_trip() -> None:
    p = torch.randn(4, 4, requires_grad=True)
    opt = Muon([p], lr=0.05)
    p.grad = torch.randn_like(p)
    opt.step()
    assert opt.state[p]["step"] == 1
    state = _serialize(opt.state_dict())

    q = p.detach().clone().requires_grad_(True)
    opt2 = Muon([q], lr=0.05)
    opt2.load_state_dict(state)
    grad = torch.randn_like(p)
    p.grad, q.grad = grad, grad.clone()
    opt.step()
    opt2.step()
    torch.testing.assert_close(p, q)


# =====================================================================================
# Cautious weight decay
# =====================================================================================


def test_cautious_mask_fires_only_where_signs_agree() -> None:
    p = torch.tensor([1.0, 1.0, -1.0, -1.0])
    u = torch.tensor([1.0, -1.0, 1.0, -1.0])
    torch.testing.assert_close(cautious_decay_mask(p, u), torch.tensor([1.0, 0.0, 0.0, 1.0]))


def test_cautious_mask_fraction_is_about_half_for_random_directions() -> None:
    p = torch.randn(10_000)
    u = torch.randn(10_000)
    assert 0.45 < cautious_mask_fraction(p, u) < 0.55
    assert cautious_mask_fraction(torch.empty(0), torch.empty(0)) == 0.0


def test_cautious_decay_skips_coordinates_where_it_would_fight_the_update() -> None:
    p = torch.tensor([[1.0, 1.0]], dtype=torch.float64, requires_grad=True)
    opt = AdamW(
        [p],
        lr=0.1,
        weight_decay=0.5,
        betas=(0.0, 0.0),
        eps=1e-30,
        cautious_weight_decay=True,
    )
    # grad = [+1, -1]: update = [+1, -1] (beta1 = 0, so m = g and v = g^2 -> u = sign(g)).
    # Coordinate 0: u>0, p>0 -> decay fires. Coordinate 1: u<0, p>0 -> decay skipped.
    p.grad = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    opt.step()
    expected = torch.tensor([[1.0 - 0.1 * 0.5 - 0.1, 1.0 + 0.1]], dtype=torch.float64)
    torch.testing.assert_close(p, expected, atol=1e-9, rtol=1e-9)


def test_cautious_decay_never_decays_more_than_plain_decay() -> None:
    torch.manual_seed(0)
    plain = torch.randn(32, 32, requires_grad=True)
    caut = plain.detach().clone().requires_grad_(True)
    o_plain = Muon([plain], lr=0.01, weight_decay=0.1)
    o_caut = Muon([caut], lr=0.01, weight_decay=0.1, cautious_weight_decay=True)
    for _ in range(20):
        g = torch.randn(32, 32)
        for p, opt in ((plain, o_plain), (caut, o_caut)):
            opt.zero_grad()
            p.grad = g.clone()
            opt.step()
    assert float(caut.detach().norm()) >= float(plain.detach().norm())


def test_wd_scale_multiplies_the_decay() -> None:
    p = torch.tensor([[1.0]], dtype=torch.float64, requires_grad=True)
    opt = AdamW([p], lr=0.1, weight_decay=0.5, betas=(0.0, 0.0), eps=1e-30)
    opt.param_groups[0]["wd_scale"] = 0.0
    p.grad = torch.zeros_like(p)
    opt.step()
    torch.testing.assert_close(p, torch.tensor([[1.0]], dtype=torch.float64))


# =====================================================================================
# The parameter router
# =====================================================================================


def _model(**kwargs: object) -> NanoScaleLM:
    base: dict[str, object] = {
        "vocab_size": 32,
        "n_layers": 2,
        "d_model": 16,
        "n_heads": 2,
        "n_kv_heads": 1,
        "max_seq_len": 8,
    }
    base.update(kwargs)
    return NanoScaleLM(ModelConfig.model_validate(base))


def test_router_sends_hidden_matrices_to_muon_and_everything_else_to_adamw() -> None:
    split = split_parameters(_model())

    for name in split.muon_names:
        assert "embed_tokens" not in name
        assert "lm_head" not in name
        assert name.endswith(".weight")
    assert any("q_proj" in n for n in split.muon_names)
    assert any("gate_proj" in n for n in split.muon_names)
    assert any("o_proj" in n for n in split.muon_names)

    assert any("embed_tokens" in n for n in split.adamw_names)
    assert any("lm_head" in n for n in split.adamw_names)
    assert any("norm" in n for n in split.adamw_names)

    for p in split.muon:
        assert p.ndim == 2
    assert all(
        p.ndim == 1 or "embed" in n or "head" in n
        for p, n in zip(split.adamw, split.adamw_names, strict=True)
    )


def test_router_covers_every_parameter_exactly_once() -> None:
    model = _model()
    split = split_parameters(model)
    routed = {id(p) for p in split.muon} | {id(p) for p in split.adamw}
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    assert routed == expected
    assert len(split.muon) + len(split.adamw) == len(routed)
    total = split.summary()["muon_params"] + split.summary()["adamw_params"]
    assert total == model.num_parameters()


def test_router_handles_weight_tying_without_double_stepping() -> None:
    model = _model(tie_embeddings=True)
    split = split_parameters(model)
    ids = [id(p) for p in split.muon + split.adamw]
    assert len(ids) == len(set(ids)), "a tied tensor must be routed exactly once"


def test_router_sends_mtp_unembeddings_to_adamw() -> None:
    split = split_parameters(_model(n_mtp_heads=2))
    mtp_names = [n for n in split.muon_names + split.adamw_names if n.startswith("mtp.")]
    assert mtp_names
    assert all(n in split.adamw_names for n in mtp_names)


def test_router_skips_frozen_parameters() -> None:
    model = _model()
    model.embed_tokens.weight.requires_grad_(False)
    split = split_parameters(model)
    assert not any("embed_tokens" in n for n in split.adamw_names)


# =====================================================================================
# The composite optimizer
# =====================================================================================


def test_build_optimizer_muon_mode_creates_both_groups() -> None:
    opt = build_optimizer(_model(), OptimConfig(name="muon"))
    assert set(opt.optimizers) == {"muon", "adamw"}
    assert isinstance(opt.optimizers["muon"], Muon)
    assert isinstance(opt.optimizers["adamw"], AdamW)
    assert len(opt) == 2


def test_build_optimizer_adamw_mode_uses_one_group() -> None:
    model = _model()
    opt = build_optimizer(model, OptimConfig(name="adamw"))
    assert set(opt.optimizers) == {"adamw"}
    routed = sum(len(g["params"]) for g in opt.param_groups)
    assert routed == len([p for p in model.parameters() if p.requires_grad])


def test_build_optimizer_uses_separate_learning_rates() -> None:
    cfg = OptimConfig(name="muon", lr=0.02, adamw_lr=3e-4)
    opt = build_optimizer(_model(), cfg)
    assert opt.optimizers["muon"].param_groups[0]["lr"] == 0.02
    assert opt.optimizers["adamw"].param_groups[0]["lr"] == 3e-4


def test_composite_step_updates_every_parameter() -> None:
    model = _model(zero_init_output=False)
    opt = build_optimizer(model, OptimConfig(name="muon"))
    before = [p.detach().clone() for p in model.parameters()]
    ids = torch.randint(0, 32, (2, 8))
    loss = model(ids, targets=ids).loss
    assert loss is not None
    backward(loss)
    opt.step()
    changed = sum(
        not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters(), strict=True)
    )
    assert changed >= len(before) - 1  # a zero-gradient tensor may legitimately not move


def test_composite_set_lr_is_idempotent() -> None:
    opt = build_optimizer(_model(), OptimConfig(name="muon", lr=0.02, adamw_lr=1e-3))
    opt.set_lr(0.5)
    opt.set_lr(0.5)
    assert opt.optimizers["muon"].param_groups[0]["lr"] == pytest.approx(0.01)
    assert opt.optimizers["adamw"].param_groups[0]["lr"] == pytest.approx(5e-4)
    opt.set_lr(1.0)
    assert opt.optimizers["muon"].param_groups[0]["lr"] == pytest.approx(0.02)


def test_composite_reports_learning_rates() -> None:
    opt = build_optimizer(_model(), OptimConfig(name="muon"))
    assert set(opt.learning_rates()) == {"lr_muon", "lr_adamw"}


def test_composite_weight_decay_scale() -> None:
    opt = build_optimizer(_model(), OptimConfig(name="muon"))
    opt.set_weight_decay_scale(0.25)
    assert all(g["wd_scale"] == 0.25 for g in opt.param_groups)


def test_composite_state_dict_round_trip() -> None:
    model = _model(zero_init_output=False)
    opt = build_optimizer(model, OptimConfig(name="muon"))
    ids = torch.randint(0, 32, (2, 8))
    loss = model(ids, targets=ids).loss
    assert loss is not None
    backward(loss)
    opt.step()

    state = _serialize(opt.state_dict())
    assert set(state) == {"muon", "adamw"}

    fresh = build_optimizer(model, OptimConfig(name="muon"))
    fresh.load_state_dict(state)
    assert fresh.state_dict().keys() == state.keys()

    with pytest.raises(KeyError, match="missing entries"):
        fresh.load_state_dict({"muon": state["muon"]})


def test_composite_zero_grad() -> None:
    model = _model(zero_init_output=False)
    opt = build_optimizer(model, OptimConfig(name="muon"))
    ids = torch.randint(0, 32, (1, 4))
    loss = model(ids, targets=ids).loss
    assert loss is not None
    backward(loss)
    assert any(p.grad is not None for p in model.parameters())
    opt.zero_grad()
    assert all(p.grad is None for p in model.parameters())


def test_composite_closure_is_evaluated_once() -> None:
    calls: list[int] = []
    opt = CompositeOptimizer({"adamw": AdamW([torch.zeros(2, requires_grad=True)], lr=0.1)})

    def closure() -> float:
        calls.append(1)
        return 1.5

    assert opt.step(closure) == 1.5
    assert len(calls) == 1


def test_build_optimizer_rejects_a_model_with_no_trainable_parameters() -> None:
    frozen = nn.Linear(2, 2)
    for p in frozen.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable parameters"):
        build_optimizer(frozen, OptimConfig(name="muon"))
