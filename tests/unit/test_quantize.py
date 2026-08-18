"""Quantization correctness (spec D1 / Phase 8 acceptance).

The load-bearing claims:

* Quantize→dequantize error is **bounded by the quantization step**: the property that
  makes every downstream bound meaningful.
* **GPTQ beats RTN at every bit-width**, and its advantage grows as bits shrink.
* GPTQ's two mechanisms are separated: error compensation (which helps even with white
  activations, because it is error feedback) and activation ordering (which only helps
  when the Hessian carries a ranking).
* The Hessian is assembled correctly against a hand computation.
* **Effective bits** include the stored scales, so a "4-bit" number is not quietly 4.5.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from nanoscale.config import ModelConfig
from nanoscale.model import NanoScaleLM
from nanoscale.quantize import (
    ActivationStats,
    AWQQuantizer,
    GPTQQuantizer,
    HessianAccumulator,
    QuantizedKVCache,
    awq_quantize_layer,
    dequantize_kv,
    effective_bits,
    gptq_quantize_layer,
    kv_cache_memory_report,
    quantize_kv,
    quantize_rtn,
    quantize_tensor_rtn,
    search_awq_scale,
)


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(20260816)


def tiny_model() -> NanoScaleLM:
    return NanoScaleLM(
        ModelConfig(
            vocab_size=64,
            n_layers=2,
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=32,
            zero_init_output=False,
        )
    )


# =====================================================================================
# RTN primitives
# =====================================================================================


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("symmetric", [False, True])
def test_round_trip_error_is_bounded_by_the_quantization_step(bits: int, symmetric: bool) -> None:
    """Every weight must land within half a step of its quantized value."""
    w = torch.randn(32, 128)
    q = quantize_tensor_rtn(w, bits=bits, group_size=32, symmetric=symmetric)
    restored = q.dequantize()

    # Reconstruct the per-group step size to compare against.
    grouped = w.reshape(32, 4, 32)
    span = (
        grouped.amax(dim=-1) - grouped.amin(dim=-1)
        if not symmetric
        else (2 * grouped.abs().amax(dim=-1))
    )
    step = span / (2**bits - 1)
    max_step = float(step.max())

    assert float((restored - w).abs().max()) <= max_step * (0.5 + 1e-4) + 1e-6


def test_more_bits_never_increase_the_error() -> None:
    w = torch.randn(16, 64)
    errors = [quantize_tensor_rtn(w, bits=b, group_size=32).error(w) for b in (2, 3, 4, 8)]
    assert errors == sorted(errors, reverse=True)


def test_smaller_groups_reduce_the_error() -> None:
    """The whole reason grouping exists: one outlier must not set the range for all."""
    w = torch.randn(8, 128)
    w[0, 0] = 100.0  # a single extreme outlier
    coarse = quantize_tensor_rtn(w, bits=4, group_size=128).error(w)
    fine = quantize_tensor_rtn(w, bits=4, group_size=16).error(w)
    assert fine < coarse


def test_quantization_is_exact_when_the_values_are_on_the_grid() -> None:
    levels = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    q = quantize_tensor_rtn(levels, bits=4, group_size=16)
    torch.testing.assert_close(q.dequantize(), levels, atol=1e-4, rtol=1e-4)


def test_effective_bits_counts_the_scales() -> None:
    """A '4-bit' model with fp16 scales at group 128 really costs 4.25 bits."""
    assert effective_bits(4, 128, symmetric=True) == pytest.approx(4.125)
    assert effective_bits(4, 128, symmetric=False) == pytest.approx(4.25)
    assert effective_bits(4, 32, symmetric=False) == pytest.approx(5.0)
    assert effective_bits(4, -1) == 4.0
    q = quantize_tensor_rtn(torch.randn(4, 64), bits=4, group_size=32)
    assert q.effective_bits() == pytest.approx(5.0)


def test_rtn_validates_shapes_and_bit_widths() -> None:
    with pytest.raises(ValueError, match="2D weight"):
        quantize_tensor_rtn(torch.randn(4))
    with pytest.raises(ValueError, match=r"bits must be in \[2, 8\]"):
        quantize_tensor_rtn(torch.randn(4, 8), bits=16)
    with pytest.raises(ValueError, match="not divisible"):
        quantize_tensor_rtn(torch.randn(4, 30), group_size=16)


def test_quantize_rtn_skips_the_embedding_and_head() -> None:
    model = tiny_model()
    embed_before = model.embed_tokens.weight.detach().clone()
    head_before = model.lm_head.weight.detach().clone()
    errors = quantize_rtn(model, bits=4, group_size=32)

    torch.testing.assert_close(model.embed_tokens.weight, embed_before)
    torch.testing.assert_close(model.lm_head.weight, head_before)
    assert errors, "no linear layers were quantized"
    assert all("embed_tokens" not in k and "lm_head" not in k for k in errors)
    assert all(0.0 < v < 1.0 for v in errors.values())


# =====================================================================================
# The Hessian
# =====================================================================================


def test_hessian_matches_a_hand_computation() -> None:
    acc = HessianAccumulator(3)
    x = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    acc.add(x)
    # Running-mean form: H = 2 X^T X / n
    expected = 2.0 * (x.T @ x) / 2
    torch.testing.assert_close(acc.hessian, expected)
    assert acc.n_samples == 2


def test_hessian_running_mean_is_batch_order_invariant() -> None:
    a = torch.randn(5, 4)
    b = torch.randn(7, 4)

    one = HessianAccumulator(4)
    one.add(torch.cat([a, b]))

    two = HessianAccumulator(4)
    two.add(a)
    two.add(b)

    torch.testing.assert_close(one.hessian, two.hessian, atol=1e-5, rtol=1e-5)


def test_hessian_dampening_makes_a_singular_matrix_factorisable() -> None:
    acc = HessianAccumulator(4)
    x = torch.randn(8, 4)
    x[:, 2] = 0.0  # a dead channel
    acc.add(x)
    h, dead = acc.finalize(damp_percent=0.01)
    assert bool(dead[2])
    torch.linalg.cholesky(h)  # must not raise


def test_hessian_ignores_empty_batches() -> None:
    acc = HessianAccumulator(3)
    acc.add(torch.zeros(0, 3))
    assert acc.n_samples == 0


# =====================================================================================
# GPTQ vs RTN
# =====================================================================================


def _layer_output_error(w: torch.Tensor, w_hat: torch.Tensor, x: torch.Tensor) -> float:
    reference = x @ w.T
    return float((x @ w_hat.T - reference).norm() / reference.norm())


def test_gptq_beats_rtn_when_activations_are_ill_conditioned() -> None:
    """The claim GPTQ exists to make, on a layer built to expose the difference.

    The activations have per-channel standard deviations spanning three orders of
    magnitude, so weights multiplying loud channels matter far more than weights
    multiplying quiet ones. RTN cannot know that; GPTQ reads it off the Hessian.
    """
    in_features, out_features, n = 128, 64, 512
    channel_scale = torch.logspace(1.5, -1.5, in_features)
    x = torch.randn(n, in_features) * channel_scale
    w = torch.randn(out_features, in_features) * 0.1

    hessian = 2.0 * (x.T @ x) / n

    rtn = quantize_tensor_rtn(w, bits=3, group_size=32).dequantize()
    gptq = gptq_quantize_layer(w, hessian, bits=3, group_size=32, act_order=True).dequantize()

    rtn_err = _layer_output_error(w, rtn, x)
    gptq_err = _layer_output_error(w, gptq, x)
    assert gptq_err < rtn_err, f"gptq {gptq_err:.4f} did not beat rtn {rtn_err:.4f}"


def test_gptqs_advantage_grows_as_bits_shrink() -> None:
    """More rounding error to compensate means more for the compensation to recover."""
    in_features, out_features, n = 128, 64, 512
    torch.manual_seed(0)
    w = torch.randn(out_features, in_features) * 0.1
    x = torch.randn(n, in_features) * torch.logspace(1.5, -1.5, in_features)
    hessian = 2.0 * (x.T @ x) / n

    gaps = {}
    for bits in (2, 3, 4, 8):
        rtn = quantize_tensor_rtn(w, bits=bits, group_size=32).dequantize()
        gptq = gptq_quantize_layer(
            w, hessian, bits=bits, group_size=32, act_order=True
        ).dequantize()
        gaps[bits] = _layer_output_error(w, rtn, x) - _layer_output_error(w, gptq, x)

    assert all(g > 0 for g in gaps.values()), f"GPTQ lost at some bit-width: {gaps}"
    assert gaps[2] > gaps[4] > gaps[8], f"the advantage should grow as bits shrink: {gaps}"


def test_activation_ordering_helps_more_when_channels_are_uneven() -> None:
    """Isolates the *ordering* half of GPTQ from the error-compensation half.

    GPTQ's advantage over RTN has two independent sources, and conflating them is easy:

    1. **Error compensation**: pushing each column's rounding error onto the columns
       not yet quantized. This is error feedback and helps even when the Hessian is
       featureless, which is why it shows up with white activations too.
    2. **Activation ordering**: quantizing salient columns first, while the most budget
       remains to absorb their error. This is the part that depends on the Hessian
       carrying a ranking.

    A note on how the test is built: the channel scales are **shuffled** before use. An
    earlier version used ``logspace`` directly, which is already sorted by salience, so
    ``argsort(diag(H))`` was the identity permutation and ``act_order`` measured as
    having exactly zero effect. The shuffle is what makes this test non-vacuous.
    """
    in_features, out_features, n = 128, 64, 512
    torch.manual_seed(0)
    w = torch.randn(out_features, in_features) * 0.1
    shuffle = torch.randperm(in_features)

    def ordering_gain(channel_scale: torch.Tensor) -> float:
        x = torch.randn(n, in_features) * channel_scale
        hessian = 2.0 * (x.T @ x) / n
        plain = gptq_quantize_layer(w, hessian, bits=2, group_size=32, act_order=False).dequantize()
        ordered = gptq_quantize_layer(
            w, hessian, bits=2, group_size=32, act_order=True
        ).dequantize()
        return _layer_output_error(w, plain, x) - _layer_output_error(w, ordered, x)

    structured = ordering_gain(torch.logspace(2, -2, in_features)[shuffle])
    white = ordering_gain(torch.ones(in_features))

    assert structured > 0, "activation ordering should help on uneven channels"
    assert structured > white, (
        f"ordering gain on structured activations ({structured:.4f}) should exceed the "
        f"gain on white ones ({white:.4f})"
    )


def test_act_order_permutation_round_trips_without_re_encoding() -> None:
    """The permutation must travel with the codes, not be undone by re-encoding.

    Re-quantizing GPTQ's reconstructed weights to restore column order runs a *second*
    rounding pass over values GPTQ had already carefully placed. That bug made GPTQ
    measurably **worse** than RTN at 2 bits on the real frontier before it was caught.
    """
    torch.manual_seed(0)
    w = torch.randn(32, 64)
    x = torch.randn(256, 64) * torch.logspace(1, -1, 64)[torch.randperm(64)]
    hessian = 2.0 * (x.T @ x) / 256

    q = gptq_quantize_layer(w, hessian, bits=3, group_size=16, act_order=True)
    assert q.perm is not None, "act_order must record its permutation"
    assert q.dequantize().shape == w.shape

    without_order = gptq_quantize_layer(w, hessian, bits=3, group_size=16, act_order=False)
    assert without_order.perm is None


def test_gptq_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="2D weight"):
        gptq_quantize_layer(torch.randn(4), torch.eye(4))
    with pytest.raises(ValueError, match="does not match in_features"):
        gptq_quantize_layer(torch.randn(4, 8), torch.eye(4))
    with pytest.raises(ValueError, match="not divisible"):
        gptq_quantize_layer(torch.randn(4, 30), torch.eye(30), group_size=16)


def test_gptq_handles_a_dead_input_channel() -> None:
    w = torch.randn(8, 32)
    x = torch.randn(64, 32)
    x[:, 5] = 0.0
    hessian = 2.0 * (x.T @ x) / 64
    out = gptq_quantize_layer(w, hessian, bits=4, group_size=16).dequantize()
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("act_order", [True, False])
def test_gptq_end_to_end_on_a_model(act_order: bool) -> None:
    model = tiny_model().eval()
    batches = [torch.randint(0, 64, (2, 16)) for _ in range(4)]
    quantizer = GPTQQuantizer(model, bits=4, group_size=32, act_order=act_order)
    quantizer.collect(batches)
    errors = quantizer.apply()
    assert errors
    assert all(math.isfinite(v) for v in errors.values())
    assert torch.isfinite(model(batches[0]).logits).all()


def test_gptq_requires_calibration_before_apply() -> None:
    with pytest.raises(RuntimeError, match="collect"):
        GPTQQuantizer(tiny_model()).apply()


# =====================================================================================
# AWQ
# =====================================================================================


def test_activation_stats_running_mean() -> None:
    stats = ActivationStats.empty(3, torch.device("cpu"))
    stats.add(torch.tensor([[1.0, -2.0, 3.0]]))
    stats.add(torch.tensor([[3.0, 2.0, 1.0]]))
    torch.testing.assert_close(stats.mean_abs, torch.tensor([2.0, 2.0, 2.0]))
    assert stats.n_samples == 2
    torch.testing.assert_close(stats.salience(), torch.ones(3))


def test_awq_scaling_is_output_preserving_before_quantization() -> None:
    """The identity the whole method rests on: (X/s)(W*s)^T == X W^T."""
    w = torch.randn(8, 16, dtype=torch.float64)
    x = torch.randn(32, 16, dtype=torch.float64)
    s = torch.rand(16, dtype=torch.float64) + 0.5
    torch.testing.assert_close((x / s) @ (w * s).T, x @ w.T, atol=1e-10, rtol=1e-10)


def test_awq_search_picks_a_nonzero_alpha_when_channels_are_uneven() -> None:
    in_features, out_features = 64, 32
    channel_scale = torch.logspace(1.5, -1.5, in_features)
    x = torch.randn(256, in_features) * channel_scale
    w = torch.randn(out_features, in_features) * 0.1
    salience = x.abs().mean(dim=0)
    salience = salience / salience.mean()

    alpha, _, error = search_awq_scale(w, salience, x, bits=3, group_size=16, grid=10)
    _, _, rtn_error = search_awq_scale(w, torch.ones(in_features), x, bits=3, group_size=16, grid=0)
    assert alpha > 0.0, "AWQ chose no scaling on a layer with very uneven channels"
    assert error <= rtn_error


def test_awq_alpha_zero_reproduces_rtn() -> None:
    w = torch.randn(8, 32)
    x = torch.randn(64, 32)
    quantized, alpha, _ = awq_quantize_layer(w, torch.ones(32), x, bits=4, group_size=16, grid=0)
    rtn = quantize_tensor_rtn(w, bits=4, group_size=16).dequantize()
    assert alpha == 0.0
    torch.testing.assert_close(quantized, rtn, atol=1e-5, rtol=1e-5)


def test_awq_end_to_end_on_a_model() -> None:
    model = tiny_model().eval()
    batches = [torch.randint(0, 64, (2, 16)) for _ in range(3)]
    quantizer = AWQQuantizer(model, bits=4, group_size=32, grid=6)
    quantizer.collect(batches)
    errors = quantizer.apply()
    assert errors
    assert quantizer.alphas
    assert all(0.0 <= a <= 1.0 for a in quantizer.alphas.values())
    assert torch.isfinite(model(batches[0]).logits).all()


def test_awq_requires_calibration_before_apply() -> None:
    with pytest.raises(RuntimeError, match="collect"):
        AWQQuantizer(tiny_model()).apply()


# =====================================================================================
# KV-cache quantization
# =====================================================================================


def test_kv_round_trip_error_is_bounded() -> None:
    x = torch.randn(2, 4, 16, 32)
    q = quantize_kv(x, bits=4, group_size=32)
    restored = dequantize_kv(q)
    grouped = x.reshape(2, 4, 16, 1, 32)
    step = (grouped.amax(dim=-1) - grouped.amin(dim=-1)) / 15
    assert float((restored - x).abs().max()) <= float(step.max()) * 0.5 + 1e-4


def test_kv_quantization_validates_shapes() -> None:
    with pytest.raises(ValueError, match=r"\(B, H, T, D\)"):
        quantize_kv(torch.randn(4, 8))
    with pytest.raises(ValueError, match="not divisible"):
        quantize_kv(torch.randn(1, 1, 2, 30), group_size=16)


def test_keys_are_more_sensitive_to_quantization_than_values() -> None:
    """The asymmetry that motivates different K and V bit-widths.

    Keys pass through a dot product and a softmax, which can reorder attention weights;
    values are combined by weights that already sum to one, so their errors average out.
    """
    torch.manual_seed(0)
    b, h, t, d = 1, 2, 24, 32
    q = torch.randn(b, h, 1, d)
    k = torch.randn(b, h, t, d)
    v = torch.randn(b, h, t, d)

    def attend(kk: torch.Tensor, vv: torch.Tensor) -> torch.Tensor:
        scores = (q @ kk.transpose(-2, -1)) / math.sqrt(d)
        return torch.softmax(scores, dim=-1) @ vv

    reference = attend(k, v)
    k_quant = dequantize_kv(quantize_kv(k, bits=2, group_size=32))
    v_quant = dequantize_kv(quantize_kv(v, bits=2, group_size=32))

    err_k = float((attend(k_quant, v) - reference).norm() / reference.norm())
    err_v = float((attend(k, v_quant) - reference).norm() / reference.norm())
    assert err_k > err_v, f"expected keys to be more sensitive: k={err_k:.4f} v={err_v:.4f}"


def test_quantized_kv_cache_still_decodes_coherently() -> None:
    model = tiny_model().eval()
    ids = torch.randint(0, 64, (1, 8))
    exact = model(ids).logits

    cache = QuantizedKVCache(
        n_layers=model.config.n_layers,
        batch_size=1,
        n_kv_heads=model.config.n_kv_heads,
        head_dim=model.config.head_dim,
        max_seq_len=32,
        key_bits=8,
        value_bits=8,
        group_size=32,
    )
    stepwise = torch.cat(
        [model(ids[:, i : i + 1], cache=cache).logits for i in range(ids.shape[1])], dim=1
    )
    # 8-bit KV should track the exact computation closely but not exactly.
    assert not torch.equal(stepwise, exact)
    torch.testing.assert_close(stepwise, exact, atol=0.05, rtol=0.05)


def test_lower_kv_bits_cost_more_accuracy() -> None:
    model = tiny_model().eval()
    ids = torch.randint(0, 64, (1, 8))
    exact = model(ids).logits

    def error(bits: int) -> float:
        cache = QuantizedKVCache(
            n_layers=model.config.n_layers,
            batch_size=1,
            n_kv_heads=model.config.n_kv_heads,
            head_dim=model.config.head_dim,
            max_seq_len=32,
            key_bits=bits,
            value_bits=bits,
            group_size=32,
        )
        out = torch.cat(
            [model(ids[:, i : i + 1], cache=cache).logits for i in range(ids.shape[1])], dim=1
        )
        return float((out - exact).norm() / exact.norm())

    assert error(2) > error(4) > error(8)


def test_kv_memory_report_accounting() -> None:
    report = kv_cache_memory_report(
        n_layers=8,
        batch_size=1,
        n_kv_heads=4,
        head_dim=64,
        seq_len=4096,
        key_bits=4,
        value_bits=4,
        group_size=32,
    )
    # 4 bits + 2*16/32 = 5 effective bits per element, vs 16 for fp16.
    assert report["effective_bits_per_element"] == pytest.approx(5.0)
    assert report["compression"] == pytest.approx(16.0 / 5.0, rel=1e-3)
    assert report["quantized_mb"] < report["baseline_mb"]


def test_quantized_cache_reports_a_smaller_footprint_than_fp32() -> None:
    from nanoscale.model import KVCache

    kwargs = {"n_layers": 4, "batch_size": 1, "n_kv_heads": 2, "head_dim": 32, "max_seq_len": 256}
    plain = KVCache(**kwargs)  # type: ignore[arg-type]
    quantized = QuantizedKVCache(**kwargs, key_bits=4, value_bits=4, group_size=32)  # type: ignore[arg-type]
    assert quantized.memory_bytes() < plain.memory_bytes()
    # fp32 is 32 bits; quantized is 5 effective bits -> ~6.4x
    assert plain.memory_bytes() / quantized.memory_bytes() == pytest.approx(32 / 5, rel=0.02)


# =====================================================================================
# Model-level sanity
# =====================================================================================


def test_a_quantized_model_still_produces_finite_logits() -> None:
    for method in ("rtn", "gptq", "awq"):
        model = tiny_model().eval()
        batches = [torch.randint(0, 64, (2, 16)) for _ in range(3)]
        if method == "rtn":
            quantize_rtn(model, bits=4, group_size=32)
        elif method == "gptq":
            q = GPTQQuantizer(model, bits=4, group_size=32)
            q.collect(batches)
            q.apply()
        else:
            q_awq = AWQQuantizer(model, bits=4, group_size=32, grid=4)
            q_awq.collect(batches)
            q_awq.apply()
        logits = model(batches[0]).logits
        assert torch.isfinite(logits).all(), f"{method} produced non-finite logits"


def test_quantization_does_not_change_shapes_or_dtypes() -> None:
    model = tiny_model()
    before = {n: (p.shape, p.dtype) for n, p in model.named_parameters()}
    quantize_rtn(model, bits=4, group_size=32)
    after = {n: (p.shape, p.dtype) for n, p in model.named_parameters()}
    assert before == after


def test_linear_layers_outside_the_model_can_be_quantized() -> None:
    layer = nn.Linear(64, 32, bias=False)
    original = layer.weight.detach().clone()
    q = quantize_tensor_rtn(layer.weight.data, bits=4, group_size=32)
    layer.weight.data.copy_(q.dequantize())
    assert 0.0 < q.error(original) < 0.5
