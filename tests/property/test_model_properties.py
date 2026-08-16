"""Hypothesis property tests for the model (spec D2).

Invariants that must hold for *every* configuration in the space the configs allow,
not just the handful of shapes the unit tests happen to instantiate.
"""

from __future__ import annotations

import math

import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nanoscale.config import ModelConfig
from nanoscale.model import NanoScaleLM, apply_rope, build_rope_cache, rms_normalize

_SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)

# Small enough to instantiate dozens of times, wide enough to cover the config space.
_HEAD_DIMS = st.sampled_from([4, 8, 16])
_N_HEADS = st.sampled_from([1, 2, 4])


@st.composite
def model_configs(draw: st.DrawFn) -> ModelConfig:
    """Draw a small but structurally valid model configuration."""
    n_heads = draw(_N_HEADS)
    head_dim = draw(_HEAD_DIMS)
    n_kv_heads = draw(st.sampled_from([d for d in (1, 2, 4) if d <= n_heads and n_heads % d == 0]))
    return ModelConfig(
        vocab_size=draw(st.integers(min_value=8, max_value=64)),
        n_layers=draw(st.integers(min_value=1, max_value=3)),
        d_model=n_heads * head_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=draw(st.integers(min_value=4, max_value=16)),
        norm_type=draw(st.sampled_from(["rmsnorm", "layernorm"])),
        mlp_type=draw(st.sampled_from(["swiglu", "relu2"])),
        qk_norm=draw(st.booleans()),
        tie_embeddings=draw(st.booleans()),
        zero_init_output=draw(st.booleans()),
        attn_impl=draw(st.sampled_from(["manual", "sdpa"])),
    )


@_SETTINGS
@given(cfg=model_configs(), batch=st.integers(min_value=1, max_value=3))
def test_shapes_and_dtypes_are_invariant_across_configs(cfg: ModelConfig, batch: int) -> None:
    model = NanoScaleLM(cfg).eval()
    seq = min(cfg.max_seq_len, 6)
    ids = torch.randint(0, cfg.vocab_size, (batch, seq))
    out = model(ids, targets=ids, return_hidden=True)
    assert out.logits.shape == (batch, seq, cfg.vocab_size)
    assert out.logits.dtype is torch.float32
    assert out.hidden is not None and out.hidden.shape == (batch, seq, cfg.d_model)
    assert out.loss is not None and out.loss.ndim == 0
    assert torch.isfinite(out.loss)


@_SETTINGS
@given(cfg=model_configs())
def test_analytic_parameter_count_always_matches_the_module(cfg: ModelConfig) -> None:
    assert NanoScaleLM(cfg).num_parameters() == cfg.param_count()


@_SETTINGS
@given(cfg=model_configs())
def test_incremental_decode_always_matches_full_recompute(cfg: ModelConfig) -> None:
    model = NanoScaleLM(cfg).double().eval()
    seq = min(cfg.max_seq_len, 6)
    ids = torch.randint(0, cfg.vocab_size, (1, seq))
    full = model(ids).logits
    cache = model.make_cache(1)
    steps = torch.cat([model(ids[:, i : i + 1], cache=cache).logits for i in range(seq)], dim=1)
    torch.testing.assert_close(steps, full, atol=1e-8, rtol=1e-8)


@_SETTINGS
@given(cfg=model_configs())
def test_output_never_depends_on_future_tokens(cfg: ModelConfig) -> None:
    """Causality, for every architecture variant."""
    model = NanoScaleLM(cfg.merged(zero_init_output=False)).double().eval()
    seq = min(cfg.max_seq_len, 6)
    if seq < 2:
        return
    ids = torch.randint(0, cfg.vocab_size, (1, seq))
    cut = seq // 2
    base = model(ids).logits
    other = ids.clone()
    other[:, cut:] = (other[:, cut:] + 1) % cfg.vocab_size
    torch.testing.assert_close(model(other).logits[:, :cut], base[:, :cut], atol=1e-9, rtol=1e-9)


@_SETTINGS
@given(cfg=model_configs())
def test_zero_init_always_starts_at_ln_vocab(cfg: ModelConfig) -> None:
    """Only meaningful with an untied head: a tied head cannot be zeroed (see below)."""
    model = NanoScaleLM(cfg.merged(zero_init_output=True, tie_embeddings=False)).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, min(cfg.max_seq_len, 4)))
    loss = model(ids, targets=ids).loss
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(math.log(cfg.vocab_size)), atol=1e-4, rtol=1e-4)


@_SETTINGS
@given(
    shape=st.tuples(
        st.integers(min_value=1, max_value=4),
        st.integers(min_value=1, max_value=8),
    ),
    dim=st.integers(min_value=2, max_value=32),
)
def test_rms_normalize_always_produces_unit_rms(shape: tuple[int, int], dim: int) -> None:
    x = torch.randn(*shape, dim) * 100.0
    out = rms_normalize(x, eps=1e-12)
    torch.testing.assert_close(out.pow(2).mean(dim=-1), torch.ones(*shape), atol=1e-4, rtol=1e-4)


@_SETTINGS
@given(
    head_dim=st.sampled_from([2, 4, 8, 16, 64]),
    seq=st.integers(min_value=1, max_value=32),
    heads=st.integers(min_value=1, max_value=4),
)
def test_rope_always_preserves_norms(head_dim: int, seq: int, heads: int) -> None:
    x = torch.randn(2, heads, seq, head_dim, dtype=torch.float64)
    cos, sin = build_rope_cache(head_dim, seq, dtype=torch.float64)
    rotated = apply_rope(x, cos, sin)
    torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-12, rtol=1e-12)


@_SETTINGS
@given(cfg=model_configs(), n_new=st.integers(min_value=1, max_value=5))
def test_generation_always_extends_the_prompt_with_valid_ids(cfg: ModelConfig, n_new: int) -> None:
    model = NanoScaleLM(cfg).eval()
    prompt_len = min(cfg.max_seq_len - 1, 3)
    if prompt_len < 1:
        return
    prompt = torch.randint(0, cfg.vocab_size, (2, prompt_len))
    out = model.generate(prompt, max_new_tokens=n_new, temperature=1.0)
    assert out.shape[0] == 2
    assert prompt_len < out.shape[1] <= min(cfg.max_seq_len, prompt_len + n_new)
    assert torch.equal(out[:, :prompt_len], prompt)
    assert int(out.max()) < cfg.vocab_size
    assert int(out.min()) >= 0
