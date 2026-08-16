"""Attention correctness (spec D1).

The headline test: **our from-scratch attention equals
``F.scaled_dot_product_attention``** on every configuration we ship — causal, GQA, RoPE,
padded, and single-query decode. SDPA is an optional fast path, so it has to prove it
did not change the answer; equally, if our manual path had a mask or scaling bug, this
test is what catches it.
"""

from __future__ import annotations

import math
from typing import Any, cast

import pytest
import torch
from torch.nn import functional as F

from nanoscale.config import ModelConfig
from nanoscale.model.attention import CausalSelfAttention, repeat_kv
from nanoscale.model.kv_cache import LayerKVCache
from nanoscale.model.norm import RMSNorm
from nanoscale.model.rope import build_rope_cache

ATOL = 2e-5
RTOL = 2e-5


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(20260816)


def make_config(**kwargs: object) -> ModelConfig:
    base: dict[str, object] = {
        "vocab_size": 128,
        "n_layers": 2,
        "d_model": 64,
        "n_heads": 4,
        "n_kv_heads": 2,
        "max_seq_len": 32,
    }
    base.update(kwargs)
    return ModelConfig.model_validate(base)


# ------------------------------------------------------------------------ repeat_kv


def test_repeat_kv_groups_heads_the_way_gqa_specifies() -> None:
    x = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).reshape(2, 2, 3, 4)
    out = repeat_kv(x, 3)
    assert out.shape == (2, 6, 3, 4)
    # KV head i must feed query heads [3i, 3i+3), contiguously.
    for kv_head in range(2):
        for rep in range(3):
            torch.testing.assert_close(out[:, kv_head * 3 + rep], x[:, kv_head])


def test_repeat_kv_is_a_noop_for_mha() -> None:
    x = torch.randn(1, 4, 5, 8)
    assert repeat_kv(x, 1) is x


# ---------------------------------------------------------------- manual vs SDPA


@pytest.mark.parametrize(
    ("n_heads", "n_kv_heads", "qk_norm"),
    [
        (4, 4, False),  # plain MHA
        (4, 2, False),  # GQA
        (4, 1, False),  # MQA
        (4, 2, True),  # GQA + QK-norm
        (8, 2, True),
    ],
)
def test_manual_attention_matches_sdpa(n_heads: int, n_kv_heads: int, qk_norm: bool) -> None:
    d_model = 64
    cfg_manual = make_config(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        qk_norm=qk_norm,
        attn_impl="manual",
    )
    cfg_sdpa = cfg_manual.merged(attn_impl="sdpa")

    manual = CausalSelfAttention(cfg_manual).double().eval()
    sdpa = CausalSelfAttention(cfg_sdpa).double().eval()
    sdpa.load_state_dict(manual.state_dict())

    x = torch.randn(2, 12, d_model, dtype=torch.float64)
    cos, sin = build_rope_cache(cfg_manual.head_dim, 12)
    cos, sin = cos.double(), sin.double()

    out_manual = manual(x, cos, sin)
    out_sdpa = sdpa(x, cos, sin)
    torch.testing.assert_close(out_manual, out_sdpa, atol=1e-10, rtol=1e-10)


def test_manual_attention_matches_sdpa_in_fp32() -> None:
    """The same equality has to hold at the precision training actually runs at."""
    cfg = make_config(attn_impl="manual")
    manual = CausalSelfAttention(cfg).eval()
    sdpa = CausalSelfAttention(cfg.merged(attn_impl="sdpa")).eval()
    sdpa.load_state_dict(manual.state_dict())

    x = torch.randn(3, 16, cfg.d_model)
    cos, sin = build_rope_cache(cfg.head_dim, 16)
    torch.testing.assert_close(manual(x, cos, sin), sdpa(x, cos, sin), atol=ATOL, rtol=RTOL)


def test_manual_attention_matches_sdpa_with_an_extra_mask() -> None:
    """Padding masks compose with the internal causal mask identically on both paths."""
    cfg = make_config(attn_impl="manual")
    manual = CausalSelfAttention(cfg).double().eval()
    sdpa = CausalSelfAttention(cfg.merged(attn_impl="sdpa")).double().eval()
    sdpa.load_state_dict(manual.state_dict())

    t = 10
    x = torch.randn(2, t, cfg.d_model, dtype=torch.float64)
    cos, sin = build_rope_cache(cfg.head_dim, t)
    cos, sin = cos.double(), sin.double()

    pad = torch.zeros(2, 1, 1, t, dtype=torch.float64)
    pad[0, :, :, -3:] = torch.finfo(torch.float64).min  # last 3 positions are padding

    torch.testing.assert_close(
        manual(x, cos, sin, attn_mask=pad),
        sdpa(x, cos, sin, attn_mask=pad),
        atol=1e-10,
        rtol=1e-10,
    )


def test_manual_attention_matches_a_hand_written_reference() -> None:
    """Independent of SDPA: softmax(QK^T/sqrt(d) + M)V, spelled out."""
    cfg = make_config(n_heads=2, n_kv_heads=2, qk_norm=False, d_model=32)
    attn = CausalSelfAttention(cfg).double().eval()

    t = 6
    x = torch.randn(1, t, cfg.d_model, dtype=torch.float64)
    cos, sin = build_rope_cache(cfg.head_dim, t)
    cos, sin = cos.double(), sin.double()

    from nanoscale.model.rope import apply_rope

    q, k, v = attn._project(x)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(cfg.head_dim)
    causal = torch.full((t, t), float("-inf"), dtype=torch.float64).triu(1)
    weights = torch.softmax(scores + causal, dim=-1)
    expected = attn.o_proj((weights @ v).transpose(1, 2).reshape(1, t, cfg.d_model))

    torch.testing.assert_close(attn(x, cos, sin), expected, atol=1e-10, rtol=1e-10)


# ----------------------------------------------------------------------- causality


def test_output_is_causal() -> None:
    """Changing a future token must not change any earlier output position."""
    cfg = make_config()
    attn = CausalSelfAttention(cfg).double().eval()
    t = 12
    x = torch.randn(1, t, cfg.d_model, dtype=torch.float64)
    cos, sin = build_rope_cache(cfg.head_dim, t)
    cos, sin = cos.double(), sin.double()

    base = attn(x, cos, sin)
    perturbed = x.clone()
    perturbed[:, 8:] += 5.0
    changed = attn(perturbed, cos, sin)

    torch.testing.assert_close(base[:, :8], changed[:, :8], atol=1e-10, rtol=1e-10)
    assert not torch.allclose(base[:, 8:], changed[:, 8:])


def test_causal_mask_shape_and_offset() -> None:
    mask = CausalSelfAttention.build_causal_mask(
        3, 7, device=torch.device("cpu"), dtype=torch.float32
    )
    assert mask.shape == (1, 1, 3, 7)
    neg = torch.finfo(torch.float32).min
    # Queries occupy the last 3 of 7 key positions, i.e. absolute positions 4, 5, 6.
    assert (mask[0, 0, 0, :5] == 0).all()
    assert (mask[0, 0, 0, 5:] == neg).all()
    assert (mask[0, 0, 2] == 0).all()


# ------------------------------------------------------------------- cached decode


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_incremental_decode_matches_full_recompute(impl: str) -> None:
    """Token-for-token equality between cached decoding and a full forward pass."""
    cfg = make_config(attn_impl=impl, max_seq_len=32)
    attn = CausalSelfAttention(cfg).double().eval()

    t = 10
    x = torch.randn(1, t, cfg.d_model, dtype=torch.float64)
    cos_all, sin_all = build_rope_cache(cfg.head_dim, t)
    cos_all, sin_all = cos_all.double(), sin_all.double()

    full = attn(x, cos_all, sin_all)

    cache = LayerKVCache(
        keys=torch.zeros(1, cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim, dtype=torch.float64),
        values=torch.zeros(1, cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim, dtype=torch.float64),
    )
    steps = [
        attn(x[:, i : i + 1], cos_all[i : i + 1], sin_all[i : i + 1], cache=cache) for i in range(t)
    ]
    incremental = torch.cat(steps, dim=1)

    torch.testing.assert_close(incremental, full, atol=1e-10, rtol=1e-10)


def test_prefill_then_decode_matches_full_recompute() -> None:
    """The realistic serving pattern: one chunked prefill, then single-token steps."""
    cfg = make_config(max_seq_len=32)
    attn = CausalSelfAttention(cfg).double().eval()

    t = 12
    prefill = 7
    x = torch.randn(1, t, cfg.d_model, dtype=torch.float64)
    cos, sin = build_rope_cache(cfg.head_dim, t)
    cos, sin = cos.double(), sin.double()
    full = attn(x, cos, sin)

    cache = LayerKVCache(
        keys=torch.zeros(1, cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim, dtype=torch.float64),
        values=torch.zeros(1, cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim, dtype=torch.float64),
    )
    chunks = [attn(x[:, :prefill], cos[:prefill], sin[:prefill], cache=cache)]
    for i in range(prefill, t):
        chunks.append(attn(x[:, i : i + 1], cos[i : i + 1], sin[i : i + 1], cache=cache))

    torch.testing.assert_close(torch.cat(chunks, dim=1), full, atol=1e-10, rtol=1e-10)


# ------------------------------------------------------------------------- QK-norm


def test_qk_norm_bounds_the_attention_logits() -> None:
    """The stability claim: QK-norm keeps logits bounded as q/k activations blow up."""
    x = torch.randn(1, 8, 64) * 50.0  # deliberately huge activations
    cos, sin = build_rope_cache(16, 8)

    def max_logit(qk_norm: bool) -> float:
        cfg = make_config(d_model=64, n_heads=4, n_kv_heads=4, qk_norm=qk_norm)
        attn = CausalSelfAttention(cfg).eval()
        from nanoscale.model.rope import apply_rope

        q, k, _ = attn._project(x)
        q, k = attn.q_norm(q), attn.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        return float(((q @ k.transpose(-2, -1)) * attn.scale).abs().max().detach())

    assert max_logit(True) < max_logit(False) / 10


def test_qk_norm_flag_controls_parameters() -> None:
    with_norm = CausalSelfAttention(make_config(qk_norm=True))
    without = CausalSelfAttention(make_config(qk_norm=False))
    delta = sum(p.numel() for p in with_norm.parameters()) - sum(
        p.numel() for p in without.parameters()
    )
    assert delta == 2 * make_config().head_dim


def test_qk_norm_is_applied_before_rope() -> None:
    """Pin the documented ordering; the learned gain does not commute with rotation."""
    cfg = make_config(qk_norm=True, n_heads=2, n_kv_heads=2, d_model=32)
    attn = CausalSelfAttention(cfg).double().eval()
    q_norm = cast(RMSNorm, attn.q_norm)
    k_norm = cast(RMSNorm, attn.k_norm)
    with torch.no_grad():
        q_norm.weight.fill_(2.0)
        k_norm.weight.copy_(torch.linspace(0.5, 2.0, cfg.head_dim).double())

    from nanoscale.model.rope import apply_rope

    t = 5
    x = torch.randn(1, t, cfg.d_model, dtype=torch.float64)
    cos, sin = build_rope_cache(cfg.head_dim, t)
    cos, sin = cos.double(), sin.double()

    q, k, v = attn._project(x)
    norm_then_rope = apply_rope(k_norm(k), cos, sin)
    rope_then_norm = k_norm(apply_rope(k, cos, sin))
    assert not torch.allclose(norm_then_rope, rope_then_norm), (
        "the test is vacuous if the two orders coincide"
    )

    expected_q = apply_rope(q_norm(q), cos, sin)
    scores = (expected_q @ norm_then_rope.transpose(-2, -1)) * attn.scale
    causal = torch.full((t, t), float("-inf"), dtype=torch.float64).triu(1)
    weights = torch.softmax(scores + causal, dim=-1)
    expected = attn.o_proj((weights @ v).transpose(1, 2).reshape(1, t, cfg.d_model))

    torch.testing.assert_close(attn(x, cos, sin), expected, atol=1e-10, rtol=1e-10)


# -------------------------------------------------------------------------- shapes


@pytest.mark.parametrize(("batch", "seq"), [(1, 1), (1, 8), (4, 16), (3, 32)])
def test_output_shapes(batch: int, seq: int) -> None:
    cfg = make_config(max_seq_len=32)
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(batch, seq, cfg.d_model)
    cos, sin = build_rope_cache(cfg.head_dim, seq)
    assert attn(x, cos, sin).shape == (batch, seq, cfg.d_model)


def test_dropout_only_applies_in_training_mode() -> None:
    cfg = make_config(dropout=0.5)
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 8, cfg.d_model)
    cos, sin = build_rope_cache(cfg.head_dim, 8)

    attn.eval()
    torch.testing.assert_close(attn(x, cos, sin), attn(x, cos, sin))

    attn.train()
    torch.manual_seed(0)
    a = attn(x, cos, sin)
    torch.manual_seed(1)
    b = attn(x, cos, sin)
    assert not torch.allclose(a, b)


def test_sdpa_dispatch_is_actually_used() -> None:
    """Guard against the 'sdpa' flag silently falling through to the manual path."""
    cfg = make_config(attn_impl="sdpa")
    attn = CausalSelfAttention(cfg).eval()
    calls: list[int] = []
    original = F.scaled_dot_product_attention

    def spy(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(1)
        result: torch.Tensor = original(*args, **kwargs)
        return result

    F.scaled_dot_product_attention = spy
    try:
        cos, sin = build_rope_cache(cfg.head_dim, 4)
        attn(torch.randn(1, 4, cfg.d_model), cos, sin)
    finally:
        F.scaled_dot_product_attention = original
    assert calls, "attn_impl='sdpa' did not reach scaled_dot_product_attention"
