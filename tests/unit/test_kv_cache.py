"""KV-cache correctness (spec D1).

The contract: **cached incremental decoding produces token-for-token identical logits
to a full recompute.** A subtly wrong cache still emits plausible text, so this has to
be tested at the level of exact numbers, not eyeballed samples.
"""

from __future__ import annotations

import pytest
import torch

from nanoscale.config import ModelConfig
from nanoscale.model import KVCache, build_model


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(20260816)


def make_config(**kwargs: object) -> ModelConfig:
    base: dict[str, object] = {
        "vocab_size": 64,
        "n_layers": 3,
        "d_model": 32,
        "n_heads": 4,
        "n_kv_heads": 2,
        "max_seq_len": 24,
        "zero_init_output": False,  # a zero-init model is trivially equal to itself
    }
    base.update(kwargs)
    return ModelConfig.model_validate(base)


# --------------------------------------------------------------- storage mechanics


def test_cache_shapes_reflect_gqa() -> None:
    cache = KVCache(n_layers=3, batch_size=2, n_kv_heads=2, head_dim=8, max_seq_len=16)
    assert len(cache) == 3
    assert cache[0].keys.shape == (2, 2, 16, 8)
    assert cache.length == 0
    assert cache[0].capacity == 16


def test_append_advances_length_and_returns_the_prefix() -> None:
    cache = KVCache(n_layers=1, batch_size=1, n_kv_heads=2, head_dim=4, max_seq_len=8)
    layer = cache[0]
    k1, v1 = layer.append(torch.ones(1, 2, 3, 4), torch.full((1, 2, 3, 4), 2.0))
    assert layer.length == 3
    assert k1.shape == (1, 2, 3, 4)
    k2, _ = layer.append(torch.full((1, 2, 2, 4), 5.0), torch.zeros(1, 2, 2, 4))
    assert layer.length == 5
    assert k2.shape == (1, 2, 5, 4)
    torch.testing.assert_close(k2[:, :, :3], torch.ones(1, 2, 3, 4))
    torch.testing.assert_close(k2[:, :, 3:], torch.full((1, 2, 2, 4), 5.0))
    torch.testing.assert_close(v1, torch.full((1, 2, 3, 4), 2.0))


def test_overflow_raises_a_clear_error() -> None:
    cache = KVCache(n_layers=1, batch_size=1, n_kv_heads=1, head_dim=2, max_seq_len=4)
    with pytest.raises(ValueError, match="KV cache overflow"):
        cache[0].append(torch.zeros(1, 1, 5, 2), torch.zeros(1, 1, 5, 2))


def test_reset_and_truncate() -> None:
    cache = KVCache(n_layers=2, batch_size=1, n_kv_heads=1, head_dim=2, max_seq_len=8)
    for layer in cache.layers:
        layer.append(torch.randn(1, 1, 6, 2), torch.randn(1, 1, 6, 2))
    assert cache.length == 6
    cache.truncate(4)
    assert cache.length == 4
    cache.truncate(6)  # beyond the current length but within capacity: a no-op
    assert cache.length == 4
    cache.reset()
    assert cache.length == 0
    with pytest.raises(ValueError, match="Cannot truncate"):
        cache.truncate(99)  # beyond capacity is an error, not a silent clamp
    with pytest.raises(ValueError, match="Cannot truncate"):
        cache.truncate(-1)


def test_clone_is_independent() -> None:
    cache = KVCache(n_layers=2, batch_size=1, n_kv_heads=1, head_dim=2, max_seq_len=8)
    for layer in cache.layers:
        layer.append(torch.ones(1, 1, 3, 2), torch.ones(1, 1, 3, 2))
    copy = cache.clone()
    assert copy.length == 3
    copy[0].append(torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2))
    assert cache[0].length == 3, "mutating the clone must not touch the original"
    copy[0].keys.fill_(7.0)
    assert not torch.allclose(cache[0].keys, copy[0].keys)


def test_memory_accounting() -> None:
    cache = KVCache(
        n_layers=4,
        batch_size=2,
        n_kv_heads=2,
        head_dim=16,
        max_seq_len=128,
        dtype=torch.float32,
    )
    # 4 layers x 2 tensors x (2*2*128*16) elements x 4 bytes
    assert cache.memory_bytes() == 4 * 2 * (2 * 2 * 128 * 16) * 4
    assert cache.used_bytes() == 0
    for layer in cache.layers:
        layer.append(torch.zeros(2, 2, 64, 16), torch.zeros(2, 2, 64, 16))
    assert cache.used_bytes() == cache.memory_bytes() // 2


def test_gqa_halves_the_cache_versus_mha() -> None:
    """The GQA memory win, stated as a number."""
    mha = KVCache(n_layers=8, batch_size=1, n_kv_heads=8, head_dim=64, max_seq_len=512)
    gqa = KVCache(n_layers=8, batch_size=1, n_kv_heads=4, head_dim=64, max_seq_len=512)
    assert gqa.memory_bytes() * 2 == mha.memory_bytes()


# ------------------------------------------------------------- end-to-end equality


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_full_model_incremental_decode_matches_full_recompute(impl: str) -> None:
    cfg = make_config(attn_impl=impl)
    model = build_model(cfg).double().eval()

    t = 12
    ids = torch.randint(0, cfg.vocab_size, (2, t))
    full = model(ids).logits

    cache = model.make_cache(2)
    steps = [model(ids[:, i : i + 1], cache=cache).logits for i in range(t)]
    incremental = torch.cat(steps, dim=1)

    torch.testing.assert_close(incremental, full, atol=1e-9, rtol=1e-9)


def test_prefill_then_decode_matches_full_recompute() -> None:
    cfg = make_config()
    model = build_model(cfg).double().eval()

    t, prefill = 14, 9
    ids = torch.randint(0, cfg.vocab_size, (1, t))
    full = model(ids).logits

    cache = model.make_cache(1)
    chunks = [model(ids[:, :prefill], cache=cache).logits]
    for i in range(prefill, t):
        chunks.append(model(ids[:, i : i + 1], cache=cache).logits)

    torch.testing.assert_close(torch.cat(chunks, dim=1), full, atol=1e-9, rtol=1e-9)


def test_generation_with_and_without_cache_is_identical() -> None:
    """Greedy decoding must not depend on whether the cache is used."""
    cfg = make_config()
    model = build_model(cfg).double().eval()
    prompt = torch.randint(0, cfg.vocab_size, (2, 4))

    cached = model.generate(prompt, max_new_tokens=8, temperature=0.0, use_cache=True)
    uncached = model.generate(prompt, max_new_tokens=8, temperature=0.0, use_cache=False)
    assert torch.equal(cached, uncached)


def test_cache_can_be_reused_after_reset() -> None:
    cfg = make_config()
    model = build_model(cfg).double().eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6))

    cache = model.make_cache(1)
    first = model(ids, cache=cache).logits
    cache.reset()
    second = model(ids, cache=cache).logits
    torch.testing.assert_close(first, second, atol=1e-12, rtol=1e-12)


def test_truncate_rolls_back_to_an_earlier_prefix() -> None:
    """The rollback speculative decoding depends on."""
    cfg = make_config()
    model = build_model(cfg).double().eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 10))

    reference = model(ids).logits

    cache = model.make_cache(1)
    model(ids[:, :6], cache=cache)
    # Speculatively consume tokens that then get rejected.
    model(torch.randint(0, cfg.vocab_size, (1, 3)), cache=cache)
    cache.truncate(6)
    rest = model(ids[:, 6:], cache=cache).logits

    torch.testing.assert_close(rest, reference[:, 6:], atol=1e-9, rtol=1e-9)
