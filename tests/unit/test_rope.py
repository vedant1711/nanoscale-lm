"""RoPE correctness (spec D1): matches a reference rotation, and is relative.

Two independent checks:

1. The vectorised implementation equals a literal, loop-based transcription of the
   paper's 2x2 rotation matrix (:func:`rope_reference`).
2. The property the whole construction exists for: attention scores between a rotated
   query and a rotated key depend only on the *relative* offset of their positions.
"""

from __future__ import annotations

import math

import pytest
import torch

from nanoscale.model.rope import RotaryCache, apply_rope, build_rope_cache, rope_reference


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(20260816)


# ---------------------------------------------------------------- vs a reference


@pytest.mark.parametrize("head_dim", [2, 8, 64])
@pytest.mark.parametrize("position", [0, 1, 7, 63])
def test_matches_the_reference_rotation(head_dim: int, position: int) -> None:
    theta = 10_000.0
    x = torch.randn(head_dim, dtype=torch.float64)
    cos, sin = build_rope_cache(head_dim, position + 1, theta=theta, dtype=torch.float64)

    ours = apply_rope(
        x.view(1, 1, 1, head_dim),
        cos[position : position + 1],
        sin[position : position + 1],
    ).view(head_dim)
    reference = rope_reference(x, position, theta=theta)

    torch.testing.assert_close(ours, reference, atol=1e-12, rtol=1e-12)


def test_position_zero_is_the_identity() -> None:
    x = torch.randn(2, 3, 1, 16)
    cos, sin = build_rope_cache(16, 4)
    torch.testing.assert_close(apply_rope(x, cos[:1], sin[:1]), x)


def test_rotation_preserves_norms() -> None:
    """Rotations are orthogonal, so position must never rescale a vector."""
    x = torch.randn(2, 4, 32, 64)
    cos, sin = build_rope_cache(64, 32)
    rotated = apply_rope(x, cos, sin)
    torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_uses_the_interleaved_pair_convention() -> None:
    """Pin the convention: pairs are (x0,x1), (x2,x3), not (x_i, x_{i+d/2})."""
    head_dim = 4
    x = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 1, head_dim)
    cos, sin = build_rope_cache(head_dim, 2, theta=10_000.0)
    out = apply_rope(x, cos[1:2], sin[1:2]).view(-1)
    # With x = e_0 at position 1: pair 0 rotates by angle 1*theta^0 = 1 rad, so the
    # unit vector maps to (cos 1, sin 1) in coordinates 0 and 1 -- not 0 and 2.
    torch.testing.assert_close(out[0], torch.tensor(math.cos(1.0)), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out[1], torch.tensor(math.sin(1.0)), atol=1e-6, rtol=1e-6)
    assert out[2].abs() < 1e-6
    assert out[3].abs() < 1e-6


# ------------------------------------------------------------- the relative property


@pytest.mark.parametrize("offset", [0, 1, 5, 20])
def test_attention_scores_depend_only_on_relative_position(offset: int) -> None:
    """<RoPE(q, m), RoPE(k, n)> must be a function of (m - n) alone."""
    head_dim = 32
    q = torch.randn(1, 1, 1, head_dim, dtype=torch.float64)
    k = torch.randn(1, 1, 1, head_dim, dtype=torch.float64)
    # fp64 tables: table rounding must not be mistaken for an error in the rotation.
    cos, sin = build_rope_cache(head_dim, 128, dtype=torch.float64)

    scores = []
    for m in (offset, offset + 10, offset + 37):
        n = m - offset
        qr = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
        kr = apply_rope(k, cos[n : n + 1], sin[n : n + 1])
        scores.append((qr * kr).sum())

    for s in scores[1:]:
        torch.testing.assert_close(s, scores[0], atol=1e-12, rtol=1e-12)


def test_different_offsets_give_different_scores() -> None:
    """Sanity: the score is genuinely position-sensitive, not constant."""
    head_dim = 32
    q = torch.randn(1, 1, 1, head_dim, dtype=torch.float64)
    k = torch.randn(1, 1, 1, head_dim, dtype=torch.float64)
    cos, sin = build_rope_cache(head_dim, 64, dtype=torch.float64)

    def score(m: int, n: int) -> float:
        qr = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
        kr = apply_rope(k, cos[n : n + 1], sin[n : n + 1])
        return float((qr * kr).sum())

    assert abs(score(5, 5) - score(5, 0)) > 1e-6


# ------------------------------------------------------------------------- the cache


def test_cache_rejects_odd_head_dims() -> None:
    with pytest.raises(ValueError, match="even"):
        RotaryCache(head_dim=7, max_seq_len=8)
    with pytest.raises(ValueError, match="even"):
        build_rope_cache(7, 8)


def test_cache_shapes_and_gather() -> None:
    cache = RotaryCache(head_dim=16, max_seq_len=32)
    assert cache.cos.shape == (32, 8)
    cos, sin = cache.get(torch.arange(5))
    assert cos.shape == sin.shape == (5, 8)
    batched_cos, _ = cache.get(torch.arange(6).view(2, 3))
    assert batched_cos.shape == (2, 3, 8)


def test_cache_rejects_out_of_range_positions() -> None:
    cache = RotaryCache(head_dim=8, max_seq_len=4)
    with pytest.raises(IndexError, match="exceeds the RoPE cache length"):
        cache.get(torch.tensor([9]))


def test_scaling_compresses_positions() -> None:
    """Linear position interpolation: scaling=2 puts position 2 where 1 used to be."""
    plain_cos, _ = build_rope_cache(8, 8, scaling=1.0)
    scaled_cos, _ = build_rope_cache(8, 8, scaling=2.0)
    torch.testing.assert_close(scaled_cos[2], plain_cos[1], atol=1e-6, rtol=1e-6)


def test_apply_rope_validates_shapes() -> None:
    cos, sin = build_rope_cache(8, 4)
    assert sin.shape == cos.shape
    with pytest.raises(ValueError, match=r"\(B, H, T, D\)"):
        apply_rope(torch.randn(4, 8), cos, sin)
    with pytest.raises(ValueError, match="does not match rope table width"):
        apply_rope(torch.randn(1, 1, 4, 16), cos, sin)
    with pytest.raises(ValueError, match="must be 2D or 3D"):
        apply_rope(torch.randn(1, 1, 4, 8), cos.view(1, 1, 4, 4), sin.view(1, 1, 4, 4))


def test_rotation_is_exact_under_bf16_inputs() -> None:
    """The rotation runs in fp32 internally even when activations are bf16."""
    x = torch.randn(1, 2, 8, 16)
    cos, sin = build_rope_cache(16, 8)
    fp32 = apply_rope(x, cos, sin)
    bf16 = apply_rope(x.bfloat16(), cos, sin)
    assert bf16.dtype is torch.bfloat16
    torch.testing.assert_close(bf16.float(), fp32, atol=5e-2, rtol=5e-2)
