r"""Rotary Position Embeddings, implemented directly from the paper (spec B2).

Reference: Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*
(arXiv:2104.09864).

The construction
----------------
Split each head vector into ``d/2`` consecutive coordinate pairs and treat each pair as
a point in the plane. At position ``m``, rotate pair ``i`` by angle ``m * θ_i`` where

.. math::  θ_i = base^{-2i/d},  i = 0 … d/2-1

so each pair rotates at its own frequency, geometrically spaced from 1 radian per step
down to ``base^{-1}``. Written as a matrix, pair ``i`` of ``x`` at position ``m`` maps to

.. math::

    \begin{pmatrix} \cos mθ_i & -\sin mθ_i \\
                     \sin mθ_i & \cos mθ_i \end{pmatrix}
    \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}

Why it works
------------
Rotations are orthogonal, so ``‖RoPE(x, m)‖ = ‖x‖``: position never changes a vector's
magnitude. And because rotations compose additively,

.. math::  \langle RoPE(q, m), RoPE(k, n) \rangle = f(q, k, m - n)

the attention score between a query at ``m`` and a key at ``n`` depends only on their
**relative** offset. That is the whole point, and it is tested directly in
``tests/unit/test_rope.py`` rather than asserted.

Implementation notes
--------------------
* This module uses the paper's *interleaved-pair* convention (pairs are ``(x0,x1)``,
  ``(x2,x3)``, …). LLaMA and Hugging Face use a "rotate-half" convention that pairs
  ``x_i`` with ``x_{i+d/2}``. The two are related by a fixed permutation of the head
  dimension and are mathematically equivalent; a model trained under either is
  identical up to relabelling its head coordinates, but they are **not**
  interchangeable at the level of loaded weights, so the convention is stated here and
  pinned by a test.
* The cache is built once in fp32 and reused. Rotation is applied at fp32 or better
  even under bf16 autocast: the angles are the one place in the model where a few ULPs
  of error compound across positions. (Promotion never *demotes* an fp64 input -- see
  :mod:`nanoscale.model.numerics`.)
"""

from __future__ import annotations

import torch
from torch import nn

from nanoscale.model.numerics import accumulation_dtype

__all__ = ["RotaryCache", "apply_rope", "build_rope_cache", "rope_reference"]


class RotaryCache(nn.Module):
    """Precomputed ``cos``/``sin`` tables of shape ``(max_seq_len, head_dim // 2)``.

    Registered as an ``nn.Module`` with **non-persistent** buffers so that
    ``model.to(device)`` and ``model.double()`` move the tables automatically, while
    checkpoints stay free of a few thousand floats that are cheaper to recompute than
    to store.
    """

    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        *,
        theta: float = 10_000.0,
        scaling: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        """Build the rotation tables for a given head dimension and context length."""
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE's pair rotation, got {head_dim}.")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scaling = scaling
        cos, sin = build_rope_cache(
            head_dim, max_seq_len, theta=theta, scaling=scaling, device=device
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def get(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather the rotation tables for arbitrary absolute positions.

        Args:
            positions: Integer tensor of shape ``(T,)`` or ``(B, T)``.

        Returns:
            ``(cos, sin)``, each shaped like ``positions`` with a trailing
            ``head_dim // 2`` axis.
        """
        if positions.numel() and int(positions.max()) >= self.max_seq_len:
            raise IndexError(
                f"position {int(positions.max())} exceeds the RoPE cache length "
                f"{self.max_seq_len}; increase model.max_seq_len."
            )
        return self.cos[positions], self.sin[positions]


def build_rope_cache(
    head_dim: int,
    max_seq_len: int,
    *,
    theta: float = 10_000.0,
    scaling: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(cos, sin)`` tables of shape ``(max_seq_len, head_dim // 2)``.

    Args:
        head_dim: Per-head dimension; must be even.
        max_seq_len: Number of positions to precompute.
        theta: Base frequency (``10000`` in the paper).
        scaling: Linear position-interpolation factor. ``scaling > 1`` compresses
            positions, which is the simplest way to extend context beyond training
            length (Chen et al., "Extending Context Window via Position Interpolation").
        device: Device for the tables.
        dtype: Precision of the tables. fp32 is right for training and inference; the
            correctness tests build them in fp64 so that table rounding cannot be
            mistaken for an error in the rotation itself.

    Returns:
        A ``(cos, sin)`` pair.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE's pair rotation, got {head_dim}.")
    # θ_i = theta^{-2i/d} for i = 0 .. d/2 - 1
    exponent = torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim
    inv_freq = 1.0 / (theta**exponent)
    positions = torch.arange(max_seq_len, device=device, dtype=dtype) / scaling
    angles = torch.outer(positions, inv_freq)  # (T, d/2)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` by the given angles, using the paper's interleaved-pair convention.

    Args:
        x: ``(B, H, T, D)`` queries or keys.
        cos: ``(T, D//2)`` or ``(B, T, D//2)`` cosine table.
        sin: Matching sine table.

    Returns:
        A tensor shaped like ``x``, rotated position-wise.

    The rotation itself is done at fp32 or better and cast back, so the angles stay
    exact under bf16/fp16 autocast.
    """
    if x.ndim != 4:
        raise ValueError(f"apply_rope expects (B, H, T, D), got shape {tuple(x.shape)}.")
    b, h, t, d = x.shape
    if d != 2 * cos.shape[-1]:
        raise ValueError(f"head_dim {d} does not match rope table width {cos.shape[-1]} * 2.")

    orig_dtype = x.dtype
    acc = accumulation_dtype(orig_dtype)
    pairs = x.to(acc).reshape(b, h, t, d // 2, 2)
    x_even = pairs[..., 0]
    x_odd = pairs[..., 1]

    if cos.ndim == 2:  # (T, D/2) shared across the batch
        cos_b = cos.reshape(1, 1, t, d // 2).to(acc)
        sin_b = sin.reshape(1, 1, t, d // 2).to(acc)
    elif cos.ndim == 3:  # (B, T, D/2) per-sequence positions
        cos_b = cos.unsqueeze(1).to(acc)
        sin_b = sin.unsqueeze(1).to(acc)
    else:
        raise ValueError(f"rope tables must be 2D or 3D, got {cos.ndim}D.")

    rotated_even = x_even * cos_b - x_odd * sin_b
    rotated_odd = x_even * sin_b + x_odd * cos_b
    out = torch.stack((rotated_even, rotated_odd), dim=-1).reshape(b, h, t, d)
    return out.to(orig_dtype)


def rope_reference(x: torch.Tensor, position: int, *, theta: float = 10_000.0) -> torch.Tensor:
    """A deliberately slow, literal transcription of the paper's rotation.

    Used only by tests as an independent reference for :func:`apply_rope`: it builds the
    2x2 rotation matrix for each coordinate pair and applies it with an explicit loop,
    with no broadcasting, caching or vectorisation to hide a mistake in.

    Args:
        x: A single head vector of shape ``(D,)``.
        position: The absolute position ``m``.
        theta: Base frequency.

    Returns:
        The rotated vector, shape ``(D,)``.
    """
    d = x.shape[-1]
    out = torch.empty_like(x, dtype=torch.float64)
    xf = x.double()
    for i in range(d // 2):
        angle = position * (theta ** (-2.0 * i / d))
        c, s = (
            torch.cos(torch.tensor(angle, dtype=torch.float64)),
            torch.sin(torch.tensor(angle, dtype=torch.float64)),
        )
        a, b = xf[2 * i], xf[2 * i + 1]
        out[2 * i] = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
