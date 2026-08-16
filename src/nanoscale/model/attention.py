r"""Causal self-attention with GQA, RoPE, QK-norm and a KV cache (spec B2).

This module implements attention from first principles.
``F.scaled_dot_product_attention`` is available as an optional fast path *behind* the
same interface, and
``tests/unit/test_attention.py`` asserts the two agree numerically on every
configuration (causal, GQA, RoPE, cached decode). The manual path is the reference; the
SDPA path is an optimisation that has to prove it did not change the answer.

The pieces
----------
**Scaled dot-product attention** (Vaswani et al., arXiv:1706.03762):

.. math::  \mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right)V

with ``M`` the causal mask (``0`` where attention is allowed, ``-inf`` otherwise). The
``1/√d_h`` scaling keeps the logits' variance independent of head width, without which
the softmax saturates as ``d_h`` grows.

**Grouped-query attention** (Ainslie et al., arXiv:2305.13245) gives each *group* of
query heads one shared key/value head. With ``n_heads=8, n_kv_heads=4`` the KV cache
halves, which at decode time — where the bottleneck is memory bandwidth, not FLOPs — is
close to a 2x throughput win for a quality cost that is small at this scale. MQA
(``n_kv_heads=1``) and MHA (``n_kv_heads=n_heads``) are the endpoints of the same knob.

**QK-norm** RMS-normalizes queries and keys before the dot product. This bounds the
logit magnitudes entering the softmax regardless of how large the q/k activations grow,
which is the failure mode that produces attention-entropy collapse and loss spikes in
small-model training. It is part of the modded-nanoGPT speedrun stack and is on by
default here, with an ablation flag so Phase 5 can measure it.

**Ordering.** QK-norm is applied *before* RoPE. RoPE is norm-preserving, so the two
commute as far as the normalization is concerned — but the learned QK gain does not
commute with the rotation, so the order is a real choice and is pinned here and by test.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from nanoscale.config import ModelConfig
from nanoscale.model.kv_cache import LayerKVCache
from nanoscale.model.norm import RMSNorm
from nanoscale.model.numerics import to_accumulation
from nanoscale.model.rope import apply_rope

__all__ = ["CausalSelfAttention", "repeat_kv"]


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand ``n_kv_heads`` to ``n_heads`` by repeating each KV head ``n_rep`` times.

    Args:
        x: ``(B, n_kv_heads, T, D)``.
        n_rep: Number of query heads per KV head.

    Returns:
        ``(B, n_kv_heads * n_rep, T, D)``, with head ``i`` of the input mapped to heads
        ``[i*n_rep, (i+1)*n_rep)`` of the output — the grouping the GQA paper specifies.
    """
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    return x[:, :, None].expand(b, n_kv, n_rep, t, d).reshape(b, n_kv * n_rep, t, d)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with GQA, RoPE, QK-norm and KV caching."""

    def __init__(self, config: ModelConfig) -> None:
        """Build the q/k/v/o projections for one attention layer."""
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_groups = config.n_kv_groups
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.d_model, bias=False)

        if config.qk_norm:
            self.q_norm: nn.Module = RMSNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm: nn.Module = RMSNorm(self.head_dim, eps=config.norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.attn_dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    # ------------------------------------------------------------------ projections

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project ``x`` to per-head q, k, v of shape ``(B, H, T, D)``."""
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    # --------------------------------------------------------------- attention cores

    def _attend_manual(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """The from-scratch attention core: scores, mask, softmax, weighted sum.

        The softmax is computed at fp32 or better regardless of the input dtype. Under
        bf16 the exponentials of attention logits lose enough precision to shift the
        argmax on near-ties, which shows up as nondeterministic generations.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores + mask
        weights = torch.softmax(to_accumulation(scores), dim=-1).to(q.dtype)
        if self.attn_dropout_p > 0.0 and self.training:
            weights = F.dropout(weights, p=self.attn_dropout_p)
        return torch.matmul(weights, v)

    def _attend_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        """The fused fast path. Must match :meth:`_attend_manual` within tolerance."""
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=is_causal and mask is None,
            scale=self.scale,
        )

    # ------------------------------------------------------------------------ masks

    @staticmethod
    def build_causal_mask(
        q_len: int,
        kv_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Additive causal mask of shape ``(1, 1, q_len, kv_len)``.

        The queries are assumed to occupy the *last* ``q_len`` positions of the ``kv_len``
        key positions, which is exactly the situation during cached decoding: a query at
        cache-relative position ``kv_len - q_len + i`` may attend to keys ``0 … that``.
        """
        offset = kv_len - q_len
        q_pos = torch.arange(q_len, device=device).unsqueeze(1) + offset
        k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
        allowed = k_pos <= q_pos
        mask = torch.zeros(q_len, kv_len, device=device, dtype=dtype)
        mask.masked_fill_(~allowed, torch.finfo(dtype).min)
        return mask.view(1, 1, q_len, kv_len)

    # ---------------------------------------------------------------------- forward

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        cache: LayerKVCache | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run causal self-attention over ``x``.

        Args:
            x: ``(B, T, d_model)`` input activations.
            cos: RoPE cosine table for the *query* positions, ``(T, head_dim//2)``.
            sin: Matching sine table.
            cache: If given, new keys/values are appended and attention runs over the
                whole cached prefix. This is the decode path.
            attn_mask: Optional additive mask, broadcastable to ``(B, H, T, T_kv)``.
                Supply this for padding; causality is handled internally.

        Returns:
            ``(B, T, d_model)``.
        """
        b, t, _ = x.shape
        q, k, v = self._project(x)

        # QK-norm before RoPE: see the module docstring on why the order is a real choice.
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.append(k, v)

        kv_len = k.shape[2]
        k = repeat_kv(k, self.n_kv_groups)
        v = repeat_kv(v, self.n_kv_groups)

        # A single query attending to a full cached prefix needs no mask at all: every
        # cached position is by construction in the past.
        needs_causal = t > 1
        mask = attn_mask
        if needs_causal:
            causal = self.build_causal_mask(t, kv_len, device=x.device, dtype=q.dtype)
            mask = causal if mask is None else mask + causal

        if self.config.attn_impl == "sdpa":
            out = self._attend_sdpa(q, k, v, mask, is_causal=needs_causal)
        else:
            out = self._attend_manual(q, k, v, mask)

        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        projected: torch.Tensor = self.resid_dropout(self.o_proj(out))
        return projected

    @property
    def output_projection(self) -> nn.Linear:
        """The residual-writing projection (zero-initialised when configured)."""
        return self.o_proj

    def extra_repr(self) -> str:
        """Describe the layer for ``print(model)``."""
        return (
            f"n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}, "
            f"head_dim={self.head_dim}, qk_norm={self.config.qk_norm}, "
            f"impl={self.config.attn_impl}"
        )
