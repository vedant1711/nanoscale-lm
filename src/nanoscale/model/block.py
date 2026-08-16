"""The transformer block: pre-norm residual attention + MLP (spec B2).

.. code-block:: text

    x -> x + Attn(Norm(x))
      -> x + MLP(Norm(x))

**Pre-norm** (normalize the branch input, not the residual output) is what makes deep
stacks trainable without a warmup-dependent knife-edge: the residual stream is an
identity path from embedding to output, so gradients reach layer 0 undamped. Post-norm
was the original Transformer's choice and is measurably harder to train at depth.
"""

from __future__ import annotations

import torch
from torch import nn

from nanoscale.config import ModelConfig
from nanoscale.model.attention import CausalSelfAttention
from nanoscale.model.kv_cache import LayerKVCache
from nanoscale.model.mlp import build_mlp
from nanoscale.model.norm import build_norm

__all__ = ["TransformerBlock"]


class TransformerBlock(nn.Module):
    """One pre-norm transformer block."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        """Build the attention and MLP sublayers with their pre-norms."""
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = build_norm(config.norm_type, config.d_model, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = build_norm(config.norm_type, config.d_model, eps=config.norm_eps)
        self.mlp = build_mlp(
            config.mlp_type, config.d_model, config.ffn_dim, dropout=config.dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        cache: LayerKVCache | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply attention and MLP with residual connections."""
        x = x + self.attn(self.attn_norm(x), cos, sin, cache=cache, attn_mask=attn_mask)
        mlp_out: torch.Tensor = self.mlp(self.mlp_norm(x))
        return x + mlp_out
