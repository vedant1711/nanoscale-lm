"""The NanoScale-LM decoder-only transformer, implemented from scratch."""

from __future__ import annotations

from nanoscale.model.attention import CausalSelfAttention, repeat_kv
from nanoscale.model.block import TransformerBlock
from nanoscale.model.kv_cache import KVCache, LayerKVCache
from nanoscale.model.lm import (
    IGNORE_INDEX,
    LMOutput,
    NanoScaleLM,
    build_model,
    sample_next_token,
)
from nanoscale.model.mlp import ReLU2MLP, SwiGLU, build_mlp
from nanoscale.model.mtp import MTPHead, MultiTokenPredictionHeads
from nanoscale.model.norm import LayerNorm, RMSNorm, build_norm, rms_normalize
from nanoscale.model.rope import RotaryCache, apply_rope, build_rope_cache, rope_reference

__all__ = [
    "IGNORE_INDEX",
    "CausalSelfAttention",
    "KVCache",
    "LMOutput",
    "LayerKVCache",
    "LayerNorm",
    "MTPHead",
    "MultiTokenPredictionHeads",
    "NanoScaleLM",
    "RMSNorm",
    "ReLU2MLP",
    "RotaryCache",
    "SwiGLU",
    "TransformerBlock",
    "apply_rope",
    "build_mlp",
    "build_model",
    "build_norm",
    "build_rope_cache",
    "repeat_kv",
    "rms_normalize",
    "rope_reference",
    "sample_next_token",
]
