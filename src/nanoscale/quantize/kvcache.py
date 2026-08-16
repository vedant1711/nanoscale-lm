"""KV-cache quantization (spec B7).

Why the KV cache, and not just the weights
-------------------------------------------
Weight-only quantization shrinks a fixed cost. The KV cache is a cost that **grows with
context**, and past a few thousand tokens it dominates: for the ``micro`` tier at 4k
context, the cache is larger than the entire model. Decoding is memory-bandwidth-bound,
so halving the bytes read per step is close to halving decode latency — which is why
KV-cache quantization is the one compression technique whose benefit *increases* with
sequence length.

Per-token, per-head grouping
-----------------------------
The K and V tensors are quantized along ``head_dim`` in groups, with scales stored per
``(batch, head, position, group)``. This grouping is the right one because the thing
that varies wildly is the *position*: a single token with an outlier activation would
otherwise set the scale for every position in the cache. Quantizing per position
confines that damage to the token that caused it.

Asymmetry between K and V
--------------------------
K and V are not equally sensitive. Keys go through a dot product and then a softmax,
which amplifies error: a perturbed key can reorder the attention weights. Values are
combined linearly by weights that already sum to one, so their errors average out. The
literature (KIVI, KVQuant) exploits this by quantizing V more aggressively than K.
:class:`QuantizedKVCache` supports separate bit-widths for exactly that reason, and
``tests/unit/test_quantize.py`` measures the sensitivity difference rather than assuming
it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from nanoscale.model.kv_cache import KVCache, LayerKVCache

__all__ = [
    "QuantizedKVCache",
    "dequantize_kv",
    "kv_cache_memory_report",
    "quantize_kv",
]


@dataclass(frozen=True, slots=True)
class QuantizedKV:
    """Quantized K or V with its per-position, per-group scales."""

    codes: Tensor  # (B, H, T, D) integer levels
    scales: Tensor  # (B, H, T, n_groups)
    zeros: Tensor  # (B, H, T, n_groups)
    bits: int
    group_size: int


def quantize_kv(x: Tensor, *, bits: int = 4, group_size: int = 32) -> QuantizedKV:
    """Quantize a ``(B, H, T, D)`` K or V tensor along ``head_dim``, per position.

    Args:
        x: The tensor to quantize.
        bits: Bit-width (2-8).
        group_size: Elements per scale group along ``head_dim``.
    """
    if x.ndim != 4:
        raise ValueError(f"expected (B, H, T, D), got {tuple(x.shape)}.")
    b, h, t, d = x.shape
    group = d if group_size in (-1, 0) else min(group_size, d)
    if d % group != 0:
        raise ValueError(f"head_dim={d} is not divisible by group_size={group}.")

    qmax = 2**bits - 1
    grouped = x.detach().float().reshape(b, h, t, d // group, group)
    lo = grouped.amin(dim=-1, keepdim=True)
    hi = grouped.amax(dim=-1, keepdim=True)
    scales = ((hi - lo) / qmax).clamp_min(1e-8)
    zeros = torch.round(-lo / scales)
    codes = torch.clamp(torch.round(grouped / scales) + zeros, 0, qmax)

    return QuantizedKV(
        codes=codes.to(torch.uint8 if bits <= 8 else torch.int32).reshape(b, h, t, d),
        scales=scales.squeeze(-1),
        zeros=zeros.squeeze(-1),
        bits=bits,
        group_size=group,
    )


def dequantize_kv(q: QuantizedKV, *, dtype: torch.dtype = torch.float32) -> Tensor:
    """Invert :func:`quantize_kv`."""
    b, h, t, d = q.codes.shape
    n_groups = d // q.group_size
    codes = q.codes.float().reshape(b, h, t, n_groups, q.group_size)
    out = (codes - q.zeros.unsqueeze(-1)) * q.scales.unsqueeze(-1)
    return out.reshape(b, h, t, d).to(dtype)


class QuantizedKVCache(KVCache):
    """A KV cache that stores quantized keys/values and dequantizes on read.

    This is a *simulation* of the memory win, not a kernel-level implementation: PyTorch
    has no int4 matmul on CPU, so the cache stores uint8 codes and dequantizes to fp32
    when attention reads it. That is honest about what is being measured — the
    **accuracy cost** of quantized KV is exactly real, and the **memory footprint** is
    computed analytically by :meth:`memory_bytes` rather than measured from an
    allocator. The latency win a real int4 kernel would deliver is *not* claimed here;
    the docs say so.
    """

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        key_bits: int = 4,
        value_bits: int = 4,
        group_size: int = 32,
    ) -> None:
        """Allocate a cache whose stored keys/values are quantized on write."""
        super().__init__(
            n_layers,
            batch_size,
            n_kv_heads,
            head_dim,
            max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.group_size = group_size
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.compute_dtype = dtype
        # Replace each layer's storage with a quantizing wrapper.
        self.layers = [
            _QuantizedLayerCache(
                layer,
                key_bits=key_bits,
                value_bits=value_bits,
                group_size=group_size,
                dtype=dtype,
            )
            for layer in self.layers
        ]

    def memory_bytes(self) -> int:
        """Analytic footprint of the *quantized* representation.

        Counted as ``codes + scales + zeros`` at their true bit-widths, so a "4-bit"
        cache with group size 32 and fp16 scale/zero is charged 4 + 32/32 = 5 bits per
        element, not 4. Same accounting rule as the weight-quantization frontier.
        """
        per_element_key = self.key_bits + 2 * 16 / self.group_size
        per_element_value = self.value_bits + 2 * 16 / self.group_size
        elements = self.n_layers * self.batch_size * self.n_kv_heads * self.max_seq_len
        elements *= self.head_dim
        return int(elements * (per_element_key + per_element_value) / 8)


class _QuantizedLayerCache(LayerKVCache):
    """One layer's cache, quantizing on append and dequantizing on read."""

    def __init__(
        self,
        base: LayerKVCache,
        *,
        key_bits: int,
        value_bits: int,
        group_size: int,
        dtype: torch.dtype,
    ) -> None:
        """Wrap an existing float layer cache."""
        super().__init__(keys=base.keys, values=base.values, length=base.length)
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.group_size = group_size
        self.compute_dtype = dtype

    def append(self, new_keys: Tensor, new_values: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize then immediately dequantize, then store — simulating lossy storage.

        Round-tripping through the quantizer on write is what makes the *accuracy* cost
        real while keeping the tensors in a dtype PyTorch can actually attend over.
        """
        qk = dequantize_kv(
            quantize_kv(new_keys, bits=self.key_bits, group_size=self.group_size),
            dtype=new_keys.dtype,
        )
        qv = dequantize_kv(
            quantize_kv(new_values, bits=self.value_bits, group_size=self.group_size),
            dtype=new_values.dtype,
        )
        return super().append(qk, qv)


def kv_cache_memory_report(
    *,
    n_layers: int,
    batch_size: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    key_bits: int = 4,
    value_bits: int = 4,
    group_size: int = 32,
    baseline_bits: int = 16,
) -> dict[str, float]:
    """Analytic KV-cache footprint at fp16 vs the quantized configuration."""
    elements = n_layers * batch_size * n_kv_heads * seq_len * head_dim
    baseline = elements * 2 * baseline_bits / 8
    overhead = 2 * 16 / group_size
    quantized = elements * ((key_bits + overhead) + (value_bits + overhead)) / 8
    return {
        "seq_len": float(seq_len),
        "baseline_mb": baseline / 1024**2,
        "quantized_mb": quantized / 1024**2,
        "compression": baseline / max(1.0, quantized),
        "effective_bits_per_element": (key_bits + value_bits) / 2 + overhead,
    }
