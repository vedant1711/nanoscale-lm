"""KV cache for incremental decoding (spec B2).

Why it exists
-------------
Autoregressive decoding without a cache recomputes the keys and values for every prefix
token at every step: generating ``n`` tokens costs ``O(n²)`` attention work *and*
``O(n²)`` projection work. With a cache, the projections for a step are ``O(1)`` in the
prefix length and only the attention scores scale with it. This is the single largest
constant-factor win in inference, and it is also what makes the cache the memory
bottleneck that Phase 8's KV-cache quantization attacks.

Correctness contract
--------------------
Cached incremental decoding must produce **token-for-token identical** logits to a full
recompute over the same sequence. That is tested directly in
``tests/unit/test_kv_cache.py``; a cache that is subtly wrong is otherwise very easy to
ship, because the outputs still look like text.

Layout
------
Storage is preallocated to ``(batch, n_kv_heads, max_seq_len, head_dim)`` per layer, so
decoding never reallocates. ``n_kv_heads`` rather than ``n_heads`` is exactly the GQA
memory win: with 8 query heads sharing 4 KV heads, the cache is half the size.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["KVCache", "LayerKVCache"]


@dataclass(slots=True)
class LayerKVCache:
    """Preallocated key/value storage for one attention layer."""

    keys: torch.Tensor  # (B, n_kv_heads, max_seq_len, head_dim)
    values: torch.Tensor
    length: int = 0

    def append(
        self, new_keys: torch.Tensor, new_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new keys/values and return views over the whole cached prefix.

        Args:
            new_keys: ``(B, n_kv_heads, T_new, head_dim)``.
            new_values: Same shape.

        Returns:
            ``(keys, values)`` covering positions ``[0, length)`` after the append.

        Raises:
            ValueError: If the append would overflow the preallocated context.
        """
        t_new = new_keys.shape[2]
        end = self.length + t_new
        if end > self.keys.shape[2]:
            raise ValueError(
                f"KV cache overflow: {self.length} cached + {t_new} new > "
                f"{self.keys.shape[2]} capacity. Increase model.max_seq_len."
            )
        self.keys[:, :, self.length : end] = new_keys.to(self.keys.dtype)
        self.values[:, :, self.length : end] = new_values.to(self.values.dtype)
        self.length = end
        return self.keys[:, :, :end], self.values[:, :, :end]

    def reset(self) -> None:
        """Forget the cached prefix without freeing the storage."""
        self.length = 0

    @property
    def capacity(self) -> int:
        """Maximum number of cached positions."""
        return int(self.keys.shape[2])


class KVCache:
    """A per-layer KV cache for one batch of sequences.

    Args:
        n_layers: Number of transformer layers.
        batch_size: Batch size the cache is sized for.
        n_kv_heads: Number of key/value heads (GQA: may be fewer than query heads).
        head_dim: Per-head dimension.
        max_seq_len: Cache capacity in tokens.
        device: Storage device.
        dtype: Storage dtype. Phase 8 exploits this to store quantized K/V.
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
    ) -> None:
        """Preallocate storage for every layer."""
        shape = (batch_size, n_kv_heads, max_seq_len, head_dim)
        self.layers: list[LayerKVCache] = [
            LayerKVCache(
                keys=torch.zeros(shape, device=device, dtype=dtype),
                values=torch.zeros(shape, device=device, dtype=dtype),
            )
            for _ in range(n_layers)
        ]
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def __getitem__(self, layer: int) -> LayerKVCache:
        """Return the cache for one layer."""
        return self.layers[layer]

    def __len__(self) -> int:
        """Number of cached layers."""
        return self.n_layers

    @property
    def length(self) -> int:
        """Number of cached positions (layer 0 is authoritative; all layers agree)."""
        return self.layers[0].length if self.layers else 0

    def reset(self) -> None:
        """Forget every cached prefix."""
        for layer in self.layers:
            layer.reset()

    def memory_bytes(self) -> int:
        """Total bytes of *allocated* cache storage.

        Reported by the benchmark harness: at long context this dominates the memory
        footprint of a small model, which is the entire motivation for KV-cache
        quantization in Phase 8.
        """
        total = 0
        for layer in self.layers:
            total += layer.keys.numel() * layer.keys.element_size()
            total += layer.values.numel() * layer.values.element_size()
        return total

    def used_bytes(self) -> int:
        """Bytes corresponding to the positions actually filled."""
        if not self.layers:
            return 0
        fraction = self.length / max(1, self.max_seq_len)
        return int(self.memory_bytes() * fraction)

    def clone(self) -> KVCache:
        """Deep-copy the cache.

        Speculative decoding needs this: the draft model advances a cache
        speculatively, and rejected tokens have to be rolled back.
        """
        copy = object.__new__(KVCache)
        copy.layers = [
            LayerKVCache(keys=lyr.keys.clone(), values=lyr.values.clone(), length=lyr.length)
            for lyr in self.layers
        ]
        copy.n_layers = self.n_layers
        copy.batch_size = self.batch_size
        copy.max_seq_len = self.max_seq_len
        return copy

    def truncate(self, length: int) -> None:
        """Roll the cache back to ``length`` positions.

        Used by speculative decoding to discard rejected draft tokens without
        recomputing the accepted prefix.
        """
        if length < 0 or length > self.max_seq_len:
            raise ValueError(f"Cannot truncate to {length}; capacity is {self.max_seq_len}.")
        for layer in self.layers:
            layer.length = min(layer.length, length)
