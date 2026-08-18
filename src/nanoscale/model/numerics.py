"""Precision helpers for the reduction-heavy parts of the model.

Several operations: the RMS reduction inside a norm, the attention softmax, the RoPE
rotation, are computed at higher precision than the surrounding activations, because
a few ULPs of error there compound in ways that show up as loss spikes or
nondeterministic generations under bf16 autocast.

The subtlety this module exists for: the naive way to write that is ``x.float()``, which
*promotes* bf16/fp16 as intended but silently **demotes** float64. That breaks the
double-precision reference tests that give the numerical-correctness suite its teeth,
and it would quietly discard precision for anyone running an fp64 experiment. So the
rule is "promote to at least fp32, never demote".
"""

from __future__ import annotations

import torch

__all__ = ["accumulation_dtype", "to_accumulation"]


def accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the dtype to accumulate in: at least fp32, never lower than the input."""
    if dtype in (torch.float64, torch.complex128):
        return dtype
    return torch.float32


def to_accumulation(x: torch.Tensor) -> torch.Tensor:
    """Cast ``x`` up to its accumulation dtype (a no-op when already there)."""
    return x.to(accumulation_dtype(x.dtype))
