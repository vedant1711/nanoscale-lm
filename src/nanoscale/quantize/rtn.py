r"""Round-to-nearest weight quantization — the baseline everything else must beat.

The primitive
-------------
Uniform affine quantization maps a group of weights to ``2^b`` levels:

.. math::

   s = \frac{\max(w) - \min(w)}{2^b - 1}, \qquad z = \mathrm{round}\!\left(\frac{-\min(w)}{s}\right)

   \hat{w} = s \cdot \big(\mathrm{clamp}(\mathrm{round}(w/s) + z,\; 0,\; 2^b-1) - z\big)

**Asymmetric** (with a zero-point ``z``) fits the observed ``[min, max]`` exactly.
**Symmetric** drops ``z`` and centres the range on zero, which costs a bit of resolution
for weights that are not zero-centred but makes the dequantization a single multiply —
worth it on hardware where that matters.

Grouping
--------
Sharing one ``(s, z)`` across an entire weight matrix is disastrous: a single outlier
sets the range and every other weight collapses onto a handful of levels. Real
implementations use **groups** along the input dimension — 64 or 128 weights per group —
so an outlier only degrades its own group. The cost is the stored scales themselves:
at 4 bits with group size 128 and fp16 scales, the overhead is
``(16 + 16) / 128 = 0.25`` bits per weight, so the *effective* bit-width is 4.25 rather
than 4. :func:`effective_bits` computes this, and the frontier figure plots against it —
quoting "4-bit" while ignoring the scales is the most common way these comparisons are
made to look better than they are.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

__all__ = [
    "QuantizedTensor",
    "dequantize",
    "effective_bits",
    "quantize_rtn",
    "quantize_tensor_rtn",
]


@dataclass(slots=True)
class QuantizedTensor:
    """Integer codes plus the per-group scales and zero-points needed to invert them."""

    codes: Tensor  # (out, in) integer levels, stored as int32 for simplicity
    scales: Tensor  # (out, n_groups)
    zeros: Tensor  # (out, n_groups)
    bits: int
    group_size: int
    symmetric: bool
    original_shape: tuple[int, ...]

    def dequantize(self) -> Tensor:
        """Reconstruct the float weight matrix."""
        return dequantize(self)

    def effective_bits(self, scale_bits: int = 16) -> float:
        """Bits per weight including the stored scales and zero-points."""
        return effective_bits(
            self.bits, self.group_size, symmetric=self.symmetric, scale_bits=scale_bits
        )

    def error(self, original: Tensor) -> float:
        """Relative Frobenius reconstruction error against the original weights."""
        return float((self.dequantize() - original).norm() / original.norm().clamp_min(1e-12))


def effective_bits(
    bits: int, group_size: int, *, symmetric: bool = False, scale_bits: int = 16
) -> float:
    """Bits per weight once the per-group scales (and zero-points) are counted.

    A "4-bit" model with group size 64 and fp16 scales actually costs 4.5 bits per
    weight asymmetric, or 4.25 symmetric. The frontier figure plots against this number
    rather than the nominal one.
    """
    if group_size <= 0:
        return float(bits)
    per_group = scale_bits if symmetric else 2 * scale_bits
    return bits + per_group / group_size


def quantize_tensor_rtn(
    weight: Tensor,
    *,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = False,
) -> QuantizedTensor:
    """Quantize a 2D weight matrix by round-to-nearest, grouped along the input dim.

    Args:
        weight: ``(out_features, in_features)``.
        bits: Bit-width (2-8).
        group_size: Weights per scale group along the input dimension; ``-1`` means one
            group per output channel (per-channel quantization).
        symmetric: Drop the zero-point and centre the range on zero.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2D weight, got shape {tuple(weight.shape)}.")
    if not 2 <= bits <= 8:
        raise ValueError(f"bits must be in [2, 8], got {bits}.")

    out_features, in_features = weight.shape
    group = in_features if group_size in (-1, 0) else min(group_size, in_features)
    if in_features % group != 0:
        raise ValueError(
            f"in_features={in_features} is not divisible by group_size={group}; "
            "pick a group size that divides the layer width."
        )
    n_groups = in_features // group

    w = weight.detach().float().reshape(out_features, n_groups, group)
    qmax = 2**bits - 1

    if symmetric:
        max_abs = w.abs().amax(dim=-1, keepdim=True)
        scales = (2 * max_abs / qmax).clamp_min(1e-8)
        zeros = torch.full_like(scales, (qmax + 1) / 2)
    else:
        w_min = w.amin(dim=-1, keepdim=True)
        w_max = w.amax(dim=-1, keepdim=True)
        scales = ((w_max - w_min) / qmax).clamp_min(1e-8)
        zeros = torch.round(-w_min / scales)

    codes = torch.clamp(torch.round(w / scales) + zeros, 0, qmax)

    return QuantizedTensor(
        codes=codes.to(torch.int32).reshape(out_features, in_features),
        scales=scales.squeeze(-1),
        zeros=zeros.squeeze(-1),
        bits=bits,
        group_size=group,
        symmetric=symmetric,
        original_shape=tuple(weight.shape),
    )


def dequantize(q: QuantizedTensor) -> Tensor:
    """Invert :func:`quantize_tensor_rtn`."""
    out_features, in_features = q.original_shape
    n_groups = in_features // q.group_size
    codes = q.codes.float().reshape(out_features, n_groups, q.group_size)
    scales = q.scales.unsqueeze(-1)
    zeros = q.zeros.unsqueeze(-1)
    return ((codes - zeros) * scales).reshape(out_features, in_features)


def quantize_rtn(
    model: nn.Module,
    *,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = False,
    skip: tuple[str, ...] = ("embed_tokens", "lm_head"),
) -> dict[str, float]:
    """Quantize every eligible ``nn.Linear`` in ``model`` in place, RTN.

    Returns a mapping ``layer name -> relative reconstruction error``.

    Embeddings and the LM head are skipped by default, which is standard practice for
    weight-only quantization: the embedding is a lookup table whose rows are used one at
    a time (so it is not a matmul bottleneck), and the head's output feeds a softmax
    where quantization error turns directly into a distribution shift.
    """
    errors: dict[str, float] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(pattern in name for pattern in skip):
            continue
        original = module.weight.detach().clone()
        q = quantize_tensor_rtn(
            module.weight.data, bits=bits, group_size=group_size, symmetric=symmetric
        )
        module.weight.data.copy_(q.dequantize().to(module.weight.dtype))
        errors[name] = q.error(original)
    return errors
