r"""Normalization layers (spec B2): RMSNorm by default, LayerNorm as an ablation.

RMSNorm (Zhang & Sennrich, arXiv:1910.07467) drops the mean-centering of LayerNorm and
keeps only the rescaling:

.. math::  \mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + ε}} \odot g

The claim from the paper, and the reason essentially every modern LM uses it, is that
re-centering contributes little to the stabilisation while costing a mean, a
subtraction and an extra pass over the activations. There is no bias term, which is
also standard in modern decoder stacks.

The reduction is always done at fp32 or better, even under bf16/fp16 autocast: a sum
of squares over a few thousand elements in bf16 loses enough precision to matter, and
this is one of the cheapest places to be careful. See :mod:`nanoscale.model.numerics`
for why that is "promote to at least fp32" rather than ``.float()``.
"""

from __future__ import annotations

import torch
from torch import nn

from nanoscale.model.numerics import to_accumulation

__all__ = ["LayerNorm", "RMSNorm", "build_norm", "rms_normalize"]


def rms_normalize(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Parameter-free RMS normalization over the last dimension, computed in fp32+."""
    dtype = x.dtype
    xf = to_accumulation(x)
    scale = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * scale).to(dtype)


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization with an optional learned gain.

    Args:
        dim: Size of the normalized (last) dimension.
        eps: Numerical floor inside the square root.
        elementwise_affine: If True, learn a per-feature gain initialised to 1.
    """

    def __init__(self, dim: int, *, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        """Create an RMSNorm over the last dimension of size ``dim``."""
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension and apply the learned gain."""
        out = rms_normalize(x, self.eps)
        if self.weight is not None:
            out = out * self.weight
        return out

    def extra_repr(self) -> str:
        """Describe the layer for ``print(model)``."""
        return f"dim={self.dim}, eps={self.eps}, affine={self.elementwise_affine}"


class LayerNorm(nn.Module):
    """Bias-free LayerNorm, available as a controlled ablation against RMSNorm.

    Kept bias-free so that the only difference from :class:`RMSNorm` is the
    mean-centering, which is what makes an A/B between them interpretable.
    """

    def __init__(self, dim: int, *, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        """Create a bias-free LayerNorm over the last dimension of size ``dim``."""
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Center and rescale the last dimension, then apply the learned gain."""
        dtype = x.dtype
        xf = to_accumulation(x)
        mean = xf.mean(dim=-1, keepdim=True)
        centered = xf - mean
        scale = torch.rsqrt(centered.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = (centered * scale).to(dtype)
        if self.weight is not None:
            out = out * self.weight
        return out

    def extra_repr(self) -> str:
        """Describe the layer for ``print(model)``."""
        return f"dim={self.dim}, eps={self.eps}, affine={self.elementwise_affine}"


def build_norm(
    kind: str, dim: int, *, eps: float = 1e-5, elementwise_affine: bool = True
) -> nn.Module:
    """Construct the normalization layer named by a :class:`ModelConfig` field."""
    if kind == "rmsnorm":
        return RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
    if kind == "layernorm":
        return LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
    raise ValueError(f"Unknown norm_type {kind!r}; expected 'rmsnorm' or 'layernorm'.")
