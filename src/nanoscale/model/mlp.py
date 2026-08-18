r"""Feed-forward blocks (spec B2): SwiGLU by default, ReLU² as the speedrun ablation.

**SwiGLU** (Shazeer, *GLU Variants Improve Transformer*, arXiv:2002.05202):

.. math::  \mathrm{SwiGLU}(x) = \big(\mathrm{SiLU}(xW_{gate}) \odot xW_{up}\big) W_{down}

The gate is the point: one branch decides *how much* of the other branch to let
through, per feature. It costs a third matrix, which is why the hidden width is
conventionally set to ``8/3 · d_model``: that keeps the parameter count level with a
classic ``4 · d_model`` two-matrix MLP so the comparison is fair.

**ReLU²** (used by the modded-nanoGPT speedrun) is the ungated alternative
``ReLU(xW_up)² W_down``. It is cheaper per parameter and, in the speedrun setting,
competitive. Both are implemented so Phase 5 can *measure* the difference at ``micro``
scale rather than repeat someone else's claim.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["ReLU2MLP", "SwiGLU", "build_mlp"]


class SwiGLU(nn.Module):
    """Gated feed-forward block: ``(SiLU(x W_gate) ⊙ x W_up) W_down``."""

    def __init__(self, d_model: int, d_ff: int, *, dropout: float = 0.0) -> None:
        """Create a SwiGLU MLP with hidden width ``d_ff``."""
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated MLP."""
        out: torch.Tensor = self.dropout(
            self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        )
        return out

    @property
    def output_projection(self) -> nn.Linear:
        """The residual-writing projection (zero-initialised when configured)."""
        return self.down_proj


class ReLU2MLP(nn.Module):
    """Ungated feed-forward block: ``ReLU(x W_up)² W_down`` (modded-nanoGPT variant)."""

    def __init__(self, d_model: int, d_ff: int, *, dropout: float = 0.0) -> None:
        """Create a ReLU-squared MLP with hidden width ``d_ff``."""
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the squared-ReLU MLP."""
        hidden = F.relu(self.up_proj(x))
        out: torch.Tensor = self.dropout(self.down_proj(hidden * hidden))
        return out

    @property
    def output_projection(self) -> nn.Linear:
        """The residual-writing projection (zero-initialised when configured)."""
        return self.down_proj


def build_mlp(kind: str, d_model: int, d_ff: int, *, dropout: float = 0.0) -> nn.Module:
    """Construct the MLP named by a :class:`ModelConfig` field."""
    if kind == "swiglu":
        return SwiGLU(d_model, d_ff, dropout=dropout)
    if kind == "relu2":
        return ReLU2MLP(d_model, d_ff, dropout=dropout)
    raise ValueError(f"Unknown mlp_type {kind!r}; expected 'swiglu' or 'relu2'.")
