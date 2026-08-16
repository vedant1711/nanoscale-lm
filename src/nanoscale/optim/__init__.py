"""Optimizers: a from-scratch AdamW and Muon (Newton-Schulz orthogonalization)."""

from __future__ import annotations

from nanoscale.optim.adamw import AdamW
from nanoscale.optim.cautious import cautious_decay_mask, cautious_mask_fraction
from nanoscale.optim.muon import (
    NS_COEFFS,
    Muon,
    muon_update_scale,
    newton_schulz_orthogonalize,
)
from nanoscale.optim.router import (
    ADAMW_NAME_PATTERNS,
    CompositeOptimizer,
    ParamSplit,
    build_optimizer,
    split_parameters,
)

__all__ = [
    "ADAMW_NAME_PATTERNS",
    "NS_COEFFS",
    "AdamW",
    "CompositeOptimizer",
    "Muon",
    "ParamSplit",
    "build_optimizer",
    "cautious_decay_mask",
    "cautious_mask_fraction",
    "muon_update_scale",
    "newton_schulz_orthogonalize",
    "split_parameters",
]
