"""Speculative decoding: draft-target sampling and Medusa-style self-speculation."""

from __future__ import annotations

from nanoscale.specdec.accept_rule import (
    acceptance_probability,
    expected_acceptance_rate,
    residual_distribution,
    sample_accept_reject,
)
from nanoscale.specdec.medusa import (
    MedusaSampler,
    TreeCandidates,
    build_candidate_tree,
    build_tree_attention_mask,
    tree_position_ids,
)
from nanoscale.specdec.spec_sampling import (
    SpeculativeResult,
    SpeculativeSampler,
    apply_sampling_transforms,
    autoregressive_baseline,
)

__all__ = [
    "MedusaSampler",
    "SpeculativeResult",
    "SpeculativeSampler",
    "TreeCandidates",
    "acceptance_probability",
    "apply_sampling_transforms",
    "autoregressive_baseline",
    "build_candidate_tree",
    "build_tree_attention_mask",
    "expected_acceptance_rate",
    "residual_distribution",
    "sample_accept_reject",
    "tree_position_ids",
]
