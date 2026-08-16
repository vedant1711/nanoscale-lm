"""Evaluation: perplexity, preference head-to-heads and generation-quality diagnostics."""

from __future__ import annotations

from nanoscale.eval.preference_eval import (
    CompletionScore,
    HeadToHeadResult,
    head_to_head,
    repetition_rate,
    score_completion,
)

__all__ = [
    "CompletionScore",
    "HeadToHeadResult",
    "head_to_head",
    "repetition_rate",
    "score_completion",
]
