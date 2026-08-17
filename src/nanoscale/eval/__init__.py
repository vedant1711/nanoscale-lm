"""Evaluation: perplexity with error bars, a tiny offline benchmark, and gen quality."""

from __future__ import annotations

from nanoscale.eval.perplexity import PerplexityResult, perplexity, token_nll
from nanoscale.eval.preference_eval import (
    CompletionScore,
    HeadToHeadResult,
    head_to_head,
    repetition_rate,
    score_completion,
)
from nanoscale.eval.tiny_bench import (
    TASKS,
    BenchmarkResult,
    MultipleChoiceQuestion,
    run_tiny_bench,
    score_choice,
)

__all__ = [
    "TASKS",
    "BenchmarkResult",
    "CompletionScore",
    "HeadToHeadResult",
    "MultipleChoiceQuestion",
    "PerplexityResult",
    "head_to_head",
    "perplexity",
    "repetition_rate",
    "run_tiny_bench",
    "score_choice",
    "score_completion",
    "token_nll",
]
