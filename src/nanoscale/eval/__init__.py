"""Evaluation: perplexity with error bars, a tiny offline benchmark, and gen quality."""

from __future__ import annotations

from nanoscale.eval.metrics import (
    BitsPerByteResult,
    CalibrationResult,
    DiversityResult,
    bits_per_byte,
    calibration,
    distinct_n,
    self_bleu,
)
from nanoscale.eval.minimal_pairs import (
    PHENOMENA,
    MinimalPair,
    MinimalPairResult,
    PhenomenonScore,
    generate_pairs,
    run_minimal_pairs,
    wilson_interval,
)
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
    "PHENOMENA",
    "TASKS",
    "BenchmarkResult",
    "BitsPerByteResult",
    "CalibrationResult",
    "CompletionScore",
    "DiversityResult",
    "HeadToHeadResult",
    "MinimalPair",
    "MinimalPairResult",
    "MultipleChoiceQuestion",
    "PerplexityResult",
    "PhenomenonScore",
    "bits_per_byte",
    "calibration",
    "distinct_n",
    "generate_pairs",
    "head_to_head",
    "perplexity",
    "repetition_rate",
    "run_minimal_pairs",
    "run_tiny_bench",
    "score_choice",
    "score_completion",
    "self_bleu",
    "token_nll",
    "wilson_interval",
]
