"""Compression and anomaly detection: the language model as a probability source."""

from __future__ import annotations

from nanoscale.compress.coder import (
    FREQ_BITS,
    ArithmeticDecoder,
    ArithmeticEncoder,
    CoderStats,
    probs_to_frequencies,
)
from nanoscale.compress.lm_compress import (
    AnomalyReport,
    CompressionResult,
    compress,
    decompress,
    score_lines,
    token_surprisal,
)

__all__ = [
    "FREQ_BITS",
    "AnomalyReport",
    "ArithmeticDecoder",
    "ArithmeticEncoder",
    "CoderStats",
    "CompressionResult",
    "compress",
    "decompress",
    "probs_to_frequencies",
    "score_lines",
    "token_surprisal",
]
