"""Measurement harnesses: controlled ablations, throughput, memory and latency."""

from __future__ import annotations

from nanoscale.bench.ablation import (
    NOISE_THRESHOLD,
    AblationResult,
    AblationSuite,
    Variant,
    describe_difference,
)
from nanoscale.bench.harness import (
    BenchHarness,
    BenchRow,
    model_memory_bytes,
    peak_memory_bytes,
)
from nanoscale.bench.multiseed import (
    ALPHA,
    MIN_EFFECT_SIZE,
    ArmStatistics,
    MultiSeedComparison,
    MultiSeedSuite,
    cohens_d,
    holm_bonferroni,
    variance_ratio_test,
    welch_t_test,
)

__all__ = [
    "ALPHA",
    "MIN_EFFECT_SIZE",
    "NOISE_THRESHOLD",
    "AblationResult",
    "AblationSuite",
    "ArmStatistics",
    "BenchHarness",
    "BenchRow",
    "MultiSeedComparison",
    "MultiSeedSuite",
    "Variant",
    "cohens_d",
    "describe_difference",
    "holm_bonferroni",
    "model_memory_bytes",
    "peak_memory_bytes",
    "variance_ratio_test",
    "welch_t_test",
]
