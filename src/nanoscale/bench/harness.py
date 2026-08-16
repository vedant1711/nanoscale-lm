"""The unified benchmark harness (spec Phase 10).

Produces one results table across every model variant: base, distilled, quantized,
speculative, and speculative+quantized — on the same prompts, the same seeds and the
same machine.

Measurement discipline
----------------------
* **Warmup iterations are discarded.** The first call through a PyTorch model pays
  lazy-init and allocator costs that have nothing to do with steady-state throughput.
* **The median is reported, not the mean.** A single scheduler hiccup on a shared laptop
  produces an outlier that a mean happily absorbs and a median does not.
* **Prefill and decode are timed separately.** Prefill is parallel over the prompt;
  decode is sequential. A combined tokens/second figure is dominated by whichever the
  prompt length happens to favour, which makes it useless for comparing decode
  optimisations.
* **Peak memory is measured where it can be, and computed where it cannot.** On CUDA
  the allocator reports it directly; on CPU there is no equivalent, so the harness
  records the analytic model + KV-cache footprint and labels it as such rather than
  quoting an RSS figure that includes the Python interpreter.
"""

from __future__ import annotations

import gc
import json
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from nanoscale.model import NanoScaleLM
from nanoscale.utils import git_sha, hardware_string
from nanoscale.utils.logging import get_logger

__all__ = ["BenchHarness", "BenchRow", "model_memory_bytes", "peak_memory_bytes"]

log = get_logger("nanoscale.bench")


def model_memory_bytes(model: NanoScaleLM, *, weight_bits: float | None = None) -> int:
    """Analytic weight footprint in bytes.

    ``weight_bits`` overrides the stored dtype, which is what makes a quantized variant
    report its *representation* size rather than the fp32 tensors it is simulated with.
    Simulating int4 in fp32 is honest about latency (there is no int4 CPU kernel) but
    would report a 4-bit model as 32-bit if the footprint were read off the tensors.
    """
    if weight_bits is None:
        return sum(p.numel() * p.element_size() for p in model.parameters())
    quantizable = model.num_parameters(non_embedding=True)
    embedding = model.num_parameters() - quantizable
    element_size = next(model.parameters()).element_size()
    return int(quantizable * weight_bits / 8 + embedding * element_size)


def peak_memory_bytes() -> int | None:
    """Peak allocator memory since the last reset, or ``None`` when unavailable."""
    if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
        return int(torch.cuda.max_memory_allocated())
    return None


@dataclass(slots=True)
class BenchRow:
    """One benchmarked variant."""

    variant: str
    params: int
    weight_mb: float
    kv_mb: float
    prefill_ms_p50: float
    decode_tokens_per_s_p50: float
    latency_ms_p50: float
    latency_ms_p95: float
    generated_tokens: int
    perplexity: float | None = None
    acceptance_rate: float | None = None
    mean_accepted_length: float | None = None
    peak_memory_mb: float | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialise."""
        return asdict(self)


@dataclass(slots=True)
class BenchHarness:
    """Runs a set of named decode callables and aggregates their timings."""

    warmup_iters: int = 2
    measure_iters: int = 5
    rows: list[BenchRow] = field(default_factory=list)

    def time_variant(
        self,
        name: str,
        run: Callable[[int], dict[str, float]],
        *,
        params: int,
        weight_mb: float,
        kv_mb: float,
        perplexity: float | None = None,
        notes: str = "",
    ) -> BenchRow:
        """Time one variant.

        Args:
            name: Variant label.
            run: Called with the iteration index; must return a dict with at least
                ``prefill_s``, ``decode_s`` and ``generated_tokens``, optionally
                ``acceptance_rate`` and ``mean_accepted_length``.
            params: Parameter count.
            weight_mb: Analytic weight footprint.
            kv_mb: Analytic KV-cache footprint at the benchmarked context length.
            perplexity: Quality number to carry into the table.
            notes: Free-text caveat shown in the results table.
        """
        for i in range(self.warmup_iters):
            run(i)

        gc.collect()
        if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
            torch.cuda.reset_peak_memory_stats()

        prefill: list[float] = []
        decode_rate: list[float] = []
        latency: list[float] = []
        acceptance: list[float] = []
        accepted_length: list[float] = []
        generated = 0

        for i in range(self.measure_iters):
            start = time.perf_counter()
            stats = run(self.warmup_iters + i)
            total = time.perf_counter() - start

            prefill.append(stats["prefill_s"] * 1000.0)
            tokens = stats["generated_tokens"]
            decode_rate.append(tokens / max(1e-9, stats["decode_s"]))
            latency.append(total * 1000.0)
            generated = int(tokens)
            if "acceptance_rate" in stats:
                acceptance.append(stats["acceptance_rate"])
            if "mean_accepted_length" in stats:
                accepted_length.append(stats["mean_accepted_length"])

        peak = peak_memory_bytes()
        row = BenchRow(
            variant=name,
            params=params,
            weight_mb=round(weight_mb, 3),
            kv_mb=round(kv_mb, 3),
            prefill_ms_p50=round(statistics.median(prefill), 3),
            decode_tokens_per_s_p50=round(statistics.median(decode_rate), 2),
            latency_ms_p50=round(statistics.median(latency), 2),
            latency_ms_p95=round(_percentile(latency, 95), 2),
            generated_tokens=generated,
            perplexity=round(perplexity, 4) if perplexity is not None else None,
            acceptance_rate=round(statistics.median(acceptance), 4) if acceptance else None,
            mean_accepted_length=(
                round(statistics.median(accepted_length), 4) if accepted_length else None
            ),
            peak_memory_mb=round(peak / 1024**2, 3) if peak is not None else None,
            notes=notes,
        )
        self.rows.append(row)
        log.info(
            "%-34s %8.1f tok/s decode | p50 %7.1f ms | p95 %7.1f ms | %6.2f MB weights",
            name,
            row.decode_tokens_per_s_p50,
            row.latency_ms_p50,
            row.latency_ms_p95,
            row.weight_mb,
        )
        return row

    def payload(self, **extra: object) -> dict[str, object]:
        """The full results table with provenance."""
        return {
            "git_sha": git_sha(),
            "hardware": hardware_string(),
            "warmup_iters": self.warmup_iters,
            "measure_iters": self.measure_iters,
            "peak_memory_available": peak_memory_bytes() is not None,
            "rows": [row.to_dict() for row in self.rows],
            **extra,
        }

    def write_json(self, path: str | Path, **extra: object) -> Path:
        """Write the results table as JSON."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.payload(**extra), indent=2) + "\n", encoding="utf-8")
        return dest

    def markdown_table(self) -> str:
        """Render the results table as markdown."""
        header = (
            "| variant | params | weights | KV @ ctx | prefill p50 | decode tok/s | "
            "latency p50 | latency p95 | val ppl | accept |"
        )
        lines = [header, "|" + "---|" * 10]
        for row in self.rows:
            ppl = f"{row.perplexity:.4f}" if row.perplexity is not None else "—"
            acc = f"{row.acceptance_rate:.3f}" if row.acceptance_rate is not None else "—"
            lines.append(
                f"| {row.variant} | {row.params:,} | {row.weight_mb:.2f} MB | "
                f"{row.kv_mb:.2f} MB | {row.prefill_ms_p50:.1f} ms | "
                f"{row.decode_tokens_per_s_p50:.1f} | {row.latency_ms_p50:.1f} ms | "
                f"{row.latency_ms_p95:.1f} ms | {ppl} | {acc} |"
            )
        return "\n".join(lines)


def _percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile — well-defined for the small samples used here.

    Interpolating between order statistics implies a continuity the sample does not
    have; with five measurements, p95 is simply the largest.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = round(pct / 100.0 * len(ordered) + 0.5) - 1
    index = min(len(ordered) - 1, max(0, rank))
    return ordered[index]
