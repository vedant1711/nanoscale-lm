"""Lossless text compression and surprisal-based anomaly detection with a small LM.

This is the module that makes NanoScale-LM useful for something other than demonstrating
that it works. Both capabilities fall out of the same forward pass, and both are things a
40M-parameter CPU model can do that a 175B one cannot do economically:

**Compression.** The model's next-token distribution drives an arithmetic coder
(:mod:`nanoscale.compress.coder`), so its cross-entropy becomes an actual file size. On
in-domain text this beats every classical compressor by a wide margin, because a general
compressor can only exploit repetition while a language model exploits *meaning*; it
knows "the cat sat on the" is followed by a small set of words, which no substitution
dictionary can represent.

**Anomaly detection.** The per-token surprisal ``-log p(token | context)`` is the same
quantity, un-summed. A line the model finds unlikely is a line unlike its training
distribution. For a model trained on one system's logs, that is a definition of "anomalous
log line" that needs no labels, no rules and no threshold tuning beyond a percentile.

The economics are the interesting part and are worked out in
``scripts/compression_bench.py``: the model must be shipped alongside the archive, so a
neural compressor only pays off above a break-even volume. Below it, use ``xz``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import torch

from nanoscale.compress.coder import (
    ArithmeticDecoder,
    ArithmeticEncoder,
    probs_to_frequencies,
)
from nanoscale.model import NanoScaleLM
from nanoscale.tokenizer import BPETokenizer

__all__ = [
    "AnomalyReport",
    "CompressionResult",
    "compress",
    "decompress",
    "score_lines",
    "token_surprisal",
]


@dataclass
class CompressionResult:
    """Outcome of compressing one text."""

    n_bytes_in: int
    n_bytes_out: int
    n_tokens: int
    ideal_bits: float
    payload: bytes = b""

    @property
    def bits_per_byte(self) -> float:
        """Achieved rate; the number that can be checked against the file on disk."""
        return self.n_bytes_out * 8 / max(1, self.n_bytes_in)

    @property
    def ideal_bits_per_byte(self) -> float:
        """The model's cross-entropy, i.e. the rate a perfect coder would reach."""
        return self.ideal_bits / max(1, self.n_bytes_in)

    @property
    def ratio(self) -> float:
        """Compression ratio against raw UTF-8."""
        return self.n_bytes_in / max(1, self.n_bytes_out)

    @property
    def coder_overhead(self) -> float:
        """Fraction of the output spent on coder inefficiency rather than on entropy."""
        ideal = self.ideal_bits / 8
        return (self.n_bytes_out - ideal) / max(1.0, ideal)

    def summary(self) -> dict[str, float | int]:
        """Flat numbers for the results table."""
        return {
            "bytes_in": self.n_bytes_in,
            "bytes_out": self.n_bytes_out,
            "tokens": self.n_tokens,
            "bits_per_byte": round(self.bits_per_byte, 5),
            "ideal_bits_per_byte": round(self.ideal_bits_per_byte, 5),
            "ratio": round(self.ratio, 3),
            "coder_overhead": round(self.coder_overhead, 5),
        }


class _Stepper:
    """Feeds the model one token at a time, reusing a KV cache.

    **Why a cache is safe here and a batched one-shot pass is not.**

    The decoder must reconstruct the encoder's probabilities closely enough that the
    *quantised integer frequencies* match; a single differing frequency desynchronises
    the arithmetic and corrupts everything after it. So the question is how much float
    drift the quantisation can absorb. With a 16k vocabulary and 16-bit frequencies the
    bucket width is 2.0e-5 in probability.

    Two candidate speedups were measured against that budget:

    * **One-shot teacher-forced encode** (score the whole sequence in one pass, since the
      encoder knows all the tokens). Measured drift against the stepwise path:
      **1.7e-6**: only 12x below the bucket width. That is not a safety margin, it is a
      coin flip at scale, and it is *rejected*: at 16k symbols per position the chance
      that some probability sits within 1.7e-6 of a boundary is not small.
    * **Incremental decoding with a KV cache.** The repository already pins this against
      full recomputation at atol=1e-9, **20,000x** below the bucket width. And the
      stronger argument is structural: encoder and decoder both use *this* path, so they
      execute identical operations on identical inputs and agree exactly, with the 1e-9
      figure only bounding the difference from the uncached path.

    The general lesson is worth stating because it applies to any neural codec: integer
    quantisation of the probability table is what makes the scheme robust to
    floating-point non-determinism, and the size of that robustness is a number you can
    measure rather than hope for.
    """

    def __init__(self, model: NanoScaleLM, device: torch.device, window: int) -> None:
        """Prepare a single-sequence cache for ``model``."""
        self.model = model
        self.device = device
        self.window = window
        self.ids: list[int] = []
        self.cache = model.make_cache(1)

    def step(self, token: int) -> torch.Tensor:
        """Feed one token and return the next-token distribution over the vocabulary."""
        self.ids.append(token)

        if len(self.ids) > self.window:
            # The cache is full. Slide it: keep the trailing half of the context and
            # re-prefill. Both encoder and decoder hit this at exactly the same token
            # index and rebuild from exactly the same ids, so they stay in lockstep.
            self.ids = self.ids[-(self.window // 2) :]
            self.cache.reset()
            ctx = torch.tensor([self.ids[:-1]], dtype=torch.long, device=self.device)
            if ctx.numel():
                self.model(ctx, cache=self.cache)

        x = torch.tensor([[token]], dtype=torch.long, device=self.device)
        logits = self.model(x, cache=self.cache).logits[0, -1]
        return torch.softmax(logits.float(), dim=-1)


@torch.no_grad()
def compress(
    model: NanoScaleLM,
    tokenizer: BPETokenizer,
    text: str,
    *,
    device: torch.device | None = None,
    window: int | None = None,
) -> CompressionResult:
    """Compress ``text`` losslessly using the model as the probability source."""
    dev = device or next(model.parameters()).device
    win = window or int(getattr(model.config, "max_seq_len", 256))
    model.eval()

    tokens = tokenizer.encode(text, add_bos=True)
    encoder = ArithmeticEncoder()
    stepper = _Stepper(model, dev, win)
    # Token 0 is BOS, which the decoder also starts from, so it is never encoded.
    for i in range(1, len(tokens)):
        probs = stepper.step(tokens[i - 1])
        encoder.encode(tokens[i], probs_to_frequencies(probs.cpu()))
    payload = encoder.finish()

    return CompressionResult(
        n_bytes_in=len(text.encode("utf-8")),
        n_bytes_out=len(payload),
        n_tokens=len(tokens) - 1,
        ideal_bits=encoder.stats.ideal_bits,
        payload=payload,
    )


@torch.no_grad()
def decompress(
    model: NanoScaleLM,
    tokenizer: BPETokenizer,
    payload: bytes,
    n_tokens: int,
    *,
    device: torch.device | None = None,
    window: int | None = None,
) -> str:
    """Reconstruct the original text from ``payload``.

    ``n_tokens`` must be transmitted alongside the payload; the coder has no end-of-stream
    symbol. A production format would either reserve one or write the count in a header;
    the benchmark reports both so the accounting stays honest.
    """
    dev = device or next(model.parameters()).device
    win = window or int(getattr(model.config, "max_seq_len", 256))
    model.eval()

    decoder = ArithmeticDecoder(payload)
    stepper = _Stepper(model, dev, win)
    ids = [tokenizer.bos_id]
    for _ in range(n_tokens):
        probs = stepper.step(ids[-1])
        ids.append(decoder.decode(probs_to_frequencies(probs.cpu())))
    return tokenizer.decode(ids[1:])


# ---------------------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------------------


@torch.no_grad()
def token_surprisal(
    model: NanoScaleLM,
    tokenizer: BPETokenizer,
    text: str,
    *,
    device: torch.device | None = None,
) -> list[tuple[str, float]]:
    """Per-token surprisal in bits: ``(token_text, -log2 p(token | context))``.

    The same quantity the compressor sums. High surprisal means the model did not expect
    this token here, which for a domain-trained model means the text departs from the
    domain.
    """
    dev = device or next(model.parameters()).device
    model.eval()
    ids = tokenizer.encode(text, add_bos=True)
    x = torch.tensor([ids], device=dev)
    logits = model(x).logits[0]
    logprobs = torch.log_softmax(logits.float(), dim=-1)

    out: list[tuple[str, float]] = []
    for i in range(1, len(ids)):
        bits = -float(logprobs[i - 1, ids[i]]) / math.log(2)
        out.append((tokenizer.decode([ids[i]]), bits))
    return out


@dataclass
class AnomalyReport:
    """Per-line surprisal scores, ranked."""

    lines: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        """Mean bits/token across scored lines."""
        return sum(self.scores) / max(1, len(self.scores))

    def ranked(self) -> list[tuple[float, str]]:
        """Lines sorted most-anomalous first."""
        return sorted(zip(self.scores, self.lines, strict=True), reverse=True)

    def flagged(self, *, percentile: float = 95.0) -> Iterator[tuple[float, str]]:
        """Lines above the given surprisal percentile.

        A percentile rather than an absolute bit threshold, because the natural scale of
        surprisal depends on the domain and on how well the model was trained. A
        percentile turns "how surprising is surprising" into an explicit budget for how
        many lines an operator is willing to look at.
        """
        if not self.scores:
            return
        ranked = self.ranked()
        cut = int(len(ranked) * (1.0 - percentile / 100.0))
        yield from ranked[: max(1, cut)]


@torch.no_grad()
def score_lines(
    model: NanoScaleLM,
    tokenizer: BPETokenizer,
    lines: Sequence[str],
    *,
    device: torch.device | None = None,
) -> AnomalyReport:
    """Mean bits-per-token for each line, independently.

    Scored independently rather than as one stream so that a single anomalous line cannot
    raise the surprisal of the lines that follow it, which would smear one anomaly across
    a whole window and is a real failure mode of naive streaming scores.

    Length-normalised, because an unnormalised total scores every long line as anomalous.
    """
    dev = device or next(model.parameters()).device
    model.eval()
    report = AnomalyReport()
    for line in lines:
        if not line.strip():
            continue
        ids = tokenizer.encode(line, add_bos=True)
        if len(ids) < 2:
            continue
        x = torch.tensor([ids], device=dev)
        logits = model(x).logits[0]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        target = torch.tensor(ids[1:], device=dev)
        nll = -logprobs[:-1].gather(-1, target.unsqueeze(-1)).squeeze(-1)
        report.lines.append(line)
        report.scores.append(float(nll.mean()) / math.log(2))
    return report
