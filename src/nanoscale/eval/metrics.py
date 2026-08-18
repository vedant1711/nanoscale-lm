"""Tokenizer-independent and calibration metrics.

Perplexity is the field's default headline number and it has a serious flaw for the kind
of comparison this project wants to make: **it depends on the tokenizer**. A model with a
larger vocabulary needs fewer tokens to express the same text, so each token carries more
information and its per-token perplexity looks worse: even if it predicts the underlying
*text* better. Comparing perplexities across models with different tokenizers is
meaningless, and it is done constantly.

Bits-per-byte fixes this by normalizing to the one quantity both models must agree on:

.. code-block:: text

    BPB = (total negative log-likelihood in nats / ln 2) / total UTF-8 bytes

The numerator is the information the model assigns to the text, in bits. The denominator
is the text's length in bytes, which no tokenizer can change. This is what the Pile paper
and the Chinchilla paper report for cross-model comparison, and it is what makes the
"our 40M model against GPT-2's 124M" comparison in ``scripts/external_baseline.py``
a fair one rather than an artefact of vocabulary size.

The second metric here is **calibration**. A model can be accurate and still badly
miscalibrated, assigning 99% confidence to predictions that are right 70% of the time.
Expected calibration error bins predictions by confidence and measures the gap between
confidence and accuracy in each bin. It matters for a small model specifically because
overconfidence is the failure mode that makes generation degenerate: a model certain of a
wrong continuation will not recover from it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from nanoscale.model import IGNORE_INDEX
from nanoscale.train.data import Batch

__all__ = [
    "BitsPerByteResult",
    "CalibrationResult",
    "bits_per_byte",
    "calibration",
    "distinct_n",
    "self_bleu",
]


@dataclass(frozen=True)
class BitsPerByteResult:
    """Bits per UTF-8 byte, with the token-level figures it was derived from."""

    bits_per_byte: float
    stderr: float
    n_bytes: int
    n_tokens: int
    nll_nats: float

    @property
    def bytes_per_token(self) -> float:
        """How much text each token carried: the tokenizer's compression rate."""
        return self.n_bytes / max(1, self.n_tokens)

    @property
    def token_perplexity(self) -> float:
        """The tokenizer-dependent number, kept for continuity with older results."""
        return math.exp(min(self.nll_nats / max(1, self.n_tokens), 20.0))

    def summary(self) -> dict[str, float | int]:
        """Flat numbers for the results table."""
        return {
            "bits_per_byte": round(self.bits_per_byte, 5),
            "bits_per_byte_stderr": round(self.stderr, 5),
            "bytes_per_token": round(self.bytes_per_token, 4),
            "token_perplexity": round(self.token_perplexity, 4),
            "n_bytes": self.n_bytes,
            "n_tokens": self.n_tokens,
        }

    def __str__(self) -> str:
        """Render as ``bpb ± stderr``."""
        return (
            f"{self.bits_per_byte:.4f} ± {self.stderr:.4f} bits/byte "
            f"({self.n_bytes:,} bytes, {self.bytes_per_token:.2f} bytes/token)"
        )


@torch.no_grad()
def bits_per_byte(
    model: nn.Module,
    batches: Sequence[Batch],
    *,
    n_bytes: int,
    device: torch.device | None = None,
) -> BitsPerByteResult:
    """Compute bits-per-byte over ``batches`` covering ``n_bytes`` of source text.

    Args:
        model: Anything returning ``.logits`` from ``model(input_ids)``.
        batches: Evaluation batches whose targets use ``IGNORE_INDEX`` for padding.
        n_bytes: The UTF-8 length of the text these batches were tokenized from. The
            caller must supply this because the batches are packed token windows and no
            longer know what text they came from.
        device: Optional device to move batches to.

    Returns:
        A :class:`BitsPerByteResult`.

    The standard error is propagated from the per-token NLL spread, scaled by the same
    ``tokens/bytes`` factor as the mean; this treats byte count as exact, which it is.
    """
    if n_bytes <= 0:
        raise ValueError(f"n_bytes must be positive, got {n_bytes}")

    total_nll = 0.0
    total_sq = 0.0
    n_tokens = 0
    for batch in batches:
        if device is not None:
            batch = batch.to(device)
        logits = model(batch.inputs).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        targets = batch.targets
        valid = targets != IGNORE_INDEX
        gathered = logprobs.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        nll = -gathered[valid]
        total_nll += float(nll.sum())
        total_sq += float((nll**2).sum())
        n_tokens += int(valid.sum())

    if n_tokens == 0:
        raise ValueError("no valid target tokens in the supplied batches")

    mean = total_nll / n_tokens
    var = max(0.0, total_sq / n_tokens - mean**2)
    # Standard error of the *total* NLL, then converted to bits and divided by bytes.
    total_stderr_nats = math.sqrt(var * n_tokens)
    return BitsPerByteResult(
        bits_per_byte=total_nll / math.log(2) / n_bytes,
        stderr=total_stderr_nats / math.log(2) / n_bytes,
        n_bytes=n_bytes,
        n_tokens=n_tokens,
        nll_nats=total_nll,
    )


@dataclass(frozen=True)
class CalibrationResult:
    """Expected calibration error and the reliability curve behind it."""

    ece: float
    mce: float
    accuracy: float
    mean_confidence: float
    n: int
    bin_edges: tuple[float, ...] = ()
    bin_confidence: tuple[float, ...] = ()
    bin_accuracy: tuple[float, ...] = ()
    bin_count: tuple[int, ...] = ()

    @property
    def overconfidence(self) -> float:
        """Mean confidence minus accuracy. Positive means the model is overconfident."""
        return self.mean_confidence - self.accuracy

    def summary(self) -> dict[str, float | int]:
        """Flat numbers for the results table."""
        return {
            "ece": round(self.ece, 5),
            "mce": round(self.mce, 5),
            "top1_accuracy": round(self.accuracy, 5),
            "mean_confidence": round(self.mean_confidence, 5),
            "overconfidence": round(self.overconfidence, 5),
            "n_predictions": self.n,
        }


@torch.no_grad()
def calibration(
    model: nn.Module,
    batches: Sequence[Batch],
    *,
    n_bins: int = 15,
    device: torch.device | None = None,
) -> CalibrationResult:
    """Expected calibration error of next-token prediction.

    Bins every prediction by the model's confidence in its top-1 token, then compares the
    mean confidence in each bin against how often that bin was actually right.

    .. code-block:: text

        ECE = Σ_b (n_b / N) · | accuracy(b) − confidence(b) |
        MCE = max_b | accuracy(b) − confidence(b) |

    A perfectly calibrated model has ECE 0: among predictions it made with 80%
    confidence, exactly 80% are correct. ECE is reported alongside accuracy because the
    two are independent; a model can be accurate and overconfident, which is precisely
    the state that produces confident degenerate generation.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")

    conf_chunks: list[torch.Tensor] = []
    hit_chunks: list[torch.Tensor] = []
    for batch in batches:
        if device is not None:
            batch = batch.to(device)
        logits = model(batch.inputs).logits
        probs = torch.softmax(logits.float(), dim=-1)
        targets = batch.targets
        valid = targets != IGNORE_INDEX
        top_p, top_i = probs.max(dim=-1)
        conf_chunks.append(top_p[valid].cpu())
        hit_chunks.append((top_i == targets)[valid].float().cpu())

    conf = torch.cat(conf_chunks)
    hit = torch.cat(hit_chunks)
    n = int(conf.numel())
    if n == 0:
        raise ValueError("no valid target tokens in the supplied batches")

    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_conf: list[float] = []
    bin_acc: list[float] = []
    bin_n: list[int] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        # Upper-inclusive on the last bin so confidence exactly 1.0 is counted.
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        count = int(mask.sum())
        if count == 0:
            bin_conf.append(0.0)
            bin_acc.append(0.0)
            bin_n.append(0)
            continue
        c = float(conf[mask].mean())
        a = float(hit[mask].mean())
        gap = abs(a - c)
        ece += (count / n) * gap
        mce = max(mce, gap)
        bin_conf.append(c)
        bin_acc.append(a)
        bin_n.append(count)

    return CalibrationResult(
        ece=ece,
        mce=mce,
        accuracy=float(hit.mean()),
        mean_confidence=float(conf.mean()),
        n=n,
        bin_edges=tuple(float(x) for x in edges),
        bin_confidence=tuple(bin_conf),
        bin_accuracy=tuple(bin_acc),
        bin_count=tuple(bin_n),
    )


@dataclass(frozen=True)
class DiversityResult:
    """Lexical diversity of a set of generations."""

    distinct: dict[int, float] = field(default_factory=dict)
    self_bleu: float = 0.0
    n_samples: int = 0

    def summary(self) -> dict[str, float | int]:
        """Flat numbers for the results table."""
        out: dict[str, float | int] = {
            f"distinct_{k}": round(v, 5) for k, v in self.distinct.items()
        }
        out["self_bleu"] = round(self.self_bleu, 5)
        out["n_samples"] = self.n_samples
        return out


def distinct_n(texts: Sequence[str], *, n: int = 2) -> float:
    """Fraction of ``n``-grams across ``texts`` that are unique.

    The standard diversity diagnostic from Li et al. 2016. A model that has collapsed onto
    a few phrasings scores low here even when its perplexity looks fine, which is why it
    is reported next to perplexity rather than instead of it.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    seen: set[tuple[str, ...]] = set()
    total = 0
    for text in texts:
        words = text.split()
        for i in range(len(words) - n + 1):
            seen.add(tuple(words[i : i + n]))
            total += 1
    return len(seen) / max(1, total)


def _ngrams(words: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def self_bleu(texts: Sequence[str], *, max_n: int = 4) -> float:
    """Mean BLEU of each generation against all the others.

    Measures how similar a model's outputs are *to each other*. High self-BLEU means the
    model produces the same thing regardless of prompt: mode collapse, which perplexity
    cannot see. Implemented directly (uniform weights, brevity penalty) rather than pulled
    from a library, in keeping with the rest of the project.

    Returns 0.0 for fewer than two samples, where the quantity is undefined.
    """
    if len(texts) < 2:
        return 0.0

    tokenized = [t.split() for t in texts]
    scores: list[float] = []
    for i, cand in enumerate(tokenized):
        if not cand:
            continue
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i and tokenized[j]]
        if not refs:
            continue

        log_precisions: list[float] = []
        for n in range(1, max_n + 1):
            cand_ngrams = _ngrams(cand, n)
            if not cand_ngrams:
                continue
            # Clipped counts: a candidate n-gram counts at most as often as it appears in
            # the best-matching reference, which is what stops repetition scoring highly.
            best = 0
            for ref in refs:
                ref_ngrams = _ngrams(ref, n)
                overlap = 0
                remaining = list(ref_ngrams)
                for g in cand_ngrams:
                    if g in remaining:
                        remaining.remove(g)
                        overlap += 1
                best = max(best, overlap)
            # Smoothing: a zero at any order would zero the whole geometric mean.
            log_precisions.append(math.log((best + 1e-9) / len(cand_ngrams)))

        if not log_precisions:
            continue
        ref_len = min((len(r) for r in refs), key=lambda x: (abs(x - len(cand)), x))
        brevity = 1.0 if len(cand) > ref_len else math.exp(1 - ref_len / max(1, len(cand)))
        scores.append(brevity * math.exp(sum(log_precisions) / len(log_precisions)))

    return sum(scores) / max(1, len(scores))
