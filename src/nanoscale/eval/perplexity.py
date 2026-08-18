r"""Perplexity, with the definition stated and the uncertainty reported.

Perplexity is the exponentiated mean negative log-likelihood **per token**:

.. math::  \mathrm{PPL} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N} \log p(x_i \mid x_{<i})\right)

Two things that make published perplexities incomparable, both pinned here:

* **Per token or per word?** This module is always per *token*, and tokens depend on the
  tokenizer; a model with a 1k vocabulary and one with 50k are not comparable by
  perplexity even on identical text. Every perplexity in this repo is measured with the
  committed tokenizer and is only comparable to other numbers measured the same way.
* **Token-weighted or batch-weighted?** Averaging per-batch losses gives short trailing
  batches equal weight to full ones. This module weights by token count, which is the
  definition above.

Error bars
----------
Following Miller, *Adding Error Bars to Evals* (arXiv:2411.00640): a single number from a
finite evaluation set is an estimate, and reporting it without a standard error invites
reading noise as signal. :func:`perplexity` returns the standard error of the mean
negative log-likelihood alongside the point estimate, propagated to a perplexity
interval. At the sizes used here the interval is often wide enough to make small
differences uninterpretable, which is exactly what it is for.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from nanoscale.model import IGNORE_INDEX
from nanoscale.train.data import Batch

__all__ = ["PerplexityResult", "perplexity", "token_nll"]


@dataclass(frozen=True, slots=True)
class PerplexityResult:
    """A perplexity estimate with its uncertainty."""

    nll: float
    nll_stderr: float
    n_tokens: int

    @property
    def perplexity(self) -> float:
        """``exp(nll)``."""
        return math.exp(min(self.nll, 20.0))

    @property
    def perplexity_low(self) -> float:
        """Lower end of the ±1 standard-error interval."""
        return math.exp(min(self.nll - self.nll_stderr, 20.0))

    @property
    def perplexity_high(self) -> float:
        """Upper end of the ±1 standard-error interval."""
        return math.exp(min(self.nll + self.nll_stderr, 20.0))

    def summary(self) -> dict[str, float | int]:
        """Flat numbers for the results table."""
        return {
            "nll": round(self.nll, 6),
            "nll_stderr": round(self.nll_stderr, 6),
            "perplexity": round(self.perplexity, 4),
            "perplexity_low": round(self.perplexity_low, 4),
            "perplexity_high": round(self.perplexity_high, 4),
            "n_tokens": self.n_tokens,
        }

    def __str__(self) -> str:
        """Render as ``ppl (low–high)``."""
        return (
            f"{self.perplexity:.4f} "
            f"({self.perplexity_low:.4f}–{self.perplexity_high:.4f}, n={self.n_tokens:,})"
        )


@torch.no_grad()
def token_nll(
    model: nn.Module, batch: Batch, *, device: torch.device | None = None
) -> tuple[torch.Tensor, int]:
    """Per-token negative log-likelihoods for one batch, and the token count."""
    if device is not None:
        batch = batch.to(device)
    logits = model(batch.inputs).logits
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    targets = batch.targets
    valid = targets != IGNORE_INDEX
    gathered = logprobs.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return -gathered[valid], int(valid.sum())


@torch.no_grad()
def perplexity(
    model: nn.Module,
    batches: Sequence[Batch],
    *,
    device: torch.device | None = None,
) -> PerplexityResult:
    """Token-weighted perplexity with a standard error.

    The standard error is computed over the per-token negative log-likelihoods, which
    treats tokens as the sampling unit. That understates the true uncertainty slightly
    because tokens within a sequence are correlated: a caveat worth knowing, and better
    than reporting no interval at all.
    """
    was_training = model.training
    model.eval()
    try:
        chunks: list[torch.Tensor] = []
        total_tokens = 0
        for batch in batches:
            nll, count = token_nll(model, batch, device=device)
            if count:
                chunks.append(nll.detach().cpu())
                total_tokens += count
        if not chunks:
            return PerplexityResult(nll=float("nan"), nll_stderr=float("nan"), n_tokens=0)

        all_nll = torch.cat(chunks)
        mean = float(all_nll.mean())
        stderr = float(all_nll.std(unbiased=True) / math.sqrt(max(1, all_nll.numel())))
        return PerplexityResult(nll=mean, nll_stderr=stderr, n_tokens=total_tokens)
    finally:
        model.train(was_training)
