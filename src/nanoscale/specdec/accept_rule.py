r"""The speculative-sampling acceptance rule (spec B8): the correctness crown jewel.

References: Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*
(arXiv:2211.17192) and Chen et al., *Accelerating Large Language Model Decoding with
Speculative Sampling* (arXiv:2302.01318).

The claim
---------
Given a draft distribution ``q`` and a target distribution ``p``, the following procedure
emits tokens **exactly** from ``p``, while usually consulting ``p`` only once per several
tokens:

1. Draw ``x ~ q``.
2. Accept ``x`` with probability ``min(1, p(x)/q(x))``.
3. If rejected, draw a replacement from the **normalised residual**
   ``p'(x) = max(0, p(x) − q(x)) / Σ_v max(0, p(v) − q(v))``.

The proof
---------
Fix a token ``x``. It can be emitted by either branch:

*Accepted:* probability ``q(x) · min(1, p(x)/q(x)) = min(q(x), p(x))``.

*Resampled:* the rejection happens with probability
``β = 1 − Σ_v min(q(v), p(v))``, and the residual distribution assigns ``x`` mass
``max(0, p(x) − q(x)) / β``: because the normaliser
``Σ_v max(0, p(v) − q(v)) = 1 − Σ_v min(p(v), q(v)) = β``. So the resampled branch
contributes exactly ``max(0, p(x) − q(x))``.

Summing the two branches:

.. math::  \min(q(x), p(x)) + \max(0, p(x) - q(x)) = p(x)

for every ``x``, since if ``p(x) \ge q(x)`` the terms are ``q(x) + p(x) - q(x)``, and
otherwise they are ``p(x) + 0``. The output distribution is ``p``, exactly, **not
approximately**. Speculative decoding is lossless in distribution regardless of how bad
the draft model is; a bad draft only lowers the acceptance rate, i.e. the speedup.

This module keeps the rule as pure functions on probability vectors, with no model
involved, so ``tests/unit/test_specdec.py`` can verify the claim statistically over
100k samples against direct sampling from ``p``.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "acceptance_probability",
    "expected_acceptance_rate",
    "residual_distribution",
    "sample_accept_reject",
]


def acceptance_probability(target_probs: Tensor, draft_probs: Tensor, tokens: Tensor) -> Tensor:
    """``min(1, p(x)/q(x))`` for the drafted tokens.

    Args:
        target_probs: ``(..., V)`` target distribution ``p``.
        draft_probs: ``(..., V)`` draft distribution ``q``.
        tokens: ``(...)`` drafted token indices.

    Returns:
        ``(...)`` acceptance probabilities in ``[0, 1]``.
    """
    p = target_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    q = draft_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    # A draft token with q = 0 cannot have been sampled from q; guard anyway so the
    # rule never produces NaN if a caller passes a mismatched pair.
    return torch.clamp(p / q.clamp_min(1e-10), max=1.0)


def residual_distribution(target_probs: Tensor, draft_probs: Tensor) -> Tensor:
    """The normalised residual ``(p − q)₊ / Σ(p − q)₊`` used on rejection.

    When ``p`` and ``q`` are identical the residual is all zeros and has no valid
    normalisation; the rejection branch is then unreachable in exact arithmetic, but
    floating point can still get there, so this falls back to ``p``.
    """
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    total = residual.sum(dim=-1, keepdim=True)
    degenerate = total <= 1e-10
    safe = torch.where(degenerate, target_probs, residual)
    return safe / safe.sum(dim=-1, keepdim=True).clamp_min(1e-10)


def expected_acceptance_rate(target_probs: Tensor, draft_probs: Tensor) -> Tensor:
    r"""``Σ_v min(p(v), q(v))``: the probability a drafted token is accepted.

    This equals ``1 − TV(p, q)``, the total-variation *agreement* between the two
    distributions, which is the cleanest statement of what determines the speedup: a
    draft model helps exactly to the extent that its distribution overlaps the target's.
    """
    return torch.minimum(target_probs, draft_probs).sum(dim=-1)


def sample_accept_reject(
    target_probs: Tensor,
    draft_probs: Tensor,
    draft_token: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply the modified rejection rule to one drafted token per row.

    Args:
        target_probs: ``(B, V)`` target distribution.
        draft_probs: ``(B, V)`` draft distribution.
        draft_token: ``(B,)`` token sampled from ``draft_probs``.
        generator: RNG, for reproducibility.

    Returns:
        ``(token, accepted)``: the emitted token per row and a boolean acceptance mask.
    """
    alpha = acceptance_probability(target_probs, draft_probs, draft_token)
    uniform = torch.rand(alpha.shape, generator=generator, device=alpha.device)
    accepted = uniform < alpha

    residual = residual_distribution(target_probs, draft_probs)
    replacement = torch.multinomial(residual, num_samples=1, generator=generator).squeeze(-1)

    token = torch.where(accepted, draft_token, replacement)
    return token, accepted
