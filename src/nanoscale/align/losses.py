r"""Preference-optimization losses, implemented from the papers (spec B5).

DPO
---
Rafailov et al., *Direct Preference Optimization* (arXiv:2305.18290). The insight is
that the RLHF objective's optimal policy has a closed form in terms of the reward, so
the reward can be *eliminated* and the preference likelihood written directly in terms
of the policy:

.. math::

   \mathcal{L}_{DPO} = -\mathbb{E}\left[\log\sigma\left(
       \beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
     - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]

The bracketed quantities are the **implicit rewards**. Two consequences that the
implementation has to respect:

* ``π_ref`` is a *frozen copy of the SFT model*, and it must be frozen and in eval mode.
  A reference that drifts turns the objective into something with no fixed point.
* The log-probabilities are **summed over the response tokens, not averaged**. That is
  what the derivation gives, and it is also the origin of the failure mode below.

**Length exploitation.** Because the reward is a *sum* of per-token log-ratios, a longer
response has more terms to accumulate advantage over. DPO can therefore reduce its loss
by making chosen responses longer rather than better. This is well documented and is
precisely what Phase 6's diagnostic measures, which is why
:func:`dpo_loss` returns the response lengths alongside the rewards.

SimPO
-----
Meng et al., *SimPO: Simple Preference Optimization with a Reference-Free Reward*. Two
changes, each addressing one of the above:

.. math::

   \mathcal{L}_{SimPO} = -\mathbb{E}\left[\log\sigma\left(
       \frac{\beta}{|y_w|}\log\pi_\theta(y_w|x)
     - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|x) - \gamma\right)\right]

1. **Length normalisation**: dividing by the response length makes the reward an
   *average* log-probability, which removes the mechanical incentive to lengthen.
2. **No reference model**: the reward is the policy's own length-normalised
   log-likelihood, so there is no frozen second copy in memory. At ``micro`` scale that
   is a ~40M-parameter saving; at production scale it is half the memory of the run.
3. A **target margin** ``γ``, which asks the model not merely to prefer the chosen
   response but to prefer it by a margin.

Both losses here operate on already-computed sequence log-probabilities so that they can
be unit-tested against hand-computed values on tiny fixtures, with no model involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

__all__ = [
    "PreferenceLossOutput",
    "dpo_loss",
    "sequence_logprobs",
    "simpo_loss",
]


@dataclass(frozen=True, slots=True)
class PreferenceLossOutput:
    """A preference loss plus the diagnostics Phase 6 reports."""

    loss: Tensor
    chosen_reward: Tensor
    rejected_reward: Tensor
    margin: Tensor
    accuracy: Tensor

    def stats(self) -> dict[str, float]:
        """Scalar diagnostics for the metric log."""
        return {
            "loss": float(self.loss.detach()),
            "chosen_reward": float(self.chosen_reward.detach().mean()),
            "rejected_reward": float(self.rejected_reward.detach().mean()),
            "reward_margin": float(self.margin.detach().mean()),
            "reward_accuracy": float(self.accuracy.detach().mean()),
        }


def sequence_logprobs(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    *,
    average: bool = False,
) -> Tensor:
    """Sum (or mean) the log-probabilities of ``targets`` over masked positions.

    Args:
        logits: ``(B, T, V)`` next-token logits.
        targets: ``(B, T)`` token IDs.
        mask: ``(B, T)`` 1 on response tokens, 0 on prompt/padding.
        average: If True, divide by the number of masked tokens (SimPO's length
            normalisation). If False, sum (DPO).

    Returns:
        ``(B,)`` sequence log-probabilities.
    """
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1).clamp_min(0)).squeeze(-1)
    masked = token_logprobs * mask
    totals = masked.sum(dim=-1)
    if not average:
        return totals
    counts = mask.sum(dim=-1).clamp_min(1.0)
    return totals / counts


def dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    ref_chosen_logps: Tensor,
    ref_rejected_logps: Tensor,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> PreferenceLossOutput:
    r"""Direct Preference Optimization loss.

    Args:
        policy_chosen_logps: ``(B,)`` summed log-probs of the chosen responses under π_θ.
        policy_rejected_logps: Same for the rejected responses.
        ref_chosen_logps: Same under the frozen reference π_ref.
        ref_rejected_logps: Same for the rejected responses under π_ref.
        beta: Inverse temperature on the implicit reward. Larger β means the policy is
            held more tightly to the reference.
        label_smoothing: cDPO-style smoothing, which assumes a fraction of preference
            labels are flipped. ``0`` recovers standard DPO.

    Returns:
        A :class:`PreferenceLossOutput`. ``chosen_reward`` is ``β·(logπ_θ − logπ_ref)``
        for the chosen response, i.e. the *implicit reward* the objective is fitting.
    """
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps)
    logits = chosen_reward - rejected_reward

    if label_smoothing > 0.0:
        # cDPO: mix in the loss of the flipped preference, weighted by the assumed
        # label-noise rate. With eps=0.5 the objective becomes symmetric and uninformative.
        losses = (
            -F.logsigmoid(logits) * (1.0 - label_smoothing)
            - F.logsigmoid(-logits) * label_smoothing
        )
    else:
        losses = -F.logsigmoid(logits)

    return PreferenceLossOutput(
        loss=losses.mean(),
        chosen_reward=chosen_reward,
        rejected_reward=rejected_reward,
        margin=logits,
        accuracy=(logits > 0).float(),
    )


def simpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    *,
    beta: float = 2.0,
    gamma: float = 0.5,
) -> PreferenceLossOutput:
    r"""SimPO loss: reference-free, length-normalised, with a target margin.

    Args:
        policy_chosen_logps: ``(B,)`` **length-normalised** log-probs of the chosen
            responses (i.e. produced with ``average=True``).
        policy_rejected_logps: Same for the rejected responses.
        beta: Reward scale.
        gamma: Target margin. The model is asked to prefer the chosen response by at
            least ``γ/β`` in average log-probability, not merely to prefer it.

    Returns:
        A :class:`PreferenceLossOutput`.
    """
    chosen_reward = beta * policy_chosen_logps
    rejected_reward = beta * policy_rejected_logps
    logits = chosen_reward - rejected_reward - gamma
    losses = -F.logsigmoid(logits)
    return PreferenceLossOutput(
        loss=losses.mean(),
        chosen_reward=chosen_reward,
        rejected_reward=rejected_reward,
        # Report the margin *without* the target offset, so it is comparable to DPO's.
        margin=chosen_reward - rejected_reward,
        accuracy=(chosen_reward > rejected_reward).float(),
    )
