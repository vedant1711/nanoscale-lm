r"""Distillation objectives (spec B6).

Three objectives, and the reason they differ is a single asymmetry in the KL divergence.

Forward KL: mode-covering
---------------------------
Hinton et al., *Distilling the Knowledge in a Neural Network* (arXiv:1503.02531):

.. math::

   \mathcal{L} = \alpha H(y, q_\theta)
              + (1-\alpha)\,\tau^2\,KL(p_\tau \,\|\, q_{\theta,\tau})

Minimising ``KL(p ‖ q)`` penalises ``q`` for putting *low* probability where ``p`` has
mass; the integrand is ``p log(p/q)``, which blows up wherever ``p > 0`` and ``q → 0``.
So the student is forced to **cover every mode** of the teacher, including the long tail
the teacher itself is unsure about. A student with less capacity than the teacher cannot
cover that tail without smearing probability across it, and the result is a model that
generates plausible-looking but incoherent text: it has learned the teacher's
*uncertainty* as well as its knowledge.

The ``τ²`` factor is not cosmetic. Dividing logits by ``τ`` scales the KD gradients by
``1/τ²``, so multiplying by ``τ²`` keeps the two loss terms' gradient magnitudes
comparable as ``τ`` changes: otherwise tuning ``τ`` silently retunes ``α`` as well.

Reverse KL: mode-seeking
--------------------------
Gu et al., *MiniLLM* (arXiv:2306.08543) minimise ``KL(q_θ ‖ p)`` instead. The integrand
is ``q log(q/p)``, which is only large where ``q`` has mass, so the student is penalised
for putting probability where the *teacher* does not, and is free to ignore the
teacher's tail entirely. It concentrates on the modes it can actually represent. For a
student strictly smaller than its teacher, that is the right trade.

The catch is that the expectation is under ``q_θ``, which appears in the sampling
distribution, so it cannot be computed by evaluating a fixed batch. It requires
**on-policy** rollouts and a policy-gradient estimator, which is what makes this the
expensive-but-better option, and what :func:`reverse_kl_policy_gradient` implements.

Sequence KD
-----------
Kim & Rush (arXiv:1606.07947): sample from the teacher, then train the student by plain
MLE on those samples. It is the cheapest of the three (no teacher forward pass during
training, just a one-off generation phase) and it sidesteps the KL asymmetry by
approximating the teacher's *distribution over sequences* with samples from it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

__all__ = [
    "DistillLossOutput",
    "forward_kl_loss",
    "reverse_kl_policy_gradient",
    "sequence_kd_loss",
    "token_kl",
]


@dataclass(frozen=True, slots=True)
class DistillLossOutput:
    """A distillation loss and its component diagnostics."""

    loss: Tensor
    ce: Tensor
    kd: Tensor
    extra: dict[str, float]

    def stats(self) -> dict[str, float]:
        """Scalar diagnostics for the metric log."""
        return {
            "loss": float(self.loss.detach()),
            "ce": float(self.ce.detach()),
            "kd": float(self.kd.detach()),
            **self.extra,
        }


def token_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    mask: Tensor,
    *,
    temperature: float = 1.0,
    reverse: bool = False,
) -> Tensor:
    r"""Mean per-token KL between the student and teacher distributions.

    Args:
        student_logits: ``(B, T, V)``.
        teacher_logits: ``(B, T, V)``, treated as a constant (detached).
        mask: ``(B, T)`` 1 where the position counts.
        temperature: Softens both distributions before comparing.
        reverse: If False compute ``KL(teacher ‖ student)`` (mode-covering); if True
            compute ``KL(student ‖ teacher)`` (mode-seeking).

    Returns:
        A scalar, averaged over masked positions.
    """
    s_logp = torch.log_softmax(student_logits.float() / temperature, dim=-1)
    t_logp = torch.log_softmax(teacher_logits.float().detach() / temperature, dim=-1)

    if reverse:
        # KL(q || p) = sum_v q(v) (log q(v) - log p(v))
        per_token = (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1)
    else:
        # KL(p || q) = sum_v p(v) (log p(v) - log q(v))
        per_token = (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1)

    denom = mask.sum().clamp_min(1.0)
    return (per_token * mask).sum() / denom


def forward_kl_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    *,
    temperature: float = 2.0,
    alpha: float = 0.5,
    ignore_index: int = -100,
) -> DistillLossOutput:
    r"""Hinton-style token-level knowledge distillation.

    ``L = α·H(y, q) + (1-α)·τ²·KL(p_τ ‖ q_τ)``: hard cross-entropy against the true
    next token, blended with the softened teacher distribution.
    """
    masked_targets = torch.where(mask.bool(), targets, torch.full_like(targets, ignore_index))
    ce = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]).float(),
        masked_targets.reshape(-1),
        ignore_index=ignore_index,
    )
    kd = token_kl(student_logits, teacher_logits, mask, temperature=temperature, reverse=False)
    # tau^2 keeps the KD gradient magnitude comparable across temperatures.
    loss = alpha * ce + (1.0 - alpha) * (temperature**2) * kd
    return DistillLossOutput(loss=loss, ce=ce, kd=kd, extra={"temperature": temperature})


def sequence_kd_loss(
    student_logits: Tensor,
    teacher_samples: Tensor,
    mask: Tensor,
    *,
    ignore_index: int = -100,
) -> DistillLossOutput:
    """SeqKD: plain MLE on sequences sampled from the teacher (Kim & Rush).

    No teacher forward pass is needed here; the teacher's contribution is entirely in
    having produced ``teacher_samples``, which is why this is the cheapest objective to
    train once the samples exist.
    """
    targets = torch.where(
        mask.bool(), teacher_samples, torch.full_like(teacher_samples, ignore_index)
    )
    ce = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]).float(),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )
    zero = ce.new_zeros(())
    return DistillLossOutput(loss=ce, ce=ce, kd=zero, extra={})


def reverse_kl_policy_gradient(
    student_logits: Tensor,
    teacher_logits: Tensor,
    sampled: Tensor,
    mask: Tensor,
    *,
    baseline: float = 0.0,
    length_normalize: bool = True,
    single_step_reg: bool = True,
) -> DistillLossOutput:
    r"""MiniLLM-style on-policy reverse KL, estimated by policy gradient.

    The objective is ``KL(q_θ ‖ p)`` where the expectation is over trajectories sampled
    from the student itself. Writing the per-token reward as
    ``r_t = log p(y_t|y_<t) − log q(y_t|y_<t)``, the gradient of the negative objective is

    .. math::  \nabla = \mathbb{E}_{y \sim q_\theta}\left[\sum_t (R_t - b)\,
                        \nabla \log q_\theta(y_t \mid y_{<t})\right]

    with ``R_t = \sum_{t' \ge t} r_{t'}`` the reward-to-go and ``b`` a baseline.

    Two details from the paper that matter in practice:

    * **Reward-to-go, not total reward.** Using the whole trajectory's reward for every
      token credits each action for outcomes that preceded it, which is pure variance.
    * **Single-step regularisation.** The pure policy-gradient term is high-variance and
      biased toward short sequences (fewer tokens, less accumulated negative reward). The
      paper adds a differentiable per-step ``KL(q(·|y_<t) ‖ p(·|y_<t))`` computed over
      the *whole vocabulary*, which supplies a low-variance signal at every position and
      removes the length shortcut.

    Args:
        student_logits: ``(B, T, V)``: differentiable.
        teacher_logits: ``(B, T, V)``: detached.
        sampled: ``(B, T)`` tokens sampled from the student.
        mask: ``(B, T)`` 1 on generated positions.
        baseline: Scalar baseline subtracted from the reward-to-go.
        length_normalize: Divide by the number of generated tokens per sequence.
        single_step_reg: Add the differentiable per-step reverse KL term.
    """
    s_logp_all = torch.log_softmax(student_logits.float(), dim=-1)
    t_logp_all = torch.log_softmax(teacher_logits.float().detach(), dim=-1)

    s_logp = s_logp_all.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    t_logp = t_logp_all.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

    # r_t = log p - log q, detached: it is the reward, not part of the gradient path.
    rewards = (t_logp - s_logp).detach() * mask
    # Reward-to-go: reverse-cumsum along time.
    reward_to_go = torch.flip(torch.cumsum(torch.flip(rewards, dims=[1]), dim=1), dims=[1])
    advantages = (reward_to_go - baseline) * mask

    per_seq = (advantages * s_logp * mask).sum(dim=1)
    if length_normalize:
        per_seq = per_seq / mask.sum(dim=1).clamp_min(1.0)
    pg = -per_seq.mean()

    reg = pg.new_zeros(())
    if single_step_reg:
        reg = token_kl(student_logits, teacher_logits, mask, reverse=True)

    loss = pg + reg
    return DistillLossOutput(
        loss=loss,
        ce=pg,
        kd=reg,
        extra={
            "mean_reward": float((rewards.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)).mean()),
            "mean_advantage": float(advantages.sum() / mask.sum().clamp_min(1.0)),
        },
    )
