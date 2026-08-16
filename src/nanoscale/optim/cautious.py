r"""Cautious weight decay (spec B3, the Feb-2026 improvement).

Standard decoupled weight decay shrinks every coordinate toward zero on every step,
including the coordinates the optimizer is actively trying to grow. Those two forces
cancel, and what you observe is not "regularisation" but a tug-of-war that wastes part
of the update and biases the effective learning rate in a way that depends on the
gradient magnitude.

**Cautious weight decay** applies the decay only where it *agrees* with the optimizer.
Write the parameter update as ``p ← p - lr·u`` (so the optimizer moves by ``-lr·u``) and
the decay as ``p ← p - lr·λ·p`` (which moves by ``-lr·λ·p``, i.e. toward zero). The two
point the same way exactly when ``sign(u) == sign(p)``, i.e. when ``u · p > 0``. So the
mask is

.. math::  \text{mask}_i = \mathbb{1}[\,u_i \cdot p_i > 0\,]

and decay is applied only on the masked coordinates. Where they disagree, decay is
skipped entirely rather than partially cancelling the update.

This is the same "only act where the signs agree" idea as Cautious Optimizers
(Liang et al., C-AdamW/C-Lion), applied to the decay term rather than to the momentum
step. It is reported to help Muon in particular, which is why it is a flag on both
optimizers here rather than a property of one.

The complementary half of the 2026 recipe is a **decaying** ``λ`` schedule: decay hard
early, then release the constraint so the final weights are not artificially shrunk.
That is exposed as ``wd_scale`` on each param group and driven by the trainer, so the
optimizers themselves stay schedule-free.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["cautious_decay_mask", "cautious_mask_fraction"]


def cautious_decay_mask(param: Tensor, update: Tensor) -> Tensor:
    """Return a 0/1 mask selecting coordinates where decay agrees with the update.

    Args:
        param: The parameter tensor ``p``.
        update: The optimizer's update direction ``u``, where the step is ``p -= lr·u``.

    Returns:
        A tensor of the same shape and dtype as ``param``, holding ``1`` where
        ``u_i · p_i > 0`` and ``0`` elsewhere.
    """
    return (update * param > 0).to(param.dtype)


def cautious_mask_fraction(param: Tensor, update: Tensor) -> float:
    """Fraction of coordinates on which cautious decay would fire.

    Logged during the Phase-5 ablation: for a randomly-oriented update this sits near
    0.5, and watching it move away from 0.5 is how you tell the flag is doing anything.
    """
    if param.numel() == 0:
        return 0.0
    return float(torch.count_nonzero(update * param > 0)) / param.numel()
