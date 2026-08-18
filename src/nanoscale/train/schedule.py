r"""Learning-rate schedules (spec B4).

Both schedules return a **multiplier in [0, 1]** applied to each parameter group's peak
LR, rather than an absolute learning rate. That is what lets Muon and AdamW run at
learning rates an order of magnitude apart under one shared schedule.

Cosine with linear warmup (default)
-----------------------------------
.. math::

    \text{mult}(t) = \begin{cases}
        t / T_w & t < T_w \\
        f + (1-f)\cdot\tfrac12\left(1 + \cos\pi\tfrac{t - T_w}{T - T_w}\right) & t \ge T_w
    \end{cases}

Warmup exists because Adam's second-moment estimate is badly calibrated for the first
few dozen steps: ``v`` is near zero, so the effective step is enormous, and a transformer
that takes one enormous step at init often never recovers. The cosine tail matters
because a model still at a high LR when the token budget runs out has not converged;
annealing recovers a large fraction of the remaining loss.

Warmup-stable-decay (WSD, flag)
--------------------------------
Warmup, then a long **constant** phase, then a short decay tail. Its advantage over
cosine is practical rather than theoretical: cosine's shape depends on the total step
count chosen up front, so extending a run means re-planning the whole schedule. WSD's
stable phase can be extended arbitrarily, and you can branch a decay tail off any
checkpoint to get a usable model. That is what makes it the natural fit for free-tier
compute, where a session can be pre-empted at any point.
"""

from __future__ import annotations

import math

from nanoscale.config import ScheduleConfig

__all__ = ["lr_multiplier", "make_schedule", "weight_decay_multiplier"]


def lr_multiplier(step: int, total_steps: int, config: ScheduleConfig) -> float:
    """Return the LR multiplier in ``[0, 1]`` for ``step`` out of ``total_steps``.

    Args:
        step: Zero-based optimizer step.
        total_steps: Planned total number of steps.
        config: Schedule configuration.

    Returns:
        A multiplier applied to every parameter group's peak learning rate.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}.")
    step = max(0, min(step, total_steps))
    warmup = max(1, round(config.warmup_frac * total_steps))
    floor = config.min_lr_frac

    if step < warmup:
        # Linear warmup, starting at one warmup-step's worth rather than exactly zero
        # so the very first step still makes progress.
        return (step + 1) / warmup

    if config.name == "constant":
        return 1.0

    if config.name == "cosine":
        span = max(1, total_steps - warmup)
        progress = min(1.0, (step - warmup) / span)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    if config.name == "wsd":
        decay_steps = max(1, round(config.decay_frac * total_steps))
        decay_start = max(warmup, total_steps - decay_steps)
        if step < decay_start:
            return 1.0
        progress = min(1.0, (step - decay_start) / max(1, total_steps - decay_start))
        # 1-sqrt decay: sharper than linear at the start of the tail, which recovers
        # more loss per step than a linear ramp at the same budget.
        return floor + (1.0 - floor) * (1.0 - math.sqrt(progress))

    raise ValueError(f"Unknown schedule {config.name!r}.")


def weight_decay_multiplier(step: int, total_steps: int, *, enabled: bool) -> float:
    """Decaying-``λ`` schedule for cautious weight decay (spec B3).

    Linear from 1 to 0 across the run when enabled, so the constraint is strongest early
    (when it regularises) and released by the end (so the final weights are not
    artificially shrunk). Returns a constant 1.0 when disabled.
    """
    if not enabled:
        return 1.0
    if total_steps <= 0:
        return 1.0
    return max(0.0, 1.0 - step / total_steps)


def make_schedule(config: ScheduleConfig, total_steps: int) -> list[float]:
    """Materialise the whole multiplier schedule: handy for plotting and for tests."""
    return [lr_multiplier(s, total_steps, config) for s in range(total_steps)]
