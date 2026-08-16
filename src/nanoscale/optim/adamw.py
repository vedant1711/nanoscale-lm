r"""AdamW, implemented from scratch (spec B3).

Reference: Loshchilov & Hutter, *Decoupled Weight Decay Regularization*
(arXiv:1711.05101), which is the "W" in AdamW: weight decay is applied directly to the
parameter rather than folded into the gradient, so it is not rescaled by Adam's
second-moment normalisation. That distinction is the whole point of the algorithm and
is the thing most from-scratch reimplementations get wrong.

The update, at step ``t``:

.. math::

    m_t &= β_1 m_{t-1} + (1-β_1) g_t \\
    v_t &= β_2 v_{t-1} + (1-β_2) g_t^2 \\
    p_t &= p_{t-1} - λ\,\text{lr}\,p_{t-1}
           - \frac{\text{lr}}{1-β_1^t}\cdot
             \frac{m_t}{\sqrt{v_t}/\sqrt{1-β_2^t} + ε}

Note the placement of ``ε``: it is added to the *bias-corrected* denominator, outside
the square root. PyTorch does it this way, and matching it exactly is what lets
``tests/unit/test_optim.py`` assert agreement with ``torch.optim.AdamW`` to
1e-12 over hundreds of steps rather than "approximately".
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from nanoscale.optim.cautious import cautious_decay_mask

__all__ = ["AdamW"]


class AdamW(Optimizer):
    """Decoupled-weight-decay Adam.

    Args:
        params: Parameters or parameter groups.
        lr: Learning rate.
        betas: ``(β1, β2)`` exponential-decay rates for the first and second moments.
        eps: Added to the bias-corrected denominator (outside the square root).
        weight_decay: Decoupled decay coefficient ``λ``.
        cautious_weight_decay: If True, apply decay only to coordinates where it agrees
            with the optimizer's own update direction. See
            :mod:`nanoscale.optim.cautious`.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        cautious_weight_decay: bool = False,
    ) -> None:
        """Create an AdamW optimizer."""
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}.")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas must lie in [0, 1), got {betas}.")
        if eps < 0.0:
            # torch.optim.AdamW allows eps == 0; matching it keeps the parity test honest.
            raise ValueError(f"eps must be non-negative, got {eps}.")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}.")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "cautious_weight_decay": cautious_weight_decay,
            "wd_scale": 1.0,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        """Perform one optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"] * group.get("wd_scale", 1.0)
            cautious = group.get("cautious_weight_decay", False)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients.")

                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]

                # m_t and v_t.
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**t
                bias_correction2 = 1.0 - beta2**t

                # denom = sqrt(v_t)/sqrt(bc2) + eps  -- eps outside the sqrt, as torch does.
                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)
                step_size = lr / bias_correction1
                update = exp_avg / denom  # the direction we will move against

                if wd != 0.0:
                    if cautious:
                        mask = cautious_decay_mask(p, update)
                        p.add_(p * mask, alpha=-lr * wd)
                    else:
                        p.add_(p, alpha=-lr * wd)

                p.add_(update, alpha=-step_size)

        return loss
