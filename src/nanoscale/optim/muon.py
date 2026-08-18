r"""Muon: Momentum Orthogonalized by Newton–Schulz (spec B3).

Reference: Keller Jordan et al., *Muon: An optimizer for hidden layers in neural
networks*, and the modded-nanoGPT speedrun lineage. Muon was the single largest
wall-clock lever in that speedrun (larger than any architecture change), which is why
the spec singles it out.

The idea
--------
Adam normalises each *coordinate* by its own running gradient magnitude. For a weight
**matrix** that is the wrong geometry: what matters is the matrix's action as a linear
map, and a momentum buffer for a matrix is typically dominated by one or two singular
directions. Steps taken along it stretch the map anisotropically: one direction moves
far, the rest barely move.

Muon instead replaces the momentum matrix ``M`` with the nearest **semi-orthogonal**
matrix, i.e. the orthogonal factor of its polar decomposition:

.. math::  M = U Σ V^\top \;\longrightarrow\; O = U V^\top

Every singular value becomes 1, so the update moves *equally* in every direction the
momentum has support on. Empirically this is what buys the speedup.

Computing it cheaply
--------------------
An SVD per step per matrix would be far too slow. Instead Muon runs a fixed number of
**Newton–Schulz iterations** of a quintic polynomial in ``X``:

.. math::  X \leftarrow a X + b (X X^\top) X + c (X X^\top)^2 X

with ``(a, b, c) = (3.4445, -4.7750, 2.0315)``. Applied to a spectrally-normalised
``X``, this iteration pushes every singular value toward 1. The coefficients are
deliberately *not* the textbook Newton–Schulz values ``(1.5, -0.5, 0)``: they are tuned
so the map overshoots near zero, which converges far faster for the small singular
values that dominate a real momentum matrix. The cost is that convergence is not
monotone and the fixed point sits near ``1 ± 0.3`` rather than exactly 1, which is
fine, because only the *direction* is used. ``tests/unit/test_optim.py`` measures the
resulting singular values directly.

Every iteration is two matmuls of the smaller dimension, so five steps on a
``d × 4d`` matrix cost a small fraction of one forward pass.

Scope
-----
Muon applies to 2D hidden weight matrices only. Embeddings, the LM head, norm gains and
biases go to AdamW; that is the documented split from the speedrun, and it is what
:mod:`nanoscale.optim.router` implements. The reason is that Muon's whole argument is
about a matrix's action as a linear map; an embedding table is a lookup, and a norm gain
is a vector of independent scalars. Neither has the geometry the orthogonalisation is
exploiting.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any, Literal

import torch
from torch import Tensor
from torch.optim import Optimizer

from nanoscale.optim.cautious import cautious_decay_mask

__all__ = ["NS_COEFFS", "Muon", "newton_schulz_orthogonalize"]

#: The quintic Newton–Schulz coefficients used by the modded-nanoGPT speedrun.
NS_COEFFS: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)

ScaleMode = Literal["shape", "rms"]


def newton_schulz_orthogonalize(
    matrix: Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
    coeffs: tuple[float, float, float] = NS_COEFFS,
    compute_dtype: torch.dtype | None = None,
) -> Tensor:
    """Approximate the orthogonal polar factor of ``matrix`` by Newton–Schulz iteration.

    Args:
        matrix: A 2D tensor ``(rows, cols)``.
        steps: Number of quintic iterations. Five is the speedrun default; the singular
            values are already tightly clustered by then.
        eps: Floor on the spectral-normalisation denominator.
        coeffs: ``(a, b, c)`` of the quintic ``a X + b (XXᵀ)X + c (XXᵀ)²X``.
        compute_dtype: Precision for the iteration. Defaults to the input dtype;
            production runs use bf16 because only the direction matters, while the
            tests run fp64 to measure the singular values honestly.

    Returns:
        A tensor shaped like ``matrix`` whose singular values are all close to 1.

    Raises:
        ValueError: If ``matrix`` is not 2D or ``steps`` is negative.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Newton-Schulz expects a 2D matrix, got shape {tuple(matrix.shape)}.")
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}.")

    a, b, c = coeffs
    dtype = compute_dtype if compute_dtype is not None else matrix.dtype
    x = matrix.to(dtype)

    # Spectral normalisation. The Frobenius norm upper-bounds the spectral norm, so
    # this guarantees every singular value starts in (0, 1] -- the basin the tuned
    # quintic is designed for.
    x = x / (x.norm() + eps)

    # Operate on the wide orientation so each iteration's Gram matrix is the smaller
    # of the two possible ones: O(min(m,n)^2 * max(m,n)) instead of the other way round.
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.mT

    for _ in range(steps):
        gram = x @ x.mT
        poly = b * gram + c * (gram @ gram)
        x = a * x + poly @ x

    if transposed:
        x = x.mT
    out: Tensor = x.to(matrix.dtype)
    return out


def muon_update_scale(rows: int, cols: int, mode: ScaleMode = "shape") -> float:
    """Shape-aware scaling applied to an orthogonalized update.

    Orthogonalization throws away the momentum's magnitude entirely, so something has to
    put a sensible scale back. Two published choices:

    * ``"shape"`` (default, the original speedrun): ``sqrt(max(1, rows/cols))``. This
      keeps the update's Frobenius norm comparable across the tall and wide matrices in
      a transformer block.
    * ``"rms"`` (the Moonlight/Kimi adjustment): ``0.2 · sqrt(max(rows, cols))``, chosen
      so the update's RMS matches what AdamW would produce on the same tensor. This is
      what makes a single learning rate transferable between the two optimizers, which
      matters for the Phase-5 A/B.
    """
    if mode == "shape":
        return math.sqrt(max(1.0, rows / cols))
    if mode == "rms":
        return 0.2 * math.sqrt(max(rows, cols))
    raise ValueError(f"Unknown scale mode {mode!r}; expected 'shape' or 'rms'.")


class Muon(Optimizer):
    """Momentum-orthogonalized SGD for 2D hidden weight matrices.

    Args:
        params: Parameters or parameter groups. Every parameter must be 2D.
        lr: Learning rate.
        momentum: Heavy-ball momentum coefficient.
        nesterov: Use Nesterov-style lookahead momentum.
        ns_steps: Newton–Schulz iterations per step.
        weight_decay: Decoupled decay coefficient.
        cautious_weight_decay: Apply decay only where it agrees with the update.
        scale_mode: See :func:`muon_update_scale`.
        compute_dtype: Precision for the Newton–Schulz iteration.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        cautious_weight_decay: bool = False,
        scale_mode: ScaleMode = "shape",
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        """Create a Muon optimizer."""
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must lie in [0, 1), got {momentum}.")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be at least 1, got {ns_steps}.")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "weight_decay": weight_decay,
            "cautious_weight_decay": cautious_weight_decay,
            "scale_mode": scale_mode,
            "compute_dtype": compute_dtype,
            "wd_scale": 1.0,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only accepts 2D parameters, got shape {tuple(p.shape)}. "
                        "Route embeddings, heads, norms and biases to AdamW; see "
                        "nanoscale.optim.router."
                    )

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        """Perform one optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"] * group.get("wd_scale", 1.0)
            cautious = group.get("cautious_weight_decay", False)
            scale_mode: ScaleMode = group["scale_mode"]
            compute_dtype = group["compute_dtype"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients.")

                state = self.state[p]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["step"] = 0
                state["step"] += 1

                buf: Tensor = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                # Nesterov lookahead: step from where the momentum is about to be.
                direction = grad.add(buf, alpha=momentum) if nesterov else buf

                update = newton_schulz_orthogonalize(
                    direction, steps=ns_steps, compute_dtype=compute_dtype
                )
                update = update * muon_update_scale(p.shape[0], p.shape[1], scale_mode)

                if wd != 0.0:
                    if cautious:
                        mask = cautious_decay_mask(p, update)
                        p.add_(p * mask, alpha=-lr * wd)
                    else:
                        p.add_(p, alpha=-lr * wd)

                p.add_(update, alpha=-lr)

        return loss
