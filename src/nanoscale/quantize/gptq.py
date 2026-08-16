r"""GPTQ: second-order, error-compensating weight quantization (spec B7).

Reference: Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative
Pre-trained Transformers* (arXiv:2210.17323), which is the practical descendant of
Optimal Brain Quantization.

The problem RTN gets wrong
---------------------------
Round-to-nearest minimises the error **in the weights**. What actually matters is the
error **in the layer's output**:

.. math::  \arg\min_{\hat{W}} \; \big\| WX - \hat{W}X \big\|_2^2

Those are the same objective only if ``X`` is white. It is not: transformer activations
have wildly different variances per channel, so a weight multiplying a high-variance
input channel matters far more than one multiplying a near-dead channel. RTN treats them
identically.

The algorithm
-------------
Expanding the objective gives a quadratic form in the weight error with Hessian
``H = 2XXᵀ`` (per output row, and shared across rows because every row sees the same
inputs). GPTQ quantizes **column by column**, and after fixing column ``j`` it
distributes that column's rounding error over the *not-yet-quantized* columns in the
direction that keeps the layer output closest to the original:

.. math::

   \delta_{j} = \frac{w_j - \mathrm{quant}(w_j)}{[H^{-1}]_{jj}}, \qquad
   W_{:,j+1:} \mathrel{-}= \delta_{j}\,[H^{-1}]_{j,\,j+1:}

The later columns absorb the earlier columns' mistakes. That single change is what turns
4-bit from unusable into production-viable.

Implementation details that matter
-----------------------------------
* **Cholesky, not an explicit inverse.** The update needs rows of ``H⁻¹`` in a fixed
  order, and the Cholesky factor of ``H⁻¹`` supplies exactly those while being
  numerically stable — a direct inverse of a near-singular Hessian is not.
* **Dampening.** Calibration Hessians are routinely rank-deficient (fewer samples than
  channels, dead channels). Adding ``λ·mean(diag(H))`` to the diagonal is what makes the
  factorisation exist at all.
* **Lazy batched updates.** The error is propagated inside a column block first and only
  applied to the rest of the matrix once per block, which turns a long chain of rank-1
  updates into one matmul.
* **Activation ordering** (``act_order``): quantize the highest-salience columns first,
  while the remaining budget for absorbing error is largest. This is the single biggest
  quality lever at 3 bits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from nanoscale.quantize.rtn import QuantizedTensor
from nanoscale.utils.logging import get_logger

__all__ = ["GPTQQuantizer", "HessianAccumulator", "gptq_quantize_layer"]

log = get_logger("nanoscale.quantize.gptq")


@dataclass(slots=True)
class HessianAccumulator:
    """Accumulates ``H = 2 XᵀX`` over calibration batches for one linear layer.

    ``X`` here is the layer's *input* with shape ``(n_tokens, in_features)``, so ``H`` is
    ``(in_features, in_features)`` and is shared across all output rows — every row of
    the weight matrix multiplies the same inputs.

    The running mean form (rather than a plain sum) keeps the magnitude independent of
    how many calibration tokens were used, so ``damp_percent`` means the same thing
    regardless of calibration-set size.
    """

    in_features: int
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    hessian: Tensor = field(init=False)
    n_samples: int = 0

    def __post_init__(self) -> None:
        """Allocate the accumulator."""
        self.hessian = torch.zeros(
            (self.in_features, self.in_features), device=self.device, dtype=torch.float32
        )

    def add(self, inputs: Tensor) -> None:
        """Accumulate one batch of layer inputs, shaped ``(..., in_features)``."""
        x = inputs.detach().reshape(-1, self.in_features).float()
        n_new = x.shape[0]
        if n_new == 0:
            return
        total = self.n_samples + n_new
        # Running mean: H <- H * (old/total) + 2 X^T X / total
        self.hessian *= self.n_samples / total
        self.hessian += (2.0 / total) * (x.T @ x)
        self.n_samples = total

    def finalize(self, damp_percent: float = 0.01) -> tuple[Tensor, Tensor]:
        """Return ``(H, dead_mask)`` with dampening applied and dead channels handled.

        A channel that is exactly zero across the whole calibration set contributes a
        zero row and column. Left alone it makes ``H`` singular; the standard treatment
        is to set its diagonal to 1 and zero the corresponding weights, since a weight
        multiplying an always-zero input cannot affect the output anyway.
        """
        h = self.hessian.clone()
        dead = torch.diag(h) == 0
        h[dead, dead] = 1.0
        damp = damp_percent * torch.mean(torch.diag(h))
        h += torch.eye(self.in_features, device=h.device) * damp
        return h, dead


def _quantize_column(
    column: Tensor, scale: Tensor, zero: Tensor, qmax: int
) -> tuple[Tensor, Tensor]:
    """Quantize one column to integer codes and back; returns ``(codes, dequantized)``."""
    codes = torch.clamp(torch.round(column / scale) + zero, 0, qmax)
    return codes, (codes - zero) * scale


def _group_params(block: Tensor, bits: int, symmetric: bool) -> tuple[Tensor, Tensor]:
    """Compute per-output-row ``(scale, zero)`` for one group of columns."""
    qmax = 2**bits - 1
    if symmetric:
        max_abs = block.abs().amax(dim=1, keepdim=True)
        scale = (2 * max_abs / qmax).clamp_min(1e-8)
        zero = torch.full_like(scale, (qmax + 1) / 2)
    else:
        w_min = block.amin(dim=1, keepdim=True)
        w_max = block.amax(dim=1, keepdim=True)
        scale = ((w_max - w_min) / qmax).clamp_min(1e-8)
        zero = torch.round(-w_min / scale)
    return scale, zero


def gptq_quantize_layer(
    weight: Tensor,
    hessian: Tensor,
    *,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = False,
    damp_percent: float = 0.01,
    act_order: bool = True,
    block_size: int = 128,
) -> QuantizedTensor:
    """Quantize one weight matrix with GPTQ's error-compensating column sweep.

    Args:
        weight: ``(out_features, in_features)``.
        hessian: ``(in_features, in_features)`` layer-input Hessian, already dampened.
        bits: Bit-width.
        group_size: Columns per scale group; ``-1`` for one group per row.
        symmetric: Symmetric quantization.
        damp_percent: Additional dampening applied here (the accumulator may have
            applied its own).
        act_order: Quantize high-salience columns first.
        block_size: Columns per lazy-update block.

    Returns:
        A :class:`QuantizedTensor` whose ``dequantize()`` gives the quantized weights.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2D weight, got {tuple(weight.shape)}.")
    out_features, in_features = weight.shape
    if hessian.shape != (in_features, in_features):
        raise ValueError(
            f"hessian shape {tuple(hessian.shape)} does not match in_features={in_features}."
        )

    group = in_features if group_size in (-1, 0) else min(group_size, in_features)
    if in_features % group != 0:
        raise ValueError(f"in_features={in_features} not divisible by group_size={group}.")
    qmax = 2**bits - 1

    w = weight.detach().float().clone()
    h = hessian.detach().float().clone()

    dead = torch.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0
    if damp_percent > 0:
        h += torch.eye(in_features, device=h.device) * (damp_percent * torch.diag(h).mean())

    perm = torch.arange(in_features, device=w.device)
    if act_order:
        # Highest Hessian diagonal = highest activation energy = quantize first, while
        # the most un-quantized columns remain to absorb the error.
        perm = torch.argsort(torch.diag(h), descending=True)
        w = w[:, perm]
        h = h[perm][:, perm]

    # Cholesky of H^-1 gives the rows of H^-1 we need, stably.
    h_inv = torch.cholesky_inverse(torch.linalg.cholesky(h))
    h_chol = torch.linalg.cholesky(h_inv, upper=True)

    codes = torch.zeros_like(w)
    scales = torch.zeros((out_features, in_features // group), device=w.device)
    zeros = torch.zeros_like(scales)

    for block_start in range(0, in_features, block_size):
        block_end = min(block_start + block_size, in_features)
        width = block_end - block_start

        block_w = w[:, block_start:block_end].clone()
        block_err = torch.zeros_like(block_w)
        block_h = h_chol[block_start:block_end, block_start:block_end]

        for i in range(width):
            col = block_start + i
            if col % group == 0:
                g = col // group
                # Scales come from the *current* (already partially compensated) weights,
                # which is what lets later groups absorb earlier groups' error.
                end = min(col + group, in_features)
                scale, zero = _group_params(w[:, col:end], bits, symmetric)
                scales[:, g] = scale.squeeze(-1)
                zeros[:, g] = zero.squeeze(-1)

            g = col // group
            scale = scales[:, g : g + 1]
            zero = zeros[:, g : g + 1]
            column = block_w[:, i : i + 1]

            code, dequantized = _quantize_column(column, scale, zero, qmax)
            codes[:, col : col + 1] = code

            d = block_h[i, i]
            error = (column - dequantized) / d
            # Push this column's error onto the not-yet-quantized columns in this block.
            if i + 1 < width:
                block_w[:, i + 1 :] -= error @ block_h[i, i + 1 :].unsqueeze(0)
            block_err[:, i : i + 1] = error

        # One matmul applies the whole block's accumulated error to the remaining columns.
        if block_end < in_features:
            w[:, block_end:] -= block_err @ h_chol[block_start:block_end, block_end:]

    # Codes and scales are in the permuted layout; the permutation travels with them so
    # `dequantize` can invert it exactly. Re-encoding the reconstructed weights instead
    # would run a second rounding pass over values GPTQ had already carefully placed --
    # measurably worse at low bit-widths, and a bug this frontier caught.
    return QuantizedTensor(
        codes=codes.to(torch.int32),
        scales=scales,
        zeros=zeros,
        bits=bits,
        group_size=group,
        symmetric=symmetric,
        original_shape=(out_features, in_features),
        perm=perm if act_order else None,
    )


class GPTQQuantizer:
    """Runs GPTQ over a model: collect Hessians on calibration data, then quantize."""

    def __init__(
        self,
        model: nn.Module,
        *,
        bits: int = 4,
        group_size: int = 128,
        symmetric: bool = False,
        damp_percent: float = 0.01,
        act_order: bool = True,
        block_size: int = 128,
        skip: tuple[str, ...] = ("embed_tokens", "lm_head"),
    ) -> None:
        """Prepare the quantizer and locate the eligible linear layers."""
        self.model = model
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.damp_percent = damp_percent
        self.act_order = act_order
        self.block_size = block_size
        self.layers: dict[str, nn.Linear] = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and not any(p in name for p in skip)
        }
        self.accumulators: dict[str, HessianAccumulator] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> GPTQQuantizer:
        """Attach forward hooks that accumulate each layer's input Hessian."""
        for name, module in self.layers.items():
            acc = HessianAccumulator(module.in_features, device=module.weight.device)
            self.accumulators[name] = acc

            def hook(
                _module: nn.Module,
                inputs: tuple[Tensor, ...],
                _output: Tensor,
                _acc: HessianAccumulator = acc,
            ) -> None:
                _acc.add(inputs[0])

            self._handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, *exc: object) -> None:
        """Remove the hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @torch.no_grad()
    def collect(self, batches: list[Tensor]) -> None:
        """Run calibration batches through the model to fill the Hessians."""
        was_training = self.model.training
        self.model.eval()
        with self:
            for batch in batches:
                self.model(batch)
        self.model.train(was_training)

    @torch.no_grad()
    def apply(self) -> dict[str, float]:
        """Quantize every collected layer in place; returns per-layer relative error."""
        if not self.accumulators:
            raise RuntimeError("call collect() before apply().")
        errors: dict[str, float] = {}
        for name, module in self.layers.items():
            acc = self.accumulators[name]
            if acc.n_samples == 0:
                log.warning("layer %s saw no calibration data; skipping", name)
                continue
            hessian, _ = acc.finalize(self.damp_percent)
            original = module.weight.detach().clone()
            q = gptq_quantize_layer(
                module.weight.data,
                hessian,
                bits=self.bits,
                group_size=self.group_size,
                symmetric=self.symmetric,
                damp_percent=0.0,  # already applied by finalize()
                act_order=self.act_order,
                block_size=self.block_size,
            )
            module.weight.data.copy_(q.dequantize().to(module.weight.dtype))
            errors[name] = float(
                (module.weight.data - original).norm() / original.norm().clamp_min(1e-12)
            )
        return errors
