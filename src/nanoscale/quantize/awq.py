r"""AWQ-style activation-aware weight scaling (spec B7).

Reference: Lin et al., *AWQ: Activation-aware Weight Quantization* (arXiv:2306.00978).

The observation
---------------
Not all weights matter equally, and which ones matter is determined by the
**activations**, not by the weights. Keeping just ~1% of weight channels — those
multiplying high-magnitude activation channels — in fp16 recovers most of the quality
lost to 4-bit quantization. But mixed-precision storage is awkward on real hardware.

The trick
---------
Achieve the same protection with a **purely mathematical** transformation. For a linear
layer, scaling input channel ``j`` by ``s_j`` and the corresponding weight column by
``1/s_j`` leaves the output identical:

.. math::  (X \oslash s)\,(W \odot s)^\top = X W^\top

Now quantize ``W ⊙ s``. A salient channel with a large ``s_j`` has its weights scaled
*up* before quantization, so relative to its group's scale it lands on a coarser part of
the grid — meaning its rounding error, once divided back by ``s_j``, is smaller. Salient
channels are protected, everything stays uniformly quantized, and inference needs no
mixed-precision kernel.

Choosing the scales
-------------------
AWQ parameterises ``s = mean(|X_j|)^α`` and grid-searches ``α ∈ [0, 1]``, picking the
value that minimises the actual output error ``‖WX − Ŵ X‖``. ``α = 0`` recovers plain
RTN; ``α = 1`` scales fully by activation magnitude. The search is over a single scalar
per layer, so it is cheap — and searching on the true objective rather than a proxy is
what makes it robust.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from nanoscale.quantize.rtn import quantize_tensor_rtn
from nanoscale.utils.logging import get_logger

__all__ = ["AWQQuantizer", "ActivationStats", "awq_quantize_layer", "search_awq_scale"]

log = get_logger("nanoscale.quantize.awq")


@dataclass(slots=True)
class ActivationStats:
    """Per-input-channel activation magnitudes collected on calibration data."""

    mean_abs: Tensor  # (in_features,)
    n_samples: int = 0

    @classmethod
    def empty(cls, in_features: int, device: torch.device) -> ActivationStats:
        """An all-zero accumulator."""
        return cls(mean_abs=torch.zeros(in_features, device=device, dtype=torch.float32))

    def add(self, inputs: Tensor) -> None:
        """Accumulate the running mean of ``|x|`` per channel."""
        x = inputs.detach().reshape(-1, self.mean_abs.shape[0]).float().abs()
        n_new = x.shape[0]
        if n_new == 0:
            return
        total = self.n_samples + n_new
        self.mean_abs *= self.n_samples / total
        self.mean_abs += x.sum(dim=0) / total
        self.n_samples = total

    def salience(self) -> Tensor:
        """Normalised per-channel salience, safe against all-zero channels."""
        s = self.mean_abs.clamp_min(1e-8)
        return s / s.mean().clamp_min(1e-8)


def search_awq_scale(
    weight: Tensor,
    salience: Tensor,
    sample_input: Tensor,
    *,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = False,
    grid: int = 20,
) -> tuple[float, Tensor, float]:
    r"""Grid-search the AWQ scale exponent ``α``.

    Args:
        weight: ``(out, in)``.
        salience: ``(in,)`` normalised per-channel activation magnitude.
        sample_input: ``(n, in)`` calibration activations used to score candidates.
        bits: Weight bit-width.
        group_size: Weights per scale group along the input dimension.
        symmetric: Symmetric (zero-point-free) quantization.
        grid: Number of ``α`` values searched in ``[0, 1]``. ``0`` evaluates only
            ``α = 0``, which is exactly plain RTN and is what the tests use as the
            control arm.

    Returns:
        ``(best_alpha, best_scales, best_error)`` where ``best_error`` is the relative
        output error ``‖WX − ŴX‖ / ‖WX‖``.

    Scoring on the **output** error rather than the weight error is the point: it is the
    same objective GPTQ targets, reached by a different mechanism.
    """
    reference = sample_input @ weight.T
    ref_norm = reference.norm().clamp_min(1e-12)

    best_alpha = 0.0
    best_scales = torch.ones_like(salience)
    best_error = float("inf")

    for i in range(grid + 1):
        alpha = i / grid if grid > 0 else 0.0
        scales = salience.pow(alpha).clamp(1e-4, 1e4)
        # Normalising keeps the scaled weights in the same overall range across alphas,
        # so the comparison is about *relative* channel emphasis, not global magnitude.
        scales = scales / scales.mean().clamp_min(1e-8)

        scaled = weight * scales.unsqueeze(0)
        q = quantize_tensor_rtn(scaled, bits=bits, group_size=group_size, symmetric=symmetric)
        restored = q.dequantize() / scales.unsqueeze(0)

        error = float((sample_input @ restored.T - reference).norm() / ref_norm)
        if error < best_error:
            best_alpha, best_scales, best_error = alpha, scales, error

    return best_alpha, best_scales, best_error


def awq_quantize_layer(
    weight: Tensor,
    salience: Tensor,
    sample_input: Tensor,
    *,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = False,
    grid: int = 20,
) -> tuple[Tensor, float, float]:
    """Quantize one layer with the searched AWQ scaling.

    Returns:
        ``(quantized_weight, alpha, relative_output_error)``.
    """
    alpha, scales, error = search_awq_scale(
        weight,
        salience,
        sample_input,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        grid=grid,
    )
    scaled = weight * scales.unsqueeze(0)
    q = quantize_tensor_rtn(scaled, bits=bits, group_size=group_size, symmetric=symmetric)
    return q.dequantize() / scales.unsqueeze(0), alpha, error


class AWQQuantizer:
    """Collects activation statistics, then quantizes with searched per-channel scaling."""

    def __init__(
        self,
        model: nn.Module,
        *,
        bits: int = 4,
        group_size: int = 128,
        symmetric: bool = False,
        grid: int = 20,
        max_calib_rows: int = 2048,
        skip: tuple[str, ...] = ("embed_tokens", "lm_head"),
    ) -> None:
        """Prepare the quantizer and locate the eligible linear layers."""
        self.model = model
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.grid = grid
        self.max_calib_rows = max_calib_rows
        self.layers: dict[str, nn.Linear] = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and not any(p in name for p in skip)
        }
        self.stats: dict[str, ActivationStats] = {}
        self.samples: dict[str, list[Tensor]] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.alphas: dict[str, float] = {}

    def __enter__(self) -> AWQQuantizer:
        """Attach hooks collecting per-channel magnitudes and a slice of raw inputs."""
        for name, module in self.layers.items():
            self.stats[name] = ActivationStats.empty(module.in_features, module.weight.device)
            self.samples[name] = []

            def hook(
                _module: nn.Module,
                inputs: tuple[Tensor, ...],
                _output: Tensor,
                _name: str = name,
            ) -> None:
                x = inputs[0].detach().reshape(-1, self.layers[_name].in_features)
                self.stats[_name].add(x)
                # Keep a bounded slice of real activations to score candidate scales on.
                held = self.samples[_name]
                budget = self.max_calib_rows - sum(t.shape[0] for t in held)
                if budget > 0:
                    held.append(x[:budget].float().clone())

            self._handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, *exc: object) -> None:
        """Remove the hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @torch.no_grad()
    def collect(self, batches: list[Tensor]) -> None:
        """Run calibration batches through the model."""
        was_training = self.model.training
        self.model.eval()
        with self:
            for batch in batches:
                self.model(batch)
        self.model.train(was_training)

    @torch.no_grad()
    def apply(self) -> dict[str, float]:
        """Quantize every collected layer in place; returns per-layer relative error."""
        if not self.stats:
            raise RuntimeError("call collect() before apply().")
        errors: dict[str, float] = {}
        for name, module in self.layers.items():
            samples = self.samples.get(name) or []
            if not samples:
                log.warning("layer %s saw no calibration data; skipping", name)
                continue
            sample_input = torch.cat(samples, dim=0)
            quantized, alpha, error = awq_quantize_layer(
                module.weight.data.float(),
                self.stats[name].salience(),
                sample_input,
                bits=self.bits,
                group_size=self.group_size,
                symmetric=self.symmetric,
                grid=self.grid,
            )
            module.weight.data.copy_(quantized.to(module.weight.dtype))
            self.alphas[name] = alpha
            errors[name] = error
        return errors
