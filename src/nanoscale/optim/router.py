"""Parameter routing and the composite optimizer (spec B3).

The documented Muon split
-------------------------
Muon's argument is about the *action of a matrix as a linear map*: orthogonalizing the
momentum makes the update move equally in every direction the map can move. That
argument applies to the hidden matmul weights and to nothing else in the model:

* **Embeddings** are a lookup table. Row ``i`` is only ever touched by token ``i``, so
  there is no shared linear map to orthogonalize; treating it as a matrix mixes
  unrelated tokens' updates.
* **The LM head** has the same structure transposed, and additionally has a vocabulary
  dimension whose scale differs from everything else in the network.
* **Norm gains and biases** are 1D — a vector of independent scalars, where per-coordinate
  adaptivity is exactly what you want and Muon is undefined anyway.

So: 2D hidden weights to Muon, everything else to AdamW. This is the split the
modded-nanoGPT speedrun uses, and the two groups carry separate learning rates because
their natural scales differ by roughly an order of magnitude.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from nanoscale.config import OptimConfig
from nanoscale.optim.adamw import AdamW
from nanoscale.optim.muon import Muon

__all__ = [
    "ADAMW_NAME_PATTERNS",
    "CompositeOptimizer",
    "ParamSplit",
    "build_optimizer",
    "split_parameters",
]

#: Parameter-name fragments that force a tensor into the AdamW group even if it is 2D.
ADAMW_NAME_PATTERNS: tuple[str, ...] = (
    "embed_tokens",
    "lm_head",
    # MTP unembeddings are LM heads by another name.
    "mtp.heads",
)


@dataclass(slots=True)
class ParamSplit:
    """The result of routing a model's parameters into optimizer groups."""

    muon: list[Tensor] = field(default_factory=list)
    adamw: list[Tensor] = field(default_factory=list)
    muon_names: list[str] = field(default_factory=list)
    adamw_names: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        """Parameter counts per group, for logging and for the run manifest."""
        return {
            "muon_tensors": len(self.muon),
            "adamw_tensors": len(self.adamw),
            "muon_params": sum(p.numel() for p in self.muon),
            "adamw_params": sum(p.numel() for p in self.adamw),
        }


def split_parameters(
    model: nn.Module,
    *,
    adamw_patterns: Iterable[str] = ADAMW_NAME_PATTERNS,
) -> ParamSplit:
    """Route a model's trainable parameters into the Muon and AdamW groups.

    A parameter goes to Muon iff it is exactly 2D **and** its name matches none of
    ``adamw_patterns``. Everything else — 1D gains and biases, embeddings, the LM head,
    MTP unembeddings, and any >2D tensor — goes to AdamW.

    Tied parameters are routed once: the same tensor appearing under two names would
    otherwise be stepped twice per optimizer step.
    """
    patterns = tuple(adamw_patterns)
    split = ParamSplit()
    seen: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:  # weight tying: step each tensor exactly once
            continue
        seen.add(id(param))

        forced_adamw = any(pat in name for pat in patterns)
        if param.ndim == 2 and not forced_adamw:
            split.muon.append(param)
            split.muon_names.append(name)
        else:
            split.adamw.append(param)
            split.adamw_names.append(name)

    return split


class CompositeOptimizer:
    """Runs several optimizers as one, so the training loop stays optimizer-agnostic.

    Deliberately *not* a subclass of :class:`torch.optim.Optimizer`: the base class
    assumes a single ``param_groups`` list and a single flat state dict, and pretending
    to satisfy that contract while holding two independent optimizers is how subtle
    checkpoint bugs happen. This class exposes exactly what the trainer needs —
    ``step``, ``zero_grad``, ``state_dict``, ``load_state_dict``, and per-group LR
    control — and nothing it cannot honour.
    """

    def __init__(self, optimizers: dict[str, Optimizer]) -> None:
        """Wrap a mapping of ``name -> optimizer``."""
        self.optimizers = optimizers

    def __iter__(self) -> Iterator[Optimizer]:
        """Iterate over the wrapped optimizers."""
        return iter(self.optimizers.values())

    def __len__(self) -> int:
        """Number of wrapped optimizers."""
        return len(self.optimizers)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        """All parameter groups across all wrapped optimizers."""
        return [g for opt in self.optimizers.values() for g in opt.param_groups]

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Step every wrapped optimizer."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self.optimizers.values():
            opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients on every wrapped optimizer."""
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)

    def set_lr(self, scale: float) -> dict[str, float]:
        """Scale every group's LR relative to its configured peak.

        The peak is captured on the first call as ``initial_lr``, which is what makes
        this idempotent: calling ``set_lr(0.5)`` twice leaves the LR at half of peak,
        not a quarter.
        """
        current: dict[str, float] = {}
        for name, opt in self.optimizers.items():
            for i, group in enumerate(opt.param_groups):
                if "initial_lr" not in group:
                    group["initial_lr"] = group["lr"]
                group["lr"] = group["initial_lr"] * scale
                current[f"lr_{name}" if i == 0 else f"lr_{name}_{i}"] = group["lr"]
        return current

    def set_weight_decay_scale(self, scale: float) -> None:
        """Scale every group's weight decay (the decaying-``λ`` half of cautious WD)."""
        for opt in self.optimizers.values():
            for group in opt.param_groups:
                group["wd_scale"] = scale

    def learning_rates(self) -> dict[str, float]:
        """Current LR per optimizer, for logging."""
        return {
            f"lr_{name}": opt.param_groups[0]["lr"]
            for name, opt in self.optimizers.items()
            if opt.param_groups
        }

    def state_dict(self) -> dict[str, Any]:
        """Serialise every wrapped optimizer under its name."""
        return {name: opt.state_dict() for name, opt in self.optimizers.items()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state written by :meth:`state_dict`."""
        missing = set(self.optimizers) - set(state)
        if missing:
            raise KeyError(f"optimizer state is missing entries for {sorted(missing)}.")
        for name, opt in self.optimizers.items():
            opt.load_state_dict(state[name])


def build_optimizer(model: nn.Module, config: OptimConfig) -> CompositeOptimizer:
    """Build the optimizer stack described by ``config``.

    ``config.name == "adamw"`` routes everything to AdamW — the A/B baseline for
    Phase 5. ``config.name == "muon"`` uses the documented split.
    """
    split = split_parameters(model)

    if config.name == "adamw":
        params = split.muon + split.adamw
        return CompositeOptimizer(
            {
                "adamw": AdamW(
                    params,
                    lr=config.adamw_lr,
                    betas=config.betas,
                    eps=config.eps,
                    weight_decay=config.weight_decay,
                    cautious_weight_decay=config.cautious_weight_decay,
                )
            }
        )

    optimizers: dict[str, Optimizer] = {}
    if split.muon:
        optimizers["muon"] = Muon(
            split.muon,
            lr=config.lr,
            momentum=config.muon_momentum,
            nesterov=config.muon_nesterov,
            ns_steps=config.muon_ns_steps,
            weight_decay=config.weight_decay,
            cautious_weight_decay=config.cautious_weight_decay,
        )
    if split.adamw:
        optimizers["adamw"] = AdamW(
            split.adamw,
            lr=config.adamw_lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            cautious_weight_decay=config.cautious_weight_decay,
        )
    if not optimizers:
        raise ValueError("The model has no trainable parameters.")
    return CompositeOptimizer(optimizers)
