"""Checkpoint save and resume (spec B4).

A checkpoint that only stores weights is not a resume point. To continue a run
*identically* — which spec Phase 4 requires and a test enforces — you need:

* model weights,
* **optimizer state** (Adam's moments, Muon's momentum buffers): dropping these
  restarts the optimizer cold and produces a visible loss bump,
* the **step counter and token count**, which drive the LR schedule, and
* the **RNG states**, so dropout and any sampling continue the same stream.

The data order does not need storing because it is a pure function of
``(seed, epoch)`` — see :class:`~nanoscale.train.data.TokenBatcher`.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn

from nanoscale.config import ExperimentConfig
from nanoscale.utils.logging import get_logger


class Stateful(Protocol):
    """Anything with a torch-style ``state_dict``/``load_state_dict`` pair."""

    def state_dict(self) -> dict[str, Any]:
        """Serialise internal state."""
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore internal state."""
        ...


__all__ = ["CHECKPOINT_VERSION", "TrainState", "load_checkpoint", "save_checkpoint"]

log = get_logger("nanoscale.train.checkpoint")

CHECKPOINT_VERSION = 1


@dataclass(slots=True)
class TrainState:
    """Everything about a run that is not a weight."""

    step: int = 0
    tokens: int = 0
    epoch: int = 0
    best_val_loss: float = float("inf")
    history: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainState:
        """Deserialise, tolerating older checkpoints missing newer fields."""
        return cls(
            step=int(payload.get("step", 0)),
            tokens=int(payload.get("tokens", 0)),
            epoch=int(payload.get("epoch", 0)),
            best_val_loss=float(payload.get("best_val_loss", float("inf"))),
            history=list(payload.get("history", [])),
        )


def _rng_states() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(states: dict[str, Any] | None) -> None:
    if not states:
        return
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(
        states["torch"].cpu() if torch.is_tensor(states["torch"]) else states["torch"]
    )
    if states.get("cuda") is not None and torch.cuda.is_available():  # pragma: no cover
        torch.cuda.set_rng_state_all(states["cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Stateful | None = None,
    state: TrainState | None = None,
    config: ExperimentConfig | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a resumable checkpoint and return its path.

    Args:
        path: Destination ``.pt`` file (parents are created).
        model: The model whose ``state_dict`` to save.
        optimizer: Anything with a ``state_dict()``; ``None`` writes a weights-only
            checkpoint (which is what the published inference artifacts use).
        state: Step/token counters and history.
        config: The experiment config, so a checkpoint is self-describing.
        extra: Any additional payload (e.g. a tokenizer path).
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "state": (state or TrainState()).to_dict(),
        "rng": _rng_states(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if config is not None:
        payload["config"] = config.dump_inputs(mode="json")
    if extra:
        payload["extra"] = extra
    torch.save(payload, dest)
    return dest


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module | None = None,
    optimizer: Stateful | None = None,
    restore_rng: bool = True,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[TrainState, dict[str, Any]]:
    """Load a checkpoint, optionally restoring model/optimizer/RNG in place.

    Returns:
        ``(state, payload)`` — the training state and the raw checkpoint dict, so
        callers can read ``config`` or ``extra`` without a second load.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        ValueError: If the checkpoint version is not understood.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"checkpoint {src} does not exist.")
    payload: dict[str, Any] = torch.load(src, map_location=map_location, weights_only=False)

    version = payload.get("version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"checkpoint {src} has version {version!r}; this build reads "
            f"version {CHECKPOINT_VERSION}."
        )

    if model is not None:
        model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        _restore_rng(payload.get("rng"))

    state = TrainState.from_dict(payload.get("state", {}))
    log.info("loaded checkpoint %s at step %d (%s tokens)", src, state.step, f"{state.tokens:,}")
    return state, payload


def load_config_from_checkpoint(path: str | Path) -> ExperimentConfig:
    """Read just the experiment config out of a checkpoint."""
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    if "config" not in payload:
        raise KeyError(f"checkpoint {path} does not embed a config.")
    return ExperimentConfig.model_validate(payload["config"])
