"""Small shims over torch APIs whose shipped type stubs are incomplete."""

from __future__ import annotations

import torch

__all__ = ["backward"]


def backward(loss: torch.Tensor, **kwargs: object) -> None:
    """Call ``Tensor.backward``, which torch ships without type annotations."""
    loss.backward(**kwargs)  # type: ignore[no-untyped-call]
