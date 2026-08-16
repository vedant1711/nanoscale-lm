"""Device and dtype resolution.

Every code path in NanoScale has a CPU fallback (spec A3.1), so device selection is
centralised here and always resolvable to ``cpu``.
"""

from __future__ import annotations

import platform
from typing import Literal

import torch

__all__ = ["autocast_context", "hardware_string", "resolve_device", "resolve_dtype"]

DeviceSpec = Literal["auto", "cpu", "cuda", "mps"]


def resolve_device(spec: DeviceSpec | str = "auto") -> torch.device:
    """Resolve a device spec to a concrete :class:`torch.device`.

    ``"auto"`` prefers CUDA, then Apple MPS, then CPU. An explicit request for an
    unavailable accelerator falls back to CPU rather than crashing, because the whole
    project is meant to run on free/no hardware.
    """
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if spec == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(spec)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """Resolve an AMP dtype name, downgrading unsupported requests to fp32.

    bf16 autocast on CPU is supported by PyTorch but is slow and numerically noisy at
    this scale, and fp16 has no CPU autocast path at all, so both degrade to fp32 on
    CPU. MPS supports fp16 autocast only.
    """
    if name == "fp32":
        return torch.float32
    if device.type == "cpu":
        return torch.float32
    if name == "bf16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if device.type == "mps" else torch.float32
    if name == "fp16":
        return torch.float16
    return torch.float32


def autocast_context(device: torch.device, dtype: torch.dtype) -> torch.amp.autocast:
    """Return an autocast context; a no-op when the dtype is fp32."""
    enabled = dtype in (torch.float16, torch.bfloat16)
    return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def hardware_string() -> str:
    """A one-line hardware description recorded in every run manifest."""
    if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"cuda:{name} ({mem_gb:.1f} GiB) | {platform.platform()}"
    if torch.backends.mps.is_available():
        return f"mps:apple-silicon | {platform.platform()}"
    return f"cpu:{platform.processor() or platform.machine()} | {platform.platform()}"
