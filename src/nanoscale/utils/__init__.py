"""Cross-cutting utilities: seeding, devices, manifests, logging and plotting."""

from __future__ import annotations

from nanoscale.utils.device import (
    autocast_context,
    hardware_string,
    resolve_device,
    resolve_dtype,
)
from nanoscale.utils.logging import MetricLogger, get_logger, setup_logging
from nanoscale.utils.manifest import Manifest, git_sha, write_manifest
from nanoscale.utils.seed import derive_seed, seed_all, seed_worker, temporary_seed

__all__ = [
    "Manifest",
    "MetricLogger",
    "autocast_context",
    "derive_seed",
    "get_logger",
    "git_sha",
    "hardware_string",
    "resolve_device",
    "resolve_dtype",
    "seed_all",
    "seed_worker",
    "setup_logging",
    "temporary_seed",
    "write_manifest",
]
