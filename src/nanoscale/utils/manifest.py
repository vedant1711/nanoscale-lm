"""Run manifests: the reproducibility contract (spec A3.4).

Every run writes a ``manifest.json`` recording the git SHA, config hash, seed, library
versions, hardware string and token budget. Every committed number in ``results/`` is
traceable back to one of these, which is what makes the "every claim is measured"
constraint (spec A3.5) enforceable rather than aspirational.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import torch

from nanoscale import __version__
from nanoscale.utils.device import hardware_string

__all__ = ["Manifest", "git_sha", "write_manifest"]


def git_sha(short: bool = True) -> str:
    """Return the current git SHA, or ``"unknown"`` outside a repository."""
    args = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
    if not short:
        args = ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return "unknown"
    if out.returncode != 0:
        return "unknown"
    return out.stdout.strip() or "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return False
    return bool(out.stdout.strip())


@dataclass(slots=True)
class Manifest:
    """A machine-readable record of exactly what produced an artifact."""

    run_name: str
    phase: str
    seed: int
    config_hash: str
    config: dict[str, Any]
    git_sha: str = field(default_factory=git_sha)
    git_dirty: bool = field(default_factory=_git_dirty)
    nanoscale_version: str = __version__
    python_version: str = field(default_factory=platform.python_version)
    torch_version: str = field(default_factory=lambda: torch.__version__)
    hardware: str = field(default_factory=hardware_string)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    token_budget: int | None = None
    tokens_consumed: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def finish(self, **metrics: float) -> Manifest:
        """Stamp the finish time and merge in final metrics."""
        self.finished_at = time.time()
        self.metrics.update(metrics)
        return self

    @property
    def wall_clock_s(self) -> float:
        """Elapsed wall-clock seconds (so far, or total if finished)."""
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at

    def to_dict(self) -> dict[str, Any]:
        """Serialise, including the derived wall-clock field."""
        payload = asdict(self)
        payload["wall_clock_s"] = round(self.wall_clock_s, 3)
        return payload

    def write(self, out_dir: str | Path) -> Path:
        """Write ``manifest.json`` into ``out_dir`` and return its path."""
        dest = Path(out_dir)
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "manifest.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")
        return path


class ConfigLike(Protocol):
    """Structural type for a NanoScale config, avoiding a utils -> config import cycle."""

    def config_hash(self) -> str:
        """Stable hash of the configuration."""
        ...

    def dump_inputs(self, *, mode: Literal["python", "json"] = ...) -> dict[str, Any]:
        """Dump the config's input fields."""
        ...


def write_manifest(
    out_dir: str | Path,
    *,
    run_name: str,
    phase: str,
    seed: int,
    config: ConfigLike | Mapping[str, Any],
    token_budget: int | None = None,
    **metrics: float,
) -> Manifest:
    """Build, finish and write a manifest in one call.

    Args:
        out_dir: Destination directory.
        run_name: Human-readable run name.
        phase: Spec phase this artifact belongs to (e.g. ``"phase4-pretrain"``).
        seed: The run's global seed.
        config: A NanoScale config object or a plain mapping.
        token_budget: Token budget for the run, if applicable.
        **metrics: Final metrics recorded alongside the manifest.
    """
    if isinstance(config, Mapping):
        cfg_dump = dict(config)
        cfg_hash = ""
    else:
        cfg_hash = config.config_hash()
        cfg_dump = config.dump_inputs(mode="json")
    manifest = Manifest(
        run_name=run_name,
        phase=phase,
        seed=seed,
        config_hash=cfg_hash,
        config=cfg_dump,
        token_budget=token_budget,
    )
    manifest.finish(**metrics)
    manifest.write(out_dir)
    return manifest
