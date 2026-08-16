"""Metric logging to JSONL + CSV, plus a small console formatter.

Tracking is deliberately dependency-free (spec A6): every run writes ``metrics.jsonl``
and ``metrics.csv`` under its run directory, and the figure scripts in ``scripts/``
read those files. Weights & Biases is supported behind a flag but is never required.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import TracebackType
from typing import Any

__all__ = ["MetricLogger", "get_logger", "setup_logging"]

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root console logging once, idempotently."""
    root = logging.getLogger()
    if any(getattr(h, "_nanoscale", False) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt="%H:%M:%S"))
    handler._nanoscale = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, ensuring console logging is configured."""
    setup_logging()
    return logging.getLogger(name)


class MetricLogger:
    """Append-only metric sink writing JSONL and CSV side by side.

    The CSV is rewritten whenever a new column appears, so ad-hoc metrics can be added
    mid-run without corrupting the file. Rows are flushed immediately: a run killed by
    a Colab timeout still leaves usable partial curves.
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        name: str = "metrics",
        use_wandb: bool = False,
        wandb_project: str = "nanoscale-lm",
        wandb_run_name: str | None = None,
        wandb_config: dict[str, Any] | None = None,
    ) -> None:
        """Open a metric sink in ``out_dir``, optionally mirroring rows to W&B."""
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.dir / f"{name}.jsonl"
        self.csv_path = self.dir / f"{name}.csv"
        self.rows: list[dict[str, Any]] = []
        self._columns: list[str] = []
        self._t0 = time.time()
        self._log = get_logger("nanoscale.metrics")
        self._wandb: Any = None
        if use_wandb:  # pragma: no cover - optional dependency, never required
            try:
                import wandb  # type: ignore[import-not-found]

                self._wandb = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config=wandb_config or {},
                    dir=str(self.dir),
                )
            except Exception as exc:
                self._log.warning("W&B disabled (%s); falling back to CSV/JSONL only.", exc)

    def log(self, *, step: int, console: bool = False, **metrics: Any) -> dict[str, Any]:
        """Record one row of metrics; optionally echo a formatted line to the console."""
        row: dict[str, Any] = {"step": step, "elapsed_s": round(time.time() - self._t0, 3)}
        row.update(metrics)
        self.rows.append(row)
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        new_cols = [k for k in row if k not in self._columns]
        if new_cols:
            self._columns.extend(new_cols)
            self._rewrite_csv()
        else:
            with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=self._columns).writerow(row)
        if self._wandb is not None:  # pragma: no cover - optional
            self._wandb.log({k: v for k, v in row.items() if k != "step"}, step=step)
        if console:
            self._log.info(self.format_row(row))
        return row

    def format_row(self, row: dict[str, Any]) -> str:
        """Render a metric row as a compact aligned console line."""
        parts: list[str] = [f"step {int(row['step']):>6d}"]
        for key, value in row.items():
            if key in ("step", "elapsed_s"):
                continue
            if isinstance(value, float):
                fmt = f"{key} {value:>9.4f}" if abs(value) >= 1e-3 else f"{key} {value:>9.3e}"
                parts.append(fmt)
            else:
                parts.append(f"{key} {value}")
        parts.append(f"t {row['elapsed_s']:.1f}s")
        return " | ".join(parts)

    def _rewrite_csv(self) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._columns)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

    def summary(self, **metrics: Any) -> Path:
        """Write a ``summary.json`` with the run's headline numbers."""
        path = self.dir / "summary.json"
        path.write_text(json.dumps(metrics, indent=2, default=float) + "\n", encoding="utf-8")
        return path

    def close(self) -> None:
        """Finish any external tracker."""
        if self._wandb is not None:  # pragma: no cover - optional
            self._wandb.finish()
            self._wandb = None

    def __enter__(self) -> MetricLogger:
        """Enter the context; the sink is usable immediately after construction."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close any external tracker on exit; the CSV/JSONL are already flushed."""
        self.close()


def wandb_enabled() -> bool:
    """True when the ``NANOSCALE_WANDB`` env flag opts a run into W&B."""
    return os.environ.get("NANOSCALE_WANDB", "").lower() in ("1", "true", "yes")
