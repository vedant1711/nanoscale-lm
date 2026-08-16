"""Shared matplotlib styling for every committed figure.

One style module means every figure in ``results/`` looks like it came from the same
project, and it keeps the per-figure scripts to the data and the axes labels.

Matplotlib is imported with the ``Agg`` backend so figure generation works headlessly in
CI and on a Colab worker without a display.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

__all__ = [
    "COLORS",
    "annotate_provenance",
    "load_metrics",
    "new_figure",
    "save_figure",
]

#: A colour-blind-safe qualitative palette (Okabe-Ito), used for every categorical series.
COLORS: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)

_RC = {
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "lines.linewidth": 1.8,
}


def new_figure(
    *, nrows: int = 1, ncols: int = 1, figsize: tuple[float, float] = (7.0, 4.2)
) -> tuple[Figure, Any]:
    """Create a styled figure and axes."""
    # matplotlib types rcParams keys as a giant Literal union; a plain str-keyed dict
    # is the readable way to express a style, so the cast stays local to this line.
    plt.rcParams.update(_RC)  # type: ignore[arg-type]
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, constrained_layout=True)
    return fig, axes


def annotate_provenance(fig: Figure, *, script: str, extra: str = "") -> None:
    """Stamp the figure with the script and git SHA that produced it (spec F5).

    The layout rect is shrunk first so the stamp sits in reserved space rather than on
    top of the x-axis label.
    """
    from nanoscale.utils.manifest import git_sha

    note = f"{script} @ {git_sha()}"
    if extra:
        note = f"{note} · {extra}"
    engine = fig.get_layout_engine()
    if engine is not None:
        # ConstrainedLayoutEngine.set accepts `rect`; the base LayoutEngine stub does not.
        engine.set(rect=(0.0, 0.045, 1.0, 0.955))  # type: ignore[call-arg]
    fig.text(0.995, 0.008, note, ha="right", va="bottom", fontsize=6, alpha=0.55)


def save_figure(fig: Figure, path: str | Path, *, script: str, extra: str = "") -> Path:
    """Stamp, save and close a figure; returns the path."""
    annotate_provenance(fig, script=script, extra=extra)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)
    return dest


def load_metrics(path: str | Path) -> list[dict[str, Any]]:
    """Read a ``metrics.jsonl`` written by :class:`~nanoscale.utils.logging.MetricLogger`."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def series(rows: Sequence[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    """Extract ``(steps, values)`` for the rows that carry ``key``."""
    steps: list[float] = []
    values: list[float] = []
    for row in rows:
        if key in row and row[key] is not None:
            steps.append(float(row["step"]))
            values.append(float(row[key]))
    return steps, values
