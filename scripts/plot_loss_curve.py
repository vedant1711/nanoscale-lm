"""Plot a training run's loss curve from its committed ``metrics.jsonl``.

Usage::

    python scripts/plot_loss_curve.py runs/nano/pretrain --out results/curves/nano.png

Every number on the figure comes from the run directory, and the figure is stamped with
the git SHA that produced it (spec F5).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from nanoscale.utils.plotting import COLORS, load_metrics, new_figure, save_figure, series


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="A training run directory.")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--title", type=str, default=None, help="Figure title.")
    args = parser.parse_args()

    rows = load_metrics(args.run_dir / "metrics.jsonl")
    if not rows:
        raise SystemExit(f"no metrics found in {args.run_dir}")
    summary_path = args.run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    train_steps, train_loss = series(rows, "loss")
    val_steps, val_loss = series(rows, "val_loss")

    fig, axes = new_figure(ncols=2, figsize=(11.0, 4.2))
    left, right = axes

    left.plot(train_steps, train_loss, color=COLORS[0], label="train")
    if val_loss:
        left.plot(val_steps, val_loss, color=COLORS[1], marker="o", ms=3.5, label="validation")
    if train_loss:
        left.axhline(
            train_loss[0],
            color="0.6",
            ls=":",
            lw=1.2,
            label=f"init = ln(V) = {train_loss[0]:.2f}",
        )
    left.set_xlabel("optimizer step")
    left.set_ylabel("cross-entropy (nats/token)")
    left.set_title(args.title or f"Loss — {args.run_dir.name}")
    left.legend()

    if train_loss:
        right.plot(
            train_steps,
            [math.exp(min(v, 20.0)) for v in train_loss],
            color=COLORS[0],
            label="train",
        )
    if val_loss:
        right.plot(
            val_steps,
            [math.exp(min(v, 20.0)) for v in val_loss],
            color=COLORS[1],
            marker="o",
            ms=3.5,
            label="validation",
        )
    right.set_yscale("log")
    right.set_xlabel("optimizer step")
    right.set_ylabel("perplexity (log scale)")
    right.set_title("Perplexity")
    right.legend()

    extra = ""
    if summary:
        extra = (
            f"{summary.get('params', 0):,} params · "
            f"{summary.get('tokens', 0):,} tokens · "
            f"val ppl {summary.get('final_val_perplexity', float('nan')):.2f}"
        )
    out = args.out or Path("results/curves") / f"{args.run_dir.name}_loss.png"
    path = save_figure(fig, out, script="scripts/plot_loss_curve.py", extra=extra)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
