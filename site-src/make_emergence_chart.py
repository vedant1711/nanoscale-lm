"""Emergence chart with collision-free right-hand labels.

The first version placed each curve's label at the y of its final data point. Where curves
converge -- and seven of nine end above 78% -- the labels landed on top of each other and
became unreadable. This version runs a small label-layout pass: sort by preferred y, then
push labels apart until each has room, and draw a leader line back to the curve so the
association survives being moved.
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(__file__).parent
ACC, WARN = "#12655F", "#8F4715"
PALETTE = ["#12655F", "#2C6A33", "#3A6EA5", "#7A5EA8", "#B08A2E", "#5A8F7B", "#8F4715"]


def layout(preferred: list[float], *, gap: float, lo: float, hi: float) -> list[float]:
    """Spread labels so none is within ``gap`` of another, staying inside [lo, hi].

    Two passes: sweep down resolving overlaps in order, then sweep up to pull anything
    that got pushed past the bottom back into range. This is the standard fix and it is
    stable, which matters because the labels must keep their vertical ordering or the
    leader lines will cross.
    """
    order = sorted(range(len(preferred)), key=lambda i: preferred[i])
    y = list(preferred)

    prev = lo - gap
    for i in order:
        y[i] = max(y[i], prev + gap)
        prev = y[i]

    prev = hi + gap
    for i in reversed(order):
        y[i] = min(y[i], prev - gap)
        prev = y[i]
    return y


def build() -> str:
    """Render the chart as inline SVG."""
    d = json.loads((ROOT / "results/emergence/emergence.json").read_text())
    probes, trends = d["probes"], d["trends"]
    names = list(probes[0]["phenomena"].keys())

    w, h = 720, 430
    left, right, top, bottom = 56, 196, 40, 84
    xs = [p["tokens"] / 1e6 for p in probes]
    xmax = max(xs)

    def px(v: float) -> float:
        return left + v / xmax * (w - left - right)

    def py(v: float) -> float:
        return h - bottom - (v - 25) / (103 - 25) * (h - top - bottom)

    s: list[str] = []
    for v in (25, 50, 75, 100):
        s.append(
            f'<line x1="{left}" y1="{py(v):.1f}" x2="{w - right}" y2="{py(v):.1f}" '
            f'stroke="currentColor" stroke-width=".7" opacity=".13"/>'
        )
        s.append(
            f'<text x="{left - 8}" y="{py(v) + 3.5:.1f}" text-anchor="end" opacity=".6">{v}%</text>'
        )
    s.append(
        f'<line x1="{left}" y1="{py(50):.1f}" x2="{w - right}" y2="{py(50):.1f}" '
        f'stroke="currentColor" stroke-width="1.3" stroke-dasharray="5 3" opacity=".55"/>'
    )
    s.append(
        f'<text x="{left + 6}" y="{py(50) - 6:.1f}" font-size="9.5" opacity=".7">chance</text>'
    )

    ordered = sorted(names, key=lambda n: -trends[n]["spearman_rho"])
    finals = [probes[-1]["phenomena"][n] * 100 for n in ordered]
    label_y = layout([py(v) for v in finals], gap=26.0, lo=top + 8, hi=h - bottom - 4)

    for i, name in enumerate(ordered):
        ys = [p["phenomena"][name] * 100 for p in probes]
        rho, pv = trends[name]["spearman_rho"], trends[name]["p_value"]
        flat = rho <= 0.05
        col = WARN if flat else PALETTE[i % len(PALETTE)]
        dash = ' stroke-dasharray="4 3"' if flat else ""
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys, strict=True))
        s.append(
            f'<polyline points="{pts}" fill="none" stroke="{col}" '
            f'stroke-width="{2.4 if flat else 1.7}"{dash} opacity="{1 if flat else 0.9}"/>'
        )

        # Leader from the curve's end to its (possibly displaced) label.
        ey, ly = py(ys[-1]), label_y[i]
        s.append(
            f'<path d="M {px(xs[-1]):.1f} {ey:.1f} L {w - right + 6:.1f} {ly:.1f}" '
            f'fill="none" stroke="{col}" stroke-width=".9" opacity=".5"/>'
        )
        s.append(f'<circle cx="{px(xs[-1]):.1f}" cy="{ey:.1f}" r="2.6" fill="{col}"/>')
        s.append(
            f'<text x="{w - right + 11:.1f}" y="{ly + 2:.1f}" font-size="9.5" fill="{col}">'
            f"{name}</text>"
        )
        s.append(
            f'<text x="{w - right + 11:.1f}" y="{ly + 13:.1f}" font-size="8" fill="{col}" '
            f'opacity=".72">rho={rho:+.2f} p={pv:.3f}</text>'
        )

    for i in range(5):
        v = xmax * i / 4
        s.append(
            f'<text x="{px(v):.1f}" y="{h - bottom + 17}" text-anchor="middle" opacity=".6">{v:.0f}M</text>'
        )
    s.append(
        f'<text x="{(left + w - right) / 2:.0f}" y="{h - bottom + 35}" text-anchor="middle" '
        f'opacity=".6">training tokens</text>'
    )
    s.append(
        f'<text x="{left}" y="{top - 16}" font-size="11" font-weight="600">'
        f"Minimal-pair accuracy through one training run, 12.8M params, 16 probes</text>"
    )
    s.append(
        f'<text x="{left}" y="{h - 16}" font-size="9.5" opacity=".7">'
        f"Dashed amber: no positive trend with training. "
        f"Validation loss over the same run: 3.106 to 1.774.</text>"
    )

    return (
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Seven of nine grammatical '
        f"phenomena improve with training while negation stays exactly at chance and "
        f'agreement with an attractor trends slightly downward." style="width:100%;height:auto">'
        f'<g font-family="ui-monospace, monospace" font-size="10" fill="currentColor">'
        + "\n".join(s)
        + "</g></svg>"
    )


if __name__ == "__main__":
    (OUT / "chart_emergence.svg").write_text(build())
    print("wrote chart_emergence.svg")
