"""Generate theme-aware inline SVG charts from the committed run data.

matplotlib PNGs are baked for one background colour and would look wrong in the artifact's
dark theme, so the charts that matter are re-drawn as inline SVG using `currentColor`.
"""

from __future__ import annotations

import json
import math
import pathlib

ROOT = pathlib.Path(
    "/Users/vedantsomani/STLP Mac Projects/Hackathons/Data Science Projects/"
    "nanoscale_lm_project/nanoscale-lm"
)
OUT = pathlib.Path(__file__).parent
ACC = "#12655F"


def load_metrics(
    path: pathlib.Path,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Read a metrics.jsonl into (train, val) step/loss pairs."""
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    train = [(r["step"], r["loss"]) for r in rows if "loss" in r]
    val = [(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]
    return train, val


def loss_chart(
    train: list[tuple[int, float]],
    val: list[tuple[int, float]],
    *,
    chance: float,
    title: str,
    sub: str,
    w: int = 660,
    h: int = 250,
) -> str:
    """Line chart of train/validation loss against the uniform-guess floor."""
    L, R, T, B = 52, 16, 30, 44
    xs = [s for s, _ in train]
    x0, x1 = 0, max(xs)
    y0 = 0.0
    y1 = math.ceil(max(chance, max(v for _, v in train)) * 1.06)

    def px(s: float) -> float:
        return L + (s - x0) / (x1 - x0) * (w - L - R)

    def py(v: float) -> float:
        return h - B - (v - y0) / (y1 - y0) * (h - T - B)

    s = [
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{title}. {sub}" style="width:100%;height:auto">'
    ]
    s.append('<g font-family="ui-monospace, monospace" font-size="10" fill="currentColor">')

    # gridlines + y labels
    steps = 5
    for i in range(steps + 1):
        v = y0 + (y1 - y0) * i / steps
        y = py(v)
        s.append(
            f'<line x1="{L}" y1="{y:.1f}" x2="{w - R}" y2="{y:.1f}" '
            f'stroke="currentColor" stroke-width=".7" opacity=".14"/>'
        )
        s.append(
            f'<text x="{L - 8}" y="{y + 3.5:.1f}" text-anchor="end" opacity=".6">{v:.1f}</text>'
        )

    # x labels
    for i in range(5):
        sx = x0 + (x1 - x0) * i / 4
        s.append(
            f'<text x="{px(sx):.1f}" y="{h - B + 16}" text-anchor="middle" opacity=".6">'
            f"{int(sx):,}</text>"
        )
    s.append(
        f'<text x="{(L + w - R) / 2:.0f}" y="{h - 6}" text-anchor="middle" opacity=".55">step</text>'
    )
    s.append(
        f'<text x="16" y="{(T + h - B) / 2:.0f}" text-anchor="middle" opacity=".55" '
        f'transform="rotate(-90 16 {(T + h - B) / 2:.0f})">cross-entropy (nats/token)</text>'
    )

    # chance line
    yc = py(chance)
    s.append(
        f'<line x1="{L}" y1="{yc:.1f}" x2="{w - R}" y2="{yc:.1f}" stroke="currentColor" '
        f'stroke-width="1" stroke-dasharray="4 3" opacity=".45"/>'
    )
    s.append(
        f'<text x="{w - R - 4}" y="{yc - 6:.1f}" text-anchor="end" opacity=".65">'
        f"ln(V) = {chance:.2f} &#8212; uniform-guess floor</text>"
    )

    # train polyline
    pts = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in train)
    s.append(
        f'<polyline points="{pts}" fill="none" stroke="currentColor" stroke-width="1.2" opacity=".45"/>'
    )

    # val polyline + points
    if val:
        vp = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in val)
        s.append(f'<polyline points="{vp}" fill="none" stroke="{ACC}" stroke-width="2"/>')
        for a, b in val:
            s.append(f'<circle cx="{px(a):.1f}" cy="{py(b):.1f}" r="2.6" fill="{ACC}"/>')
        la, lb = val[-1]
        s.append(
            f'<circle cx="{px(la):.1f}" cy="{py(lb):.1f}" r="4.6" fill="none" '
            f'stroke="{ACC}" stroke-width="1.4"/>'
        )
        s.append(
            f'<text x="{px(la) - 8:.1f}" y="{py(lb) - 10:.1f}" text-anchor="end" fill="{ACC}" '
            f'font-size="11">{lb:.3f} &#183; ppl {math.exp(lb):.2f}</text>'
        )

    # legend
    s.append(f'<text x="{L}" y="{T - 12}" font-size="11" font-weight="600">{title}</text>')
    s.append(
        f'<line x1="{w - R - 150}" y1="{T - 16}" x2="{w - R - 132}" y2="{T - 16}" stroke="currentColor" stroke-width="1.2" opacity=".45"/>'
    )
    s.append(f'<text x="{w - R - 127}" y="{T - 13}" opacity=".7">train</text>')
    s.append(
        f'<line x1="{w - R - 88}" y1="{T - 16}" x2="{w - R - 70}" y2="{T - 16}" stroke="{ACC}" stroke-width="2"/>'
    )
    s.append(f'<text x="{w - R - 65}" y="{T - 13}" opacity=".7">validation</text>')

    s.append("</g></svg>")
    return "\n".join(s)


def frontier_chart(w: int = 660, h: int = 260) -> str:
    """Perplexity vs effective bits, for RTN / GPTQ / AWQ."""
    data = {
        "RTN": [(2.5, 1.5405), (3.5, 1.4783), (4.5, 1.4767), (8.5, 1.4764)],
        "GPTQ": [(2.5, 1.4997), (3.5, 1.4764), (4.5, 1.4766), (8.5, 1.4764)],
        "AWQ": [(2.5, 1.5468), (3.5, 1.4782), (4.5, 1.4765), (8.5, 1.4764)],
    }
    fp32 = 1.4764
    L, R, T, B = 60, 100, 30, 44
    x0, x1 = 2.0, 9.0
    y0, y1 = 1.470, 1.555

    def px(v: float) -> float:
        return L + (v - x0) / (x1 - x0) * (w - L - R)

    def py(v: float) -> float:
        return h - B - (v - y0) / (y1 - y0) * (h - T - B)

    styles = {
        "RTN": ("currentColor", "0", 0.55),
        "GPTQ": (ACC, "0", 1.0),
        "AWQ": ("currentColor", "4 3", 0.45),
    }
    s = [
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Perplexity against effective bit-width for round-to-nearest, GPTQ and AWQ: all three tie from 3.5 bits upward, and GPTQ separates clearly at 2.5 bits." style="width:100%;height:auto">'
    ]
    s.append('<g font-family="ui-monospace, monospace" font-size="10" fill="currentColor">')

    for i in range(5):
        v = y0 + (y1 - y0) * i / 4
        y = py(v)
        s.append(
            f'<line x1="{L}" y1="{y:.1f}" x2="{w - R}" y2="{y:.1f}" stroke="currentColor" stroke-width=".7" opacity=".14"/>'
        )
        s.append(
            f'<text x="{L - 8}" y="{y + 3.5:.1f}" text-anchor="end" opacity=".6">{v:.3f}</text>'
        )
    for b in (2.5, 3.5, 4.5, 8.5):
        s.append(
            f'<text x="{px(b):.1f}" y="{h - B + 16}" text-anchor="middle" opacity=".6">{b}</text>'
        )
    s.append(
        f'<text x="{(L + w - R) / 2:.0f}" y="{h - 6}" text-anchor="middle" opacity=".55">effective bits per weight (incl. stored scales)</text>'
    )
    s.append(
        f'<text x="16" y="{(T + h - B) / 2:.0f}" text-anchor="middle" opacity=".55" transform="rotate(-90 16 {(T + h - B) / 2:.0f})">validation perplexity</text>'
    )

    yf = py(fp32)
    s.append(
        f'<line x1="{L}" y1="{yf:.1f}" x2="{w - R}" y2="{yf:.1f}" stroke="currentColor" stroke-width="1" stroke-dasharray="3 3" opacity=".4"/>'
    )
    s.append(f'<text x="{w - R + 6}" y="{yf + 3.5:.1f}" opacity=".6">fp32 = {fp32}</text>')

    for name, pts in data.items():
        col, dash, op = styles[name]
        line = " ".join(f"{px(a):.1f},{py(min(b, y1)):.1f}" for a, b in pts)
        s.append(
            f'<polyline points="{line}" fill="none" stroke="{col}" stroke-width="{2 if name == "GPTQ" else 1.3}" stroke-dasharray="{dash}" opacity="{op}"/>'
        )
        for a, b in pts:
            s.append(
                f'<circle cx="{px(a):.1f}" cy="{py(min(b, y1)):.1f}" r="{3 if name == "GPTQ" else 2.4}" fill="{col}" opacity="{op}"/>'
            )
        ea, eb = pts[-1]
        s.append(
            f'<text x="{px(ea) + 8:.1f}" y="{py(eb) - 6:.1f}" fill="{col}" opacity="{op}" font-size="10.5">{name}</text>'
        )

    # annotate the separation
    s.append(
        f'<text x="{px(2.5) + 10:.1f}" y="{py(1.5405) - 4:.1f}" opacity=".7" font-size="10">RTN 1.5405</text>'
    )
    s.append(
        f'<text x="{px(2.5) + 10:.1f}" y="{py(1.4997) + 14:.1f}" fill="{ACC}" font-size="10">GPTQ 1.4997</text>'
    )
    s.append("</g></svg>")
    return "\n".join(s)


nano_train, nano_val = load_metrics(ROOT / "runs/nano/pretrain/metrics.jsonl")
(OUT / "chart_nano.svg").write_text(
    loss_chart(
        nano_train,
        nano_val,
        chance=math.log(1024),
        title="nano · 5.0M params · toy corpus · 95 s on a laptop CPU",
        sub="Loss falls from ln(1024) to 0.385.",
    )
)

micro_train, micro_val = load_metrics(ROOT / "runs/micro/tinystories/metrics.jsonl")
(OUT / "chart_micro.svg").write_text(
    loss_chart(
        micro_train,
        micro_val,
        chance=math.log(16384),
        title="micro · 40.4M params · TinyStories · Apple Silicon GPU",
        sub="Loss falls from ln(16384).",
    )
)

(OUT / "chart_frontier.svg").write_text(frontier_chart())
print("nano:", len(nano_train), "train pts,", len(nano_val), "val pts")
print("micro:", len(micro_train), "train pts,", len(micro_val), "val pts")
print("charts written")
