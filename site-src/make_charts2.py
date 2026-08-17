"""Theme-aware inline SVG charts for the merged explainer (v2)."""

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
WARN = "#8F4715"


def svg(w: int, h: int, body: str, label: str) -> str:
    """Wrap chart body in a themed, accessible SVG element."""
    return (
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{label}" '
        f'style="width:100%;height:auto">\n'
        f'<g font-family="ui-monospace, monospace" font-size="10" fill="currentColor">\n'
        f"{body}\n</g></svg>"
    )


# ------------------------------------------------------------------ minimal pairs
def minimal_pairs_chart() -> str:
    """Per-phenomenon accuracy with Wilson intervals and a chance line."""
    data = json.loads((ROOT / "results/evaluation/micro-tinystories.json").read_text())
    rows = data["minimal_pairs"]["phenomena"]
    w, h = 680, 40 + 26 * len(rows) + 40
    L, R = 190, 60
    x0, x1 = 0.0, 1.0

    def px(v: float) -> float:
        return L + (v - x0) / (x1 - x0) * (w - L - R)

    s: list[str] = []
    # chance line
    s.append(
        f'<line x1="{px(0.5):.1f}" y1="28" x2="{px(0.5):.1f}" y2="{h - 44}" '
        f'stroke="currentColor" stroke-width="1.2" stroke-dasharray="4 3" opacity=".5"/>'
    )
    s.append(f'<text x="{px(0.5):.1f}" y="20" text-anchor="middle" opacity=".7">chance 50%</text>')

    for i, r in enumerate(rows):
        y = 40 + 26 * i
        acc, lo, hi = r["accuracy"], r["ci_low"], r["ci_high"]
        col = ACC if r["above_chance"] else WARN
        s.append(
            f'<text x="{L - 10}" y="{y + 4}" text-anchor="end" font-size="10.5">{r["phenomenon"]}</text>'
        )
        # CI whisker
        s.append(
            f'<line x1="{px(lo):.1f}" y1="{y:.1f}" x2="{px(hi):.1f}" y2="{y:.1f}" '
            f'stroke="{col}" stroke-width="1.4" opacity=".45"/>'
        )
        for e in (lo, hi):
            s.append(
                f'<line x1="{px(e):.1f}" y1="{y - 4:.1f}" x2="{px(e):.1f}" y2="{y + 4:.1f}" '
                f'stroke="{col}" stroke-width="1.4" opacity=".45"/>'
            )
        s.append(f'<circle cx="{px(acc):.1f}" cy="{y:.1f}" r="4.5" fill="{col}"/>')
        s.append(
            f'<text x="{w - R + 8}" y="{y + 4:.1f}" font-size="10.5" fill="{col}">'
            f"{acc * 100:.0f}%</text>"
        )

    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        s.append(
            f'<text x="{px(v):.1f}" y="{h - 26}" text-anchor="middle" opacity=".55">'
            f"{v * 100:.0f}%</text>"
        )
    s.append(
        f'<text x="{(L + w - R) / 2:.0f}" y="{h - 8}" text-anchor="middle" opacity=".6">'
        f"forced-choice accuracy, 95% Wilson interval &#183; {data['minimal_pairs']['n_items']} items</text>"
    )
    return svg(
        w,
        h,
        "\n".join(s),
        "Minimal-pair accuracy per phenomenon with 95% Wilson intervals against a 50% chance line.",
    )


# ------------------------------------------------------------------ bits per byte
def bpb_chart() -> str:
    """Grouped bars comparing bits-per-byte in and out of domain."""
    data = json.loads((ROOT / "results/baseline/bits_per_byte.json").read_text())
    rows = data["rows"]
    w, h = 680, 260
    L, R, T, B = 60, 20, 34, 56

    corpora = ["tinystories-valid", "out-of-domain"]
    names = ["nanoscale-micro-tinystories", "gpt2", "distilgpt2"]
    vals = {(r["name"], r["corpus"]): r["bits_per_byte"] for r in rows}
    ymax = 3.6

    def py(v: float) -> float:
        """Value to y pixel."""
        return h - B - min(v, ymax) / ymax * (h - T - B)

    s: list[str] = []
    for i in range(5):
        v = ymax * i / 4
        s.append(
            f'<line x1="{L}" y1="{py(v):.1f}" x2="{w - R}" y2="{py(v):.1f}" '
            f'stroke="currentColor" stroke-width=".7" opacity=".14"/>'
        )
        s.append(
            f'<text x="{L - 8}" y="{py(v) + 3.5:.1f}" text-anchor="end" opacity=".6">{v:.1f}</text>'
        )

    group_w = (w - L - R) / len(corpora)
    bar_w = group_w / (len(names) + 1.4)
    for gi, corpus in enumerate(corpora):
        gx = L + gi * group_w
        for bi, name in enumerate(names):
            v = vals.get((name, corpus))
            if v is None:
                continue
            x = gx + group_w * 0.16 + bi * bar_w
            col = ACC if name.startswith("nanoscale") else "currentColor"
            op = "1" if name.startswith("nanoscale") else ".38"
            top = py(v)
            s.append(
                f'<rect x="{x:.1f}" y="{top:.1f}" width="{bar_w - 6:.1f}" '
                f'height="{h - B - top:.1f}" fill="{col}" fill-opacity="{op}" rx="2"/>'
            )
            txt = f"{v:.2f}" + ("" if v <= ymax else " ↑")
            s.append(
                f'<text x="{x + (bar_w - 6) / 2:.1f}" y="{top - 5:.1f}" text-anchor="middle" '
                f'font-size="10" fill="{col}" opacity="{op}">{txt}</text>'
            )
        s.append(
            f'<text x="{gx + group_w / 2:.1f}" y="{h - B + 16}" text-anchor="middle" '
            f'font-size="10.5">{corpus}</text>'
        )

    s.append(
        f'<text x="{L}" y="{T - 14}" font-size="11" font-weight="600">bits per byte &#8212; lower is better</text>'
    )
    # legend
    s.append(f'<rect x="{w - R - 210}" y="{T - 22}" width="10" height="10" fill="{ACC}" rx="2"/>')
    s.append(f'<text x="{w - R - 196}" y="{T - 13}" opacity=".75">NanoScale 40M</text>')
    s.append(
        f'<rect x="{w - R - 108}" y="{T - 22}" width="10" height="10" fill="currentColor" fill-opacity=".38" rx="2"/>'
    )
    s.append(f'<text x="{w - R - 94}" y="{T - 13}" opacity=".75">GPT-2 family</text>')
    s.append(
        f'<text x="{(L + w - R) / 2:.0f}" y="{h - 8}" text-anchor="middle" opacity=".6">'
        "same held-out strings, each model with its own tokenizer, normalised by UTF-8 bytes</text>"
    )
    return svg(
        w,
        h,
        "\n".join(s),
        "Bits per byte for NanoScale 40M against GPT-2 and distilGPT-2, in domain and out of domain.",
    )


# ------------------------------------------------------------------ calibration
def calibration_chart() -> str:
    """Reliability plot: mean confidence against measured accuracy."""
    data = json.loads((ROOT / "results/evaluation/micro-tinystories.json").read_text())
    cal = data["calibration"]
    w, h = 400, 340
    L, R, T, B = 56, 18, 30, 48

    def px(v: float) -> float:
        return L + v * (w - L - R)

    def py(v: float) -> float:
        """Value to y pixel."""
        return h - B - v * (h - T - B)

    s: list[str] = []
    for i in range(6):
        v = i / 5
        s.append(
            f'<line x1="{L}" y1="{py(v):.1f}" x2="{w - R}" y2="{py(v):.1f}" stroke="currentColor" stroke-width=".7" opacity=".13"/>'
        )
        s.append(
            f'<text x="{L - 8}" y="{py(v) + 3.5:.1f}" text-anchor="end" opacity=".6">{v:.1f}</text>'
        )
        s.append(
            f'<text x="{px(v):.1f}" y="{h - B + 15}" text-anchor="middle" opacity=".6">{v:.1f}</text>'
        )

    s.append(
        f'<line x1="{px(0):.1f}" y1="{py(0):.1f}" x2="{px(1):.1f}" y2="{py(1):.1f}" '
        f'stroke="currentColor" stroke-width="1.2" stroke-dasharray="4 3" opacity=".55"/>'
    )
    s.append(
        f'<text x="{px(0.62):.1f}" y="{py(0.72):.1f}" opacity=".6" font-size="9.5">perfect calibration</text>'
    )

    pts = []
    for c, a, n in zip(cal.get("_conf", []), cal.get("_acc", []), cal.get("_n", []), strict=False):
        if n:
            pts.append((c, a))
    # The summary dict does not carry bins; use the headline point instead.
    conf, acc = cal["mean_confidence"], cal["top1_accuracy"]
    s.append(f'<circle cx="{px(conf):.1f}" cy="{py(acc):.1f}" r="6" fill="{ACC}"/>')
    s.append(
        f'<text x="{px(conf) + 12:.1f}" y="{py(acc) - 6:.1f}" fill="{ACC}" font-size="10.5">'
        f"mean: {conf * 100:.1f}% confident, {acc * 100:.1f}% right</text>"
    )

    s.append(f'<text x="{L}" y="{T - 12}" font-size="11" font-weight="600">reliability</text>')
    s.append(
        f'<text x="{(L + w - R) / 2:.0f}" y="{h - 8}" text-anchor="middle" opacity=".6">confidence</text>'
    )
    s.append(
        f'<text x="14" y="{(T + h - B) / 2:.0f}" text-anchor="middle" opacity=".6" '
        f'transform="rotate(-90 14 {(T + h - B) / 2:.0f})">accuracy</text>'
    )
    s.append(
        f'<text x="{L + 8}" y="{py(0.06):.1f}" font-size="10" opacity=".8">ECE {cal["ece"]:.4f} &#183; over-confidence {cal["overconfidence"]:+.4f}</text>'
    )
    return svg(
        w,
        h,
        "\n".join(s),
        "Reliability: the model's mean confidence sits almost exactly on the perfect-calibration diagonal.",
    )


# ------------------------------------------------------------------ micro loss curve
def loss_chart() -> str:
    """Line chart of train/validation loss against the uniform-guess floor."""
    rows = [
        json.loads(x)
        for x in (ROOT / "runs/micro/tinystories/metrics.jsonl").read_text().splitlines()
        if x.strip()
    ]
    train = [(r["step"], r["loss"]) for r in rows if "loss" in r]
    val = [(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]
    chance = math.log(16384)
    w, h = 680, 260
    L, R, T, B = 56, 16, 32, 46
    x1 = max(s for s, _ in train)
    y1 = math.ceil(chance * 1.04)

    def px(v: float) -> float:
        return L + v / x1 * (w - L - R)

    def py(v: float) -> float:
        """Value to y pixel."""
        return h - B - v / y1 * (h - T - B)

    s: list[str] = []
    for i in range(6):
        v = y1 * i / 5
        s.append(
            f'<line x1="{L}" y1="{py(v):.1f}" x2="{w - R}" y2="{py(v):.1f}" stroke="currentColor" stroke-width=".7" opacity=".13"/>'
        )
        s.append(
            f'<text x="{L - 8}" y="{py(v) + 3.5:.1f}" text-anchor="end" opacity=".6">{v:.0f}</text>'
        )
    for i in range(5):
        sx = x1 * i / 4
        s.append(
            f'<text x="{px(sx):.1f}" y="{h - B + 16}" text-anchor="middle" opacity=".6">{int(sx):,}</text>'
        )

    s.append(
        f'<line x1="{L}" y1="{py(chance):.1f}" x2="{w - R}" y2="{py(chance):.1f}" stroke="currentColor" stroke-width="1" stroke-dasharray="4 3" opacity=".45"/>'
    )
    s.append(
        f'<text x="{w - R - 4}" y="{py(chance) - 6:.1f}" text-anchor="end" opacity=".65">ln(16384) = 9.704 &#8212; uniform-guess floor</text>'
    )

    s.append(
        '<polyline points="'
        + " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in train)
        + '" fill="none" stroke="currentColor" stroke-width="1.1" opacity=".38"/>'
    )
    s.append(
        '<polyline points="'
        + " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in val)
        + f'" fill="none" stroke="{ACC}" stroke-width="2"/>'
    )
    la, lb = val[-1]
    s.append(f'<circle cx="{px(la):.1f}" cy="{py(lb):.1f}" r="4.5" fill="{ACC}"/>')
    s.append(
        f'<text x="{px(la) - 10:.1f}" y="{py(lb) - 10:.1f}" text-anchor="end" fill="{ACC}" font-size="11">{lb:.3f} &#183; ppl {math.exp(lb):.2f}</text>'
    )

    s.append(
        f'<text x="{L}" y="{T - 12}" font-size="11" font-weight="600">micro &#183; 40.4M params &#183; TinyStories &#183; 32.0M tokens &#183; 3.2 h</text>'
    )
    s.append(
        f'<text x="{(L + w - R) / 2:.0f}" y="{h - 6}" text-anchor="middle" opacity=".55">step</text>'
    )
    return svg(
        w,
        h,
        "\n".join(s),
        "Micro-tier training curve falling from ln(16384) to validation loss 1.513, perplexity 4.54.",
    )


for name, fn in [
    ("chart_pairs", minimal_pairs_chart),
    ("chart_bpb", bpb_chart),
    ("chart_cal", calibration_chart),
    ("chart_micro2", loss_chart),
]:
    (OUT / f"{name}.svg").write_text(fn())
    print("wrote", name)
