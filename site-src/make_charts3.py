"""Multi-seed ablation chart + the DPO likelihood-collapse diagram."""

from __future__ import annotations

import json
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


def seeds_chart(suite: str, title: str) -> str:
    """Per-seed final losses with mean and standard-deviation band."""
    data = json.loads((ROOT / f"results/ablations/{suite}_multiseed.json").read_text())
    arms = list(data["arms"].values())
    w, h = 680, 300
    L, R, T, B = 62, 20, 40, 80

    allv = [v for a in arms for v in a["losses"]]
    y0, y1 = min(allv) * 0.995, max(allv) * 1.005

    def py(v: float) -> float:
        return h - B - (v - y0) / (y1 - y0) * (h - T - B)

    n = len(arms)
    slot = (w - L - R) / n

    s: list[str] = []
    for i in range(5):
        v = y0 + (y1 - y0) * i / 4
        s.append(
            f'<line x1="{L}" y1="{py(v):.1f}" x2="{w - R}" y2="{py(v):.1f}" '
            f'stroke="currentColor" stroke-width=".7" opacity=".13"/>'
        )
        s.append(
            f'<text x="{L - 8}" y="{py(v) + 3.5:.1f}" text-anchor="end" opacity=".6">{v:.3f}</text>'
        )

    for i, a in enumerate(arms):
        cx = L + slot * (i + 0.5)
        col = ACC if i == 0 else "currentColor"
        op = "1" if i == 0 else ".75"
        mean, sd = a["mean_val_loss"], a["stdev"]
        # spread band
        s.append(
            f'<rect x="{cx - 26:.1f}" y="{py(mean + sd):.1f}" width="52" '
            f'height="{abs(py(mean - sd) - py(mean + sd)):.1f}" fill="{col}" '
            f'fill-opacity=".13" rx="3"/>'
        )
        s.append(
            f'<line x1="{cx - 30:.1f}" y1="{py(mean):.1f}" x2="{cx + 30:.1f}" '
            f'y2="{py(mean):.1f}" stroke="{col}" stroke-width="2.2" opacity="{op}"/>'
        )
        for j, v in enumerate(a["losses"]):
            jx = cx - 18 + (j % 5) * 9
            s.append(f'<circle cx="{jx:.1f}" cy="{py(v):.1f}" r="3" fill="{col}" opacity=".85"/>')
        label = a["variant"]
        if len(label) > 22:
            label = label[:21] + "…"
        s.append(
            f'<text x="{cx:.1f}" y="{h - B + 18}" text-anchor="middle" font-size="10">{label}</text>'
        )
        s.append(
            f'<text x="{cx:.1f}" y="{h - B + 34}" text-anchor="middle" font-size="9.5" '
            f'opacity=".65">sd {sd:.4f}</text>'
        )
        steps = a["mean_steps_to_target"]
        if steps is not None:
            s.append(
                f'<text x="{cx:.1f}" y="{h - B + 50}" text-anchor="middle" font-size="9.5" '
                f'fill="{col}" opacity="{op}">{steps:.0f} steps → target</text>'
            )

    s.append(f'<text x="{L}" y="{T - 18}" font-size="11" font-weight="600">{title}</text>')
    s.append(
        f'<text x="{L}" y="{T - 4}" font-size="9.5" opacity=".65">'
        f"bar: mean &#183; band: ±1 sd &#183; dots: individual seeds (n={len(arms[0]['losses'])})</text>"
    )
    return svg(
        w,
        h,
        "\n".join(s),
        f"{title}: per-seed final validation losses with mean and standard deviation.",
    )


def dpo_collapse() -> str:
    """The measured Δ log p for DPO with and without the NLL anchor."""
    w, h = 660, 250
    L, R, T, B = 150, 96, 40, 40
    rows = [
        ("DPO", -0.0454, -4.2385, "0–7–33"),
        ("DPO + NLL anchor", +0.0104, -3.9170, "3–0–37"),
    ]
    lo, hi = -4.6, 0.6

    def px(v: float) -> float:
        return L + (v - lo) / (hi - lo) * (w - L - R)

    s: list[str] = []
    s.append(
        f'<line x1="{px(0):.1f}" y1="{T}" x2="{px(0):.1f}" y2="{h - B}" stroke="currentColor" stroke-width="1" opacity=".4"/>'
    )
    s.append(
        f'<text x="{px(0):.1f}" y="{T - 8}" text-anchor="middle" opacity=".65">no change</text>'
    )

    for i, (name, dch, drej, h2h) in enumerate(rows):
        y = T + 34 + i * 84
        s.append(
            f'<text x="{L - 12}" y="{y + 4}" text-anchor="end" font-size="11" font-weight="600">{name}</text>'
        )
        # chosen
        col_ch = ACC if dch > 0 else WARN
        s.append(
            f'<line x1="{px(0):.1f}" y1="{y:.1f}" x2="{px(dch):.1f}" y2="{y:.1f}" stroke="{col_ch}" stroke-width="9" opacity=".85" stroke-linecap="round"/>'
        )
        s.append(
            f'<text x="{px(dch) + (10 if dch > 0 else -10):.1f}" y="{y + 4:.1f}" text-anchor="{"start" if dch > 0 else "end"}" font-size="10" fill="{col_ch}">chosen {dch:+.4f}</text>'
        )
        # rejected
        yr = y + 26
        s.append(
            f'<line x1="{px(0):.1f}" y1="{yr:.1f}" x2="{px(drej):.1f}" y2="{yr:.1f}" stroke="currentColor" stroke-width="9" opacity=".35" stroke-linecap="round"/>'
        )
        s.append(
            f'<text x="{px(drej) - 10:.1f}" y="{yr + 4:.1f}" text-anchor="end" font-size="10" opacity=".7">rejected {drej:+.4f}</text>'
        )
        s.append(
            f'<text x="{L - 12}" y="{y + 30}" text-anchor="end" font-size="9.5" opacity=".65">head-to-head {h2h}</text>'
        )

    s.append(
        f'<text x="{(L + w - R) / 2:.0f}" y="{h - 8}" text-anchor="middle" opacity=".6">change in mean per-token log-probability over the run</text>'
    )
    return svg(
        w,
        h,
        "\n".join(s),
        "DPO alone lowers the log-probability of both chosen and rejected responses; adding the NLL anchor raises the chosen one.",
    )


(OUT / "chart_seeds_opt.svg").write_text(seeds_chart("optimizer", "Optimizer, 5 seeds per arm"))
(OUT / "chart_seeds_arch.svg").write_text(
    seeds_chart("architecture", "Architecture, 5 seeds per arm")
)
(OUT / "chart_dpo.svg").write_text(dpo_collapse())
print("wrote 3 charts")
