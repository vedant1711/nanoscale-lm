r"""When does each grammatical capability appear during training?

The evaluation in ``scripts/evaluate.py`` reports one number per phenomenon at the end of
training. That is a photograph of a process, and the process is the more interesting
object: a 40M model ends at 94% on simple subject-verb agreement and 44% -- *below chance*
-- on agreement across an attractor, and a single endpoint cannot say whether the model
ever knew better, or whether the attractor case was always broken.

This script answers that by evaluating the full minimal-pair suite at intervals *during* a
single training run, producing an emergence curve per phenomenon. One run, many
measurements, so the cost is the training run plus a few seconds of evaluation per probe
point.

Three things the curves can show that an endpoint cannot:

1. **Emergence order.** Which capabilities are cheap (learnable from local statistics) and
   which are expensive (need long-range structure). This is the ordering that a
   parameter-count or loss curve completely hides.
2. **Non-monotonicity.** A phenomenon that rises then *falls* means the model found a
   heuristic that pays off on the training distribution and costs accuracy on the probe.
   Agreement attraction is the prime suspect: linear recency is a cheap rule that works
   whenever nothing intervenes.
3. **Whether below-chance is a floor or a trajectory.** A phenomenon sitting at 44% could
   be heading up slowly or have converged there. Only the curve distinguishes them.

Usage::

    python scripts/emergence.py --steps 4000 --probe-every 250
    python scripts/emergence.py --replay
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanoscale.config import ExperimentConfig, load_experiment
from nanoscale.eval import PHENOMENA, generate_pairs, run_minimal_pairs, wilson_interval
from nanoscale.tokenizer import BPETokenizer
from nanoscale.train import Trainer
from nanoscale.utils import get_logger, git_sha, hardware_string

log = get_logger("nanoscale.emergence")

ROOT = _ROOT
RESULTS = ROOT / "results" / "emergence"


def build_config(steps: int, probe_every: int) -> ExperimentConfig:
    """A small TinyStories model: big enough to learn syntax, fast enough to probe often."""
    return load_experiment(
        tier="micro",
        overrides=[
            "name=emergence",
            "model.n_layers=6",
            "model.d_model=256",
            "model.n_heads=4",
            "model.n_kv_heads=2",
            "model.max_seq_len=256",
            "model.attn_impl=sdpa",
            "tokenizer.vocab_size=16384",
            "data.source=textfile",
            "data.paths=['data/tinystories/train.txt','data/tinystories/valid.txt']",
            "data.doc_separator=<|endoftext|>",
            "data.seq_len=256",
            "data.val_fraction=0.01",
            "train.device=mps",
            f"train.max_steps={steps}",
            "train.token_budget=null",
            "train.batch_size=24",
            "train.grad_accum=1",
            f"train.eval_interval={probe_every}",
            "train.log_interval=100",
            "train.ckpt_interval=1000000",
        ],
    )


def measure(args: argparse.Namespace) -> dict[str, Any]:
    """Train once, evaluating the minimal-pair suite at intervals."""
    cfg = build_config(args.steps, args.probe_every)
    tok = BPETokenizer.load(args.tokenizer)
    trainer = Trainer(cfg, tokenizer=tok, out_dir=ROOT / "runs" / "emergence")
    pairs = generate_pairs(n_per_phenomenon=args.pairs, seed=1337)

    probes: list[dict[str, Any]] = []
    tokens_per_step = cfg.data.seq_len * cfg.train.batch_size * cfg.train.grad_accum
    started = time.perf_counter()

    for stop in range(args.probe_every, args.steps + 1, args.probe_every):
        # `stop_at_step` continues the *same* run rather than restarting, so the curve is
        # one trajectory rather than a series of independent short runs.
        result = trainer.train(stop_at_step=stop)
        mp = run_minimal_pairs(trainer.model, tok, pairs=pairs, device=trainer.device)
        phenomena = {s.phenomenon: round(s.accuracy, 5) for s in mp.scores}
        row: dict[str, Any] = {
            "step": stop,
            "tokens": stop * tokens_per_step,
            "val_loss": round(result.final_val_loss, 5),
            "macro": round(mp.overall, 5),
            "phenomena": phenomena,
            "elapsed_s": round(time.perf_counter() - started, 1),
        }
        probes.append(row)
        log.info(
            "step %5d (%9s tokens) val %.4f macro %.3f | attractor %.2f simple %.2f",
            stop,
            f"{row['tokens']:,}",
            row["val_loss"],
            row["macro"],
            phenomena.get("agreement_attractor", float("nan")),
            phenomena.get("agreement_simple", float("nan")),
        )

    names = [n for n in PHENOMENA if n in probes[0]["phenomena"]]
    tokens_axis = [float(p["tokens"]) for p in probes]
    trends = {}
    for name in names:
        ys = [float(p["phenomena"][name]) for p in probes]
        rho, pv = spearman(tokens_axis, ys)
        trends[name] = {
            "spearman_rho": round(rho, 4),
            "p_value": round(pv, 5),
            "first": ys[0],
            "last": ys[-1],
            "min": min(ys),
            "max": max(ys),
            "ci_halfwidth": round(
                (
                    wilson_interval(int(ys[-1] * args.pairs), args.pairs)[1]
                    - wilson_interval(int(ys[-1] * args.pairs), args.pairs)[0]
                )
                / 2,
                4,
            ),
        }

    return {
        "git_sha": git_sha(),
        "hardware": hardware_string(),
        "params": trainer.model.num_parameters(),
        "trends": trends,
        "steps": args.steps,
        "probe_every": args.probe_every,
        "items_per_phenomenon": args.pairs,
        "tokens_per_step": tokens_per_step,
        "probes": probes,
    }


def spearman(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Spearman rank correlation and a two-sided p-value.

    The right statistic for this experiment. Any *single* probe point carries a binomial
    standard error of about 5 points at 100 items, so a dip of 9 points is barely one
    standard error of a difference and cannot be claimed on its own. A monotone *trend*
    across sixteen probes is a much stronger signal than any one of them, and rank
    correlation tests exactly that without assuming the curve is a straight line.

    So the reportable claim is not "accuracy dipped to 38%" — it is "this phenomenon's
    accuracy fails to rise with training while every other phenomenon's does", which is a
    statement about sixteen paired observations rather than one.
    """
    n = len(xs)
    if n < 3:
        return (float("nan"), float("nan"))

    def rank(v: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    if den == 0:
        return (0.0, 1.0)
    rho = num / den
    # t approximation, adequate for n >= 10.
    if abs(rho) >= 1.0:
        return (rho, 0.0)
    tstat = rho * math.sqrt((n - 2) / (1 - rho**2))
    p = 2.0 * _student_sf(abs(tstat), n - 2)
    return (rho, min(1.0, max(0.0, p)))


def _student_sf(t: float, df: int) -> float:
    """Upper tail of Student's t, reusing the project's incomplete-beta implementation."""
    from nanoscale.bench.multiseed import _betainc

    return 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))


def plot(payload: dict[str, Any]) -> Path:
    """One panel of curves, chance line marked."""
    probes = payload["probes"]
    tokens = [p["tokens"] / 1e6 for p in probes]
    names = [n for n in PHENOMENA if n in probes[0]["phenomena"]]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 8), height_ratios=[3, 1], sharex=True)
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(names):
        ys = [p["phenomena"][name] * 100 for p in probes]
        style = "-" if ys[-1] >= 50 else "--"
        ax.plot(tokens, ys, style, color=cmap(i % 10), lw=2, label=name, marker="o", ms=3)

    ax.axhline(50, color="black", ls=":", lw=1.2, alpha=0.6)
    ax.text(tokens[-1], 51, "chance", ha="right", fontsize=8, alpha=0.7)
    ax.set_ylabel("forced-choice accuracy (%)")
    ax.set_ylim(20, 104)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7.5, ncol=3, loc="lower right")
    ax.set_title(
        f"Capability emergence during one training run\n"
        f"{payload['params']:,} parameters, TinyStories, "
        f"{payload['items_per_phenomenon']} items per phenomenon",
        fontsize=10,
    )

    ax2.plot(tokens, [p["val_loss"] for p in probes], color="#12655F", lw=2)
    ax2.set_ylabel("val loss")
    ax2.set_xlabel("training tokens (millions)")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    path = RESULTS / "emergence.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def ensure_trends(payload: dict[str, Any], items: int) -> dict[str, Any]:
    """Compute per-phenomenon trend statistics if the payload predates them.

    Deriving these at render time rather than only at measure time is what lets the
    statistics improve without paying for the training run again — the same reason the
    multi-seed ablations store raw per-seed losses.
    """
    probes = payload["probes"]
    names = [n for n in PHENOMENA if n in probes[0]["phenomena"]]
    tokens_axis = [float(p["tokens"]) for p in probes]
    trends: dict[str, Any] = {}
    for name in names:
        ys = [float(p["phenomena"][name]) for p in probes]
        rho, pv = spearman(tokens_axis, ys)
        lo, hi = wilson_interval(round(ys[-1] * items), items)
        trends[name] = {
            "spearman_rho": round(rho, 4),
            "p_value": round(pv, 5),
            "first": ys[0],
            "last": ys[-1],
            "min": min(ys),
            "max": max(ys),
            "ci_halfwidth": round((hi - lo) / 2, 4),
        }
    payload["trends"] = trends
    return payload


def render(payload: dict[str, Any], figure: Path) -> str:
    """Write the markdown fragment."""
    probes = payload["probes"]
    names = [n for n in PHENOMENA if n in probes[0]["phenomena"]]
    first, last = probes[0], probes[-1]

    def crossed(name: str) -> str:
        """Tokens at which this phenomenon first exceeded 60% and stayed there."""
        for i, p in enumerate(probes):
            if all(q["phenomena"][name] >= 0.60 for q in probes[i:]):
                return f"{p['tokens'] / 1e6:.1f}M"
        return "never"

    def peak_drop(name: str) -> float:
        """Largest fall from a running maximum — the non-monotonicity measure."""
        best = 0.0
        worst = 0.0
        for p in probes:
            v = p["phenomena"][name]
            best = max(best, v)
            worst = max(worst, best - v)
        return worst

    lines = [
        "# Capability emergence during training",
        "",
        f"Generated by `scripts/emergence.py` at git `{payload['git_sha']}` on "
        f"`{payload['hardware']}`.",
        "",
        f"One training run of a {payload['params']:,}-parameter model on TinyStories, with "
        f"the full minimal-pair suite evaluated every {payload['probe_every']} steps "
        f"({payload['tokens_per_step'] * payload['probe_every'] / 1e6:.2f}M tokens). Chance "
        f"is 50%.",
        "",
        f"![emergence]({figure.name})",
        "",
        "| phenomenon | 1st probe | final | reaches 60% | max drop | Spearman ρ vs tokens | p |",
        "|---|---|---|---|---|---|---|",
    ]
    trends = payload.get("trends", {})
    for name in names:
        tr = trends.get(name, {})
        rho = tr.get("spearman_rho")
        pv = tr.get("p_value")
        lines.append(
            f"| {name} | {first['phenomena'][name] * 100:.0f}% | "
            f"**{last['phenomena'][name] * 100:.0f}%** | {crossed(name)} | "
            f"{peak_drop(name) * 100:.0f} pts | "
            f"{'—' if rho is None else f'{rho:+.2f}'} | "
            f"{'—' if pv is None else f'{pv:.4f}'} |"
        )

    lines += [
        "",
        f"Validation loss over the same run: {first['val_loss']:.3f} → {last['val_loss']:.3f}.",
        "",
        "## Why this is worth plotting",
        "",
        "A single end-of-training number cannot distinguish *never learned* from *learned "
        "and then unlearned*, and the two have opposite implications. A capability that "
        "rises and then falls means the model found a shortcut that pays on the training "
        "distribution and costs accuracy on the probe — which is a statement about the "
        "data, not about capacity. A capability that never moves is a statement about "
        "capacity or about the probe.",
        "",
        "**Read the Spearman column, not the individual dips.** With "
        f"{payload['items_per_phenomenon']} items per probe the binomial standard error is "
        "about 5 points, so any single point moving by 9 points is barely one standard "
        "error of a difference and cannot carry a claim. The rank correlation between "
        "accuracy and training tokens uses all "
        f"{len(payload['probes'])} probes at once and is the statistic that can.",
        "",
        "The ordering itself is the other result: phenomena learnable from local "
        "co-occurrence should saturate early and cheaply, while anything requiring "
        "structure over a distance should lag. Reading the crossing points down the table "
        "gives that ordering directly.",
        "",
        "Reproduce with: `python scripts/emergence.py --replay`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--probe-every", type=int, default=250)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument(
        "--tokenizer", type=Path, default=ROOT / "artifacts/tokenizer/micro_tinystories.json"
    )
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "emergence.json"

    if args.replay:
        if not json_path.exists():
            raise SystemExit(f"no committed results at {json_path}; run without --replay.")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        payload = measure(args)
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    payload = ensure_trends(payload, payload.get("items_per_phenomenon", args.pairs))
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    figure = plot(payload)
    md = RESULTS / "emergence.md"
    md.write_text(render(payload, figure), encoding="utf-8")
    print(f"wrote {md} and {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
