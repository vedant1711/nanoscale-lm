"""Run the Phase-5 ablations at several seeds and test the differences properly.

`scripts/ablate.py` runs one seed per arm and refuses to call any gap below 2% a result,
because with one run there is nothing to compare a gap against. That was the honest thing
to do with the data available, and it is not a measurement.

This script trains every arm at N seeds and replaces the threshold with a two-sample
Welch's t-test plus Cohen's d. A difference is only reported as real when it is
statistically significant *and* large relative to run-to-run variance — the two criteria
shown separately, so a reader who disagrees with either threshold can see the underlying
numbers.

Cost is linear in the number of seeds: at `nano` on a CPU each arm is roughly 25 seconds,
so five seeds across four architecture arms is about eight minutes.

Usage::

    python scripts/ablate_multiseed.py --seeds 5
    python scripts/ablate_multiseed.py --replay
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))  # so `scripts.ablate` resolves when run as a file

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanoscale.bench import ArmStatistics, MultiSeedComparison, MultiSeedSuite
from nanoscale.utils import get_logger
from scripts.ablate import architecture_suite, optimizer_suite

log = get_logger("nanoscale.ablate.multiseed")

ROOT = _ROOT
RESULTS = ROOT / "results" / "ablations"

DEFAULT_SEEDS = (1337, 42, 7, 2024, 31337)

#: Assembled from parts to stay inside the line limit.
_HEADER = (
    "| "
    + " | ".join(
        [
            "variant",
            "Δ mean loss",
            "p (loss)",
            "Cohen's d",
            "p (steps)",
            "var F",
            "p (var)",
            "verdict",
        ]
    )
    + " |"
)


def reanalyse(payload: dict[str, Any]) -> dict[str, Any]:
    """Recompute the comparisons from the stored per-seed losses.

    Lets the statistics be improved — a multiplicity correction, a new test — without
    retraining anything, which is the whole point of committing the raw per-seed numbers
    rather than only the summary.
    """
    from nanoscale.bench import cohens_d, holm_bonferroni, variance_ratio_test, welch_t_test

    arms = payload["arms"]
    baseline_key = payload["baseline"]
    base = arms[baseline_key]

    def stats(a: dict[str, Any]) -> ArmStatistics:
        return ArmStatistics(
            name=a["variant"],
            display=a["variant"],
            losses=tuple(a["losses"]),
            steps_to_target=tuple(a["steps_to_target"]),
        )

    base_stat = stats(base)
    comparisons: list[MultiSeedComparison] = []
    for key, arm in arms.items():
        if key == baseline_key:
            continue
        a = stats(arm)
        t_, df, pv = welch_t_test(list(a.losses), list(base_stat.losses))
        a_steps = [float(s) for s in a.steps_to_target if s is not None]
        b_steps = [float(s) for s in base_stat.steps_to_target if s is not None]
        st, _, sp = welch_t_test(a_steps, b_steps)
        vf, vp = variance_ratio_test(list(a.losses), list(base_stat.losses))
        comparisons.append(
            MultiSeedComparison(
                baseline=base_stat,
                challenger=a,
                t=t_,
                df=df,
                p_value=pv,
                effect_size=cohens_d(list(a.losses), list(base_stat.losses)),
                steps_t=st,
                steps_p=sp,
                variance_f=vf,
                variance_p=vp,
            )
        )

    flags = holm_bonferroni([c.p_value for c in comparisons])
    from dataclasses import replace as _replace

    comparisons = [
        _replace(c, survives_correction=f) for c, f in zip(comparisons, flags, strict=True)
    ]
    payload["comparisons"] = [c.summary() for c in comparisons]
    payload["correction"] = "holm-bonferroni"
    return payload


def plot(payload: dict[str, Any], name: str) -> Path:
    """Per-arm mean with a standard-error bar and the individual seeds overlaid."""
    arms = payload["arms"]
    labels = [a["variant"] for a in arms.values()]
    means = [a["mean_val_loss"] for a in arms.values()]
    errs = [a["stderr"] for a in arms.values()]
    seeds = [a["losses"] for a in arms.values()]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = list(range(len(labels)))
    ax.bar(x, means, yerr=errs, capsize=5, color="#4C9A94", alpha=0.75, zorder=2)
    for i, losses in enumerate(seeds):
        ax.scatter([i] * len(losses), losses, color="#1B3A38", s=18, zorder=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
    ax.set_ylabel("final validation loss")
    ax.set_title(
        f"{name} ablation — {len(payload['seeds'])} seeds per arm\n"
        "bars: mean ± standard error · dots: individual seeds",
        fontsize=10,
    )
    lo = min(min(s) for s in seeds)
    hi = max(max(s) for s in seeds)
    pad = (hi - lo) * 0.25 or 0.01
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(axis="y", alpha=0.2, zorder=1)
    fig.tight_layout()

    path = RESULTS / f"{name}_multiseed.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_finding(payload: dict[str, Any], name: str, figure: Path) -> Path:
    """Write the markdown fragment for this suite."""
    lines = [
        f"# Ablation — {name} (multi-seed)",
        "",
        f"**Question.** {payload['question']}",
        "",
        f"![{name} multi-seed]({figure.name})",
        "",
        f"Every arm trained at {len(payload['seeds'])} seeds "
        f"({', '.join(str(s) for s in payload['seeds'])}). Arms differ only in the named "
        f"field; seed controls initialisation and data order together.",
        "",
        "| variant | mean val loss | ± stderr | seeds | mean steps → target |",
        "|---|---|---|---|---|",
    ]
    for arm in payload["arms"].values():
        steps = arm["mean_steps_to_target"]
        lines.append(
            f"| {arm['variant']} | **{arm['mean_val_loss']:.4f}** | {arm['stderr']:.4f} | "
            f"{arm['n_seeds']} | {steps if steps is not None else '—'} |"
        )

    lines += [
        "",
        "## Significance",
        "",
        f"Two-sided Welch's t-test against the baseline arm at α={payload['alpha']}, "
        f"**Holm-Bonferroni corrected** across the "
        f"{len(payload['comparisons'])} comparison"
        f"{'s' if len(payload['comparisons']) != 1 else ''} in this suite, "
        f"with Cohen's d alongside. A difference counts as real only when it "
        f"survives the correction *and* has |d| ≥ {payload['min_effect_size']} — with low "
        f"enough variance a 0.1% gap becomes significant and stays irrelevant.",
        "",
        "Three separate questions are tested, because a single comparison of mean loss "
        "cannot answer them all: does the arm reach a *better* loss, does it get there in "
        "*fewer steps*, and is it *more consistent* across seeds?",
        "",
        _HEADER,
        "|---|---|---|---|---|---|---|---|",
    ]

    def fmt(v: float | None, spec: str = ".4f") -> str:
        return "—" if v is None else format(v, spec)

    for c in payload["comparisons"]:
        lines.append(
            f"| {c['variant']} | {c['delta_vs_baseline']:+.4f} | {fmt(c.get('p_value'))} | "
            f"{fmt(c.get('cohens_d'), '+.2f')} | {fmt(c.get('steps_p'))} | "
            f"{fmt(c.get('variance_f'), '.1f')} | {fmt(c.get('variance_p'))} | "
            f"**{c['verdict']}** |"
        )

    lines += [
        "",
        "## How to read this",
        "",
        "The single-seed version of this experiment compared arms with a fixed 2% rule, "
        "which was an assumption rather than a measurement — with one run per arm there is "
        "no way to estimate run-to-run variance, so there is nothing to compare a gap "
        "against. With several seeds that variance is measured directly, and the question "
        "becomes whether the between-arm gap is large relative to it.",
        "",
        "**A `no difference` verdict here is a real result, not a missing one.** It says "
        "the experiment had the resolution to detect a difference of this size and did not "
        "find one. It does not say the technique does not work — these are 5M-parameter "
        "runs over 400 steps, and a stability aid has little to stabilise at that scale.",
        "",
        "**The `verdict` column refers to mean final loss only.** Read the other two "
        "p-values beside it. An arm can reach the same loss while getting there in half "
        "the steps, or with a fraction of the run-to-run spread, and both are results the "
        "mean comparison is structurally unable to report.",
        "",
        "Reproduce with: `python scripts/ablate_multiseed.py --replay`",
    ]

    path = RESULTS / f"{name}_multiseed.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["optimizer", "architecture", "all"], default="all")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--target", type=float, default=0.8)
    parser.add_argument("--seeds", type=int, default=5, help="How many seeds per arm.")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    seeds = DEFAULT_SEEDS[: args.seeds]

    wanted = []
    if args.suite in ("optimizer", "all"):
        wanted.append(("optimizer", optimizer_suite(args.steps, args.target)))
    if args.suite in ("architecture", "all"):
        wanted.append(("architecture", architecture_suite(args.steps, args.target)))

    for name, base in wanted:
        json_path = RESULTS / f"{name}_multiseed.json"
        if args.replay:
            if not json_path.exists():
                raise SystemExit(f"no committed results at {json_path}; run without --replay.")
            payload = reanalyse(json.loads(json_path.read_text(encoding="utf-8")))
            json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        else:
            suite = MultiSeedSuite(suite=base, seeds=seeds, out_dir=RESULTS)
            per_variant = suite.run()
            comparisons = suite.analyse(per_variant)
            suite.write_json(per_variant, comparisons)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            for c in comparisons:
                log.info("[%s] %s", name, c.sentence())

        figure = plot(payload, name)
        finding = write_finding(payload, name, figure)
        print(f"[{name}] wrote {json_path}, {figure}, {finding}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
