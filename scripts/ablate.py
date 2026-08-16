"""Run the Phase-5 controlled ablations and write figures + written findings.

Two suites, matching the spec:

* ``optimizer`` — AdamW-only vs the Muon+AdamW split (spec E3: "the optimizer was the
  biggest lever").
* ``architecture`` — QK-norm on/off, zero-init output projections on/off, and
  SwiGLU vs ReLU² (spec E2's speedrun stack).

Usage::

    python scripts/ablate.py                 # both suites
    python scripts/ablate.py --suite optimizer
    python scripts/ablate.py --steps 200     # quicker

Everything is written to ``results/ablations/``: one JSON table, one PNG and one
markdown finding per suite.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from nanoscale.bench.ablation import AblationResult, AblationSuite, Variant, describe_difference
from nanoscale.train import TrainResult
from nanoscale.utils.plotting import COLORS, new_figure, save_figure

RESULTS = Path("results/ablations")


def optimizer_suite(steps: int, target: float) -> AblationSuite:
    return AblationSuite(
        name="optimizer",
        question=(
            "Does routing hidden matmul weights to Muon beat sending everything to AdamW, "
            "at equal step budget and seed?"
        ),
        target_loss=target,
        base_overrides=(
            "train.device=cpu",
            f"train.max_steps={steps}",
            "train.token_budget=null",
            "train.log_interval=5",
            f"train.eval_interval={max(10, steps // 8)}",
            "train.ckpt_interval=100000",
        ),
        variants=[
            Variant(
                name="adamw",
                label="AdamW only",
                overrides=("train.optim.name=adamw",),
                note="Every parameter goes to AdamW. The baseline.",
            ),
            Variant(
                name="muon",
                label="Muon + AdamW",
                overrides=("train.optim.name=muon",),
                note="2D hidden matrices to Muon; embeddings, head, norms to AdamW.",
            ),
        ],
    )


def architecture_suite(steps: int, target: float) -> AblationSuite:
    base = (
        "train.device=cpu",
        f"train.max_steps={steps}",
        "train.token_budget=null",
        "train.log_interval=5",
        f"train.eval_interval={max(10, steps // 8)}",
        "train.ckpt_interval=100000",
    )
    return AblationSuite(
        name="architecture",
        question=(
            "Do the modded-nanoGPT speedrun's architecture choices — QK-norm, zero-init "
            "output projections, SwiGLU — measurably help at this scale?"
        ),
        target_loss=target,
        base_overrides=base,
        variants=[
            Variant(name="default", label="default (QK-norm, zero-init, SwiGLU)"),
            Variant(
                name="no_qk_norm",
                label="− QK-norm",
                overrides=("model.qk_norm=false",),
                note="Removes the RMS normalization of q and k before the dot product.",
            ),
            Variant(
                name="no_zero_init",
                label="− zero-init output",
                overrides=("model.zero_init_output=false",),
                note="Falls back to GPT-2's std/sqrt(2L) residual init.",
            ),
            Variant(
                name="relu2",
                label="ReLU² instead of SwiGLU",
                overrides=("model.mlp_type=relu2",),
                note="Ungated MLP; cheaper per parameter.",
            ),
        ],
    )


def plot(suite: AblationSuite, results: Sequence[AblationResult]) -> Path:
    fig, axes = new_figure(ncols=2, figsize=(11.5, 4.4))
    left, right = axes

    for i, res in enumerate(results):
        steps = [r["step"] for r in res.result.history if "loss" in r]
        loss = [r["loss"] for r in res.result.history if "loss" in r]
        elapsed = [r["elapsed_s"] for r in res.result.history if "loss" in r]
        color = COLORS[i % len(COLORS)]
        left.plot(steps, loss, color=color, label=res.variant.display())
        right.plot(elapsed, loss, color=color, label=res.variant.display())

    for ax, xlabel, title in (
        (left, "optimizer step", "Loss vs steps"),
        (right, "wall-clock seconds", "Loss vs wall clock"),
    ):
        ax.axhline(
            suite.target_loss, color="0.6", ls=":", lw=1.2, label=f"target = {suite.target_loss}"
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("training loss (nats/token)")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle(f"Ablation: {suite.name}", fontweight="bold")
    return save_figure(
        fig,
        RESULTS / f"{suite.name}.png",
        script="scripts/ablate.py",
        extra=f"tier={suite.tier} · single seed",
    )


def write_finding(suite: AblationSuite, results: Sequence[AblationResult], figure: Path) -> Path:
    baseline = results[0]
    lines = [
        f"# Ablation — {suite.name}",
        "",
        f"**Question.** {suite.question}",
        "",
        f"![{suite.name}]({figure.name})",
        "",
        "| variant | val loss | val ppl | steps → target | seconds → target | tok/s |",
        "|---|---|---|---|---|---|",
    ]
    for res in results:
        row = res.row()
        lines.append(
            f"| {row['variant']} | {row['final_val_loss']} | {row['val_ppl']} | "
            f"{row['steps_to_target'] if row['steps_to_target'] is not None else '—'} | "
            f"{row['seconds_to_target'] if row['seconds_to_target'] is not None else '—'} | "
            f"{row['tokens_per_s']} |"
        )
    lines += ["", "## Findings", ""]
    for res in results[1:]:
        lines.append(f"- {describe_difference(baseline, res)}")
        if res.variant.note:
            lines.append(f"  <br/>*{res.variant.note}*")
    lines += [
        "",
        "## How to read this",
        "",
        "All arms share one seed, one data order, one schedule and a fixed step budget; "
        "they differ only in the fields named in the variant. Runs are "
        f"`{suite.tier}` tier on CPU.",
        "",
        "**Steps-to-target is the trustworthy column; wall-clock is not.** These runs "
        "were executed sequentially on a shared laptop, so tokens/s is sensitive to "
        "whatever else the machine was doing. A per-step cost difference that is real "
        "(Muon adds five Newton-Schulz matmuls per 2D weight) is therefore mixed with "
        "measurement noise here. Treat the seconds columns as indicative and the step "
        "counts as the result.",
        "",
        "**These are single-seed results on a ~5M-parameter model trained on a synthetic "
        "corpus.** They are directional confirmations (or non-confirmations) of published "
        "findings obtained at 100–1000× this scale, not independent evidence about them. "
        "Differences below 2% in final loss are reported as *no measurable difference*, "
        "because at one seed that is what they are. A lever that matters at scale can be "
        "invisible here — a small model in a narrow domain is exactly the regime where "
        "stability aids have little to stabilise.",
        "",
        f"Reproduce with: `python scripts/ablate.py --suite {suite.name}`",
        "",
    ]
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{suite.name}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def replay(suite: AblationSuite) -> list[AblationResult]:
    """Rebuild results from a committed JSON table, without retraining.

    Spec Phase 10 asks for the results to regenerate "from one command in replay/offline
    mode". This is that mode for the ablations: the numbers come from the committed
    artifact, so re-rendering a write-up after a wording change cannot silently produce
    different numbers from the ones that were reviewed.
    """
    path = RESULTS / f"{suite.name}.json"
    if not path.exists():
        raise SystemExit(f"no committed results at {path}; run without --replay first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_label = {v.display(): v for v in suite.variants}
    out: list[AblationResult] = []
    for row in payload["rows"]:
        run_dir = Path(row["run_dir"])
        history: list[dict[str, float]] = []
        metrics = run_dir / "metrics.jsonl"
        if metrics.exists():
            history = [json.loads(line) for line in metrics.read_text().splitlines() if line]
        out.append(
            AblationResult(
                variant=by_label.get(row["variant"], Variant(row["variant"], label=row["variant"])),
                result=TrainResult(
                    final_train_loss=row["final_train_loss"],
                    final_val_loss=row["final_val_loss"],
                    best_val_loss=row["final_val_loss"],
                    steps=0,
                    tokens=0,
                    wall_clock_s=row["wall_clock_s"],
                    tokens_per_second=row["tokens_per_s"],
                    history=history,
                ),
                steps_to_target=row["steps_to_target"],
                seconds_to_target=row["seconds_to_target"],
                run_dir=run_dir,
            )
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["optimizer", "architecture", "all"], default="all")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--target", type=float, default=0.8, help="Target loss for the crossing.")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Re-render figures and findings from the committed JSON, without retraining.",
    )
    args = parser.parse_args()

    suites = []
    if args.suite in ("optimizer", "all"):
        suites.append(optimizer_suite(args.steps, args.target))
    if args.suite in ("architecture", "all"):
        suites.append(architecture_suite(args.steps, args.target))

    for suite in suites:
        results = replay(suite) if args.replay else suite.run()
        table = suite.write_json(results)
        figure = plot(suite, results)
        finding = write_finding(suite, results, figure)
        print(f"[{suite.name}] wrote {table}, {figure}, {finding}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
