"""A controlled-A/B harness for the Phase-5 ablations (spec Phase 5).

The point of this module is to make an ablation *honest by construction*:

* every variant is built from **one** base config with a named set of overrides, so a
  variant differs from the baseline in exactly the fields the table says it does;
* every variant runs with the **same seed, same data, same schedule and same step
  budget**, so the only difference is the thing under test;
* the reported metric is not just final loss but **steps-to-target** and
  **wall-clock-to-target**, because "reaches a lower loss eventually" and "gets there
  sooner" are different claims and the optimizer literature is about the second;
* every run writes its own manifest and metrics, so a number in a results table can be
  traced back to a run directory.

A note on what these numbers can support: at ``nano`` scale, with one seed, a difference
of a few percent in final loss is noise. The write-ups say so, and the reporting helper
below refuses to describe a difference smaller than ``NOISE_THRESHOLD`` as anything but
"no measurable difference".
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanoscale.config import ExperimentConfig, load_experiment
from nanoscale.tokenizer import BPETokenizer
from nanoscale.train import Trainer, TrainResult
from nanoscale.utils import get_logger, git_sha, hardware_string

__all__ = [
    "NOISE_THRESHOLD",
    "AblationResult",
    "AblationSuite",
    "Variant",
    "describe_difference",
]

log = get_logger("nanoscale.bench.ablation")

#: Relative difference in final loss below which we decline to call a winner at this
#: scale with a single seed.
NOISE_THRESHOLD = 0.02


@dataclass(frozen=True, slots=True)
class Variant:
    """One arm of an ablation: a name, config overrides, and a one-line rationale."""

    name: str
    overrides: tuple[str, ...] = ()
    label: str = ""
    note: str = ""

    def display(self) -> str:
        """Human-readable label for figures and tables."""
        return self.label or self.name


@dataclass(slots=True)
class AblationResult:
    """The measured outcome of one variant."""

    variant: Variant
    result: TrainResult
    steps_to_target: int | None
    seconds_to_target: float | None
    run_dir: Path

    @property
    def name(self) -> str:
        """Variant name."""
        return self.variant.name

    def row(self) -> dict[str, Any]:
        """A flat row for the results table."""
        return {
            "variant": self.variant.display(),
            "final_train_loss": round(self.result.final_train_loss, 4),
            "final_val_loss": round(self.result.final_val_loss, 4),
            "val_ppl": round(math.exp(min(self.result.final_val_loss, 20.0)), 3),
            "steps_to_target": self.steps_to_target,
            "seconds_to_target": (
                round(self.seconds_to_target, 2) if self.seconds_to_target is not None else None
            ),
            "wall_clock_s": round(self.result.wall_clock_s, 2),
            "tokens_per_s": round(self.result.tokens_per_second, 1),
            "run_dir": str(self.run_dir),
        }


def _first_crossing(
    history: Sequence[dict[str, float]], target: float
) -> tuple[int | None, float | None]:
    """First ``(step, elapsed_s)`` at which the smoothed train loss drops below ``target``.

    Smoothing over a short window is deliberate: a single lucky batch dipping under the
    threshold is not "reaching" it, and without smoothing the steps-to-target metric is
    dominated by batch noise.
    """
    window: list[float] = []
    for row in history:
        if "loss" not in row:
            continue
        window.append(float(row["loss"]))
        if len(window) > 3:
            window.pop(0)
        if sum(window) / len(window) <= target:
            return int(row["step"]), float(row.get("elapsed_s", 0.0))
    return None, None


@dataclass(slots=True)
class AblationSuite:
    """Runs a set of variants against one base config and writes the artifacts."""

    name: str
    question: str
    variants: list[Variant]
    tier: str = "nano"
    base_overrides: tuple[str, ...] = ()
    target_loss: float = 1.0
    out_dir: Path = field(default_factory=lambda: Path("results/ablations"))
    runs_dir: Path = field(default_factory=lambda: Path("runs/ablations"))
    tokenizer_path: Path = field(default_factory=lambda: Path("artifacts/tokenizer/nano.json"))

    def config_for(self, variant: Variant) -> ExperimentConfig:
        """Resolve the config for one variant: tier preset + base overrides + variant."""
        return load_experiment(tier=self.tier, overrides=[*self.base_overrides, *variant.overrides])

    def run(self) -> list[AblationResult]:
        """Train every variant and return their measured results."""
        tokenizer = BPETokenizer.load(self.tokenizer_path)
        results: list[AblationResult] = []
        for variant in self.variants:
            cfg = self.config_for(variant)
            run_dir = self.runs_dir / self.name / variant.name
            log.info("[%s] running variant %r", self.name, variant.name)
            trainer = Trainer(
                cfg,
                tokenizer=tokenizer,
                out_dir=run_dir,
                phase=f"phase5-ablation-{self.name}",
                run_name=f"{self.name}/{variant.name}",
            )
            outcome = trainer.train()
            steps, seconds = _first_crossing(outcome.history, self.target_loss)
            results.append(
                AblationResult(
                    variant=variant,
                    result=outcome,
                    steps_to_target=steps,
                    seconds_to_target=seconds,
                    run_dir=run_dir,
                )
            )
        return results

    # ---------------------------------------------------------------- reporting

    def write_json(self, results: Sequence[AblationResult]) -> Path:
        """Write the machine-readable results table."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "question": self.question,
            "tier": self.tier,
            "target_loss": self.target_loss,
            "git_sha": git_sha(),
            "hardware": hardware_string(),
            "base_overrides": list(self.base_overrides),
            "rows": [r.row() for r in results],
        }
        path = self.out_dir / f"{self.name}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path


def describe_difference(
    baseline: AblationResult,
    challenger: AblationResult,
    *,
    threshold: float = NOISE_THRESHOLD,
) -> str:
    """One honest sentence comparing two arms, refusing to over-claim on small gaps."""
    base_loss = baseline.result.final_val_loss
    other_loss = challenger.result.final_val_loss
    if base_loss <= 0:
        return "baseline loss is non-positive; comparison is meaningless."
    relative = (base_loss - other_loss) / base_loss

    speed = _describe_speed(baseline, challenger)

    if abs(relative) < threshold:
        line = (
            f"**No measurable difference in final loss.** {challenger.variant.display()} "
            f"reaches {other_loss:.4f} vs {base_loss:.4f} for {baseline.variant.display()} "
            f": a {abs(relative) * 100:.1f}% gap, below the {threshold * 100:.0f}% we are "
            f"willing to call a result from a single seed at this scale."
        )
        # Converging to the same place and getting there at the same rate are different
        # claims. A variant can tie on final loss and still be clearly slower, and saying
        # only "no difference" would hide that.
        return f"{line} {speed}" if speed else line

    direction = "better" if relative > 0 else "worse"
    line = (
        f"**{challenger.variant.display()} is {abs(relative) * 100:.1f}% {direction}** on "
        f"final validation loss ({other_loss:.4f} vs {base_loss:.4f})."
    )
    return f"{line} {speed}" if speed else line


def _describe_speed(baseline: AblationResult, challenger: AblationResult) -> str:
    """Describe the steps-to-target comparison, or why it is unavailable."""
    base_steps, other_steps = baseline.steps_to_target, challenger.steps_to_target
    if other_steps is None and base_steps is not None:
        return f"{challenger.variant.display()} never reached the target loss."
    if base_steps is None or other_steps is None:
        return ""
    ratio = base_steps / max(1, other_steps)
    if abs(ratio - 1.0) < 0.15:
        return (
            f"Both reach the target loss in about the same number of steps "
            f"({other_steps} vs {base_steps})."
        )
    if ratio > 1.0:
        return (
            f"It also reaches the target loss in **{ratio:.2f}x fewer steps** "
            f"({other_steps} vs {base_steps})."
        )
    return (
        f"It needs **{1.0 / ratio:.2f}x more steps** to reach the target loss "
        f"({other_steps} vs {base_steps}), so the two converge to the same place at "
        f"different rates."
    )
