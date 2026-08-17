"""Multi-seed ablations with a real significance test.

The single-seed ablation harness in :mod:`nanoscale.bench.ablation` reports any gap
smaller than 2% as "no measurable difference". That threshold was an honest guess, not a
measurement: with one run per arm there is no way to estimate run-to-run variance, so
there is nothing to compare a gap *against*.

This module fixes that properly. Each arm is trained at several seeds, and the comparison
becomes a two-sample test rather than a subtraction:

.. code-block:: text

    Welch's t  =  (mean_a − mean_b) / sqrt(s_a²/n_a + s_b²/n_b)

Welch rather than Student because the two arms have no reason to share a variance — an
architectural change can affect run-to-run stability as well as the mean, and assuming
equal variances would be assuming away part of what is being measured.

Alongside the p-value the module reports **Cohen's d**, the effect size in units of
pooled standard deviation. This matters more than significance here: with enough seeds a
0.1% loss difference becomes statistically significant and remains practically irrelevant.
A result is only reported as real when it is both significant *and* large enough to care
about, and the two criteria are shown separately so the reader can disagree with the
threshold.

Seeds control initialization, data order and dropout together, which is what "run-to-run
variance" means for a training run. They do not control non-determinism in the accelerator
kernels, so the variance measured here is a lower bound on what a different machine would
see.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from nanoscale.bench.ablation import AblationResult, AblationSuite
from nanoscale.utils import get_logger, git_sha, hardware_string

log = get_logger("nanoscale.bench.multiseed")

__all__ = [
    "ALPHA",
    "MIN_EFFECT_SIZE",
    "ArmStatistics",
    "MultiSeedComparison",
    "MultiSeedSuite",
    "cohens_d",
    "holm_bonferroni",
    "variance_ratio_test",
    "welch_t_test",
]

#: Below this effect size a difference is called practically negligible even when the
#: p-value is small. 0.8 is Cohen's conventional "large effect" boundary; we use it
#: because at this scale anything smaller will not survive a change of hardware.
MIN_EFFECT_SIZE = 0.8

#: Significance level for the two-sided Welch test.
ALPHA = 0.05


def _student_t_sf(t: float, df: float) -> float:
    """Upper-tail probability of Student's t, via the regularized incomplete beta.

    Implemented directly rather than pulling in SciPy, which is not a dependency of this
    project and would be a heavy addition for one function. Uses the identity
    ``P(T > t) = 0.5 · I_x(df/2, 1/2)`` with ``x = df / (df + t²)``.
    """
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    p = 0.5 * _betainc(df / 2.0, 0.5, x)
    return p if t > 0 else 1.0 - p


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function ``I_x(a, b)`` by continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    # Lentz's algorithm on the continued fraction; converges in well under 200 terms here.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float, *, max_iter: int = 200, eps: float = 1e-12) -> float:
    """Continued-fraction expansion used by :func:`_betainc`."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < eps:
        d = eps
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < eps:
            d = eps
        if abs(c) < eps:
            c = eps
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < eps:
            d = eps
        if abs(c) < eps:
            c = eps
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def welch_t_test(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    """Two-sided Welch's t-test.

    Returns:
        ``(t, degrees_of_freedom, p_value)``. Returns ``(nan, nan, nan)`` when either
        sample has fewer than two observations, since variance is undefined there.
    """
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"), float("nan"))

    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)

    se_sq = va / na + vb / nb
    if se_sq <= 0.0:
        # Both arms are exactly constant. Identical means -> no difference; different
        # means -> a difference with no variance to test against, which is not a p-value
        # question.
        return (0.0, float("nan"), 1.0) if ma == mb else (float("inf"), float("nan"), 0.0)

    t = (ma - mb) / math.sqrt(se_sq)
    df = se_sq**2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p = 2.0 * _student_t_sf(abs(t), df)
    return (t, df, min(1.0, max(0.0, p)))


def variance_ratio_test(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Two-sided F-test for equality of variances. Returns ``(F, p)``.

    Added because the first multi-seed optimizer run produced a result the mean comparison
    could not express: Muon's five seeds landed within 0.003 of each other while AdamW's
    spanned 0.10, one of them having effectively failed to converge. Averaging that away
    and reporting "no significant difference in mean loss" would have discarded the most
    striking thing in the data.

    **Stability across seeds is a property worth testing directly**, not an inconvenience
    for the mean comparison. An optimizer that reaches a slightly better loss every time
    is more useful than one that sometimes reaches a much better loss and sometimes
    diverges, and a t-test on the mean cannot say that.

    F is oriented larger-variance-over-smaller so it is always ≥ 1, and the p-value is
    doubled accordingly.
    """
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"))
    va, vb = statistics.variance(a), statistics.variance(b)
    if va <= 0.0 or vb <= 0.0:
        return (float("inf"), 0.0) if va != vb else (1.0, 1.0)

    if va >= vb:
        f, df1, df2 = va / vb, len(a) - 1, len(b) - 1
    else:
        f, df1, df2 = vb / va, len(b) - 1, len(a) - 1
    # P(F > f) = I_x(df2/2, df1/2) with x = df2 / (df2 + df1*f).
    x = df2 / (df2 + df1 * f)
    p = min(1.0, 2.0 * _betainc(df2 / 2.0, df1 / 2.0, x))
    return (f, p)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Effect size in pooled standard deviations."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0.0:
        return 0.0
    return (statistics.fmean(a) - statistics.fmean(b)) / pooled


def holm_bonferroni(p_values: Sequence[float], *, alpha: float = ALPHA) -> list[bool]:
    """Holm-Bonferroni step-down correction. Returns one "reject the null" flag per input.

    Running three comparisons against one baseline at α=0.05 gives roughly a 14% chance of
    at least one false positive, so an uncorrected p=0.035 among three tests is close to
    what noise alone produces. This is not hypothetical here: the architecture suite's
    five-seed run produced exactly that, and reporting it as a finding would have been
    wrong.

    Holm rather than plain Bonferroni because it controls the same family-wise error rate
    while being uniformly more powerful: sort the p-values ascending and compare the k-th
    against ``alpha / (m - k)``, stopping at the first failure.

    NaNs (arms with too few seeds to test) are never rejected.
    """
    order = sorted(range(len(p_values)), key=lambda i: (math.isnan(p_values[i]), p_values[i]))
    m = len(p_values)
    out = [False] * m
    for k, idx in enumerate(order):
        p = p_values[idx]
        if math.isnan(p) or p > alpha / (m - k):
            break  # step-down: once one fails, every larger p-value fails too
        out[idx] = True
    return out


@dataclass(frozen=True)
class ArmStatistics:
    """Per-arm summary across seeds."""

    name: str
    display: str
    losses: tuple[float, ...]
    steps_to_target: tuple[int | None, ...]

    @property
    def mean(self) -> float:
        """Mean final validation loss across seeds."""
        return statistics.fmean(self.losses)

    @property
    def stdev(self) -> float:
        """Sample standard deviation; 0 for a single seed."""
        return statistics.stdev(self.losses) if len(self.losses) > 1 else 0.0

    @property
    def stderr(self) -> float:
        """Standard error of the mean."""
        return self.stdev / math.sqrt(len(self.losses)) if self.losses else 0.0

    @property
    def mean_steps(self) -> float | None:
        """Mean steps-to-target over the seeds that reached it."""
        hit = [s for s in self.steps_to_target if s is not None]
        return statistics.fmean(hit) if hit else None

    def summary(self) -> dict[str, Any]:
        """Flat numbers for the results table."""
        return {
            "variant": self.display,
            "n_seeds": len(self.losses),
            "mean_val_loss": round(self.mean, 5),
            "stdev": round(self.stdev, 5),
            "stderr": round(self.stderr, 5),
            "losses": [round(x, 5) for x in self.losses],
            "mean_steps_to_target": self.mean_steps,
            "steps_to_target": list(self.steps_to_target),
        }


@dataclass(frozen=True)
class MultiSeedComparison:
    """One challenger arm against the baseline, with a significance verdict."""

    baseline: ArmStatistics
    challenger: ArmStatistics
    t: float
    df: float
    p_value: float
    effect_size: float
    steps_t: float = float("nan")
    steps_p: float = float("nan")
    variance_f: float = float("nan")
    variance_p: float = float("nan")
    #: Set by :meth:`MultiSeedSuite.analyse` once the whole family of tests is known.
    survives_correction: bool = True

    @property
    def significant(self) -> bool:
        """True when p < ALPHA."""
        return self.p_value < ALPHA

    @property
    def large_enough(self) -> bool:
        """True when |Cohen's d| clears :data:`MIN_EFFECT_SIZE`."""
        return abs(self.effect_size) >= MIN_EFFECT_SIZE

    @property
    def verdict(self) -> str:
        """A short label combining significance, multiplicity and effect size."""
        if math.isnan(self.p_value):
            return "insufficient seeds"
        if not self.significant:
            return "no difference"
        if not self.survives_correction:
            return "not significant after correction"
        if not self.large_enough:
            return "significant but negligible"
        return "better" if self.challenger.mean < self.baseline.mean else "worse"

    def sentence(self) -> str:
        """One honest sentence, refusing to over- or under-claim."""
        d = self.challenger.mean - self.baseline.mean
        rel = abs(d) / self.baseline.mean * 100 if self.baseline.mean else 0.0
        stem = (
            f"{self.challenger.display}: {self.challenger.mean:.4f} ± {self.challenger.stderr:.4f} "
            f"vs baseline {self.baseline.mean:.4f} ± {self.baseline.stderr:.4f} "
            f"({rel:.1f}% {'lower' if d < 0 else 'higher'})"
        )
        if math.isnan(self.p_value):
            return f"{stem}. Too few seeds to test."
        stats = f"Welch t={self.t:.2f}, p={self.p_value:.3f}, d={self.effect_size:.2f}"
        if not self.significant:
            return (
                f"{stem}. **Not significant** ({stats}) — the gap is within run-to-run "
                f"variance, so this experiment cannot distinguish the arms."
            )
        if not self.large_enough:
            return (
                f"{stem}. **Significant but negligible** ({stats}) — the difference is real "
                f"and smaller than |d|={MIN_EFFECT_SIZE}, so it is unlikely to survive a "
                f"change of scale or hardware."
            )
        direction = "better" if d < 0 else "worse"
        return f"{stem}. **Significantly {direction}** ({stats})."

    @property
    def steps_significant(self) -> bool:
        """True when the steps-to-target difference clears ALPHA."""
        return not math.isnan(self.steps_p) and self.steps_p < ALPHA

    @property
    def variance_significant(self) -> bool:
        """True when the two arms have significantly different run-to-run variance."""
        return not math.isnan(self.variance_p) and self.variance_p < ALPHA

    def summary(self) -> dict[str, Any]:
        """Flat numbers for the results table."""

        def r(x: float, n: int = 4) -> float | None:
            return None if math.isnan(x) else round(x, n)

        return {
            "variant": self.challenger.display,
            "mean_val_loss": round(self.challenger.mean, 5),
            "stderr": round(self.challenger.stderr, 5),
            "stdev": round(self.challenger.stdev, 5),
            "delta_vs_baseline": round(self.challenger.mean - self.baseline.mean, 5),
            "t": r(self.t),
            "df": r(self.df, 2),
            "p_value": r(self.p_value, 5),
            "cohens_d": r(self.effect_size),
            "significant": self.significant,
            "survives_correction": self.survives_correction,
            "large_enough": self.large_enough,
            "verdict": self.verdict,
            "steps_t": r(self.steps_t),
            "steps_p": r(self.steps_p, 5),
            "steps_significant": self.steps_significant,
            "variance_f": r(self.variance_f, 2),
            "variance_p": r(self.variance_p, 5),
            "variance_significant": self.variance_significant,
        }


@dataclass
class MultiSeedSuite:
    """Runs an :class:`AblationSuite` at several seeds and compares arms statistically."""

    suite: AblationSuite
    seeds: tuple[int, ...] = (1337, 42, 7, 2024, 31337)
    out_dir: Path = field(default_factory=lambda: Path("results/ablations"))

    def run(self) -> dict[str, list[AblationResult]]:
        """Train every variant at every seed. Returns ``variant name -> results``."""
        per_variant: dict[str, list[AblationResult]] = {v.name: [] for v in self.suite.variants}
        for seed in self.seeds:
            log.info("[%s] seed %d of %s", self.suite.name, seed, list(self.seeds))
            seeded = AblationSuite(
                name=f"{self.suite.name}",
                question=self.suite.question,
                variants=self.suite.variants,
                tier=self.suite.tier,
                base_overrides=(*self.suite.base_overrides, f"train.seed={seed}"),
                target_loss=self.suite.target_loss,
                out_dir=self.suite.out_dir,
                runs_dir=self.suite.runs_dir / f"seed{seed}",
                tokenizer_path=self.suite.tokenizer_path,
            )
            for result in seeded.run():
                per_variant[result.name].append(result)
        return per_variant

    def analyse(self, per_variant: dict[str, list[AblationResult]]) -> list[MultiSeedComparison]:
        """Compare every non-baseline arm against the first variant."""
        stats = {
            name: ArmStatistics(
                name=name,
                display=results[0].variant.display(),
                losses=tuple(r.result.final_val_loss for r in results),
                steps_to_target=tuple(r.steps_to_target for r in results),
            )
            for name, results in per_variant.items()
            if results
        }
        baseline_name = self.suite.variants[0].name
        baseline = stats[baseline_name]

        out: list[MultiSeedComparison] = []
        for variant in self.suite.variants[1:]:
            arm = stats.get(variant.name)
            if arm is None:
                continue
            t, df, p = welch_t_test(list(arm.losses), list(baseline.losses))
            # Steps-to-target is often the cleaner signal: it is an integer count with
            # little within-arm spread, where final loss can be dominated by one seed that
            # converged badly.
            arm_steps = [float(s) for s in arm.steps_to_target if s is not None]
            base_steps = [float(s) for s in baseline.steps_to_target if s is not None]
            st, _, sp = welch_t_test(arm_steps, base_steps)
            vf, vp = variance_ratio_test(list(arm.losses), list(baseline.losses))
            out.append(
                MultiSeedComparison(
                    baseline=baseline,
                    challenger=arm,
                    t=t,
                    df=df,
                    p_value=p,
                    effect_size=cohens_d(list(arm.losses), list(baseline.losses)),
                    steps_t=st,
                    steps_p=sp,
                    variance_f=vf,
                    variance_p=vp,
                )
            )

        # Every arm in a suite is tested against the same baseline, so these are one
        # family and the per-test alpha has to be adjusted for how many were run.
        flags = holm_bonferroni([c.p_value for c in out])
        return [replace(c, survives_correction=f) for c, f in zip(out, flags, strict=True)]

    def write_json(
        self,
        per_variant: dict[str, list[AblationResult]],
        comparisons: Sequence[MultiSeedComparison],
    ) -> Path:
        """Write the machine-readable multi-seed results."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        baseline_name = self.suite.variants[0].name
        arms = {
            name: ArmStatistics(
                name=name,
                display=results[0].variant.display(),
                losses=tuple(r.result.final_val_loss for r in results),
                steps_to_target=tuple(r.steps_to_target for r in results),
            ).summary()
            for name, results in per_variant.items()
            if results
        }
        payload = {
            "name": f"{self.suite.name}_multiseed",
            "question": self.suite.question,
            "tier": self.suite.tier,
            "seeds": list(self.seeds),
            "alpha": ALPHA,
            "correction": "holm-bonferroni",
            "min_effect_size": MIN_EFFECT_SIZE,
            "baseline": baseline_name,
            "git_sha": git_sha(),
            "hardware": hardware_string(),
            "arms": arms,
            "comparisons": [c.summary() for c in comparisons],
        }
        path = self.out_dir / f"{self.suite.name}_multiseed.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path
