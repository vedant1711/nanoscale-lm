# Ablation — architecture (multi-seed)

**Question.** Do the modded-nanoGPT speedrun's architecture choices — QK-norm, zero-init output projections, SwiGLU — measurably help at this scale?

![architecture multi-seed](architecture_multiseed.png)

Every arm trained at 5 seeds (1337, 42, 7, 2024, 31337). Arms differ only in the named field; seed controls initialisation and data order together.

| variant | mean val loss | ± stderr | seeds | mean steps → target |
|---|---|---|---|---|
| default (QK-norm, zero-init, SwiGLU) | **0.3882** | 0.0015 | 5 | 53.0 |
| − QK-norm | **0.3965** | 0.0062 | 5 | 76.0 |
| − zero-init output | **0.3837** | 0.0008 | 5 | 43.0 |
| ReLU² instead of SwiGLU | **0.3916** | 0.0045 | 5 | 55.0 |

## Significance

Two-sided Welch's t-test against the baseline arm at α=0.05, **Holm-Bonferroni corrected** across the 3 comparisons in this suite, with Cohen's d alongside. A difference counts as real only when it survives the correction *and* has |d| ≥ 0.8 — with low enough variance a 0.1% gap becomes significant and stays irrelevant.

Three separate questions are tested, because a single comparison of mean loss cannot answer them all: does the arm reach a *better* loss, does it get there in *fewer steps*, and is it *more consistent* across seeds?

| variant | Δ mean loss | p (loss) | Cohen's d | p (steps) | var F | p (var) | verdict |
|---|---|---|---|---|---|---|---|
| − QK-norm | +0.0083 | 0.2559 | +0.82 | 0.0000 | 18.0 | 0.0160 | **no difference** |
| − zero-init output | -0.0045 | 0.0353 | -1.70 | 0.0004 | 3.3 | 0.2721 | **not significant after correction** |
| ReLU² instead of SwiGLU | +0.0035 | 0.4950 | +0.47 | 0.1778 | 9.3 | 0.0527 | **no difference** |

## How to read this

The single-seed version of this experiment compared arms with a fixed 2% rule, which was an assumption rather than a measurement — with one run per arm there is no way to estimate run-to-run variance, so there is nothing to compare a gap against. With several seeds that variance is measured directly, and the question becomes whether the between-arm gap is large relative to it.

**A `no difference` verdict here is a real result, not a missing one.** It says the experiment had the resolution to detect a difference of this size and did not find one. It does not say the technique does not work — these are 5M-parameter runs over 400 steps, and a stability aid has little to stabilise at that scale.

**The `verdict` column refers to mean final loss only.** Read the other two p-values beside it. An arm can reach the same loss while getting there in half the steps, or with a fraction of the run-to-run spread, and both are results the mean comparison is structurally unable to report.

Reproduce with: `python scripts/ablate_multiseed.py --replay`
