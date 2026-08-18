# Ablation, architecture

**Question.** Do the modded-nanoGPT speedrun's architecture choices, QK-norm, zero-init output projections, SwiGLU: measurably help at this scale?

![architecture](architecture.png)

| variant | val loss | val ppl | steps → target | seconds → target | tok/s |
|---|---|---|---|---|---|
| default (QK-norm, zero-init, SwiGLU) | 0.3896 | 1.476 | 50 | 22.86 | 4362.3 |
| − QK-norm | 0.3893 | 1.476 | 75 | 33.05 | 4266.5 |
| − zero-init output | 0.3842 | 1.468 | 45 | 25.14 | 3782.2 |
| ReLU² instead of SwiGLU | 0.3862 | 1.471 | 55 | 25.93 | 4190.6 |

## Findings

- **No measurable difference in final loss.** − QK-norm reaches 0.3893 vs 0.3896 for default (QK-norm, zero-init, SwiGLU); a 0.1% gap, below the 2% we are willing to call a result from a single seed at this scale. It needs **1.50x more steps** to reach the target loss (75 vs 50), so the two converge to the same place at different rates.
  <br/>*Removes the RMS normalization of q and k before the dot product.*
- **No measurable difference in final loss.** − zero-init output reaches 0.3842 vs 0.3896 for default (QK-norm, zero-init, SwiGLU); a 1.4% gap, below the 2% we are willing to call a result from a single seed at this scale. Both reach the target loss in about the same number of steps (45 vs 50).
  <br/>*Falls back to GPT-2's std/sqrt(2L) residual init.*
- **No measurable difference in final loss.** ReLU² instead of SwiGLU reaches 0.3862 vs 0.3896 for default (QK-norm, zero-init, SwiGLU); a 0.9% gap, below the 2% we are willing to call a result from a single seed at this scale. Both reach the target loss in about the same number of steps (55 vs 50).
  <br/>*Ungated MLP; cheaper per parameter.*

## How to read this

All arms share one seed, one data order, one schedule and a fixed step budget; they differ only in the fields named in the variant. Runs are `nano` tier on CPU.

**Steps-to-target is the trustworthy column; wall-clock is not.** These runs were executed sequentially on a shared laptop, so tokens/s is sensitive to whatever else the machine was doing. A per-step cost difference that is real (Muon adds five Newton-Schulz matmuls per 2D weight) is therefore mixed with measurement noise here. Treat the seconds columns as indicative and the step counts as the result.

**These are single-seed results on a ~5M-parameter model trained on a synthetic corpus.** They are directional confirmations (or non-confirmations) of published findings obtained at 100–1000× this scale, not independent evidence about them. Differences below 2% in final loss are reported as *no measurable difference*, because at one seed that is what they are. A lever that matters at scale can be invisible here; a small model in a narrow domain is exactly the regime where stability aids have little to stabilise.

Reproduce with: `python scripts/ablate.py --suite architecture`
