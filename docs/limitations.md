# Limitations

The single most important page in this documentation. Everything measured here comes from
a ~5M-parameter model trained for 95 seconds on a laptop CPU on a synthetic corpus. This
page states precisely what that does and does not support.

## The headline caveat

**Every result in this repository is a directional confirmation of a published finding,
not independent evidence about it.** The findings being confirmed were obtained at
100–1000× this scale. Reproducing the *direction* of an effect at 5M parameters shows the
implementation is correct and the mechanism is understood. It does not show the effect
holds at scale, and where the direction did *not* reproduce, this repository says so
rather than tuning until it did.

## What the `nano` model is

- **~5.0M parameters**, 6 layers × 256 width, 256-token context, 1024-token vocabulary.
- Trained on **819,200 tokens**, which is 0.8% of its own Chinchilla-optimal budget.
- Trained on a **synthetic story grammar**, not natural text. Roughly 440 word types.
  It writes coherent stories about Lily and Tom because that is the only thing it has
  ever read.
- It knows no facts, follows no general instructions, and cannot attempt HellaSwag, ARC
  or MMLU at anything above chance.

The `micro` tier (~40M parameters, FineWeb-Edu, full 20:1 budget) is the same code path
with more compute and is the tier the spec designates for reported language-modelling
results. It has not been run here; no GPU was available, so **no `micro` numbers appear
anywhere in this repository**. The recipe is committed and runnable.

## Statistical power

| Measurement | n | Uncertainty | What that means |
|---|---|---|---|
| Validation perplexity | 16,384 tokens | ±0.011 ppl | Small differences are real |
| Tiny benchmark | 28 questions | ±9 points near 50% | Differences under ~10 points are noise |
| Ablations | **1 seed** | unquantified | Differences under 2% are not reported as results |
| Preference head-to-head | 40 paired prompts |, | 3 wins vs 0 losses is suggestive, not conclusive |

Every ablation is a **single seed**. Proper practice is 3–5 seeds with a variance
estimate; that was not affordable here on CPU, and the ablation harness compensates by
refusing to name a winner on a gap below 2%.

## Where results did *not* reproduce

Recorded here rather than omitted:

- **GPTQ does not beat RTN at 4 bits** at this scale; all methods tie with fp32, because
  a 5M-parameter model has little redundancy for 4-bit rounding to destroy. The
  separation appears at 2–3 bits.
- **QK-norm, zero-init and SwiGLU-vs-ReLU² show no measurable difference in final loss.**
  A small model on a narrow domain is exactly the regime where stability aids have little
  to stabilise. QK-norm does cost 1.5× more steps to reach the target loss.
- **Speculative decoding is slower in wall-clock on this CPU**, despite reducing target
  forward passes 3×. At 5M parameters a forward pass is dominated by Python dispatch, not
  by weight loading, so the memory-bandwidth win the method exists for does not appear.
- **Muon loses to AdamW on a convex single-matrix problem.** Its advantage needs depth and
  stochasticity.

## What is simulated rather than implemented

- **Quantized inference is simulated in fp32.** There is no int4 matmul kernel on CPU, so
  quantized weights are dequantized before use. The **accuracy** cost is exactly real; the
  **memory** figures are computed analytically from the representation; **no latency claim
  is made** for quantization, because none was measured.
- **KV-cache quantization** likewise: real accuracy cost, analytic footprint, no measured
  latency win.
- **Speculative decoding is batch size 1.** Batched speculation needs per-row bookkeeping
  of divergent accepted lengths and a ragged cache; a serving-engine concern that does
  not change the algorithm.

## Evaluation-suite limits

The tiny benchmark is **saturated**: the base model scores 100% on all four tasks. A
saturated benchmark cannot rank models that are all good at it. Its role in Arc 2 is as a
**degradation detector**: "did 4-bit quantization break something?", and nothing more.

The preference judge is **programmatic**, not an LLM or a human. It scores on-topic
overlap, absence of degenerate repetition, and proper termination: exactly the properties
the synthetic preference labels encode. So it measures *did the model learn the labels*,
not *is the model better*. It is deliberately length-insensitive so a model that learned
to game DPO's length bias gains nothing from it.

## Data limits

The `nano` corpus, the instruction data and the preference data are all **synthetic**,
generated from grammars in `src/nanoscale/data/`. Advantages: deterministic, offline, no
licensing questions, and small enough that a 5M-parameter model can actually learn it.
Disadvantage: it is not language. Nothing here says anything about how these methods
behave on real text.

The preference data is **length-matched by construction** (chosen and rejected within 5%),
which is what makes the DPO length-exploitation diagnostic interpretable at all, but it
also means the length bias has less to grip than it would on real preference data where
longer responses genuinely are preferred more often.

## Measurement environment

All timings come from one shared laptop CPU running other work. Wall-clock numbers mix a
real per-step cost with scheduler noise, and the write-ups say so wherever a timing is
quoted. Step counts, forward-pass counts and token counts are the trustworthy columns.

## What this repository *does* support

- The algorithms are implemented from their papers and verified against independent
  references: attention against SDPA and against a hand-written formula, AdamW against
  `torch.optim.AdamW` to 1e-12, RoPE against a literal transcription of the paper's
  rotation, the speculative accept rule against direct sampling over 120k draws.
- The training loop is correct: it overfits a single batch, two seeded runs are identical
  to 1e-9, and resume is indistinguishable from an uninterrupted run.
- Compression composes: quantization and speculation stack, and speculation over a
  quantized target reproduces that target's greedy output exactly.
- Every number in the docs is produced by a committed script, stamped with the git SHA
  and hardware that produced it, and regenerable in replay mode.
