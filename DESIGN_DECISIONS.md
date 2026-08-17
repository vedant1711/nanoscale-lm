# Design decisions (ADR-style)

One section per decision. Each records the evidence that motivated it, what was
actually built, and the honest caveat. The E-numbers refer to the evidence table in
the project spec.

---

## D0. Configuration is a frozen, hashable, fully-documented object graph

**Context.** Every claim this project makes has to be traceable to the exact setup
that produced it. That is impossible if configuration lives in argparse namespaces
and mutable dicts.

**Decision.** All configuration is pydantic v2 with `frozen=True`, `extra="forbid"`,
and a description on every single field (there is a test that fails if a field is
undocumented). Configs hash to a stable 12-hex-character digest that is written into
the run manifest alongside the git SHA, the seed, library versions and the hardware
string.

**Consequence.** `nanoscale config hash -c <file>` is enough to identify a run.
Computed fields (`head_dim`, `ffn_dim`, ...) are stripped by `dump_inputs()` before
any round-trip, so save → load → validate is an exact identity (tested).

**Caveat.** Frozen configs mean any "just tweak it at runtime" hack has to become an
explicit `merged(...)` call. That is the point, but it is friction.

---

## D1. The size ladder pins *shapes*, and reports parameter counts honestly

**Context.** The spec's tier table gives approximate parameter figures ("~2–4M",
"~25M", "~120M") alongside exact shapes (`6 × 256 × 4`, `8 × 512 × 8`,
`12 × 768 × 12`).

**Decision.** The shapes are pinned exactly. The parameter counts that those shapes
actually produce — with SwiGLU at an 8/3 expansion, GQA, and an **untied** LM head —
are computed analytically, reported in both total and non-embedding form, and asserted
as exact constants in `tests/unit/test_config.py`.

| Tier | Shape | Total params | Non-embedding | Spec's approximate figure |
|---|---|---|---|---|
| `nano` | 6 × 256 × 4, ctx 256, vocab 1024 | 4,952,064 | 4,427,776 | ~2–4M |
| `micro` | 8 × 512 × 8, ctx 512, vocab 16384 | 40,379,904 | 23,602,688 | ~25M |
| `small` | 12 × 768 × 12, ctx 1024, vocab 32768 | 125,849,856 | 75,518,208 | ~120M (GPT-2-ish) |

The spec's `micro` and `small` figures line up well with the non-embedding and total
counts respectively; its `nano` figure is simply lower than the stated shape can
produce (6 × 256 SwiGLU blocks alone are ~4.4M parameters). Rather than shrink the
architecture away from the spec, the repo keeps the shape and states the number.

`nano`'s 1024-token vocabulary is sized to exactly what the offline toy corpus can
fill (761 learnable merges + 256 byte tokens + 7 specials). A larger vocabulary would
add dead embedding rows that cost parameters and CPU time while carrying no
information.

**Caveat.** Comparing "parameter counts" across projects is ambiguous unless you say
whether embeddings and an untied head are included. Every table in this repo says.

---

## D2. `nano` is deliberately *not* compute-optimal

**Context.** The Chinchilla-style 20:1 tokens-per-parameter heuristic (Hoffmann et al.,
arXiv:2203.15556) is what makes a training run compute-honest rather than arbitrary.

**Decision.** `micro` and `small` set their stopping budget to exactly
`20 × params` tokens. `nano` does not: 20:1 on 6.5M parameters is ~130M tokens, which
is hours of CPU, and `nano` exists to be a sub-10-minute CI and teaching run. Its
budget is derived from its step count instead, and its manifests record the fraction
of the compute-optimal budget actually covered. `nano` also trains on a synthetic
corpus rather than real text — see D4.

**Consequence.** No `nano` result is ever presented as a quality claim. It is a
correctness and plumbing tier.

---

## D3. Everything has a CPU path, and CI never needs a GPU

**Context.** Spec constraint A3.1: zero paid resources, and a laptop tier that runs
end-to-end with no GPU at all.

**Decision.** `resolve_device` degrades an unavailable accelerator request to CPU
instead of raising, `resolve_dtype` degrades bf16/fp16 autocast to fp32 on CPU, and
the CI matrix runs the whole `nano` tier on Ubuntu CPU runners across Python 3.11 and
3.12. GPU runs are *reported experiments*, never gates.

**Caveat.** Absolute throughput numbers are therefore hardware-dependent and are
always stamped with the hardware string that produced them.

---

## D4. Reductions promote to at least fp32, and never demote

**Context.** The RMS reduction, the attention softmax and the RoPE rotation all need
more precision than the activations they operate on. The reflex is `x.float()`.

**Decision.** `model/numerics.py::accumulation_dtype` returns the dtype to accumulate
in: fp32 for anything smaller, and the *input* dtype for float64 and complex128.

**Why it exists.** `x.float()` promotes bf16/fp16 as intended but silently **demotes**
float64 to fp32. That is not a style problem: the numerical test suite gets its teeth
from fp64 reference implementations, and the demotion capped their agreement at 1e-5
when the true agreement is 1e-10. A test that passes at a tolerance five orders of
magnitude looser than reality is a test that will not catch the bug it was written for.
Fixing this tightened the RoPE, RMSNorm and attention oracles by that full margin.

**Cost.** One indirection on a hot path, and one module that exists for a single rule.

---

## D5. Negative and surprising results are committed, not tuned away

**Context.** Every phase of the spec predicted an outcome. Several did not happen.

**Decision.** When a measurement contradicted the plan, the measurement was recorded —
as a results table, a docs paragraph, and where possible a test that pins the surprising
behaviour so it cannot silently change.

**What this produced.**

| Expected | Measured | Where it lives |
|---|---|---|
| GPTQ beats RTN at 4 bits "by a clear margin" | A tie; the separation appears at 2–3 bits | `results/quantization/quantization.md` |
| QK-norm / zero-init / SwiGLU improve final loss | No measurable difference (<1.4%, under the 2% threshold) | `results/ablations/architecture.md` |
| Speculative decoding is faster | 3× fewer target passes but 0.79× wall-clock on this CPU | `results/speculative/speculative.md` |
| Muon beats AdamW | True on a deep linear net, **false** on convex least squares | `test_adamw_wins_on_a_convex_single_matrix_problem` |
| BPE token counts are subadditive under concatenation | False: `"eps"`(1) + `"ep"`(1) → `"epsep"`(3) | `test_concatenation_is_not_subadditive` |
| GPTQ's advantage comes from the Hessian's off-diagonal structure | Partly: error compensation helps even with white activations | two narrower tests replacing one false one |

**Why.** A project where every prediction came true is a project that either got lucky or
stopped looking. The distinction between "the code is wrong" and "my hypothesis was
wrong" is the one that matters, and it is only visible if both kinds of failure are kept.

**Cost.** The README is longer and less triumphant than it could be.

---

## D6. Every number is generated, and lives in exactly one place

**Context.** The failure mode for a project like this is documentation that was true when
it was written.

**Decision.** Measurements are written by scripts into `results/<area>/`, as JSON (the
data), a PNG (the figure) and a Markdown fragment (the prose). `docs/results.md` is
**generated** from those fragments by `scripts/build_docs_results.py`, and CI runs it with
`--check` so a stale page fails the build. Every script supports `--replay`, which
re-renders the figure and the page from committed JSON without retraining.

**Consequence.** There is no number in the docs that a human typed. Changing a result
means rerunning its script, which rewrites the JSON, the figure and the page together.

**Cost.** An extra generation step, and prose that has to be written inside the
measurement script rather than in the document it appears in.

---

## D7. Alignment reports the diagnostic that makes the metric interpretable

**Context.** DPO's reward margin rose steadily and its preference accuracy hit 100%. The
aligned model then **lost** its head-to-head against the SFT model it started from, 0-7-33.

**Decision.** Every preference run logs the *absolute* log-probabilities of the chosen and
rejected responses, not just their difference; every run reports mean generated length
before and after; and the head-to-head judge is length-insensitive by construction.

**What it caught.** DPO was reducing its loss by pushing **both** log-probabilities down —
the chosen one merely less far. `Δ log p(chosen) = -0.0454` with `Δ log p(rejected) =
-4.2385`. A rising margin was hiding a model getting worse at everything. Implementing the
declared-but-unimplemented `sft_loss_weight` (the RPO-style NLL anchor) turned 0-7-33 into
3-0-37, with `Δ log p(chosen) = +0.0104`.

**Why it generalises.** Any objective defined on a difference can satisfy itself by moving
both terms. The fix is not a better margin metric; it is reporting the terms.

**Cost.** One extra forward pass per step when the anchor is enabled.

---

## D8. The compression levers are measured on what transfers, and the rest is disclaimed

**Context.** Quantization and speculative decoding exist to relieve **memory bandwidth**.
At 5M parameters on a CPU there is no memory-bandwidth bottleneck to relieve; a forward
pass is dominated by Python dispatch.

**Decision.** Report the hardware-independent quantities as results — target forward
passes saved, weight footprint at a given effective bit-width, exactness of the
speculative output distribution, rank ordering of the distillation objectives — and
report wall-clock throughput with an explicit statement that it does not generalise.

**Concretely.** The weight column in the benchmark table is the *representation* size
computed from effective bits including stored scales, not `sum(p.numel() * p.element_size())`
— the 4-bit rows are simulated in fp32 because there is no int4 CPU kernel, so reading the
footprint off the tensors would report a 4-bit model as 32-bit. Speculative rows include
the draft's weights and KV cache, because speculation is not free in memory.

**Cost.** The headline "3× faster" chart this project could have shown does not exist,
because it would not have been true.

---

## D9. Losslessness is proven, not asserted

**Context.** The single most counter-intuitive claim in the whole project is that
speculative decoding does not change the output distribution.

**Decision.** It is checked three ways: greedy speculation must equal greedy
autoregressive decoding token-for-token (exact equality, in the e2e smoke test); the
accept/reject rule is unit-tested against a closed-form acceptance probability; and the
sampled first-token distribution is compared to the target's over hundreds of draws, with
the comparison run at raised temperature because at T=1 this model is peaked enough that
any sampler would appear to match.

**Cost.** The distributional test is slow and statistical, so it carries an explicit
sampling-noise floor rather than a fixed tolerance.

---

## D10. Tests pin behaviour, including behaviour I did not predict

**Context.** 555 tests, `mypy --strict`, and property-based tests via hypothesis.

**Decision.** Three kinds of test, deliberately mixed:

1. **Oracle tests** — a slow, obviously-correct fp64 reference (`rope_reference`, a naive
   attention loop, closed-form optimizer solutions) that the fast path must match.
2. **Property tests** — hypothesis generates the inputs. This is what found that BPE
   concatenation is not subadditive; the counterexample it produced is now a named
   regression test.
3. **Documented-surprise tests** — `test_adamw_parity_divergence_is_chaos_not_a_formula_difference`
   measures how the divergence from `torch.optim.AdamW` grows (1e-17 → 1e-15 → 1e-12 →
   1e-5 over 200 steps) and asserts the *growth pattern*, proving chaos amplification
   rather than a formula error. A plain tolerance test would have either failed or hidden
   the question.

**Cost.** The third category is unusual and needs its rationale written into the test, so
those tests are long. That is the point: the docstring is the finding.

