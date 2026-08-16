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
| `nano` | 6 × 256 × 4, ctx 256, vocab 4096 | 6,524,928 | 4,427,776 | ~2–4M |
| `micro` | 8 × 512 × 8, ctx 512, vocab 16384 | 40,379,904 | 23,602,688 | ~25M |
| `small` | 12 × 768 × 12, ctx 1024, vocab 32768 | 125,849,856 | 75,518,208 | ~120M (GPT-2-ish) |

The spec's `micro` and `small` figures line up well with the non-embedding and total
counts respectively; its `nano` figure is simply lower than the stated shape can
produce (6 × 256 SwiGLU blocks alone are ~4.4M parameters). Rather than shrink the
architecture away from the spec, the repo keeps the shape and states the number.

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
of the compute-optimal budget actually covered.

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

*(Sections D4 onward are added as their phases land.)*
