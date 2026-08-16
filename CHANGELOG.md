# Changelog

All notable changes to NanoScale-LM are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Conventional Commits](https://www.conventionalcommits.org/).

Each entry corresponds to one phase of the build plan in `nanoscale_lm_spec.md`.

## [Unreleased]

### Phase 0 — Foundation

**Added**

- Repository scaffold under `src/nanoscale/` with the module map from the spec
  (Part G), a `uv`-managed environment and an Apache-2.0 license.
- Pydantic v2 configuration layer (`nanoscale.config`): 16 frozen, fully-documented
  config models covering every phase, with stable config hashing, computed-field-safe
  round-tripping (`dump_inputs`), deep YAML merging and dotted-path CLI overrides.
- The compute-honest size ladder (`nano` / `micro` / `small`) with exact
  parameter-count regression guards and Chinchilla-style 20:1 token budgets for the
  GPU tiers.
- Cross-cutting utilities: global seed control with derived per-stream seeds,
  device/dtype resolution with a guaranteed CPU fallback, run manifests recording
  git SHA + config hash + seed + versions + hardware, and a dependency-free
  JSONL/CSV metric logger with optional W&B mirroring.
- `nanoscale` Typer CLI: `info`, `config show|hash|save|schema|params`.
- Exported JSON Schemas for all config models under `configs/schema/`, checked for
  staleness in CI.
- Tooling: `ruff` (lint + format), `mypy --strict`, `pytest`, `pre-commit`, a
  `Makefile`, and a GitHub Actions CI matrix (Python 3.11/3.12, CPU only).

### Phase 1 — Tokenizer

**Added**

- `nanoscale.tokenizer.bpe`: byte-level BPE from scratch — train, encode, decode,
  save/load — with GPT-2 and GPT-4 pre-tokenization regexes. Training uses an
  incremental pair counter plus a `pair -> words` inverted index, so a merge touches
  only the words that contain it rather than rescanning the corpus.
- `nanoscale.tokenizer.chat`: chat templating with an SFT completion mask. Role
  markers are special tokens that ordinary `encode` can never emit, so user text
  cannot forge a turn boundary.
- `nanoscale.data.toy`: a deterministic, offline, TinyStories-style corpus so that
  CI, `make smoke` and the `nano` tier never need the network.
- `nanoscale tokenizer train|info|encode|decode` CLI, `make tokenizer`, and a
  committed 1024-token `nano` vocabulary plus its measured report in
  `results/tokenizer/`.
- Tests: exact round-trip over 15 scripts/edge cases plus hypothesis over arbitrary
  Unicode; vocabulary-layout and merge-consistency invariants; `tiktoken` parity
  split into three sharp claims (exact pre-tokenization equality, in-domain length
  parity, bounded out-of-domain degradation).

**Changed**

- `nano`'s vocabulary is now 1024 rather than 4096: that is exactly what the toy
  corpus can fill (761 merges), and dead embedding rows cost parameters and CPU time
  for nothing. `nano` is now 4,952,064 parameters.

### Phase 2 — Model

**Added**

- `nanoscale.model`: the full decoder-only transformer, from scratch.
  - `rope.py` — RoPE in the paper's interleaved-pair convention, with a slow literal
    reference implementation used only by tests, and configurable table precision.
  - `norm.py` — RMSNorm (default) and a bias-free LayerNorm ablation.
  - `mlp.py` — SwiGLU (default, 8/3 expansion) and the ReLU² speedrun variant.
  - `attention.py` — causal self-attention with GQA, RoPE, QK-norm, KV caching, and
    an optional SDPA fast path behind the same interface.
  - `kv_cache.py` — preallocated per-layer KV storage with `clone`/`truncate` for
    speculative rollback and byte-level memory accounting.
  - `block.py`, `lm.py`, `mtp.py` — pre-norm blocks, the LM with zero-init or
    depth-scaled initialisation, tanh logit soft-capping, optional multi-token
    prediction heads, and the reference generation loop.
  - `numerics.py` — "promote to at least fp32, never demote" for the reduction-heavy
    ops.
- 119 new tests: manual attention == SDPA in fp32 and fp64 across MHA/GQA/MQA with and
  without QK-norm and padding masks; cached decode == full recompute token-for-token;
  RoPE == a literal reference rotation and the relative-position property to 1e-12;
  hand-computed RMSNorm/SwiGLU values; parameter counts against the tier table;
  `ln(vocab)` initial loss; plus hypothesis properties over the whole config space.

### Phase 3 — Optimizer

**Added**

- `nanoscale.optim.adamw` — decoupled-weight-decay Adam from scratch, matching
  `torch.optim.AdamW` to 1e-12 including the exact `eps`-outside-the-sqrt placement.
- `nanoscale.optim.muon` — Muon: momentum orthogonalized by a five-step quintic
  Newton–Schulz iteration, with both the original shape-aware update scaling and the
  RMS-matching (Moonlight) alternative.
- `nanoscale.optim.cautious` — cautious weight decay: decay only where it agrees with
  the optimizer's own update direction, plus a `wd_scale` hook for a decaying λ.
- `nanoscale.optim.router` — the documented Muon/AdamW parameter split and a
  `CompositeOptimizer` that presents both as one interface to the trainer.
- 57 unit tests and 7 hypothesis properties covering all four Phase-3 acceptance
  criteria.

**Notes**

- Two honest negative results are recorded as tests rather than omitted: AdamW beats
  Muon on a convex single-matrix problem (Muon's win needs depth and stochasticity),
  and long-horizon AdamW/torch divergence at aggressive hyperparameters is chaos
  amplification, not a formula error — pinned by measuring the growth curve.

### Phase 4 — Pretraining

**Added**

- `nanoscale.train.data` — packing, contiguous train/val splitting, and a deterministic
  batcher whose position in the stream is a pure function of `(seed, batches consumed)`.
- `nanoscale.train.schedule` — cosine+warmup and warmup-stable-decay, both returning a
  multiplier so Muon and AdamW can share one schedule at different peak LRs; plus the
  decaying-λ schedule for cautious weight decay.
- `nanoscale.train.checkpoint` — resumable checkpoints carrying weights, optimizer
  state, counters and RNG state.
- `nanoscale.train.loop` — gradient accumulation, clipping, AMP with fp32 master
  weights, token-budget stopping, metric logging and manifest writing.
- `nanoscale train pretrain|generate` CLI; `make train-nano` / `make train-micro`.
- `nanoscale.utils.plotting` and `scripts/plot_loss_curve.py` /
  `scripts/sample_generations.py`, both stamping figures with the producing git SHA.
- First committed results: `results/curves/nano_loss.png`, `results/samples/nano_base.*`.

**Fixed**

- Checkpoint resume replayed the current epoch from its first batch instead of the
  correct offset, so a resumed run silently re-trained on data it had already seen and
  diverged from an uninterrupted one. Data position is now derived from the step count.

### Phase 5 — Ablations

**Added**

- `nanoscale.bench.ablation` — a controlled-A/B harness where every arm is one base
  config plus named overrides, sharing seed, data order, schedule and step budget, and
  where the reporting helper refuses to name a winner on a gap below single-seed noise.
- `scripts/ablate.py` with two suites (optimizer, architecture), a `--replay` mode that
  re-renders findings from the committed JSON without retraining, and committed results
  in `results/ablations/`.

**Findings** (nano tier, single seed — directional only)

- **Muon + AdamW reaches the target loss in 50 steps vs AdamW's 105 (2.1×)** and ends
  3.5% lower. This reproduces the spec's E3 claim qualitatively.
- QK-norm, zero-init output projections and SwiGLU-vs-ReLU² all show **no measurable
  difference in final loss** at this scale. Removing QK-norm does cost 1.5× more steps
  to reach the target, which the write-up reports rather than rounding to "no effect".

### Phase 6 — Alignment

**Added**

- `nanoscale.align.sft` — completion-masked SFT; prompt tokens are verified to receive
  exactly zero gradient.
- `nanoscale.align.losses` — DPO (with cDPO label smoothing) and SimPO, both matched
  against hand-computed values on tiny fixtures.
- `nanoscale.align.preference` — the trainer, with a properly frozen reference policy,
  the length-exploitation diagnostic, and an RPO-style auxiliary NLL term.
- `nanoscale.align.grpo` — the optional RLVR track: group-relative advantages, a
  PPO-clipped surrogate, a k3 KL penalty, and a programmatic arithmetic verifier.
- `nanoscale.eval.preference_eval` — a stated, length-insensitive programmatic judge
  and a paired head-to-head harness.
- `nanoscale.data.instruct` — instruction and preference data whose chosen/rejected
  lengths are matched by construction, so the length diagnostic is interpretable.
- `nanoscale align sft|preference|grpo` CLI and `scripts/align_pipeline.py`.

**Findings**

- All three preference arms reach 100% preference accuracy, but plain DPO's mean
  per-token log-probability of the *chosen* response falls (−0.045) while the margin
  rises: the objective satisfies itself by pushing both sides down. Adding the auxiliary
  NLL anchor flips that to +0.010 — and DPO+NLL is the only arm that **wins** the
  scripted head-to-head against the SFT model (3–0–37 vs DPO's 0–7–33).

**Fixed**

- The synthetic "repetitive" rejection could reproduce the chosen response exactly for
  single-sentence answers, feeding DPO a pair labelled both preferred and dispreferred.

**Notes**

- The spec's headline parameter figures are approximate. The shapes from the spec's
  tier table are pinned exactly; the resulting parameter counts (total and
  non-embedding) are reported honestly and asserted in tests. See
  `DESIGN_DECISIONS.md`.
