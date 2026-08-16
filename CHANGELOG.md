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

**Notes**

- The spec's headline parameter figures are approximate. The shapes from the spec's
  tier table are pinned exactly; the resulting parameter counts (total and
  non-embedding) are reported honestly and asserted in tests. See
  `DESIGN_DECISIONS.md`.
