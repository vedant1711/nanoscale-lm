# NanoScale-LM — A Small Language Model, Built From Scratch and Made Deployable

## A modern decoder-only LM (pretrain → SFT → alignment) + a full efficient-inference stack (distillation, quantization, speculative decoding), in one coherent PyTorch project

**How to use this document:** Paste the whole thing into Claude Code as the project brief, then drive it phase-by-phase: *"Execute Phase N of the NanoScale-LM spec. Do not advance to Phase N+1 until every acceptance criterion for Phase N passes."* Each phase is independently verifiable. The two arcs compose deliberately: **Arc 1 (Phases 0–6)** builds a small language model from first principles; **Arc 2 (Phases 7–10)** compresses that exact model for cheap, fast inference. The story is one sentence: *I built a language model from scratch and then made it deployable on hardware anyone can afford.*

---

# PART A — PROJECT BRIEF (read first; applies to every phase)

## A1. Role and thesis

You are a senior ML engineer building **NanoScale-LM**: a from-scratch, fully-typed PyTorch implementation of a modern small language model and the complete efficiency stack needed to serve it. This is a portfolio artifact that must read as *deep algorithmic understanding*, not framework plumbing. Every core algorithm — attention, RoPE, the optimizer step, the distillation loss, the quantization rounding, the speculative accept/reject rule — is implemented and explained by you, not imported from a high-level trainer.

The enterprise framing (for the README and the future paper): frontier pretraining needs thousands of GPUs, but the *techniques* are now reproducible at small scale, and the deployment techniques are exactly what enterprises use to cut inference cost. NanoScale-LM demonstrates the whole lifecycle on free hardware and is written so that swapping in more compute scales it up without redesign.

## A2. Why this is a defensible 2026 project (evidence base)

Each row is a design decision the code must reflect; the README tells this story.

| # | State-of-the-art finding | Design consequence in NanoScale-LM |
|---|---|---|
| E1 | Pretraining a GPT-2-quality model end-to-end is now ~$50 / ~1–3 hrs on rented 8×H100 via Karpathy's **nanochat**; a tiny (1–10M param) transformer trains in 20–60 min even on CPU. Chinchilla-optimal data-to-parameter ratio ≈ 20:1 tokens per parameter. | The project targets a **configurable size ladder** (see A4): a ~2–10M "laptop" tier that fully trains on free CPU/Colab, and a ~120–560M "Colab/Kaggle GPU" tier trained to a fixed validation-loss target on FineWeb-Edu. Token budgets follow the 20:1 rule so runs are compute-honest, not vibes. |
| E2 | The 2026 "architecture that won" for small decoder-only LMs is **RoPE + RMSNorm + SwiGLU + GQA**; the modded-nanoGPT speedrun stack further adds **QK-norm, ReLU² MLPs, zero-init projections (muP-like), untied head, value/embedding residual shortcuts, tanh logit soft-capping (Gemma-2 style), and multi-token prediction**. | The reference architecture implements RoPE, RMSNorm, SwiGLU, GQA, a KV cache, and QK-norm as the default; the speedrun tricks (ReLU² variant, zero-init, untied head, logit soft-cap, MTP head) are toggleable ablation flags so the project *demonstrates* their effect rather than just asserting it. |
| E3 | The **Muon optimizer** (Momentum Orthogonalized by Newton–Schulz) was the single largest speedrun lever (AdamW→Muon), and February 2026 work shows **cautious weight decay** improves it further and also helps in nanochat. | Implement Muon from scratch (Newton–Schulz orthogonalization of 2D weight-matrix updates) alongside a correct AdamW; default recipe = Muon for hidden matrices + AdamW for embeddings/head/scalars (the standard split), with a documented A/B showing wall-clock-to-target for each. This is a strong depth signal few portfolios have. |
| E4 | Post-training is a modular stack: **SFT → preference optimization (DPO/SimPO/KTO/ORPO) → optional RL with verifiable rewards (GRPO/GSPO/DHPO)**; RLHF-with-a-reward-model is largely superseded, and DPO has known **length-exploitation / reference-dependence** failure modes that motivate its successors. | Arc 1 ends with SFT + a **from-scratch DPO** implementation (the stable, widely-used baseline) and a **SimPO** variant (reference-free) for contrast, with length-exploitation explicitly measured. A clearly-scoped optional **GRPO-on-a-verifiable-task** track (GSM8K-style arithmetic with programmatic reward) is included and labeled current-SOTA-adjacent, with a docs note that GSPO/DHPO are the 2026 sequence-level successors to cite in the paper. |
| E5 | INT4 weight-only quantization became production-viable via **GPTQ** (second-order/Hessian error compensation, column-wise) and **AWQ** (activation-aware scaling that protects salient weight channels); **SmoothQuant** shifts activation outliers into weights to enable W8A8; **KV-cache quantization** attacks the long-context memory bottleneck. | Arc 2 implements a naive round-to-nearest PTQ baseline, then **GPTQ from scratch** (the Hessian-based algorithm) and an **AWQ-style activation-aware scaling** pass, reports the accuracy-vs-bits frontier for each, and adds **KV-cache quantization** as a memory-bandwidth win. bitsandbytes is allowed only as an external comparison point, never as the thing you're claiming to have built. |
| E6 | **Speculative decoding** is the highest-ROI single-stream inference optimization in 2026: 2–4× lossless speedup with output identical in distribution to the target model. The lineage is draft-model speculative sampling → **Medusa** (extra decoding heads, no separate draft) → **EAGLE-2/EAGLE-3** (feature-level drafting, current SOTA); it **composes with quantization**. | Implement classic **draft-target speculative sampling from scratch** with the correct modified-rejection acceptance rule (provably preserves the target distribution), then a **Medusa-style multi-head** draft as an ablation. Measure acceptance rate and wall-clock speedup, and demonstrate that quantization + speculation **stack**. EAGLE-3 is the documented "next step" for the paper. |
| E7 | On-policy / white-box distillation (**MiniLLM: reverse-KL, on-policy**; **GKD**: interpolates on/off-policy over divergences) outperforms naive forward-KL token matching because forward KL makes the student **over-cover the teacher's low-probability regions**; off-policy training suffers **exposure bias**. | The distillation module implements forward-KL token KD and **SeqKD** as baselines, then **reverse-KL on-policy distillation (MiniLLM-style)** as the headline method, with a controlled comparison of generation quality — turning a compression step into a genuine algorithmic study. |

## A3. Hard constraints (non-negotiable)

1. **Zero paid resources.** All training/eval must be feasible on: free Google Colab / Kaggle GPUs (T4/P100-class), and a CPU-only "laptop tier" that runs end-to-end with no GPU at all. No paid APIs, no paid compute, no paid datasets. Teacher models for distillation are open-weight (e.g., a small open model or your own larger NanoScale checkpoint).
2. **From scratch means from scratch.** Attention, RoPE, RMSNorm, SwiGLU, GQA, KV cache, the BPE tokenizer, Muon, the DPO/SimPO losses, GPTQ, speculative sampling, and the distillation losses are implemented in this repo. `torch.nn.functional.scaled_dot_product_attention` may be used as an optional fast path **behind** your own attention module (and your module must match it numerically in a test). Hugging Face `transformers`/`trl` may appear **only** in clearly-labeled comparison notebooks, never in the core library.
3. **Python 3.11+, PyTorch 2.x, fully typed** (`mypy --strict` passes), `ruff` for lint/format, `pytest` + `hypothesis` for tests, `uv` for env management. Config via `pydantic` v2 + YAML.
4. **Reproducibility is a graded feature.** Global seed control, deterministic dataloading, and a run manifest (git SHA, config hash, seed, library versions, hardware string, token budget) written for every run. A `make smoke` target trains the laptop-tier model end-to-end in < 10 minutes on CPU and must produce a stable loss trajectory across two runs on the same seed.
5. **Every claim is measured.** No "this makes it faster" without a benchmark in the repo producing the number. Loss curves, tokens/sec, perplexity, MMLU-style tiny-eval accuracy, acceptance rates, and bits-vs-accuracy frontiers are all generated by committed scripts.
6. **License:** Apache-2.0. Public-portfolio quality: README with architecture diagram, training curves, a results table, and a runnable Colab badge.

## A4. The size ladder (compute-honest configuration tiers)

All tiers share one architecture; only dims/depth/token-budget change (config-selected). The 20:1 token:param heuristic sets budgets.

| Tier | Params | Layers × d_model × heads | Context | Token budget (~20:1) | Trains on | Purpose |
|---|---|---|---|---|---|---|
| `nano` | ~2–4M | 6 × 256 × 4 | 256 | ~60–80M | CPU / laptop, < 10 min smoke | CI, unit tests, teaching, the from-scratch story end-to-end with no GPU |
| `micro` | ~25M | 8 × 512 × 8 | 512 | ~500M | Free Colab/Kaggle T4, a few hrs | The main reported model; real curves and evals |
| `small` | ~120M (GPT-2-ish) | 12 × 768 × 12 | 1024 | ~2–3B (loss-target track) | Kaggle P100 / multi-session, or documented as "scale-up" | Optional stretch; the "GPT-2 quality" headline if quota allows |

The `micro` tier is the default for everything reported. `small` is the aspirational/scale-up tier — the README documents its recipe even if full training is left as "run this with more compute."

## A5. System architecture (module map)

```
┌─────────────────────────────────────────────────────────────────────┐
│ ARC 2 — SERVE CHEAPLY (Phases 7–10)                                  │
│  distill/   → forward-KL, SeqKD, reverse-KL on-policy (MiniLLM)      │
│  quantize/  → RTN baseline, GPTQ (Hessian), AWQ-scaling, KV-cache q  │
│  specdec/   → draft-target spec sampling, Medusa heads, accept rule  │
│  serve/     → generation loop, KV cache mgmt, bench harness          │
├─────────────────────────────────────────────────────────────────────┤
│ ARC 1 — BUILD THE MODEL (Phases 0–6)                                 │
│  tokenizer/ → BPE train + encode/decode (byte-level)                 │
│  model/     → attention(GQA,RoPE,QK-norm,KV-cache), RMSNorm, SwiGLU, │
│               blocks, LM head, tanh logit cap, optional MTP head     │
│  optim/     → Muon (Newton–Schulz) + AdamW + param-group router      │
│  train/     → data pipeline, LR schedules, grad-accum, AMP, ckpt     │
│  align/     → SFT trainer, DPO, SimPO, (optional) GRPO-RLVR          │
│  eval/      → loss/perplexity, tiny benchmark suite, gen quality     │
├─────────────────────────────────────────────────────────────────────┤
│ CROSS-CUTTING                                                        │
│  config (pydantic+YAML) · manifest/repro · bench (tokens/s, memory) │
│  cli (typer) · datasets (streaming FineWeb-Edu / TinyStories)       │
└─────────────────────────────────────────────────────────────────────┘
```

Data flow, Arc 1: `tokenizer.train` → `train pretrain` (→ base checkpoint) → `train sft` (→ instruct checkpoint) → `align dpo|simpo` (→ aligned checkpoint) → `eval`. Arc 2 takes the aligned checkpoint as the **teacher/target** and produces `distilled`, `quantized`, and `speculative` variants, each benchmarked against the base on the same eval and bench harness.

## A6. Technology stack (all free)

| Concern | Choice | Notes |
|---|---|---|
| Core | Python 3.11, PyTorch 2.x, uv, ruff, mypy --strict | CUDA optional; everything has a CPU path |
| Config/schemas | pydantic v2 + YAML; JSON Schema exported | one config object per phase, composable |
| Data | `datasets` (streaming) for FineWeb-Edu / TinyStories / OpenWebText subset; UltraFeedback-style pairs (or self-generated) for DPO | stream to avoid disk blowup on free tiers |
| Tokenizer | your byte-level BPE (train + inference); `tiktoken` allowed only as a cross-check in a test | |
| Eval | your perplexity + a small committed benchmark (HellaSwag-tiny / ARC-easy subset / GSM8K-arith) | tiny, offline, deterministic |
| Tracking | CSV/JSONL + matplotlib plots committed as artifacts; optional Weights & Biases free tier behind a flag | never required |
| Notebooks | Colab/Kaggle quickstarts with badges | the no-clone entry point |
| Docs | README + ARCHITECTURE.md + docs/ (mkdocs-material → GitHub Pages) | methodology pages seed the paper |

---

# PART B — ALGORITHM SPECIFICATIONS (implement exactly; cite sources in docstrings)

This part pins down the math so Claude Code implements the *right* version of each algorithm. Where a formula is given, the code's docstring must reproduce it and cite the reference from Part J.

## B1. Tokenizer — byte-level BPE

Train byte-level BPE (GPT-2 style) on a corpus sample: initialize vocab with 256 byte tokens, iteratively merge the most frequent adjacent pair until target vocab size (config: 8k for `nano`, 16k–32k for larger). Implement `train`, `encode`, `decode`, special-token handling (`<bos>`, `<eos>`, `<pad>`, chat role tokens for SFT). **Correctness test:** `decode(encode(x)) == x` for arbitrary UTF-8 (hypothesis), and encoded-length parity within a tolerance of `tiktoken` on a fixed passage.

## B2. Model — modern decoder-only transformer

Implement as composable `nn.Module`s. Defaults in **bold**; ablation flags in brackets.

- **Attention:** multi-head with **grouped-query attention** (config `n_kv_heads ≤ n_heads`); **rotary position embeddings** applied to q,k (implement the rotation directly; verify against a reference rotation); **QK-norm** (RMS-normalize q,k before the dot product — training stability, per speedrun stack); causal mask; a **KV cache** for incremental decoding (correctness-tested against a full recompute). Provide an SDPA fast-path that your module must match numerically (atol test).
- **Norm:** **RMSNorm** (implement; no bias, learned gain). [LayerNorm available as ablation.]
- **MLP:** **SwiGLU** (gated: `(SiLU(xW_gate) ⊙ xW_up)W_down`). [ReLU² variant flag, per modded-nanoGPT.]
- **Embeddings:** token embedding; **untied** LM head by default [tied flag]; **zero-init** output projection and classification head (muP-like), per speedrun.
- **Logits:** optional **tanh soft-capping** (`c·tanh(logits/c)`, Gemma-2 style) [flag].
- **Multi-token prediction head:** optional auxiliary head predicting t+2 for a training-speed/-quality ablation and later reuse as a self-speculation draft (ties Arc 1 to Arc 2).
- Weight init documented and seeded; parameter count printed and asserted against the tier table.

## B3. Optimizer — Muon + AdamW

- **AdamW:** implement the standard update (decoupled weight decay) yourself; unit-test against `torch.optim.AdamW` to near-bit-equality on a toy problem.
- **Muon:** for 2D hidden weight matrices, compute the momentum buffer, then **orthogonalize the update via Newton–Schulz iteration** (fixed iteration count, the standard quintic coefficients) before applying it; scale by a shape-aware factor. Route embeddings, the LM head, norms, and scalars to AdamW; route hidden matmul weights to Muon (the documented split). Implement **cautious weight decay** (decaying schedule) as a flag (Feb-2026 improvement). Docstring cites Newton–Schulz orthogonalization and the modded-nanoGPT lineage.
- **Deliverable:** an A/B script reporting wall-clock and steps-to-target-loss for AdamW-only vs Muon+AdamW on `micro`, reproducing the "optimizer was the biggest lever" claim qualitatively.

## B4. Training — schedules, throughput, stability

Streaming tokenized data pipeline with packing to fixed context; gradient accumulation for large effective batch on small GPUs; **AMP/bf16** with a correct fp32 master-weight path; gradient clipping; **cosine decay with linear warmup** as default, **warmup-stable-decay (WSD)** as a flag (per 2026 nanoGPT recipes); checkpoint save/resume with optimizer state; token-budget-driven stopping per the 20:1 rule. Log loss, grad-norm, tokens/sec, and LR. Determinism test: two seeded `nano` runs → identical loss to fp tolerance.

## B5. Alignment — SFT, DPO, SimPO, (optional) GRPO-RLVR

- **SFT:** chat-formatted instruction tuning with loss masked to completion tokens only; verify masking with a test.
- **DPO (headline baseline):** implement the loss directly —
  `L_DPO = −E[ log σ( β( log π_θ(y_w|x) − log π_ref(y_w|x) ) − β( log π_θ(y_l|x) − log π_ref(y_l|x) ) ) ]`
  with a frozen reference policy π_ref. Log the implicit reward margin and **sequence-length of chosen vs rejected** to expose length exploitation (E4).
- **SimPO (reference-free contrast):** length-normalized reward with a target margin, no π_ref — demonstrate the reduced memory footprint and compare length-exploitation behavior against DPO.
- **Optional GRPO-RLVR track:** on a programmatically-verifiable arithmetic/GSM8K-style task, sample a group of completions per prompt, compute a verifiable 0/1 reward, and apply the **group-relative advantage** (normalize rewards within the group, no learned critic) policy-gradient update. Label it explicitly as the reasoning-RL arc; docs note **GSPO (sequence-level importance ratio)** and **DHPO (hybrid token+sequence)** as the 2026 successors to cite, and state that this track is the paper's forward-looking extension.

## B6. Distillation (Arc 2)

Teacher = the aligned `micro`/`small` checkpoint (or a small open-weight model); student = a smaller NanoScale config.
- **Baseline 1 — token-level forward-KL KD:** `L = αH(y,student) + (1−α)τ²·KL(teacher_τ ‖ student_τ)` (Hinton temperature τ).
- **Baseline 2 — SeqKD:** train student by MLE on teacher-generated sequences.
- **Headline — reverse-KL on-policy distillation (MiniLLM-style):** minimize `KL(student ‖ teacher)` on **student-sampled** trajectories via a policy-gradient formulation, motivated by avoiding over-coverage of the teacher's low-probability regions (mode-seeking). Provide a controlled comparison (perplexity + generation quality + a length/repetition diagnostic) showing where reverse-KL on-policy beats forward-KL, reproducing the MiniLLM finding at small scale. Docstrings cite MiniLLM and GKD.

## B7. Quantization (Arc 2)

- **RTN baseline:** per-channel symmetric/asymmetric round-to-nearest at 8/4 bits; report degradation.
- **GPTQ (from scratch):** layer-wise, column-by-column quantization using the **Hessian of the layer input** (`H = 2 XXᵀ`, from a few hundred calibration samples) to **compensate remaining columns for accumulated rounding error** (the OBQ/GPTQ update). Support 4-bit and 3-bit; report perplexity vs bits.
- **AWQ-style:** compute per-channel activation statistics on calibration data, derive a **per-channel scaling** that protects salient weight channels before quantization; compare against GPTQ at equal bits.
- **KV-cache quantization:** quantize stored K/V to 8/4-bit to cut the decode-time memory bandwidth/footprint; measure the long-context memory win and any quality cost.
- **Comparison:** a single figure — perplexity (and a tiny-benchmark accuracy) vs effective bits — for RTN, GPTQ, AWQ, with bitsandbytes INT8 as an external reference point.

## B8. Speculative decoding (Arc 2)

- **Draft–target speculative sampling (from scratch):** a small draft model proposes γ tokens; the target verifies them in one forward pass; accept via the **modified rejection rule** `accept with prob min(1, p_target(x)/p_draft(x))`, and on rejection **resample from the normalized residual** `(p_target − p_draft)_+`. Prove-by-test that the accepted-token distribution matches plain target sampling (distributional equivalence within statistical tolerance over many samples) — this is the correctness crown jewel.
- **Medusa-style heads:** add extra decoding heads to the target to predict multiple future tokens, verified with **tree attention** — draft-model-free speculation as an ablation; reuse the Phase-6 MTP head where possible.
- **Metrics:** mean **acceptance length**, wall-clock **tokens/sec** speedup vs autoregressive, and a demonstration that **speculation + quantization compose** (run the target quantized). Docs cite EAGLE-2/3 as the current SOTA drafting approach and the documented next step.


---

# PART C — PHASED EXECUTION PLAN

Execute strictly in order. Each phase ends with: all acceptance criteria demonstrably true; `ruff` + `mypy --strict` + `pytest` green; a conventional-commits git commit; a CHANGELOG entry; and any new numbers written into `results/` as committed artifacts. Do not scaffold future phases early.

## Phase 0 — Foundation
Repo scaffold (`src/nanoscale/…` per Part G), uv project, ruff/mypy/pytest/pre-commit, Apache-2.0, CI (lint+type+unit on 3.11 CPU). Pydantic config objects (ModelConfig, TrainConfig, TokenizerConfig, AlignConfig, QuantConfig, SpecConfig, DistillConfig) + the size-ladder presets (A4). `typer` CLI skeleton. Seed/manifest utilities.
**Accept:** `nanoscale --help` works; config round-trip + preset tests pass; CI badge live.

## Phase 1 — Tokenizer
Byte-level BPE per B1: train/encode/decode, specials, chat template.
**Accept:** `decode(encode(x))==x` on UTF-8 hypothesis cases; length parity vs `tiktoken` on a fixed passage within tolerance; a trained `nano` vocab committed.

## Phase 2 — Model
Implement B2 in full (GQA+RoPE+QK-norm attention, KV cache, RMSNorm, SwiGLU, untied/zero-init head, tanh cap flag, optional MTP head).
**Accept:** forward-shape tests; **your attention matches SDPA** within atol; **KV-cache decoding matches full recompute** token-for-token; RoPE matches a reference rotation; parameter counts match the tier table; a random-init `nano` model generates (garbage but shaped) text.

## Phase 3 — Optimizer
Implement B3: AdamW (tested vs torch), Muon with Newton–Schulz, param-group router, cautious-weight-decay flag.
**Accept:** AdamW matches `torch.optim.AdamW` on a toy convex problem; Newton–Schulz output is (near-)orthogonal on random matrices (σ≈1 test); Muon reduces a toy loss faster than AdamW on a matrix-structured problem; param router assigns groups correctly.

## Phase 4 — Pretraining
Implement B4: streaming data pipeline, packing, AMP/bf16 master weights, grad-accum, clipping, cosine+warmup (and WSD flag), ckpt save/resume, logging.
**Accept:** `nano` tier trains end-to-end on **CPU in < 10 min** (the `make smoke` target) with monotone-ish decreasing loss; two seeded runs match to fp tolerance; resume-from-checkpoint continues identically; `micro` tier launches on Colab T4 and logs tokens/sec. Commit the `micro` loss curve.

## Phase 5 — Optimizer & architecture ablations (the "depth" showcase)
Scripts that reproduce, at `micro` scale, the SOTA levers as controlled A/Bs: **AdamW vs Muon+AdamW** (steps/wall-clock to target loss), and at least two architecture toggles (e.g., **QK-norm on/off**, **zero-init head on/off**, or **SwiGLU vs ReLU²**). Each writes a figure + a short markdown finding.
**Accept:** `results/ablations/` contains committed plots and a written interpretation for each; conclusions are hedged honestly (small-scale caveat stated).

## Phase 6 — Alignment
Implement B5: SFT (completion-masked), DPO (from scratch, with length/reward-margin logging), SimPO. Optional GRPO-RLVR arithmetic track behind a flag.
**Accept:** SFT loss-masking verified by test; DPO drives implicit reward margin up and the aligned model wins a scripted head-to-head vs the SFT model on a small preference eval; **length-exploitation diagnostic** committed comparing DPO vs SimPO; if enabled, GRPO improves verified-arithmetic accuracy over the SFT baseline. This checkpoint is the **teacher/target** for Arc 2.

## Phase 7 — Distillation
Implement B6: forward-KL KD, SeqKD, and reverse-KL on-policy (MiniLLM-style). Student = smaller config; teacher = Phase-6 checkpoint.
**Accept:** all three train a student; a committed comparison (perplexity + generation-quality + repetition/length diagnostic) shows the reverse-KL on-policy student is competitive-or-better and the write-up explains *why* (mode-seeking vs mode-covering), reproducing MiniLLM's qualitative finding at small scale.

## Phase 8 — Quantization
Implement B7: RTN, GPTQ (Hessian), AWQ-style scaling, KV-cache quant.
**Accept:** GPTQ 4-bit beats RTN 4-bit on perplexity by a clear margin; the **bits-vs-accuracy frontier figure** (RTN/GPTQ/AWQ + bitsandbytes reference) is committed; KV-cache quant reduces measured decode memory with quantified quality cost; quantized model still generates coherent text at 4-bit.

## Phase 9 — Speculative decoding
Implement B8: draft–target speculative sampling with the exact accept/resample rule, plus a Medusa-style head variant.
**Accept:** the **distributional-equivalence test passes** (accepted-token empirical distribution ≈ target sampling within tolerance over N samples); mean acceptance length and **tokens/sec speedup** reported vs autoregressive; a committed demo shows **speculation over a GPTQ-quantized target** (the two levers compounding).

## Phase 10 — Serving, benchmarks, demo, docs
A minimal generation server / CLI (`nanoscale serve` and `nanoscale generate`) with KV-cache management; the unified **bench harness** producing one results table (tokens/sec, latency p50/p95, peak memory, perplexity, tiny-eval accuracy) across {base, distilled, GPTQ-4bit, speculative, speculative+GPTQ}; Colab/Kaggle quickstart notebooks; mkdocs site (quickstart, architecture, methodology, results, limitations) to GitHub Pages; README with diagram, curves, results table, and badges.
**Accept:** the results table regenerates from one command in replay/offline mode; a newcomer runs the Colab quickstart to a generated sample with zero setup; Pages site deploys; every headline number in the README is traceable to a committed script.

---

# PART D — TESTING PLAN (the implementation must be trustworthy, not just runnable)

## D1. Numerical-correctness tests (the credibility core)
- **Attention == SDPA** within atol (masked + GQA + RoPE paths).
- **KV-cache decoding == full recompute**, token-for-token, over a random sequence.
- **RoPE == reference rotation**; **RMSNorm/SwiGLU** match hand-computed values on fixtures.
- **AdamW == torch.optim.AdamW** on a toy problem; **Newton–Schulz output orthogonal** (singular values ≈ 1).
- **Tokenizer round-trip** exact; parity vs tiktoken within tolerance.
- **DPO/SimPO losses** match hand-computed values on tiny fixtures; **SFT loss masking** ignores prompt tokens (verified by gradient check on masked positions).
- **GPTQ** on a synthetic linear layer with known-optimal quantization recovers the expected error ordering vs RTN; Hessian assembled correctly on fixtures.
- **Speculative sampling distributional equivalence:** over N≥20k sampled tokens, the accepted-token distribution matches direct target sampling (χ² / total-variation within tolerance) — mathematically the most important test in the repo.

## D2. Property-based tests (hypothesis)
Shapes/dtypes invariant across configs; attention output invariant to padding beyond the causal frontier; RoPE relative-position property (attention scores depend only on relative offset for norm-matched inputs); quantize→dequantize error bounded by the quantization step; speculative acceptance probability ∈ [0,1]; distillation losses non-negative where they must be; pass counts monotone where defined.

## D3. Training-dynamics / regression tests
- **Overfit-a-batch:** every trainer (pretrain, SFT, DPO, distill) must drive loss to ~0 on a single repeated batch — the canonical "is my training loop correct" test.
- **Determinism:** seeded `nano` runs reproduce loss to fp tolerance; CI stores a golden loss trajectory and fails on drift beyond tolerance.
- **Smoke end-to-end:** `make smoke` runs tokenizer→pretrain→SFT→DPO→quantize→spec-decode on `nano` in < 10 min CPU and asserts each stage's sanity metric (loss decreased, aligned beats base on the toy pref eval, 4-bit still coherent, acceptance length > 1).

## D4. Efficiency-claim tests (Arc 2 integrity)
Every speed/memory/quality number in the README is produced by a committed benchmark with fixed seeds and logged hardware; a CI job re-runs the `nano`-scale versions and asserts the *direction* of each claim (GPTQ-4bit perplexity < RTN-4bit; speculative tokens/sec > autoregressive; distilled student smaller than teacher with bounded quality drop). Absolute numbers are tier/hardware-dependent and documented as such.

## D5. Eval-harness tests
Perplexity computed correctly (matches a hand-computed value on a fixture corpus); tiny-benchmark scorers deterministic and matching hand-labeled expected outputs; generation is reproducible under a fixed seed + temperature.

---

# PART E — DELIVERABLES & PORTFOLIO FRAMING

1. **README.md** — the one-sentence thesis; architecture diagram; the results table (base vs distilled vs quantized vs speculative on tokens/sec, memory, perplexity, tiny-eval); `micro` loss curve; the Muon-vs-AdamW and quantization-frontier figures; Colab badge; honest "trained on free hardware; here's the scale-up recipe" note.
2. **docs/methodology.md** — every algorithm with its formula and citation (Part J), the DPO length-exploitation finding, the reverse-KL-vs-forward-KL distillation result, the speculative-sampling correctness argument. Seed of the paper.
3. **docs/results.md** — all figures/tables with reproduction commands.
4. **docs/limitations.md** — small-scale caveats (conclusions are directional, not frontier-scale claims), free-tier throughput ceilings, eval-suite size, what each technique does and doesn't buy.
5. **DESIGN_DECISIONS.md** — the E1–E7 table expanded ADR-style, one section per decision.
6. **docs/enterprise-scale.md** — the "minimal now, enterprise later" appendix: same code path with more compute (larger tier, multi-GPU FSDP sketch, larger token budget, a real serving stack via vLLM/TensorRT-LLM as the production target, batched serving economics), and where each Arc-2 technique lands on the cost curve at scale. Demonstrates the thesis without building any of it.

---

# PART F — DEPLOYABLE PUBLIC DEMO (no-clone experience)

Anyone with a link should be able to *use* NanoScale-LM in-browser with no install and no paid API. Two surfaces (one interactive, one always-instant) plus a notebook path.

## F1. Surface 1 — Interactive demo on Hugging Face Spaces (Gradio, free CPU tier)
Ship a committed `demo/app.py` (Gradio; Spaces free CPU-basic tier, 2 vCPU / 16 GB RAM) loading the small **quantized** `micro` checkpoint so it runs on CPU. Tabs:
1. **Chat / generate** — type a prompt, get a completion from the aligned model; a toggle switches between the base and the aligned checkpoint so visitors *feel* what alignment did.
2. **Speed lab** — run the same prompt with autoregressive vs speculative decoding and show live tokens/sec and acceptance length — the compression story, experienced.
3. **Compression explorer** — a static-but-interactive view of the bits-vs-accuracy frontier and the model-size/latency table across variants, pulled from committed `results/`.
4. **About** — architecture diagram, the E1–E7 thesis, links to repo/paper/methodology.
Requirements: pin port 7860; keep model weights small enough for the free tier (quantized `micro`); enable "Duplicate this Space"; document the ~48h idle-sleep + cold-start behavior in the Space README. Load weights from the HF Hub model repo you publish (see F4), not committed to the Space.

## F2. Surface 2 — Static results gallery on GitHub Pages (never sleeps)
The mkdocs site (Part E) is the always-instant link for a resume: loss curves, the Muon ablation, the quantization frontier, the speculative-decoding speedup table, and the methodology pages — all pre-rendered, zero cold start. Landing page embeds a generation GIF and links to the live Space with an honest "may take ~60s to wake" note.

## F3. Surface 3 — Colab / Kaggle quickstart notebooks
"Open in Colab" badges on notebooks that (a) load your published checkpoint and generate, (b) run the `nano`-tier full pipeline end-to-end in a free session so a reviewer can *watch* it train from scratch, and (c) reproduce the quantization + speculative-decoding benchmarks. This is the "I want to run the code without cloning" path.

## F4. Published artifacts on the Hugging Face Hub (free)
Publish the `micro` base, aligned, distilled, and GPTQ-4bit checkpoints as HF model repos with model cards (architecture, training data, token budget, eval numbers, intended use, limitations). This is standard practice that makes the work citable and lets the Space and notebooks pull weights without bloating the git repo.

## F5. Honesty & anti-drift rules
Every number shown in any surface is generated by a committed script and stamped with the git SHA + hardware string; the demo never claims frontier quality — it claims a correctly-built, honestly-measured small model. A CI check asserts the results table in the docs matches the committed `results/` artifacts (the demo may not hand-edit numbers). README badges: **▶ Live Demo · 📊 Results & Methodology · ▶ Open in Colab · 🤗 Model Weights**.

---

# PART G — REPOSITORY LAYOUT

```
nanoscale-lm/
├── src/nanoscale/
│   ├── cli.py                    # typer app
│   ├── config/                   # pydantic configs + size-ladder presets
│   ├── tokenizer/                # byte-level BPE
│   ├── model/                    # attention(GQA,RoPE,QK-norm,KVcache), rmsnorm, swiglu, block, lm, mtp
│   ├── optim/                    # muon (newton-schulz), adamw, param_router
│   ├── train/                    # data, schedules, loop, checkpoint, amp
│   ├── align/                    # sft, dpo, simpo, grpo (optional)
│   ├── distill/                  # forward_kl, seqkd, minillm_onpolicy
│   ├── quantize/                 # rtn, gptq, awq, kvcache_quant
│   ├── specdec/                  # spec_sampling, medusa, accept_rule
│   ├── serve/                    # generate, kv_cache_mgmt, server
│   ├── eval/                     # perplexity, tiny_bench, gen_quality
│   └── bench/                    # throughput, memory, latency harness
├── configs/                      # nano.yaml, micro.yaml, small.yaml, align/*, quant/*, ...
├── scripts/                      # ablation runners, figure generators
├── demo/                         # app.py (Gradio), README (Space card)
├── notebooks/                    # colab_train_from_scratch.ipynb, colab_compress_and_serve.ipynb
├── tests/{unit,property,dynamics,e2e}/
├── results/                      # committed figures, tables, loss curves (source of truth)
├── docs/                         # mkdocs: quickstart, architecture, methodology, results, limitations, enterprise-scale
├── .github/workflows/            # ci.yml (lint+type+unit+smoke), pages.yml
├── Makefile                      # smoke, test, train-micro, ablate, bench, docs
└── pyproject.toml · README.md · DESIGN_DECISIONS.md · CHANGELOG.md
```

---

# PART H — DEFINITION OF DONE

A stranger with no GPU and no API keys can, in five minutes and with zero installs, either (a) open the **live Space** and chat with a language model you built from scratch, flip the base↔aligned toggle to feel what DPO did, and hit the speed lab to watch speculative decoding beat autoregressive decoding on the very same model — or (b) open the **Colab notebook** and watch the `nano` model train from raw bytes to coherent-ish text end-to-end; and in either path the **methodology page** justifies every algorithm and every number with a citation and a committed script. The repo proves, component by component, that you understand transformers, optimization, alignment, distillation, quantization, and speculative decoding at the level of the math — not the API. Build exactly that.

---

# PART I — SEQUENCING NOTES FOR CLAUDE CODE

- Build Arc 1 fully before Arc 2 — Arc 2's teacher/target is the Phase-6 checkpoint; do not start distillation/quantization against a random-init model.
- Always land the numerical-correctness test (D1) for a component **in the same phase** you build it; these tests are the project's credibility and must never be deferred.
- Keep everything runnable at `nano` tier on CPU at every phase so CI and the smoke test stay green without a GPU; treat GPU (`micro`) runs as reported experiments, not gates.
- Prefer honest, hedged write-ups: at this scale, results are *directional confirmations* of the cited SOTA findings, not frontier claims. Say so. That honesty is itself a senior signal.
- Before implementing any alignment or compression method, re-verify the current best-practice variant and cite what's current — the alignment family in particular is churning (2026 sequence-level successors GSPO and DHPO; on-policy distillation successors like GKD and DistiLLM; EAGLE-3 for speculation). Implement the stable baseline as the headline, name the current SOTA as the documented next step for the paper.

---

# PART J — REFERENCES (cite in docstrings, methodology page, and the future paper)

1. Vaswani et al., *Attention Is All You Need* (arXiv:1706.03762) — transformer core.
2. Su et al., *RoFormer: Rotary Position Embedding* (arXiv:2104.09864) — RoPE.
3. Zhang & Sennrich, *RMSNorm* (arXiv:1910.07467); Shazeer, *GLU Variants / SwiGLU* (arXiv:2002.05202).
4. Ainslie et al., *GQA: Grouped-Query Attention* (arXiv:2305.13245).
5. Karpathy, **nanochat** and **nanoGPT** (repos) — the from-scratch small-LM pipeline and $50 recipe; the 20:1 token:param heuristic (Hoffmann et al., *Chinchilla*, arXiv:2203.15556).
6. Jordan et al., **modded-nanoGPT speedrun** (repo) — QK-norm, ReLU², zero-init projections, value/embedding shortcuts, logit soft-cap, multi-token prediction; the modern small-LM template.
7. Jordan et al., **Muon optimizer** (Momentum Orthogonalized by Newton–Schulz); 2026 cautious-weight-decay and NorMuon follow-ups.
8. Rafailov et al., **DPO** (arXiv:2305.18290); Meng et al., **SimPO** (reference-free); Ethayarajh et al., **KTO**; Hong et al., **ORPO** — preference optimization family.
9. Shao et al., **GRPO** (DeepSeekMath, arXiv:2402.03300); 2026 successors **GSPO** (sequence-level importance ratio) and **DHPO** (hybrid) — reasoning RL / RLVR.
10. Gu et al., **MiniLLM: On-Policy Distillation** (reverse-KL, arXiv:2306.08543); Agarwal et al., **GKD**; Ko et al., **DistiLLM** — LM distillation.
11. Hinton et al., *Distilling the Knowledge in a Neural Network* (arXiv:1503.02531); Kim & Rush, **SeqKD** (arXiv:1606.07947).
12. Frantar et al., **GPTQ** (arXiv:2210.17323); Lin et al., **AWQ** (arXiv:2306.00978); Xiao et al., **SmoothQuant** (arXiv:2211.10438); Dettmers et al., **LLM.int8()** / bitsandbytes.
13. Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (arXiv:2211.17192); Chen et al., speculative sampling (arXiv:2302.01318); Cai et al., **Medusa** (arXiv:2401.10774); Li et al., **EAGLE-2/EAGLE-3** (arXiv:2406.16858 / 2503.01840).
14. Penedo et al., **FineWeb / FineWeb-Edu** (dataset); Eldan & Li, **TinyStories** (arXiv:2305.07759).
15. Miller (Anthropic), *Adding Error Bars to Evals* (arXiv:2411.00640) — for reporting eval numbers with uncertainty in the results table.

**Reminder to self while building:** implement the math, test the math, then explain the math. The tests in Part D are not overhead — they are the argument that you actually built this.
