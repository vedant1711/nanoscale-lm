# Architecture

## Module map

```
src/nanoscale/
├── cli.py            typer entry point; sub-apps registered per phase
├── config/           16 frozen pydantic models + the size-ladder presets
├── tokenizer/        byte-level BPE: train, encode, decode, chat template
├── model/            attention (GQA, RoPE, QK-norm, KV cache), norms, MLPs,
│                     blocks, LM head, MTP heads, numerics
├── optim/            Muon (Newton–Schulz), AdamW, cautious decay, param router
├── train/            data pipeline, LR schedules, checkpointing, the loop
├── align/            SFT, DPO/SimPO losses and trainer, GRPO-RLVR
├── distill/          forward-KL, SeqKD, on-policy reverse KL
├── quantize/         RTN, GPTQ, AWQ, KV-cache quantization
├── specdec/          accept rule, draft–target sampling, Medusa + tree attention
├── serve/            streaming generation, stop sequences, timing breakdown
├── eval/             perplexity with error bars, tiny benchmark, preference judge
├── bench/            ablation harness, throughput/latency/memory harness
├── data/             the offline toy corpus and instruction/preference data
└── utils/            seeds, devices, manifests, logging, plotting
```

## The model

```
token embedding  (vocab × d_model)
      ↓
N × ┌──────────────────────────────────────────────────────────┐
    │  x → RMSNorm → CausalSelfAttention ─────────→ + x        │
    │              (GQA · RoPE · QK-norm · KV cache)           │
    │  x → RMSNorm → SwiGLU MLP ──────────────────→ + x        │
    └──────────────────────────────────────────────────────────┘
      ↓
RMSNorm → untied LM head → [optional tanh soft-cap] → logits
                         ↘ [optional MTP heads → t+2, t+3 …]
```

Pre-norm residuals keep an identity path from embedding to output, so gradients reach
layer 0 undamped — which is what makes depth trainable without a warmup knife-edge.

## The size ladder

One architecture; only depth, width, context and token budget change.

| Tier | Params (total / non-emb.) | Shape | Context | Token budget | Trains on |
|---|---|---|---|---|---|
| `nano` | 4,952,064 / 4,427,776 | 6 × 256 × 4 | 256 | 0.8M (step-driven) | laptop CPU, 95 s |
| `micro` | 40,379,904 / 23,602,688 | 8 × 512 × 8 | 512 | 808M (20:1) | free Colab T4 |
| `small` | 125,849,856 / 75,518,208 | 12 × 768 × 12 | 1024 | 2.5B (20:1) | scale-up recipe |

These counts are asserted as exact constants in the test suite, against both the analytic
formula and the built `nn.Module`, so architecture drift fails loudly.

## Design decisions worth knowing

**The parameter counts are reported honestly.** The spec's headline figures are
approximate; this repo pins the *shapes* and reports what they actually produce, in both
total and non-embedding form. Comparing "parameter counts" across projects is ambiguous
unless you say whether embeddings and an untied head are included. Every table here says.

**`nano` is deliberately not compute-optimal.** 20:1 on 5M parameters is ~99M tokens,
which is hours of CPU. `nano` exists to be a sub-10-minute CI and teaching tier; its
manifests record what fraction of the compute-optimal budget it covered. `micro` and
`small` train to the full 20:1 budget.

**Everything has a CPU path.** An unavailable accelerator degrades to CPU rather than
raising; bf16/fp16 autocast degrades to fp32 on CPU. CI runs the whole `nano` tier on CPU
runners. GPU runs are reported experiments, never gates.

**Reductions promote to at least fp32, and never demote.** The RMS reduction, the
attention softmax and the RoPE rotation all accumulate at higher precision than the
surrounding activations. The naive `x.float()` promotes bf16 as intended but silently
*demotes* float64 — which broke the fp64 reference tests that give the numerical suite its
teeth. `model/numerics.py` exists for that one rule.

**Configuration is frozen, hashed and fully documented.** Every field carries a
description (enforced by a test), configs hash to a stable digest recorded in every run
manifest, and save → load → validate is an exact identity.

See `DESIGN_DECISIONS.md` in the repository for the ADR-style long form.
