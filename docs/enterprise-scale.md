# Minimal now, enterprise later

The thesis of this project is that the *techniques* are scale-invariant even though the
*results* are not. This page maps each component onto what changes with more compute —
and is explicit that none of it is built here.

## The same code path, more compute

| Knob | `nano` (built) | `micro` (recipe) | `small` (recipe) | Production |
|---|---|---|---|---|
| Parameters | 5.0M | 40M | 126M | 7B–70B |
| Data | synthetic grammar | FineWeb-Edu (streamed) | FineWeb-Edu | curated + filtered trillions |
| Token budget | 0.8M (step-driven) | 808M (20:1) | 2.5B (20:1) | 15T+ (over-trained) |
| Hardware | laptop CPU, 95 s | free Colab T4, hours | Kaggle P100, sessions | thousands of H100s |
| Precision | fp32 | bf16 autocast | bf16 autocast | bf16 + fp8 |
| Parallelism | none | none | none | FSDP/TP/PP |

Changing tier changes a config, not code. `nanoscale train pretrain --tier micro` runs the
identical loop against streaming FineWeb-Edu.

## What would need building for real scale

**Multi-GPU.** The training loop is single-device. FSDP would wrap the model, shard
optimizer state, and require: gradient accumulation to remain correct under sharding
(already is — the loss is scaled per micro-batch), checkpointing to become sharded
save/load, and the Muon/AdamW router to run per-shard. Muon specifically needs care: the
Newton–Schulz iteration is a matrix operation on a *whole* weight matrix, so a
tensor-sharded matrix needs an all-gather before orthogonalising, or a distributed variant.
That is a real design question this project does not answer.

**Data.** Streaming already avoids disk blowup. At scale you additionally need dedup,
quality filtering, and a shuffle buffer large enough that a single shard's topic
distribution does not leak into a batch.

**Serving.** `nanoscale serve` is a generation loop, not a serving stack. Production means
vLLM or TensorRT-LLM, which supply the three things this repository deliberately does not:
**PagedAttention** (KV-cache memory management without fragmentation), **continuous
batching** (new requests join a running batch instead of waiting for it), and **fused
kernels** including real int4 matmuls.

That last point is where the Arc-2 numbers here become conservative rather than optimistic.

## Where each Arc-2 technique lands on the cost curve

**Quantization — weight-bound cost.** GPTQ-4bit takes the weight footprint from 18.9 MB to
4.4 MB here. At 7B that is 14 GB → 3.5 GB: the difference between needing an A100 and
fitting on a consumer GPU. The cost is a one-off calibration pass. This is the highest-ROI
change for *fitting* a model somewhere cheaper.

**Distillation — everything-bound cost.** A 17.7× smaller student cuts weights, KV cache,
prefill and decode simultaneously, and it is the only technique here that reduces *prefill*
cost. The price is a training run and a real quality drop, so it pays when you serve
enough traffic to amortise the training and can tolerate the quality you measure.

**Speculative decoding — latency, not throughput.** It cuts target forward passes ~3×
here. At scale, where decode is memory-bandwidth-bound, that translates to a 2–3× latency
win for a single stream. Crucially it **does not help a saturated batched server**: if the
GPU is already busy with concurrent requests, there is no idle bandwidth for the draft to
exploit. It is a technique for interactive latency, not for aggregate throughput — and the
distinction is one that gets glossed over often.

**KV-cache quantization — context-bound cost.** The only technique whose benefit *grows*
with sequence length. At 4k context the cache here is 3.2× smaller at 4 bits. At 128k
context on a 7B model the cache exceeds the weights, and quantizing it is what makes long
context affordable at all.

## Serving economics, sketched

The numbers below are illustrative arithmetic, not measurements — this project has no
production deployment to measure.

For a single-stream interactive workload, latency is roughly
`tokens × bytes_per_forward_pass / memory_bandwidth`. Quantization divides the numerator;
speculative decoding divides the token count that hits the target. They compose, which is
why the two are stacked in the results table.

For a batched throughput workload the calculus inverts: batching already amortises weight
loading across requests, so quantization's benefit shrinks toward the arithmetic saving
alone and speculation's benefit largely disappears. **The right compression stack depends
on which regime you are in**, and a project that reported only single-stream numbers while
implying they transfer to a batched server would be overselling.

## What this project deliberately does not claim

It does not claim frontier quality, that its measurements transfer to 7B, or that a
laptop-CPU wall-clock number says anything about an H100. It claims that the lifecycle is
implemented correctly, tested at the level of the mathematics, measured honestly, and
written so that adding compute is a config change rather than a rewrite.
