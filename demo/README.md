---
title: NanoScale-LM
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.36.0
app_file: app.py
pinned: false
license: apache-2.0
---

# NanoScale-LM — live demo

A ~5M-parameter language model built entirely from scratch, plus the efficiency stack
that makes it cheap to serve: distillation, GPTQ quantization and speculative decoding.

## What you can do here

- **Chat / generate** — talk to the model and flip between the pretrained and aligned
  checkpoints to feel what DPO did.
- **Speed lab** — race autoregressive decoding against speculative decoding on the same
  prompt and watch the target-forward-pass count drop.
- **Compression explorer** — the bits-vs-accuracy frontier and the variants table, read
  straight from the repository's committed `results/`.

## Honest expectations

This model has ~5M parameters and was trained for 95 seconds on a laptop CPU on a
**synthetic story corpus**. It writes small coherent stories about Lily and Tom because
that is all it has ever read. It knows no facts, follows no general instructions, and is
not an assistant.

The claim is not that the model is good. It is that the whole lifecycle is implemented
from first principles, tested at the level of the maths, and measured honestly — and
that the identical code path scales to a real corpus on a free Colab GPU.

## Space notes

- Free CPU-basic tier (2 vCPU / 16 GB) is sufficient.
- The Space sleeps after ~48 hours idle; the first request after that pays a cold start
  of roughly a minute while the container boots.
- "Duplicate this Space" is enabled — fork it and point it at your own checkpoints.

## Source

Everything is in the [NanoScale-LM repository](https://github.com/nanoscale-lm/nanoscale-lm),
Apache-2.0. Every number shown here is produced by a committed script and stamped with
the git SHA that produced it.
