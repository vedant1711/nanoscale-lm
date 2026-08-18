# NanoScale-LM

> I built a language model from scratch and then made it deployable on hardware anyone
> can afford.

A fully-typed PyTorch implementation of a modern small decoder-only language model **and**
the complete efficiency stack needed to serve it — distillation, quantization and
speculative decoding — in one project that runs end to end on free hardware.

Every core algorithm is implemented in this repository: the BPE merges, attention with
GQA/RoPE/QK-norm, the Muon optimizer's Newton–Schulz orthogonalization, the DPO and SimPO
losses, GPTQ's Hessian error compensation, and the speculative-sampling accept/reject
rule. No high-level trainer library appears anywhere in `src/nanoscale/`.

## The result, in one figure

![nano loss curve](curves/nano_loss.png)

4,952,064 parameters. 819,200 tokens. **95 seconds on a laptop CPU.** Validation
perplexity 1.476, starting from exactly `ln(1024) = 6.93` as the zero-init scheme
predicts.

> It was a sunny day. Lily went to the park with a patient hedgehog. Lily wanted to find a
> torn map. But a torn map was too heavy to lift. She tied her scarf around it for grip. A
> patient hedgehog pushed from the other side. With one more try a torn map came loose.

## The two arcs

**Arc 1 — build the model.** Byte-level BPE → a decoder-only transformer (RoPE, RMSNorm,
SwiGLU, GQA, QK-norm, KV cache) → Muon + AdamW → pretraining → SFT, DPO, SimPO, and an
optional GRPO-RLVR track.

**Arc 2 — serve it cheaply.** That exact checkpoint compressed three ways: knowledge
distillation (17.7× smaller), GPTQ 4-bit quantization (4.3× smaller weights), and
speculative decoding (3× fewer target forward passes) — with the levers composed.

## Start here

- **[▶ Live demo](demo.html)** — generate text, step through the model's next-token
  distribution, watch it compress and flag anomalies. Real outputs, no install.
- **[📖 Full documentation, one page](explainer.html)** — every algorithm, diagram, design
  decision and measurement, written for anyone who knows Transformers and nothing past them.
- **[Quickstart](quickstart.md)** — from a clone to generated text in under ten minutes,
  no GPU.
- **[Architecture](architecture.md)** — what is built and why each choice was made.
- **[Methodology](methodology.md)** — every algorithm with its formula, its citation, and
  the test that verifies it.
- **[Results](results.md)** — every measurement, generated from committed artifacts.
- **[Limitations](limitations.md)** — read this before quoting any number from here.

## Honesty policy

Three rules this project holds itself to, because they are what make the rest worth
reading:

1. **Every number is produced by a committed script**, stamped with the git SHA and
   hardware that produced it, and regenerable in replay mode. The docs cannot drift from
   the artifacts — `docs/results.md` is generated from them and CI fails if it is stale.
2. **Negative results are reported.** GPTQ does not beat RTN at 4 bits here. QK-norm shows
   no measurable difference in final loss. Speculative decoding is *slower* in wall-clock
   on this CPU. Muon loses to AdamW on a convex problem. All of that is in the docs
   because leaving it out would make the positive results untrustworthy.
3. **The scale caveat is stated everywhere it matters.** These are directional
   confirmations of findings obtained at 100–1000× this scale, not independent evidence
   about them.
