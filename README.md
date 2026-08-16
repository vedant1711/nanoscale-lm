# NanoScale-LM

> I built a language model from scratch and then made it deployable on hardware anyone can afford.

A from-scratch, fully-typed PyTorch implementation of a modern small decoder-only language
model **and** the complete efficiency stack needed to serve it — distillation, quantization
and speculative decoding — in one coherent project that runs end-to-end on free hardware.

Every core algorithm here is implemented in this repository: attention, RoPE, the optimizer
step, the BPE merges, the DPO loss, the GPTQ rounding, the speculative accept/reject rule.
No high-level trainer library is used anywhere in `src/nanoscale/`.

---

## Status

Under construction, phase by phase, against `nanoscale_lm_spec.md`.

| Phase | Scope | State |
|---|---|---|
| 0 | Foundation: scaffold, configs, CLI, seeds/manifests, CI | ✅ |
| 1 | Byte-level BPE tokenizer | ✅ |
| 2 | Model: GQA + RoPE + QK-norm + RMSNorm + SwiGLU + KV cache | ✅ |
| 3 | Optimizer: AdamW + Muon (Newton–Schulz) | ✅ |
| 4 | Pretraining loop | ✅ |
| 5 | Optimizer & architecture ablations | ⬜ |
| 6 | Alignment: SFT, DPO, SimPO, (optional) GRPO-RLVR | ⬜ |
| 7 | Distillation: forward-KL, SeqKD, reverse-KL on-policy | ⬜ |
| 8 | Quantization: RTN, GPTQ, AWQ, KV-cache quant | ⬜ |
| 9 | Speculative decoding: draft–target, Medusa | ⬜ |
| 10 | Serving, benchmarks, demo, docs | ⬜ |

## Quickstart

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
nanoscale info
```

## First result: the `nano` tier, trained on a laptop CPU

![nano loss curve](results/curves/nano_loss.png)

4,952,064 parameters, 819,200 tokens, 95 s on
CPU at 8,598 tokens/s. Validation loss
0.390 (perplexity 1.48), starting from
exactly `ln(1024) = 6.93` as the zero-init scheme predicts.

Sampled from the committed checkpoint (`results/samples/nano_base.md`):

> It was a sunny day. Lily went to the park with a patient hedgehog. Lily wanted to find
> a torn map. But a torn map was too heavy to lift. She tied her scarf around it for
> grip. A patient hedgehog pushed from the other side. She sat down and thought about the
> problem. With one more try a torn map came loose.

The `nano` tier trains on a synthetic story corpus (see `src/nanoscale/data/toy.py`), so
this is a demonstration that the pipeline learns structure — agreement, coreference,
narrative shape — not a claim about open-domain language modelling. The `micro` tier
trains on FineWeb-Edu to a full 20:1 token budget for that.

## The size ladder

All tiers share one architecture; only depth/width/context/token budget change. Token
budgets follow the Chinchilla-style 20:1 tokens-per-parameter heuristic.

| Tier | Params | Layers × d_model × heads | Context | Token budget | Trains on |
|---|---|---|---|---|---|
| `nano` | ~5.0M | 6 × 256 × 4 | 256 | 0.8M (step-driven) | CPU / laptop, < 10 min |
| `micro` | ~40M (23.6M non-emb.) | 8 × 512 × 8 | 512 | 808M (20:1) | Free Colab/Kaggle T4 |
| `small` | ~126M | 12 × 768 × 12 | 1024 | 2.52B (20:1) | Scale-up recipe |

Run `nanoscale info` for the exact numbers this repo asserts.

## License

Apache-2.0. See [LICENSE](LICENSE).
