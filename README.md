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
| 4 | Pretraining loop | ⬜ |
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
