# Quickstart

Three paths, in increasing order of commitment.

## 1. Run it locally (10 minutes, no GPU)

```bash
git clone https://github.com/vedant1711/nanoscale-lm
cd nanoscale-lm
make install          # uv venv + editable install with dev extras
make smoke            # tokenizer -> pretrain -> SFT -> DPO -> quantize -> speculate
```

`make smoke` runs the entire pipeline on the `nano` tier on CPU in under ten minutes and
asserts a sanity metric at every stage. When it finishes:

```bash
.venv/bin/nanoscale serve generate runs/smoke/pretrain/final.pt \
    --prompt "It was a sunny day. Lily went to the park with" -n 64
```

## 2. Reproduce the reported results

```bash
make tokenizer        # train and report the committed 1k vocabulary
make train-nano       # 400 steps on CPU (~95 s), writes the loss curve and samples
make ablate           # the optimizer and architecture A/Bs
make align            # SFT -> {DPO, DPO+NLL, SimPO} + the length diagnostic
make distill          # the three distillation objectives
make quantize         # the bits-vs-accuracy frontier
make specdec          # speculative decoding vs autoregressive
make bench            # the unified variants table
make results          # regenerate docs/results.md from all of the above
```

Every one of those writes to `results/` and can be re-rendered without retraining:

```bash
python scripts/ablate.py --replay
python scripts/quantize_frontier.py --replay
```

## 3. Scale up on a free GPU

The `micro` tier is the same code path against streaming FineWeb-Edu:

```bash
uv pip install -e ".[data]"
nanoscale train pretrain --tier micro -o runs/micro/pretrain
```

~40M parameters, 808M tokens (the Chinchilla-optimal 20:1 budget), a few hours on a free
Colab T4. Nothing in the code changes, only the config.

## The CLI

```bash
nanoscale info                        # hardware and the size ladder
nanoscale config show --tier micro    # the fully-resolved configuration
nanoscale config params --tier small  # parameter breakdown

nanoscale tokenizer train --tier nano
nanoscale train pretrain --tier nano
nanoscale train generate <ckpt> --prompt "..."

nanoscale align sft <ckpt>
nanoscale align preference <ckpt> --method dpo
nanoscale align grpo <ckpt>

nanoscale serve generate <ckpt> --prompt "..."   # streams
nanoscale serve chat <ckpt>
nanoscale serve eval <ckpt>                      # perplexity + tiny benchmark
```

## Development

```bash
make check    # ruff + mypy --strict + pytest  (what CI runs)
make test     # 554 tests
make fmt      # auto-fix lint and formatting
make docs     # serve the documentation locally
```
