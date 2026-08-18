"""The unified results table across every model variant (spec Phase 10).

One table, one machine, one set of prompts and seeds, covering:

    base · distilled · GPTQ-4bit · speculative · speculative + GPTQ-4bit

Produces ``results/bench/{table.json, table.md, table.png}``.

Usage::

    python scripts/bench_all.py
    python scripts/bench_all.py --replay    # re-render from the committed JSON
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections.abc import Callable
from pathlib import Path

import torch

from nanoscale.bench import BenchHarness, model_memory_bytes
from nanoscale.config import GenerateConfig
from nanoscale.eval import perplexity, run_tiny_bench
from nanoscale.model import NanoScaleLM, build_model
from nanoscale.quantize import GPTQQuantizer, effective_bits
from nanoscale.serve import generate_text
from nanoscale.specdec import SpeculativeSampler
from nanoscale.tokenizer import BPETokenizer
from nanoscale.train import TokenBatcher, build_packed_tokens
from nanoscale.train.checkpoint import load_checkpoint, load_config_from_checkpoint
from nanoscale.utils import get_logger, resolve_device
from nanoscale.utils.plotting import COLORS, new_figure, save_figure

log = get_logger("nanoscale.scripts.bench")
RESULTS = Path("results/bench")

PROMPT = "It was a sunny day. Lily went to the park with"


def kv_mb(model: NanoScaleLM, seq_len: int, *, bits: float = 32.0) -> float:
    """Analytic KV-cache footprint in MB at ``seq_len``."""
    cfg = model.config
    elements = cfg.n_layers * cfg.n_kv_heads * seq_len * cfg.head_dim * 2
    return elements * bits / 8 / 1024**2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("runs/nano/pretrain/final.pt"))
    parser.add_argument("--aligned", type=Path, default=Path("runs/nano/dpo_nll/final.pt"))
    parser.add_argument(
        "--distilled", type=Path, default=Path("runs/nano/distill/reverse_kl/final.pt")
    )
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/nano.json"))
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--gamma", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "table.json"

    if args.replay:
        if not json_path.exists():
            raise SystemExit(f"no committed results at {json_path}; run without --replay.")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        payload = measure(args)
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    figure = plot(payload)
    report = write_report(payload, figure)
    print(f"wrote {json_path}, {figure}, {report}")
    return 0


def measure(args: argparse.Namespace) -> dict[str, object]:
    """Benchmark every variant."""
    tok = BPETokenizer.load(args.tokenizer)
    device = resolve_device("cpu")
    cfg = load_config_from_checkpoint(args.base)

    base = build_model(cfg.model).to(device)
    load_checkpoint(args.base, model=base, restore_rng=False, map_location=device)
    base.eval()

    data = build_packed_tokens(cfg.data, tok)
    eval_batches = TokenBatcher(
        data.val, seq_len=cfg.data.seq_len, batch_size=4, shuffle=False
    ).take(args.eval_batches)
    gen_cfg = GenerateConfig(max_new_tokens=args.tokens, temperature=0.8, seed=1337)
    ctx = args.tokens + 16

    harness = BenchHarness(warmup_iters=args.warmup, measure_iters=args.iters)

    def decode_fn(model: NanoScaleLM) -> Callable[[int], dict[str, float]]:
        def run(i: int) -> dict[str, float]:
            out = generate_text(model, tok, PROMPT, gen_cfg.merged(seed=1337 + i))
            return {
                "prefill_s": out.prefill_s,
                "decode_s": out.decode_s,
                "generated_tokens": float(out.generated_tokens),
            }

        return run

    # --- base ---------------------------------------------------------------------
    harness.time_variant(
        "base (fp32)",
        decode_fn(base),
        params=base.num_parameters(),
        weight_mb=model_memory_bytes(base) / 1024**2,
        kv_mb=kv_mb(base, ctx),
        perplexity=perplexity(base, eval_batches, device=device).perplexity,
        notes="the Phase-4 pretrained checkpoint",
    )
    base_bench = run_tiny_bench(base, tok)

    # --- distilled student ---------------------------------------------------------
    student_bench = None
    if args.distilled.exists():
        student_cfg = load_config_from_checkpoint(args.distilled)
        student = build_model(student_cfg.model).to(device)
        load_checkpoint(args.distilled, model=student, restore_rng=False, map_location=device)
        student.eval()
        harness.time_variant(
            "distilled (reverse-KL)",
            decode_fn(student),
            params=student.num_parameters(),
            weight_mb=model_memory_bytes(student) / 1024**2,
            kv_mb=kv_mb(student, ctx),
            perplexity=perplexity(student, eval_batches, device=device).perplexity,
            notes="Phase-7 on-policy reverse-KL student",
        )
        student_bench = run_tiny_bench(student, tok)
    else:
        student = build_model(cfg.model).to(device).eval()

    # --- GPTQ-4bit -----------------------------------------------------------------
    quantized = copy.deepcopy(base)
    calib = TokenBatcher(data.train, seq_len=cfg.data.seq_len, batch_size=4, seed=1337).take(8)
    gptq = GPTQQuantizer(quantized, bits=4, group_size=64, act_order=True)
    gptq.collect([b.inputs for b in calib])
    gptq.apply()
    quantized.eval()

    harness.time_variant(
        "GPTQ 4-bit",
        decode_fn(quantized),
        params=quantized.num_parameters(),
        weight_mb=model_memory_bytes(quantized, weight_bits=effective_bits(4, 64)) / 1024**2,
        kv_mb=kv_mb(quantized, ctx),
        perplexity=perplexity(quantized, eval_batches, device=device).perplexity,
        notes="weights simulated in fp32; the MB figure is the 4-bit representation",
    )
    quantized_bench = run_tiny_bench(quantized, tok)

    # --- speculative ----------------------------------------------------------------
    def speculative_fn(
        target: NanoScaleLM, draft: NanoScaleLM
    ) -> Callable[[int], dict[str, float]]:
        sampler = SpeculativeSampler(target, draft, gamma=args.gamma, temperature=0.8)
        prompt_ids = torch.tensor([tok.encode(PROMPT, add_bos=True)])

        def run(i: int) -> dict[str, float]:
            start = time.perf_counter()
            out = sampler.generate(
                prompt_ids,
                max_new_tokens=args.tokens,
                generator=torch.Generator().manual_seed(1337 + i),
            )
            total = time.perf_counter() - start
            return {
                "prefill_s": 0.0,
                "decode_s": total,
                "generated_tokens": float(out.generated),
                "acceptance_rate": out.acceptance_rate,
                "mean_accepted_length": out.mean_accepted_length,
            }

        return run

    harness.time_variant(
        f"speculative (γ={args.gamma})",
        speculative_fn(base, student),
        params=base.num_parameters() + student.num_parameters(),
        weight_mb=(model_memory_bytes(base) + model_memory_bytes(student)) / 1024**2,
        kv_mb=kv_mb(base, ctx) + kv_mb(student, ctx),
        perplexity=perplexity(base, eval_batches, device=device).perplexity,
        notes="lossless w.r.t. the base model; draft weights and cache add to the footprint",
    )
    harness.time_variant(
        f"speculative (γ={args.gamma}) + GPTQ 4-bit",
        speculative_fn(quantized, student),
        params=quantized.num_parameters() + student.num_parameters(),
        weight_mb=(
            model_memory_bytes(quantized, weight_bits=effective_bits(4, 64))
            + model_memory_bytes(student)
        )
        / 1024**2,
        kv_mb=kv_mb(quantized, ctx) + kv_mb(student, ctx),
        perplexity=perplexity(quantized, eval_batches, device=device).perplexity,
        notes="both levers; lossless w.r.t. the *quantized* target",
    )

    return harness.payload(
        prompt=PROMPT,
        tokens_per_request=args.tokens,
        gamma=args.gamma,
        context_len=ctx,
        tiny_bench={
            "base": base_bench.summary(),
            "distilled": student_bench.summary() if student_bench else None,
            "gptq4": quantized_bench.summary(),
        },
    )


def plot(payload: dict[str, object]) -> Path:
    """Weight footprint and decode throughput per variant."""
    rows = payload["rows"]
    assert isinstance(rows, list)
    labels = [str(r["variant"]) for r in rows]
    x = range(len(rows))

    fig, axes = new_figure(ncols=2, figsize=(12.0, 4.6))
    left, right = axes
    left.bar(
        list(x), [float(r["weight_mb"]) for r in rows], color=[COLORS[i % len(COLORS)] for i in x]
    )
    left.set_xticks(list(x))
    left.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    left.set_ylabel("weight footprint (MB)")
    left.set_title("Model size")

    right.bar(
        list(x),
        [float(r["decode_tokens_per_s_p50"]) for r in rows],
        color=[COLORS[i % len(COLORS)] for i in x],
    )
    right.set_xticks(list(x))
    right.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    right.set_ylabel("decode tokens/s (p50)")
    right.set_title("Decode throughput (this machine, CPU)")

    fig.suptitle("NanoScale-LM variants", fontweight="bold")
    return save_figure(
        fig,
        RESULTS / "table.png",
        script="scripts/bench_all.py",
        extra=f"{payload['tokens_per_request']} tokens/request · median of "
        f"{payload['measure_iters']}",
    )


def write_report(payload: dict[str, object], figure: Path) -> Path:
    """Write the committed results table."""
    rows = payload["rows"]
    tiny = payload["tiny_bench"]
    assert isinstance(rows, list) and isinstance(tiny, dict)

    lines = [
        "# Results table",
        "",
        f"Generated by `scripts/bench_all.py` at git `{payload['git_sha']}` on "
        f"`{payload['hardware']}`. Prompt: `{payload['prompt']}`, "
        f"{payload['tokens_per_request']} tokens per request, median of "
        f"{payload['measure_iters']} measured iterations after {payload['warmup_iters']} "
        "warmup iterations.",
        "",
        f"![variants]({figure.name})",
        "",
        "| variant | params | weights | KV @ ctx | prefill p50 | decode tok/s | "
        "latency p50 | latency p95 | val ppl | accept |",
        "|" + "---|" * 10,
    ]
    for row in rows:
        ppl = f"{row['perplexity']:.4f}" if row["perplexity"] is not None else ", "
        acc = f"{row['acceptance_rate']:.3f}" if row["acceptance_rate"] is not None else ", "
        lines.append(
            f"| {row['variant']} | {row['params']:,} | {row['weight_mb']:.2f} MB | "
            f"{row['kv_mb']:.2f} MB | {row['prefill_ms_p50']:.1f} ms | "
            f"{row['decode_tokens_per_s_p50']:.1f} | {row['latency_ms_p50']:.1f} ms | "
            f"{row['latency_ms_p95']:.1f} ms | {ppl} | {acc} |"
        )

    lines += [
        "",
        "### Tiny-benchmark accuracy",
        "",
        "| variant | accuracy | n | chance |",
        "|---|---|---|---|",
    ]
    for name, result in tiny.items():
        if result is None:
            continue
        lines.append(
            f"| {name} | {result['accuracy']:.1%} ± {result['accuracy_stderr']:.1%} | "
            f"{result['n_questions']} | {result['chance']:.0%} |"
        )

    lines += [
        "",
        "## How to read this table",
        "",
        "**The weight column is the representation size, not the tensor size.** The "
        "4-bit rows are simulated in fp32 because there is no int4 CPU kernel, so reading "
        "the footprint off the tensors would report a 4-bit model as 32-bit. The figure "
        "is computed from the effective bit-width including the stored scales.",
        "",
        "**Speculative rows include the draft's weights and cache in their footprint.** "
        "Speculation is not free in memory: it trades space for target forward passes. "
        "Reporting only the target's size would hide the trade.",
        "",
        "**Decode throughput here is a CPU measurement at 5M parameters and does not "
        "generalise.** Speculation reduces target forward passes; that part is real and "
        "hardware-independent, and is measured in `results/speculative/`, but at this "
        "scale a forward pass is dominated by Python dispatch rather than by weight "
        "loading, so the wall-clock win the method exists for does not appear. Quantization "
        "likewise shows no speedup because the arithmetic is still fp32.",
        "",
        "**The tiny benchmark is saturated** at 100% for the base model. It is a "
        "degradation detector for Arc 2, not a quality ladder; an unchanged score means "
        "compression did not break the capabilities it probes, and nothing stronger.",
        "",
        "Reproduce with: `python scripts/bench_all.py` (or `--replay` to re-render).",
        "",
    ]
    path = RESULTS / "table.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
