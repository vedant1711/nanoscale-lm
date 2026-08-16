"""Measure the bits-vs-accuracy frontier for RTN, GPTQ and AWQ (spec B7, Phase 8).

Produces ``results/quantization/{frontier.json, frontier.png, quantization.md}``.

The x-axis is **effective** bits — nominal bit-width plus the amortised cost of the
stored per-group scales and zero-points. Plotting against the nominal width would let a
method buy accuracy with smaller groups and appear to win for free.

Usage::

    python scripts/quantize_frontier.py runs/nano/pretrain/final.pt
    python scripts/quantize_frontier.py --replay    # re-render from committed JSON
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch

from nanoscale.model import NanoScaleLM, build_model
from nanoscale.quantize import (
    AWQQuantizer,
    GPTQQuantizer,
    effective_bits,
    kv_cache_memory_report,
    quantize_rtn,
)
from nanoscale.quantize.kvcache import QuantizedKVCache
from nanoscale.tokenizer import BPETokenizer
from nanoscale.train import TokenBatcher, build_packed_tokens, evaluate_loss
from nanoscale.train.checkpoint import load_checkpoint, load_config_from_checkpoint
from nanoscale.utils import get_logger, git_sha, hardware_string, resolve_device
from nanoscale.utils.plotting import COLORS, new_figure, save_figure

log = get_logger("nanoscale.scripts.quantize")
RESULTS = Path("results/quantization")

METHODS = ("rtn", "gptq", "awq")
BIT_WIDTHS = (2, 3, 4, 8)


def _perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


@torch.no_grad()
def kv_quality_cost(
    model: NanoScaleLM, batches: list[torch.Tensor], *, bits: int, group_size: int
) -> float:
    """Relative logit error introduced by storing the KV cache at ``bits``."""
    total = 0.0
    for ids in batches:
        exact = model(ids).logits
        cache = QuantizedKVCache(
            n_layers=model.config.n_layers,
            batch_size=ids.shape[0],
            n_kv_heads=model.config.n_kv_heads,
            head_dim=model.config.head_dim,
            max_seq_len=ids.shape[1] + 1,
            key_bits=bits,
            value_bits=bits,
            group_size=group_size,
        )
        stepwise = torch.cat(
            [model(ids[:, i : i + 1], cache=cache).logits for i in range(ids.shape[1])], dim=1
        )
        total += float((stepwise - exact).norm() / exact.norm())
    return total / max(1, len(batches))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?", default=None)
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/nano.json"))
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--calib-batches", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--replay", action="store_true", help="Re-render from committed JSON.")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "frontier.json"

    if args.replay:
        if not json_path.exists():
            raise SystemExit(f"no committed results at {json_path}; run without --replay.")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        if args.checkpoint is None:
            raise SystemExit("a checkpoint is required unless --replay is given.")
        payload = measure(args)
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    figure = plot(payload)
    write_report(payload, figure)
    print(json.dumps({"rows": len(payload["rows"]), "figure": str(figure)}, indent=2))
    return 0


def measure(args: argparse.Namespace) -> dict[str, object]:
    """Run every (method, bits) combination and record perplexity."""
    cfg = load_config_from_checkpoint(args.checkpoint)
    tok = BPETokenizer.load(args.tokenizer)
    device = resolve_device("cpu")

    base = build_model(cfg.model).to(device)
    load_checkpoint(args.checkpoint, model=base, restore_rng=False, map_location=device)
    base.eval()

    data = build_packed_tokens(cfg.data, tok)
    eval_batcher = TokenBatcher(
        data.val, seq_len=cfg.data.seq_len, batch_size=4, seed=cfg.train.seed, shuffle=False
    )
    eval_batches = eval_batcher.take(args.eval_batches)
    calib_batcher = TokenBatcher(
        data.train, seq_len=cfg.data.seq_len, batch_size=4, seed=cfg.train.seed
    )
    calib_batches = [b.inputs for b in calib_batcher.take(args.calib_batches)]

    baseline_loss = evaluate_loss(base, eval_batches, device=device)
    log.info("fp32 baseline: loss %.4f (ppl %.3f)", baseline_loss, _perplexity(baseline_loss))

    rows: list[dict[str, object]] = [
        {
            "method": "fp32",
            "bits": 32,
            "effective_bits": 32.0,
            "loss": round(baseline_loss, 5),
            "perplexity": round(_perplexity(baseline_loss), 4),
            "mean_layer_error": 0.0,
        }
    ]

    for method in METHODS:
        for bits in BIT_WIDTHS:
            model = copy.deepcopy(base)
            if method == "rtn":
                errors = quantize_rtn(model, bits=bits, group_size=args.group_size)
            elif method == "gptq":
                q = GPTQQuantizer(model, bits=bits, group_size=args.group_size, act_order=True)
                q.collect(calib_batches)
                errors = q.apply()
            else:
                q_awq = AWQQuantizer(model, bits=bits, group_size=args.group_size, grid=12)
                q_awq.collect(calib_batches)
                errors = q_awq.apply()

            loss = evaluate_loss(model, eval_batches, device=device)
            row = {
                "method": method,
                "bits": bits,
                "effective_bits": round(effective_bits(bits, args.group_size), 4),
                "loss": round(loss, 5),
                "perplexity": round(_perplexity(loss), 4),
                "mean_layer_error": round(sum(errors.values()) / max(1, len(errors)), 5),
            }
            rows.append(row)
            log.info(
                "%s %d-bit: loss %.4f (ppl %.3f), mean layer error %.4f",
                method,
                bits,
                loss,
                _perplexity(loss),
                row["mean_layer_error"],
            )

    kv_rows = []
    for bits in (2, 4, 8):
        report = kv_cache_memory_report(
            n_layers=cfg.model.n_layers,
            batch_size=1,
            n_kv_heads=cfg.model.n_kv_heads,
            head_dim=cfg.model.head_dim,
            seq_len=4096,
            key_bits=bits,
            value_bits=bits,
            group_size=32,
        )
        report["logit_error"] = kv_quality_cost(
            base, [b.inputs[:1] for b in eval_batches[:2]], bits=bits, group_size=32
        )
        report["bits"] = float(bits)
        kv_rows.append(report)
        log.info(
            "kv %d-bit: %.2fx smaller, logit error %.4f",
            bits,
            report["compression"],
            report["logit_error"],
        )

    return {
        "git_sha": git_sha(),
        "hardware": hardware_string(),
        "checkpoint": str(args.checkpoint),
        "group_size": args.group_size,
        "baseline_loss": round(baseline_loss, 5),
        "params": base.num_parameters(),
        "rows": rows,
        "kv_rows": kv_rows,
    }


def plot(payload: dict[str, object]) -> Path:
    """Perplexity vs effective bits, one line per method."""
    rows = payload["rows"]
    assert isinstance(rows, list)
    fig, axes = new_figure(ncols=2, figsize=(11.5, 4.4))
    left, right = axes

    baseline = next(r for r in rows if r["method"] == "fp32")
    for i, method in enumerate(METHODS):
        points = sorted(
            (r for r in rows if r["method"] == method), key=lambda r: r["effective_bits"]
        )
        left.plot(
            [p["effective_bits"] for p in points],
            [p["perplexity"] for p in points],
            marker="o",
            ms=4,
            color=COLORS[i],
            label=method.upper(),
        )
        right.plot(
            [p["effective_bits"] for p in points],
            [p["mean_layer_error"] for p in points],
            marker="o",
            ms=4,
            color=COLORS[i],
            label=method.upper(),
        )

    left.axhline(
        baseline["perplexity"],
        color="0.6",
        ls=":",
        lw=1.2,
        label=f"fp32 = {baseline['perplexity']:.3f}",
    )
    left.set_yscale("log")
    left.set_xlabel("effective bits per weight (including scales)")
    left.set_ylabel("validation perplexity (log scale)")
    left.set_title("Accuracy vs bits")
    left.legend(fontsize=8)

    right.set_yscale("log")
    right.set_xlabel("effective bits per weight")
    right.set_ylabel("mean relative weight error")
    right.set_title("Reconstruction error")
    right.legend(fontsize=8)

    fig.suptitle("Quantization frontier", fontweight="bold")
    return save_figure(
        fig,
        RESULTS / "frontier.png",
        script="scripts/quantize_frontier.py",
        extra=f"group_size={payload['group_size']} · {payload['params']:,} params",
    )


def write_report(payload: dict[str, object], figure: Path) -> Path:
    """Write the committed markdown write-up."""
    rows = payload["rows"]
    kv_rows = payload["kv_rows"]
    assert isinstance(rows, list) and isinstance(kv_rows, list)
    baseline = next(r for r in rows if r["method"] == "fp32")

    lines = [
        "# Quantization — RTN, GPTQ and AWQ",
        "",
        f"Generated by `scripts/quantize_frontier.py` at git `{payload['git_sha']}` from "
        f"`{payload['checkpoint']}` ({payload['params']:,} parameters, group size "
        f"{payload['group_size']}).",
        "",
        f"![frontier]({figure.name})",
        "",
        "## Weight quantization",
        "",
        "| method | nominal bits | effective bits | val perplexity | vs fp32 | mean layer error |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        delta = row["perplexity"] / baseline["perplexity"]
        lines.append(
            f"| {row['method'].upper()} | {row['bits']} | {row['effective_bits']} | "
            f"{row['perplexity']:.4f} | {delta:.3f}x | {row['mean_layer_error']} |"
        )

    # Pull out the comparison the spec asks about, and say what actually happened.
    def cell(method: str, bits: int) -> dict[str, object]:
        return next(r for r in rows if r["method"] == method and r["bits"] == bits)

    lines += [
        "",
        "## What the numbers say",
        "",
        f"**At 4 and 8 bits every method is indistinguishable from fp32** "
        f"(perplexity {cell('rtn', 4)['perplexity']:.4f} / "
        f"{cell('gptq', 4)['perplexity']:.4f} / {cell('awq', 4)['perplexity']:.4f} against "
        f"a baseline of {baseline['perplexity']:.4f}). The spec anticipated GPTQ beating "
        'RTN at 4 bits "by a clear margin"; at this scale it does not, because there is '
        "no margin left to win — a 5M-parameter model on a narrow synthetic corpus has "
        "little redundancy for 4-bit rounding to destroy in the first place. Reporting a "
        "tie is the honest outcome.",
        "",
        f"**The separation appears at 2 and 3 bits, and there GPTQ wins.** At 2 bits GPTQ "
        f"reaches {cell('gptq', 2)['perplexity']:.4f} against RTN's "
        f"{cell('rtn', 2)['perplexity']:.4f} and AWQ's {cell('awq', 2)['perplexity']:.4f}; "
        f"at 3 bits GPTQ recovers the fp32 perplexity exactly "
        f"({cell('gptq', 3)['perplexity']:.4f} vs {baseline['perplexity']:.4f}) while the "
        "other two do not.",
        "",
        "**GPTQ has the *worst* weight error and the *best* perplexity.** At 2 bits its "
        f"mean relative weight error is {cell('gptq', 2)['mean_layer_error']:.3f} against "
        f"RTN's {cell('rtn', 2)['mean_layer_error']:.3f} — nearly double — yet it produces "
        "the better model. That is not a contradiction, it is the entire thesis of the "
        "method: GPTQ minimises ‖WX − ŴX‖, the error in the layer's *output*, and will "
        "happily accept a larger perturbation to a weight that multiplies a quiet input "
        "channel in exchange for a smaller one on a loud channel. Any comparison that "
        "ranked these methods by weight error would rank them backwards.",
        "",
        "**Effective bits include the scales.** A '4-bit' model with group size "
        f"{payload['group_size']} and fp16 scale + zero-point actually costs "
        f"{4 + 2 * 16 / int(str(payload['group_size'])):.2f} bits per weight. Plotting "
        "against the nominal width would let a method buy accuracy with smaller groups "
        "and appear to win for free.",
        "",
        "## KV-cache quantization",
        "",
        "| bits | effective bits/element | cache at 4k ctx | vs fp16 | mean logit error |",
        "|---|---|---|---|---|",
    ]
    for row in kv_rows:
        lines.append(
            f"| {int(row['bits'])} | {row['effective_bits_per_element']:.2f} | "
            f"{row['quantized_mb']:.2f} MB | {row['compression']:.2f}x | "
            f"{row['logit_error']:.4f} |"
        )

    lines += [
        "",
        "The KV numbers are an **analytic footprint plus a measured accuracy cost**, not "
        "a measured latency win. PyTorch has no int4 matmul on CPU, so the cache stores "
        "codes and dequantizes on read: the quality cost is exactly real, the memory "
        "figure is computed from the representation, and no decode-speed claim is made "
        "for it here. A real int4 kernel is where the latency win would come from.",
        "",
        "## Caveats",
        "",
        "This is a ~5M-parameter model on a synthetic corpus with a 1k vocabulary. The "
        "*ordering* of the methods is the transferable finding; the absolute perplexity "
        "degradations are not, and at this scale the model is small enough that even "
        "aggressive quantization has less to destroy than it would at 7B. bitsandbytes "
        "is available as an external reference point via the `compare` extra but is not "
        "installed by default and is not what any number here comes from.",
        "",
        "Reproduce with: `python scripts/quantize_frontier.py runs/nano/pretrain/final.pt`",
        "",
    ]
    path = RESULTS / "quantization.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
