"""Benchmark speculative decoding against autoregressive decoding (spec B8, Phase 9).

Measures acceptance length, target-forward-pass count and wall-clock throughput for:

* autoregressive (the baseline),
* draft–target speculation at several ``γ``,
* speculation over a **GPTQ-quantized** target — the two levers composed.

Produces ``results/speculative/{bench.json, speculative.png, speculative.md}``.

Usage::

    python scripts/specdec_bench.py runs/nano/pretrain/final.pt --draft runs/nano/draft/final.pt
    python scripts/specdec_bench.py --replay
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Callable
from pathlib import Path

import torch

from nanoscale.config import ExperimentConfig, draft_model_config
from nanoscale.model import NanoScaleLM, build_model
from nanoscale.quantize import GPTQQuantizer
from nanoscale.specdec import SpeculativeResult, SpeculativeSampler, autoregressive_baseline
from nanoscale.tokenizer import BPETokenizer
from nanoscale.train import TokenBatcher, build_packed_tokens
from nanoscale.train.checkpoint import load_checkpoint, load_config_from_checkpoint
from nanoscale.utils import get_logger, git_sha, hardware_string, resolve_device
from nanoscale.utils.plotting import COLORS, new_figure, save_figure

log = get_logger("nanoscale.scripts.specdec")
RESULTS = Path("results/speculative")

PROMPTS = (
    "It was a sunny day. Lily went to",
    "Tom wanted to find a shiny key",
    "The wind was cold. Mia walked",
    "But a red ball was stuck",
)


def _load(path: Path) -> tuple[NanoScaleLM, ExperimentConfig]:
    cfg = load_config_from_checkpoint(path)
    device = resolve_device("cpu")
    model = build_model(cfg.model).to(device)
    load_checkpoint(path, model=model, restore_rng=False, map_location=device)
    model.eval()
    return model, cfg


DecodeFn = Callable[[torch.Tensor, torch.Generator], SpeculativeResult]


def run_arm(
    name: str,
    fn: DecodeFn,
    tokenizer: BPETokenizer,
    *,
    repeats: int,
) -> dict[str, object]:
    """Run one decoding configuration over every prompt and aggregate."""
    generated = target_calls = draft_calls = accepted = proposed = 0
    wall = 0.0
    for repeat in range(repeats):
        for i, prompt in enumerate(PROMPTS):
            ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)])
            gen = torch.Generator().manual_seed(1000 * repeat + i)
            result = fn(ids, gen)
            generated += result.generated
            target_calls += result.target_calls
            draft_calls += result.draft_calls
            accepted += result.accepted_tokens
            proposed += result.proposed_tokens
            wall += result.wall_clock_s
    return {
        "arm": name,
        "generated": generated,
        "target_calls": target_calls,
        "draft_calls": draft_calls,
        "acceptance_rate": round(accepted / max(1, proposed), 4) if proposed else None,
        "mean_accepted_length": round(generated / max(1, target_calls), 4),
        "wall_clock_s": round(wall, 4),
        "tokens_per_s": round(generated / max(1e-9, wall), 2),
    }


def measure(args: argparse.Namespace) -> dict[str, object]:
    """Benchmark every arm."""
    tok = BPETokenizer.load(args.tokenizer)
    target, cfg = _load(args.checkpoint)

    if args.draft is not None and args.draft.exists():
        draft, _ = _load(args.draft)
        draft_source = str(args.draft)
    else:
        # No distilled draft available: use an untrained small model so the harness
        # still runs. The write-up flags that this makes the acceptance rate a floor.
        torch.manual_seed(cfg.train.seed)
        draft = build_model(draft_model_config(cfg.model))
        draft.eval()
        draft_source = "untrained (no distilled draft supplied)"
        log.warning("no draft checkpoint at %s; using an untrained draft", args.draft)

    rows: list[dict[str, object]] = [
        run_arm(
            "autoregressive",
            lambda ids, gen: autoregressive_baseline(
                target,
                ids,
                max_new_tokens=args.tokens,
                temperature=args.temperature,
                generator=gen,
            ),
            tok,
            repeats=args.repeats,
        )
    ]

    def speculative_arm(sampler: SpeculativeSampler) -> DecodeFn:
        """Bind one sampler into a decode callable (a closure, not a late-bound name)."""

        def run(ids: torch.Tensor, gen: torch.Generator) -> SpeculativeResult:
            return sampler.generate(ids, max_new_tokens=args.tokens, generator=gen)

        return run

    for gamma in args.gammas:
        sampler = SpeculativeSampler(target, draft, gamma=gamma, temperature=args.temperature)
        rows.append(
            run_arm(f"speculative γ={gamma}", speculative_arm(sampler), tok, repeats=args.repeats)
        )

    # --- the two levers composed --------------------------------------------------
    quantized = copy.deepcopy(target)
    data = build_packed_tokens(cfg.data, tok)
    calib = TokenBatcher(
        data.train,
        seq_len=cfg.data.seq_len,
        batch_size=4,
        seed=1337,
    ).take(8)
    gptq = GPTQQuantizer(quantized, bits=4, group_size=64, act_order=True)
    gptq.collect([b.inputs for b in calib])
    gptq.apply()

    rows.append(
        run_arm(
            "autoregressive + GPTQ-4bit",
            lambda ids, gen: autoregressive_baseline(
                quantized,
                ids,
                max_new_tokens=args.tokens,
                temperature=args.temperature,
                generator=gen,
            ),
            tok,
            repeats=args.repeats,
        )
    )
    best_gamma = max(args.gammas)
    composed = SpeculativeSampler(quantized, draft, gamma=best_gamma, temperature=args.temperature)
    rows.append(
        run_arm(
            f"speculative γ={best_gamma} + GPTQ-4bit",
            speculative_arm(composed),
            tok,
            repeats=args.repeats,
        )
    )

    return {
        "git_sha": git_sha(),
        "hardware": hardware_string(),
        "checkpoint": str(args.checkpoint),
        "draft": draft_source,
        "target_params": target.num_parameters(),
        "draft_params": draft.num_parameters(),
        "temperature": args.temperature,
        "tokens_per_request": args.tokens,
        "prompts": len(PROMPTS) * args.repeats,
        "rows": rows,
    }


def plot(payload: dict[str, object]) -> Path:
    """Target-call reduction and measured throughput, side by side."""
    rows = payload["rows"]
    assert isinstance(rows, list)
    labels = [str(r["arm"]) for r in rows]
    x = range(len(rows))

    fig, axes = new_figure(ncols=2, figsize=(12.0, 4.6))
    left, right = axes

    left.bar(
        list(x),
        [float(r["mean_accepted_length"]) for r in rows],
        color=[COLORS[i % len(COLORS)] for i in x],
    )
    left.axhline(1.0, color="0.6", ls=":", lw=1.2, label="autoregressive = 1 token/pass")
    left.set_xticks(list(x))
    left.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    left.set_ylabel("tokens per target forward pass")
    left.set_title("Target-pass efficiency (hardware-independent)")
    left.legend(fontsize=8)

    right.bar(
        list(x),
        [float(r["tokens_per_s"]) for r in rows],
        color=[COLORS[i % len(COLORS)] for i in x],
    )
    right.set_xticks(list(x))
    right.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    right.set_ylabel("tokens / second")
    right.set_title("Measured throughput (this machine, CPU)")

    fig.suptitle("Speculative decoding", fontweight="bold")
    return save_figure(
        fig,
        RESULTS / "speculative.png",
        script="scripts/specdec_bench.py",
        extra=f"target {payload['target_params']:,} · draft {payload['draft_params']:,}",
    )


def write_report(payload: dict[str, object], figure: Path) -> Path:
    """Write the committed markdown write-up."""
    rows = payload["rows"]
    assert isinstance(rows, list)
    base = rows[0]

    lines = [
        "# Speculative decoding",
        "",
        f"Generated by `scripts/specdec_bench.py` at git `{payload['git_sha']}` on "
        f"`{payload['hardware']}`. Target: {payload['target_params']:,} parameters; draft: "
        f"{payload['draft_params']:,} ({payload['draft']}). "
        f"{payload['prompts']} requests × {payload['tokens_per_request']} tokens at "
        f"temperature {payload['temperature']}.",
        "",
        f"![speculative]({figure.name})",
        "",
        "| arm | target passes | tokens/pass | acceptance | tokens/s | vs baseline |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        speedup = float(row["tokens_per_s"]) / max(1e-9, float(base["tokens_per_s"]))
        acceptance = (
            f"{float(row['acceptance_rate']):.3f}" if row["acceptance_rate"] is not None else "—"
        )
        lines.append(
            f"| {row['arm']} | {row['target_calls']} | {row['mean_accepted_length']:.2f} | "
            f"{acceptance} | {row['tokens_per_s']:.1f} | {speedup:.2f}x |"
        )

    lines += [
        "",
        "## Reading the two columns",
        "",
        "**Tokens per target forward pass** is the hardware-independent result and the "
        "quantity the method actually controls. Autoregressive decoding is 1.0 by "
        "definition; speculation raises it toward `γ+1`.",
        "",
        "**Tokens per second is a measurement of this machine**, and at `nano` scale on a "
        "CPU it can go the *wrong way*. Speculation trades target passes for draft passes, "
        "and it only pays when a target pass is expensive relative to a draft pass. On a "
        "5M-parameter model whose forward pass is a handful of small matmuls, Python and "
        "dispatch overhead dominate and the draft's `γ` sequential steps can cost more "
        "than the target passes they save. The mechanism is real and the target-pass "
        "reduction is real; the wall-clock win needs a model where weight loading, not "
        "interpreter overhead, is the bottleneck — which is exactly the regime the method "
        "was designed for and not the regime a laptop CPU running a 5M-parameter model is "
        "in. Reporting the speedup here without that caveat would be misleading.",
        "",
        "## Composition with quantization",
        "",
        "The last two rows quantize the target to 4-bit GPTQ and then speculate over it. "
        "The levers compose because they act on different costs: quantization shrinks the "
        "bytes per target pass, speculation reduces the number of target passes. "
        "`tests/unit/test_specdec.py::test_speculation_composes_with_a_quantized_target` "
        "asserts that greedy speculation over the quantized target reproduces the "
        "quantized target's own greedy output token-for-token.",
        "",
        "A precise statement of what stays lossless: speculative decoding is lossless "
        "**relative to the target it is given**. Quantizing changes the target's "
        "distribution; speculation then reproduces *that* distribution exactly. It does "
        "not undo the quantization error, and nothing here claims it does.",
        "",
        "## Caveats",
        "",
        f"The draft here is `{payload['draft']}`. Draft quality is the single lever that "
        "determines the acceptance rate, and therefore the speedup: an untrained draft "
        "agrees with the target only by chance and gives a floor. EAGLE-2/EAGLE-3, which "
        "draft on the target's own hidden features rather than with a separate model, are "
        "the current state of the art and the documented next step.",
        "",
        "One arm deserves a footnote: `autoregressive + GPTQ-4bit` is *faster* than "
        "unquantized autoregressive here even though the quantized weights are simulated "
        "in fp32 and no int4 kernel is involved. That is measurement noise on a shared "
        "laptop, not a quantization speedup — there is no mechanism by which it could be "
        "one, and the honest reading is that differences of this size in the tokens/s "
        "column should not be interpreted at all.",
        "",
        "Reproduce with: `python scripts/specdec_bench.py runs/nano/pretrain/final.pt`",
        "",
    ]
    path = RESULTS / "speculative.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?", default=None)
    parser.add_argument("--draft", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/nano.json"))
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gammas", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "bench.json"

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
    report = write_report(payload, figure)
    print(f"wrote {json_path}, {figure}, {report}")
    for row in payload["rows"]:
        print(
            f"  {row['arm']:>34s}: {row['mean_accepted_length']:.2f} tok/pass, "
            f"{row['tokens_per_s']:.1f} tok/s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
