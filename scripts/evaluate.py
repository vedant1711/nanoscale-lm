r"""The full evaluation report for a checkpoint.

Runs every metric the project has and writes one JSON + one markdown fragment. This
replaces "perplexity and a 28-question quiz" with a report that can actually support a
claim:

* **Bits per byte** — tokenizer-independent, so it is comparable to any other model.
* **Token perplexity** — kept for continuity with earlier results, and because it is what
  everyone expects to see.
* **Minimal pairs** — nine grammatical and discourse phenomena, forced choice against a
  50% chance line, with Wilson intervals. The primary capability measurement.
* **Calibration** — ECE and over-confidence, which perplexity cannot see.
* **Generation diversity** — distinct-n, self-BLEU and repetition rate, which catch mode
  collapse that likelihood metrics reward.

Usage::

    python scripts/evaluate.py runs/micro/tinystories/final.pt \\
        --tokenizer artifacts/tokenizer/micro_tinystories.json --name micro
    python scripts/evaluate.py --replay --name micro
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nanoscale.config import GenerateConfig
from nanoscale.eval import (
    bits_per_byte,
    calibration,
    distinct_n,
    perplexity,
    repetition_rate,
    run_minimal_pairs,
    self_bleu,
)
from nanoscale.model import build_model
from nanoscale.serve import generate_text
from nanoscale.tokenizer import BPETokenizer
from nanoscale.train import Batch, TokenBatcher, build_packed_tokens
from nanoscale.train.checkpoint import load_checkpoint, load_config_from_checkpoint
from nanoscale.utils import get_logger, git_sha, hardware_string, resolve_device

log = get_logger("nanoscale.evaluate")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "evaluation"

PROMPTS = (
    "Once upon a time there was a little girl named Mia who",
    "Tom found a shiny red box in the garden. When he opened it,",
    "The sun was setting when Ben walked home. On the way he saw",
    "Lily and her brother wanted to build something. They decided to",
    "It was raining hard. The cat sat by the window and",
    "Anna had a problem she could not solve. She asked her friend",
)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run every metric against one checkpoint."""
    device = resolve_device(args.device)
    cfg = load_config_from_checkpoint(args.checkpoint)
    model = build_model(cfg.model)
    state, _ = load_checkpoint(
        args.checkpoint, model=model, restore_rng=False, map_location=str(device)
    )
    model.to(device).eval()
    tok = BPETokenizer.load(args.tokenizer)
    log.info(
        "%s: %s params, step %s, %s tokens",
        args.name,
        f"{model.num_parameters():,}",
        state.step,
        f"{state.tokens:,}",
    )

    # ---------------------------------------------------------------- likelihood
    data = build_packed_tokens(cfg.data, tok)
    batches: list[Batch] = TokenBatcher(
        data.val, seq_len=cfg.data.seq_len, batch_size=args.batch_size, shuffle=False
    ).take(args.eval_batches)
    ppl = perplexity(model, batches, device=device)

    # Bits-per-byte needs the byte length the eval tokens came from. Decoding the exact
    # evaluated token windows is the only way to get that right — estimating it from a
    # global bytes/token ratio would silently bias the metric.
    eval_bytes = sum(
        len(tok.decode([int(t) for t in row if int(t) >= 0]).encode("utf-8"))
        for b in batches
        for row in b.targets
    )
    bpb = bits_per_byte(model, batches, n_bytes=eval_bytes, device=device)
    cal = calibration(model, batches, device=device)

    # ---------------------------------------------------------------- capability
    pairs = run_minimal_pairs(
        model, tok, n_per_phenomenon=args.pairs_per_phenomenon, seed=1337, device=device
    )

    # ---------------------------------------------------------------- generation
    generations = [
        generate_text(
            model,
            tok,
            prompt,
            GenerateConfig(max_new_tokens=96, temperature=0.8, top_p=0.95, seed=1337 + i),
        ).text
        for i, prompt in enumerate(PROMPTS)
    ]
    diversity = {
        "distinct_1": distinct_n(generations, n=1),
        "distinct_2": distinct_n(generations, n=2),
        "distinct_3": distinct_n(generations, n=3),
        "self_bleu": self_bleu(generations),
        "repetition_rate": sum(repetition_rate(g) for g in generations) / len(generations),
        "mean_length_words": sum(len(g.split()) for g in generations) / len(generations),
    }

    return {
        "name": args.name,
        "git_sha": git_sha(),
        "hardware": hardware_string(),
        "checkpoint": str(args.checkpoint),
        "params": model.num_parameters(),
        "non_embedding_params": cfg.model.param_breakdown()["non_embedding"],
        "vocab_size": cfg.model.vocab_size,
        "step": state.step,
        "train_tokens": state.tokens,
        "chinchilla_fraction": round(state.tokens / (20 * model.num_parameters()), 5),
        "likelihood": {**ppl.summary(), **bpb.summary()},
        "calibration": cal.summary(),
        "minimal_pairs": pairs.summary(),
        "diversity": {k: round(v, 5) for k, v in diversity.items()},
        "samples": [
            {"prompt": p, "completion": g.strip()[:400]}
            for p, g in zip(PROMPTS, generations, strict=True)
        ],
    }


def render(payload: dict[str, Any]) -> str:
    """Write the markdown fragment consumed by the docs."""
    mp = payload["minimal_pairs"]
    lk = payload["likelihood"]
    cal = payload["calibration"]
    div = payload["diversity"]

    lines = [
        f"# Evaluation — {payload['name']}",
        "",
        f"Generated by `scripts/evaluate.py` at git `{payload['git_sha']}` on "
        f"`{payload['hardware']}` from `{payload['checkpoint']}`.",
        "",
        f"{payload['params']:,} parameters "
        f"({payload['non_embedding_params']:,} non-embedding), "
        f"{payload['train_tokens']:,} training tokens "
        f"({payload['chinchilla_fraction'] * 100:.1f}% of the Chinchilla-optimal budget).",
        "",
        "## Likelihood",
        "",
        "| metric | value |",
        "|---|---|",
        f"| bits per byte | **{lk['bits_per_byte']:.4f}** ± {lk['bits_per_byte_stderr']:.4f} |",
        f"| token perplexity | {lk['perplexity']:.4f} "
        f"({lk['perplexity_low']:.4f}–{lk['perplexity_high']:.4f}) |",
        f"| bytes per token | {lk['bytes_per_token']:.3f} |",
        f"| evaluated on | {lk['n_bytes']:,} bytes / {lk['n_tokens']:,} tokens |",
        "",
        "Bits-per-byte is the tokenizer-independent figure and the one to compare against "
        "other models; token perplexity depends on the vocabulary and is reported only for "
        "continuity.",
        "",
        "## Grammatical and discourse competence (minimal pairs)",
        "",
        f"Forced choice between a grammatical sentence and a minimally corrupted one. "
        f"Chance is 50%. {mp['n_items']:,} items across {mp['n_phenomena']} phenomena; "
        f"**{mp['n_above_chance']} of {mp['n_phenomena']}** are significantly above chance.",
        "",
        "| phenomenon | accuracy | 95% CI | n | above chance |",
        "|---|---|---|---|---|",
    ]
    for row in mp["phenomena"]:
        mark = "yes" if row["above_chance"] else "**no**"
        lines.append(
            f"| {row['phenomenon']} | **{row['accuracy'] * 100:.1f}%** | "
            f"{row['ci_low'] * 100:.1f}–{row['ci_high'] * 100:.1f}% | {row['n']} | {mark} |"
        )
    lines += [
        "",
        f"Macro-average across phenomena: **{mp['overall_macro_accuracy'] * 100:.1f}%**.",
        "",
        "## Calibration",
        "",
        "| metric | value |",
        "|---|---|",
        f"| expected calibration error | {cal['ece']:.4f} |",
        f"| maximum calibration error | {cal['mce']:.4f} |",
        f"| top-1 next-token accuracy | {cal['top1_accuracy'] * 100:.1f}% |",
        f"| mean confidence | {cal['mean_confidence'] * 100:.1f}% |",
        f"| over-confidence (conf − acc) | {cal['overconfidence']:+.4f} |",
        "",
        "A positive over-confidence means the model is more certain than it should be, "
        "which is the state that produces confident degenerate generation. Perplexity "
        "cannot distinguish it from being well-calibrated and simply wrong.",
        "",
        "## Generation quality",
        "",
        "| metric | value |",
        "|---|---|",
        f"| distinct-1 | {div['distinct_1']:.4f} |",
        f"| distinct-2 | {div['distinct_2']:.4f} |",
        f"| distinct-3 | {div['distinct_3']:.4f} |",
        f"| self-BLEU | {div['self_bleu']:.4f} |",
        f"| repetition rate | {div['repetition_rate']:.4f} |",
        f"| mean length (words) | {div['mean_length_words']:.1f} |",
        "",
        "Low distinct-n or high self-BLEU means the model produces the same thing "
        "regardless of prompt. Both are reported because a likelihood metric rewards "
        "exactly the behaviour they detect.",
        "",
        "## Samples",
        "",
    ]
    for s in payload["samples"][:3]:
        lines += [f"> **{s['prompt']}**{s['completion']}", ""]

    lines.append(f"Reproduce with: `python scripts/evaluate.py --replay --name {payload['name']}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?", default=None)
    parser.add_argument("--name", default="micro")
    parser.add_argument(
        "--tokenizer", type=Path, default=ROOT / "artifacts/tokenizer/micro_tinystories.json"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--pairs-per-phenomenon", type=int, default=100)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / f"{args.name}.json"

    if args.replay:
        if not json_path.exists():
            raise SystemExit(f"no committed results at {json_path}; run without --replay.")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        if args.checkpoint is None:
            raise SystemExit("a checkpoint is required unless --replay is given.")
        payload = evaluate(args)
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md = RESULTS / f"{args.name}.md"
    md.write_text(render(payload), encoding="utf-8")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
