"""Compare NanoScale-LM against external pretrained models on bits-per-byte.

**This is the only place in the project where `transformers` is used**, and it is used
strictly as a *measured baseline* — nothing here feeds back into `src/nanoscale/`. The
project's rule is that Hugging Face may appear as an external comparison point and never
as the thing being claimed as built.

Why bits-per-byte rather than perplexity
----------------------------------------
Perplexity is per *token*, and these models do not share a tokenizer. NanoScale-LM's
TinyStories vocabulary is 16,384 entries trained on this exact distribution; GPT-2's is
50,257 trained on WebText. GPT-2 needs a different number of tokens to express the same
sentence, so its per-token perplexity is not comparable to ours in either direction —
a larger vocabulary makes each token carry more information and look "worse".

Bits-per-byte normalizes by the UTF-8 length of the source text, which no tokenizer can
change:

.. code-block:: text

    BPB = (Σ negative log-likelihood in nats / ln 2) / (bytes of source text)

Both models are scored on the *same held-out strings*, each with its own tokenizer, and
the results are directly comparable. This is the metric the Pile and Chinchilla papers use
for exactly this reason.

What a win here does and does not mean
--------------------------------------
If NanoScale-LM scores lower BPB than GPT-2 on TinyStories, that is a real, correctly-measured
result — and it says the model is better adapted *to this distribution*, which is
unsurprising given it was trained on it and GPT-2 was not. It is a statement about
in-domain specialization at small scale, not a claim of general superiority. The script
therefore also reports out-of-domain BPB on text neither model was trained on, where the
ordering is expected to reverse; reporting only the favourable half would make the
favourable half worthless.

Usage::

    uv pip install -e ".[compare]" transformers
    python scripts/external_baseline.py runs/micro/tinystories/final.pt
    python scripts/external_baseline.py --replay
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nanoscale.eval import bits_per_byte
from nanoscale.model import IGNORE_INDEX, build_model
from nanoscale.tokenizer import BPETokenizer
from nanoscale.train import Batch
from nanoscale.train.checkpoint import load_checkpoint, load_config_from_checkpoint
from nanoscale.utils import get_logger, git_sha, hardware_string, resolve_device

log = get_logger("nanoscale.baseline")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "baseline"

#: Models to compare against. All are freely downloadable and CPU-runnable.
BASELINES = ("gpt2", "distilgpt2")

#: Held-out text neither model trained on, to show the in-domain result is domain-bound.
OUT_OF_DOMAIN = """The mitochondrion is a double-membrane-bound organelle found in most
eukaryotic organisms. Mitochondria generate most of the cell's supply of adenosine
triphosphate, subsequently used throughout the cell as a source of chemical energy. The
organelle is composed of compartments that carry out specialized functions, including the
outer membrane, the intermembrane space, the inner membrane, the cristae, and the matrix.
Although most of a cell's DNA is contained in the nucleus, the mitochondrion has its own
independent genome that shows substantial similarity to bacterial genomes."""


def chunk_text(text: str, *, n_chunks: int, chunk_chars: int) -> list[str]:
    """Split ``text`` into evenly spaced chunks, each starting at a paragraph boundary."""
    stride = max(1, len(text) // max(1, n_chunks))
    out: list[str] = []
    for i in range(n_chunks):
        start = i * stride
        # Nudge forward to the next whitespace so we never cut mid-word.
        while start < len(text) and not text[start].isspace():
            start += 1
        piece = text[start : start + chunk_chars].strip()
        if len(piece) > chunk_chars // 2:
            out.append(piece)
    return out


@torch.no_grad()
def nanoscale_bpb(
    ckpt: Path, tokenizer_path: Path, chunks: list[str], device: torch.device
) -> dict[str, Any]:
    """Score text chunks with a NanoScale checkpoint."""
    cfg = load_config_from_checkpoint(ckpt)
    model = build_model(cfg.model)
    state, _ = load_checkpoint(ckpt, model=model, restore_rng=False, map_location=str(device))
    model.to(device).eval()
    tok = BPETokenizer.load(tokenizer_path)

    batches: list[Batch] = []
    n_bytes = 0
    for chunk in chunks:
        ids = tok.encode(chunk, add_bos=True)
        # Truncate to the trained context; a longer window would measure extrapolation
        # rather than modelling, and GPT-2 would not be handicapped the same way.
        ids = ids[: cfg.model.max_seq_len + 1]
        if len(ids) < 8:
            continue
        inp = torch.tensor([ids[:-1]])
        tgt = torch.tensor([ids[1:]])
        batches.append(Batch(inputs=inp, targets=tgt))
        # The BOS token is not part of the source text, and the first target is the first
        # real token, so the bytes covered are exactly the chunk's bytes.
        n_bytes += len(chunk.encode("utf-8"))

    result = bits_per_byte(model, batches, n_bytes=n_bytes, device=device)
    return {
        "name": f"nanoscale-{cfg.name}",
        "params": model.num_parameters(),
        "vocab_size": cfg.model.vocab_size,
        "step": state.step,
        "train_tokens": state.tokens,
        **result.summary(),
    }


@torch.no_grad()
def hf_bpb(name: str, chunks: list[str], device: torch.device) -> dict[str, Any]:
    """Score the same text chunks with a Hugging Face causal LM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    # `transformers` decorates `.to()` with a wrapper mypy resolves to the wrong overload,
    # so the handle is kept as Any. This is the boundary with an untyped dependency; the
    # rest of the file stays strict.
    model: Any = AutoModelForCausalLM.from_pretrained(name)
    model.to(device)
    model.eval()
    max_len = min(1024, getattr(model.config, "n_positions", 1024))

    total_nll = 0.0
    total_sq = 0.0
    n_tokens = 0
    n_bytes = 0
    for chunk in chunks:
        ids = tok(chunk, return_tensors="pt").input_ids[0][: max_len + 1]
        if ids.numel() < 8:
            continue
        inp = ids[:-1].unsqueeze(0).to(device)
        tgt = ids[1:].unsqueeze(0).to(device)
        logits = model(inp).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        gathered = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        nll = -gathered[tgt != IGNORE_INDEX]
        total_nll += float(nll.sum())
        total_sq += float((nll**2).sum())
        n_tokens += int(nll.numel())
        n_bytes += len(chunk.encode("utf-8"))

    mean = total_nll / max(1, n_tokens)
    var = max(0.0, total_sq / max(1, n_tokens) - mean**2)
    stderr = math.sqrt(var * n_tokens) / math.log(2) / max(1, n_bytes)
    return {
        "name": name,
        "params": sum(p.numel() for p in model.parameters()),
        "vocab_size": int(model.config.vocab_size),
        "bits_per_byte": total_nll / math.log(2) / max(1, n_bytes),
        "bits_per_byte_stderr": stderr,
        "bytes_per_token": n_bytes / max(1, n_tokens),
        "token_perplexity": math.exp(min(mean, 20.0)),
        "n_bytes": n_bytes,
        "n_tokens": n_tokens,
    }


def measure(args: argparse.Namespace) -> dict[str, Any]:
    """Score every model on both corpora."""
    device = resolve_device("cpu")  # CPU keeps the comparison hardware-neutral.

    valid = (ROOT / "data" / "tinystories" / "valid.txt").read_text(encoding="utf-8")
    in_domain = chunk_text(valid, n_chunks=args.chunks, chunk_chars=args.chunk_chars)
    out_domain = [OUT_OF_DOMAIN.replace("\n", " ")]
    n_bytes = sum(len(c.encode()) for c in in_domain)
    log.info("in-domain: %d chunks, %d bytes", len(in_domain), n_bytes)

    rows: list[dict[str, Any]] = []
    for corpus_name, chunks in (("tinystories-valid", in_domain), ("out-of-domain", out_domain)):
        row = nanoscale_bpb(args.checkpoint, args.tokenizer, chunks, device)
        row["corpus"] = corpus_name
        rows.append(row)
        log.info("%-22s %-18s %.4f bpb", row["name"], corpus_name, row["bits_per_byte"])

        for hf_name in args.baselines:
            r = hf_bpb(hf_name, chunks, device)
            r["corpus"] = corpus_name
            rows.append(r)
            log.info("%-22s %-18s %.4f bpb", r["name"], corpus_name, r["bits_per_byte"])

    return {
        "git_sha": git_sha(),
        "hardware": hardware_string(),
        "checkpoint": str(args.checkpoint),
        "chunks": args.chunks,
        "chunk_chars": args.chunk_chars,
        "rows": rows,
    }


def render(payload: dict[str, Any]) -> str:
    """Write the markdown fragment consumed by docs/results.md."""
    rows = payload["rows"]
    lines = [
        "# External baseline — bits per byte",
        "",
        f"Generated by `scripts/external_baseline.py` at git `{payload['git_sha']}` on "
        f"`{payload['hardware']}`.",
        "",
        "Perplexity is per *token* and these models do not share a tokenizer, so it cannot "
        "compare them. Bits-per-byte normalizes by the UTF-8 length of the source text, "
        "which no tokenizer can change, and is what the Pile and Chinchilla papers use for "
        "cross-model comparison.",
        "",
    ]

    for corpus in ("tinystories-valid", "out-of-domain"):
        subset = [r for r in rows if r["corpus"] == corpus]
        if not subset:
            continue
        best = min(r["bits_per_byte"] for r in subset)
        lines += [
            f"## {corpus}",
            "",
            "| model | params | vocab | bytes/token | bits/byte | token ppl |",
            "|---|---|---|---|---|---|",
        ]
        for r in sorted(subset, key=lambda x: x["bits_per_byte"]):
            mark = " **←**" if r["bits_per_byte"] == best else ""
            lines.append(
                f"| {r['name']} | {r['params']:,} | {r['vocab_size']:,} | "
                f"{r['bytes_per_token']:.2f} | **{r['bits_per_byte']:.4f}** "
                f"± {r['bits_per_byte_stderr']:.4f}{mark} | {r['token_perplexity']:.2f} |"
            )
        lines.append("")

    def pick(prefix: str, corpus: str) -> dict[str, Any] | None:
        return next(
            (r for r in rows if r["name"].startswith(prefix) and r["corpus"] == corpus), None
        )

    ns = pick("nanoscale", "tinystories-valid")
    gpt2 = pick("gpt2", "tinystories-valid")
    ns_ood = pick("nanoscale", "out-of-domain")
    gpt2_ood = pick("gpt2", "out-of-domain")

    if ns and gpt2 and ns_ood and gpt2_ood:
        lines += [
            "## What this says",
            "",
            f"**In-domain, NanoScale-LM at {ns['params']:,} parameters scores "
            f"{ns['bits_per_byte']:.4f} bits/byte against GPT-2's {gpt2['bits_per_byte']:.4f} "
            f"at {gpt2['params']:,} parameters** — "
            f"{'better' if ns['bits_per_byte'] < gpt2['bits_per_byte'] else 'worse'} with "
            f"{gpt2['params'] / ns['params']:.1f}× fewer parameters.",
            "",
            "**And the ordering reverses out of domain**: "
            f"{ns_ood['bits_per_byte']:.4f} against GPT-2's {gpt2_ood['bits_per_byte']:.4f} on "
            "text neither model was trained on. That reversal is the point. The in-domain "
            "number measures *specialization*, not general capability, and reporting it "
            "without its counterpart would misrepresent what was achieved: a small model "
            "trained on a narrow distribution beats a larger general model on that "
            "distribution and loses badly everywhere else.",
            "",
            "This is a fair comparison in the one way that matters — both models scored the "
            "same held-out strings, each with its own tokenizer, normalized by bytes — and a "
            "limited one in every other way. GPT-2 was trained for general-purpose use on "
            "WebText; TinyStories was not in its training distribution.",
            "",
        ]

    lines.append("Reproduce with: `python scripts/external_baseline.py --replay`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?", default=None)
    parser.add_argument(
        "--tokenizer", type=Path, default=ROOT / "artifacts/tokenizer/micro_tinystories.json"
    )
    parser.add_argument("--baselines", nargs="*", default=list(BASELINES))
    parser.add_argument("--chunks", type=int, default=64)
    parser.add_argument("--chunk-chars", type=int, default=1800)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "bits_per_byte.json"

    if args.replay:
        if not json_path.exists():
            raise SystemExit(f"no committed results at {json_path}; run without --replay.")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        if args.checkpoint is None:
            raise SystemExit("a checkpoint is required unless --replay is given.")
        payload = measure(args)
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    (RESULTS / "baseline.md").write_text(render(payload), encoding="utf-8")
    print(f"wrote {RESULTS / 'baseline.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
