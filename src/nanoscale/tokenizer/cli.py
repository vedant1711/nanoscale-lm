"""``nanoscale tokenizer`` sub-app: train, inspect, encode and decode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from nanoscale.config import load_experiment
from nanoscale.data.toy import generate_corpus
from nanoscale.tokenizer.bpe import BPETokenizer
from nanoscale.utils import get_logger

tokenizer_app = typer.Typer(name="tokenizer", help="Train and use the byte-level BPE tokenizer.")
log = get_logger("nanoscale.tokenizer.cli")

DEFAULT_PATH = Path("artifacts/tokenizer/nano.json")


@tokenizer_app.command("train")
def train(
    out: Annotated[Path, typer.Option("--out", "-o", help="Destination JSON path.")] = DEFAULT_PATH,
    tier: Annotated[str, typer.Option("--tier", "-t", help="Size-ladder tier.")] = "nano",
    config: Annotated[Path | None, typer.Option("--config", "-c", help="YAML config.")] = None,
    corpus: Annotated[
        Path | None,
        typer.Option("--corpus", help="Text file to train on. Defaults to the offline toy corpus."),
    ] = None,
    stories: Annotated[
        int, typer.Option("--stories", help="Toy-corpus story count when --corpus is absent.")
    ] = 8000,
    seed: Annotated[int, typer.Option("--seed", help="Toy-corpus seed.")] = 1337,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log merge progress.")] = False,
) -> None:
    """Train a byte-level BPE tokenizer and write it to JSON."""
    cfg = load_experiment(config, tier=tier).tokenizer
    text = (
        corpus.read_text(encoding="utf-8")
        if corpus is not None
        else generate_corpus(seed=seed, n_stories=stories)
    )
    log.info(
        "training BPE: target vocab %d (%d merges) on %.2f MB",
        cfg.vocab_size,
        cfg.n_merges,
        len(text.encode("utf-8")) / 1e6,
    )
    tok = BPETokenizer.train(text, cfg, verbose=verbose)
    path = tok.save(out)
    sample = text[:200_000]
    typer.echo(f"merges learned    {tok.n_merges} / {cfg.n_merges}")
    typer.echo(f"compression       {tok.compression_ratio(sample):.3f} bytes/token")
    typer.echo(f"wrote             {path}")


@tokenizer_app.command("info")
def info(
    path: Annotated[Path, typer.Argument(help="Tokenizer JSON path.")] = DEFAULT_PATH,
    show: Annotated[
        int, typer.Option("--show", help="Print this many of the longest tokens.")
    ] = 20,
) -> None:
    """Print vocabulary statistics for a trained tokenizer."""
    tok = BPETokenizer.load(path)
    typer.echo(repr(tok))
    typer.echo(f"specials          {sorted(tok.special_to_id.items(), key=lambda kv: kv[1])}")
    longest = sorted(
        ((tid, piece) for tid, piece in tok.vocab.items() if tid not in tok.id_to_special),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )[:show]
    typer.echo(f"longest {show} tokens:")
    for tid, piece in longest:
        typer.echo(f"  {tid:>6}  {piece.decode('utf-8', errors='backslashreplace')!r}")


@tokenizer_app.command("encode")
def encode(
    text: Annotated[str, typer.Argument(help="Text to encode.")],
    path: Annotated[Path, typer.Option("--tokenizer", help="Tokenizer JSON.")] = DEFAULT_PATH,
    show_tokens: Annotated[bool, typer.Option("--tokens", help="Also print token strings.")] = True,
) -> None:
    """Encode text and print the token IDs."""
    tok = BPETokenizer.load(path)
    ids = tok.encode(text)
    typer.echo(json.dumps(ids))
    if show_tokens:
        typer.echo(json.dumps([tok.token_repr(i) for i in ids], ensure_ascii=False))
    n_bytes = len(text.encode("utf-8"))
    ratio = n_bytes / len(ids) if ids else 0.0
    typer.echo(f"{n_bytes} bytes -> {len(ids)} tokens ({ratio:.2f} bytes/token)")


@tokenizer_app.command("decode")
def decode(
    ids: Annotated[str, typer.Argument(help="JSON list of token IDs.")],
    path: Annotated[Path, typer.Option("--tokenizer", help="Tokenizer JSON.")] = DEFAULT_PATH,
) -> None:
    """Decode a JSON list of token IDs back to text."""
    tok = BPETokenizer.load(path)
    typer.echo(tok.decode(json.loads(ids)))
