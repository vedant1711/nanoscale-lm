"""``nanoscale serve`` sub-app: generation, chat and the unified benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import torch
import typer

from nanoscale.config import ExperimentConfig, GenerateConfig
from nanoscale.eval import perplexity, run_tiny_bench
from nanoscale.model import NanoScaleLM, build_model
from nanoscale.serve.generate import generate_text, stream_text
from nanoscale.tokenizer import BPETokenizer, Message, render_prompt
from nanoscale.train import TokenBatcher, build_packed_tokens
from nanoscale.train.checkpoint import load_checkpoint, load_config_from_checkpoint
from nanoscale.utils import get_logger, resolve_device

serve_app = typer.Typer(name="serve", help="Generate, chat and evaluate from a checkpoint.")
log = get_logger("nanoscale.serve.cli")

DEFAULT_TOKENIZER = Path("artifacts/tokenizer/nano.json")
CheckpointArg = Annotated[Path, typer.Argument(help="Checkpoint to load.")]
TokenizerOpt = Annotated[Path, typer.Option("--tokenizer", help="Trained tokenizer JSON.")]


def _load(
    checkpoint: Path, tokenizer_path: Path
) -> tuple[NanoScaleLM, BPETokenizer, ExperimentConfig]:
    """Load the model, tokenizer and embedded config from a checkpoint."""
    tok = BPETokenizer.load(tokenizer_path)
    cfg = load_config_from_checkpoint(checkpoint)
    device = resolve_device("cpu")
    model = build_model(cfg.model).to(device)
    load_checkpoint(checkpoint, model=model, restore_rng=False, map_location=device)
    model.eval()
    return model, tok, cfg


@serve_app.command("generate")
def generate(
    checkpoint: CheckpointArg,
    prompt: Annotated[str, typer.Option("--prompt", "-p")] = "It was a sunny day.",
    tokenizer_path: TokenizerOpt = DEFAULT_TOKENIZER,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", "-n")] = 64,
    temperature: Annotated[float, typer.Option("--temperature")] = 0.8,
    top_p: Annotated[float, typer.Option("--top-p")] = 0.95,
    top_k: Annotated[int, typer.Option("--top-k")] = 0,
    repetition_penalty: Annotated[float, typer.Option("--repetition-penalty")] = 1.0,
    seed: Annotated[int, typer.Option("--seed")] = 1337,
    stream: Annotated[bool, typer.Option("--stream/--no-stream")] = True,
) -> None:
    """Generate a completion, streaming tokens as they are produced."""
    model, tok, _ = _load(checkpoint, tokenizer_path)
    cfg = GenerateConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        seed=seed,
    )
    typer.echo(prompt, nl=False)
    if stream:
        for piece in stream_text(model, tok, prompt, cfg):
            sys.stdout.write(piece)
            sys.stdout.flush()
        sys.stdout.write("\n")
    else:
        out = generate_text(model, tok, prompt, cfg)
        typer.echo(out.text)
        typer.echo(json.dumps(out.summary(), indent=2))


@serve_app.command("chat")
def chat(
    checkpoint: CheckpointArg,
    tokenizer_path: TokenizerOpt = DEFAULT_TOKENIZER,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", "-n")] = 64,
    temperature: Annotated[float, typer.Option("--temperature")] = 0.7,
    seed: Annotated[int, typer.Option("--seed")] = 1337,
) -> None:
    """Interactive chat against an aligned checkpoint. Ctrl-D or 'exit' to quit."""
    model, tok, _ = _load(checkpoint, tokenizer_path)
    cfg = GenerateConfig(max_new_tokens=max_new_tokens, temperature=temperature, seed=seed)
    history: list[Message] = []
    typer.echo("NanoScale-LM chat. Type 'exit' to quit.\n")
    while True:
        try:
            user = input("you> ").strip()
        except EOFError:
            break
        if not user or user.lower() in ("exit", "quit"):
            break
        history.append(Message("user", user))
        ids = render_prompt(tok, history)
        typer.echo("bot> ", nl=False)
        pieces: list[str] = []
        for piece in stream_text(model, tok, "", cfg, prompt_ids=ids):
            pieces.append(piece)
            sys.stdout.write(piece)
            sys.stdout.flush()
        sys.stdout.write("\n\n")
        history.append(Message("assistant", "".join(pieces)))


@serve_app.command("eval")
def evaluate(
    checkpoint: CheckpointArg,
    tokenizer_path: TokenizerOpt = DEFAULT_TOKENIZER,
    eval_batches: Annotated[int, typer.Option("--batches")] = 32,
) -> None:
    """Report perplexity (with error bars) and the tiny-benchmark accuracy."""
    model, tok, cfg = _load(checkpoint, tokenizer_path)
    data = build_packed_tokens(cfg.data, tok)
    batches = TokenBatcher(data.val, seq_len=cfg.data.seq_len, batch_size=4, shuffle=False).take(
        eval_batches
    )

    ppl = perplexity(model, batches, device=torch.device("cpu"))
    bench = run_tiny_bench(model, tok)
    typer.echo(json.dumps({**ppl.summary(), **bench.summary()}, indent=2))
