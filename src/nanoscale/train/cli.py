"""``nanoscale train`` sub-app: pretraining and generation from a checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import torch
import typer

from nanoscale.config import load_experiment
from nanoscale.model import build_model
from nanoscale.tokenizer import BPETokenizer
from nanoscale.train.checkpoint import load_checkpoint
from nanoscale.train.loop import Trainer
from nanoscale.utils import get_logger, resolve_device, temporary_seed

train_app = typer.Typer(name="train", help="Pretrain a model and sample from checkpoints.")
log = get_logger("nanoscale.train.cli")

DEFAULT_TOKENIZER = Path("artifacts/tokenizer/nano.json")


@train_app.command("pretrain")
def pretrain(
    config: Annotated[Path | None, typer.Option("--config", "-c", help="YAML config.")] = None,
    tier: Annotated[str | None, typer.Option("--tier", "-t", help="Size-ladder tier.")] = None,
    set_: Annotated[
        list[str] | None, typer.Option("--set", "-s", help="Config override, key.path=value.")
    ] = None,
    tokenizer_path: Annotated[
        Path, typer.Option("--tokenizer", help="Trained tokenizer JSON.")
    ] = DEFAULT_TOKENIZER,
    out_dir: Annotated[
        Path | None, typer.Option("--out", "-o", help="Run directory (overrides the config).")
    ] = None,
    resume: Annotated[
        Path | None, typer.Option("--resume", help="Checkpoint to resume from.")
    ] = None,
    run_name: Annotated[
        str | None, typer.Option("--name", help="Run name for the manifest.")
    ] = None,
) -> None:
    """Pretrain a model from scratch."""
    cfg = load_experiment(config, tier=tier, overrides=list(set_ or []))
    tok = BPETokenizer.load(tokenizer_path) if tokenizer_path.exists() else None
    if tok is None:
        raise typer.BadParameter(
            f"no tokenizer at {tokenizer_path}; run `nanoscale tokenizer train` first."
        )
    if tok.vocab_size != cfg.model.vocab_size:
        raise typer.BadParameter(
            f"tokenizer vocab {tok.vocab_size} != model vocab {cfg.model.vocab_size}."
        )

    trainer = Trainer(cfg, tokenizer=tok, out_dir=out_dir, run_name=run_name)
    trainer.maybe_resume(resume)
    result = trainer.train()
    typer.echo(json.dumps(result.summary(), indent=2))


@train_app.command("generate")
def generate(
    checkpoint: Annotated[Path, typer.Argument(help="Checkpoint to sample from.")],
    prompt: Annotated[str, typer.Option("--prompt", "-p", help="Prompt text.")] = "Lily went to",
    tokenizer_path: Annotated[
        Path, typer.Option("--tokenizer", help="Trained tokenizer JSON.")
    ] = DEFAULT_TOKENIZER,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", "-n")] = 64,
    temperature: Annotated[float, typer.Option("--temperature")] = 0.8,
    top_k: Annotated[int, typer.Option("--top-k")] = 0,
    top_p: Annotated[float, typer.Option("--top-p")] = 0.95,
    seed: Annotated[int, typer.Option("--seed")] = 1337,
    samples: Annotated[int, typer.Option("--samples", help="Number of completions.")] = 1,
) -> None:
    """Sample a continuation from a trained checkpoint."""
    from nanoscale.train.checkpoint import load_config_from_checkpoint

    cfg = load_config_from_checkpoint(checkpoint)
    tok = BPETokenizer.load(tokenizer_path)
    device = resolve_device(cfg.train.device)
    model = build_model(cfg.model).to(device)
    load_checkpoint(checkpoint, model=model, restore_rng=False, map_location=device)
    model.eval()

    ids = torch.tensor([tok.encode(prompt, add_bos=True)], device=device)
    for i in range(samples):
        with temporary_seed(seed + i):
            gen = torch.Generator(device="cpu").manual_seed(seed + i)
            out = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_id=tok.eos_id,
                generator=gen,
            )
        typer.echo(f"--- sample {i + 1} ---")
        typer.echo(tok.decode(out[0].tolist(), skip_special=True))
