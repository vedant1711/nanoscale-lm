"""``nanoscale align`` sub-app: SFT, DPO/SimPO and the optional GRPO track."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from nanoscale.align.grpo import GRPOTrainer
from nanoscale.align.preference import PreferenceTrainer
from nanoscale.align.sft import SFTTrainer
from nanoscale.config import ExperimentConfig, load_experiment
from nanoscale.model import NanoScaleLM, build_model
from nanoscale.tokenizer import BPETokenizer
from nanoscale.train.checkpoint import load_checkpoint, load_config_from_checkpoint
from nanoscale.utils import get_logger, resolve_device

align_app = typer.Typer(name="align", help="Post-training: SFT, DPO, SimPO, GRPO.")
log = get_logger("nanoscale.align.cli")

DEFAULT_TOKENIZER = Path("artifacts/tokenizer/nano.json")

CheckpointArg = Annotated[Path, typer.Argument(help="Checkpoint to start from.")]
TokenizerOpt = Annotated[Path, typer.Option("--tokenizer", help="Trained tokenizer JSON.")]
SetOpt = Annotated[
    list[str] | None, typer.Option("--set", "-s", help="Config override, key.path=value.")
]


def _load(
    checkpoint: Path, tokenizer_path: Path, tier: str | None, overrides: list[str]
) -> tuple[BPETokenizer, ExperimentConfig, NanoScaleLM]:
    """Load the tokenizer, the experiment config and a model initialised from a checkpoint."""
    tok = BPETokenizer.load(tokenizer_path)
    try:
        cfg = load_config_from_checkpoint(checkpoint)
    except KeyError:
        cfg = load_experiment(tier=tier or "nano")
    if overrides:
        cfg = load_experiment(tier=cfg.name, overrides=overrides)
    device = resolve_device(cfg.train.device)
    model = build_model(cfg.model).to(device)
    load_checkpoint(checkpoint, model=model, restore_rng=False, map_location=device)
    return tok, cfg, model


@align_app.command("sft")
def sft(
    checkpoint: CheckpointArg,
    tokenizer_path: TokenizerOpt = DEFAULT_TOKENIZER,
    tier: Annotated[str | None, typer.Option("--tier", "-t")] = None,
    set_: SetOpt = None,
    out_dir: Annotated[Path | None, typer.Option("--out", "-o")] = None,
) -> None:
    """Instruction-tune a pretrained checkpoint with completion-masked loss."""
    tok, cfg, model = _load(checkpoint, tokenizer_path, tier, list(set_ or []))
    trainer = SFTTrainer(model, tok, cfg.align.sft, out_dir=out_dir)
    typer.echo(json.dumps(trainer.train().summary(), indent=2))


@align_app.command("preference")
def preference(
    checkpoint: CheckpointArg,
    method: Annotated[str, typer.Option("--method", "-m", help="dpo or simpo.")] = "dpo",
    tokenizer_path: TokenizerOpt = DEFAULT_TOKENIZER,
    tier: Annotated[str | None, typer.Option("--tier", "-t")] = None,
    set_: SetOpt = None,
    out_dir: Annotated[Path | None, typer.Option("--out", "-o")] = None,
) -> None:
    """Run preference optimization (DPO or SimPO) from an SFT checkpoint."""
    if method not in ("dpo", "simpo"):
        raise typer.BadParameter("method must be 'dpo' or 'simpo'.")
    overrides = [*(set_ or []), f"align.preference.method={method}"]
    tok, cfg, model = _load(checkpoint, tokenizer_path, tier, overrides)
    trainer = PreferenceTrainer(model, tok, cfg.align.preference, out_dir=out_dir)
    typer.echo(json.dumps(trainer.train().summary(), indent=2))


@align_app.command("grpo")
def grpo(
    checkpoint: CheckpointArg,
    tokenizer_path: TokenizerOpt = DEFAULT_TOKENIZER,
    tier: Annotated[str | None, typer.Option("--tier", "-t")] = None,
    set_: SetOpt = None,
    out_dir: Annotated[Path | None, typer.Option("--out", "-o")] = None,
) -> None:
    """Run the optional GRPO-RLVR track on verifiable arithmetic."""
    tok, cfg, model = _load(checkpoint, tokenizer_path, tier, list(set_ or []))
    trainer = GRPOTrainer(model, tok, cfg.align.grpo, out_dir=out_dir)
    typer.echo(json.dumps(trainer.train().summary(), indent=2))
