"""``nanoscale`` command-line interface (typer).

The CLI is the single entry point for every phase of the project. Sub-apps are
registered as phases land, so ``nanoscale --help`` always reflects what is actually
implemented rather than a promise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from nanoscale import __version__
from nanoscale.align.cli import align_app
from nanoscale.config import (
    TIERS,
    export_json_schemas,
    get_preset,
    load_experiment,
    save_experiment,
)
from nanoscale.serve.cli import serve_app
from nanoscale.tokenizer.cli import tokenizer_app
from nanoscale.train.cli import train_app
from nanoscale.utils import get_logger, hardware_string

app = typer.Typer(
    name="nanoscale",
    help=(
        "NanoScale-LM: a small language model built from scratch (Arc 1) and made "
        "deployable (Arc 2). Every algorithm in this CLI is implemented in-repo."
    ),
    add_completion=False,
)
config_app = typer.Typer(name="config", help="Inspect, resolve and export configurations.")
app.add_typer(config_app)
app.add_typer(tokenizer_app)
app.add_typer(train_app)
app.add_typer(align_app)
app.add_typer(serve_app)

log = get_logger("nanoscale.cli")

ConfigOpt = Annotated[
    Path | None, typer.Option("--config", "-c", help="YAML config file (extends a tier preset).")
]
TierOpt = Annotated[
    str | None, typer.Option("--tier", "-t", help=f"Size-ladder tier: one of {', '.join(TIERS)}.")
]
SetOpt = Annotated[
    list[str] | None,
    typer.Option("--set", "-s", help="Override a config field, e.g. --set train.max_steps=10."),
]


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", help="Print the version and exit.")] = False,
) -> None:
    """NanoScale-LM command line."""
    if version:
        typer.echo(f"nanoscale-lm {__version__}")
        raise typer.Exit
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit


@app.command()
def info() -> None:
    """Print environment, hardware and the size-ladder table."""
    import torch

    typer.echo(f"nanoscale-lm  {__version__}")
    typer.echo(f"torch         {torch.__version__}")
    typer.echo(f"hardware      {hardware_string()}")
    typer.echo("")
    header = f"{'tier':<8} {'params':>12} {'layers':>7} {'d_model':>8} {'heads':>6} "
    header += f"{'kv':>4} {'ctx':>6} {'token budget':>14}"
    typer.echo(header)
    typer.echo("-" * len(header))
    for tier in TIERS:
        cfg = get_preset(tier)
        m = cfg.model
        budget = cfg.train.token_budget or 0
        typer.echo(
            f"{tier:<8} {m.param_count():>12,} {m.n_layers:>7} {m.d_model:>8} "
            f"{m.n_heads:>6} {m.n_kv_heads:>4} {m.max_seq_len:>6} {budget:>14,}"
        )


@config_app.command("show")
def config_show(
    config: ConfigOpt = None,
    tier: TierOpt = None,
    set_: SetOpt = None,
    section: Annotated[
        str | None, typer.Option("--section", help="Only print this top-level section.")
    ] = None,
) -> None:
    """Resolve a configuration (preset + YAML + overrides) and print it as JSON."""
    cfg = load_experiment(config, tier=tier, overrides=list(set_ or []))
    payload = cfg.model_dump(mode="json")
    if section is not None:
        if section not in payload:
            raise typer.BadParameter(f"No section {section!r}; have {sorted(payload)}.")
        payload = {section: payload[section]}
    typer.echo(json.dumps(payload, indent=2, default=str))


@config_app.command("hash")
def config_hash(config: ConfigOpt = None, tier: TierOpt = None, set_: SetOpt = None) -> None:
    """Print the stable config hash recorded in run manifests."""
    cfg = load_experiment(config, tier=tier, overrides=list(set_ or []))
    typer.echo(cfg.config_hash())


@config_app.command("save")
def config_save(
    out: Annotated[Path, typer.Argument(help="Destination YAML path.")],
    config: ConfigOpt = None,
    tier: TierOpt = None,
    set_: SetOpt = None,
) -> None:
    """Write a fully-resolved configuration to YAML."""
    cfg = load_experiment(config, tier=tier, overrides=list(set_ or []))
    path = save_experiment(cfg, out)
    typer.echo(f"wrote {path}")


@config_app.command("schema")
def config_schema(
    out_dir: Annotated[Path, typer.Argument(help="Directory for the JSON Schema files.")],
) -> None:
    """Export the JSON Schema of every config model."""
    written = export_json_schemas(out_dir)
    typer.echo(f"wrote {len(written)} schema files to {out_dir}")


@config_app.command("params")
def config_params(config: ConfigOpt = None, tier: TierOpt = None, set_: SetOpt = None) -> None:
    """Print the analytic parameter breakdown for a model configuration."""
    cfg = load_experiment(config, tier=tier, overrides=list(set_ or []))
    m = cfg.model
    embed = m.vocab_size * m.d_model
    head = 0 if m.tie_embeddings else m.vocab_size * m.d_model
    total = m.param_count()
    typer.echo(f"tier              {cfg.name}")
    typer.echo(f"total params      {total:,}")
    typer.echo(f"  embeddings      {embed:,}")
    typer.echo(f"  lm head         {head:,}")
    typer.echo(f"  blocks          {total - embed - head - m.d_model:,}")
    typer.echo(f"non-embedding     {total - embed - head:,}")
    typer.echo(f"ffn_dim           {m.ffn_dim}")
    typer.echo(f"head_dim          {m.head_dim}")
    typer.echo(f"token budget 20:1 {total * 20:,}")


if __name__ == "__main__":  # pragma: no cover
    app()
