"""Phase-0 tests: the typer CLI surface."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nanoscale import __version__
from nanoscale.cli import app

runner = CliRunner()


def test_help_lists_the_app() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "NanoScale-LM" in result.stdout


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_info_prints_the_size_ladder() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    for tier in ("nano", "micro", "small"):
        assert tier in result.stdout
    assert "token budget" in result.stdout


def test_config_show_and_section() -> None:
    result = runner.invoke(app, ["config", "show", "--tier", "nano", "--section", "model"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert set(payload) == {"model"}
    assert payload["model"]["n_layers"] == 6


def test_config_show_bad_section() -> None:
    result = runner.invoke(app, ["config", "show", "--section", "nope"])
    assert result.exit_code != 0


def test_config_show_with_overrides() -> None:
    result = runner.invoke(
        app, ["config", "show", "--section", "train", "--set", "train.max_steps=5"]
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["train"]["max_steps"] == 5


def test_config_hash_is_deterministic() -> None:
    a = runner.invoke(app, ["config", "hash", "--tier", "micro"])
    b = runner.invoke(app, ["config", "hash", "--tier", "micro"])
    assert a.exit_code == 0
    assert a.stdout.strip() == b.stdout.strip()
    assert len(a.stdout.strip()) == 12


def test_config_save_and_schema(tmp_path: Path) -> None:
    out = tmp_path / "cfg.yaml"
    assert runner.invoke(app, ["config", "save", str(out), "--tier", "nano"]).exit_code == 0
    assert out.exists()

    schema_dir = tmp_path / "schema"
    assert runner.invoke(app, ["config", "schema", str(schema_dir)]).exit_code == 0
    assert (schema_dir / "ExperimentConfig.json").exists()


def test_config_params() -> None:
    result = runner.invoke(app, ["config", "params", "--tier", "micro"])
    assert result.exit_code == 0
    assert "total params" in result.stdout
    assert "non-embedding" in result.stdout
