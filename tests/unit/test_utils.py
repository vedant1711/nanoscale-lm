"""Phase-0 tests: seeding/determinism helpers, manifests, metric logging, devices."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from nanoscale.config import get_preset
from nanoscale.utils import (
    Manifest,
    MetricLogger,
    derive_seed,
    git_sha,
    hardware_string,
    resolve_device,
    resolve_dtype,
    seed_all,
    temporary_seed,
    write_manifest,
)


def _draw() -> tuple[float, float, float]:
    import random

    return (random.random(), float(np.random.rand()), float(torch.rand(1).item()))


def test_seed_all_makes_all_three_rngs_reproducible() -> None:
    seed_all(123)
    first = _draw()
    seed_all(123)
    assert _draw() == first
    seed_all(124)
    assert _draw() != first


def test_temporary_seed_restores_the_stream() -> None:
    seed_all(7)
    before = _draw()
    seed_all(7)
    with temporary_seed(999):
        inner = _draw()
    after = _draw()
    assert after == before, "the outer RNG stream must be untouched by temporary_seed"
    with temporary_seed(999):
        assert _draw() == inner


def test_derive_seed_is_stable_and_distinct() -> None:
    assert derive_seed(1337, "data", 0) == derive_seed(1337, "data", 0)
    assert derive_seed(1337, "data", 0) != derive_seed(1337, "data", 1)
    assert derive_seed(1337, "data", 0) != derive_seed(1338, "data", 0)
    assert 0 <= derive_seed(1337, "x") < 2**31 - 1


def test_resolve_device_always_resolves() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda", "mps")
    # Requesting an unavailable accelerator degrades to CPU instead of crashing.
    if not torch.cuda.is_available():
        assert resolve_device("cuda").type == "cpu"


def test_resolve_dtype_downgrades_on_cpu() -> None:
    cpu = torch.device("cpu")
    assert resolve_dtype("fp32", cpu) is torch.float32
    assert resolve_dtype("bf16", cpu) is torch.float32
    assert resolve_dtype("fp16", cpu) is torch.float32


def test_hardware_string_is_nonempty() -> None:
    assert len(hardware_string()) > 3


def test_manifest_records_the_reproducibility_contract(tmp_path: Path) -> None:
    cfg = get_preset("nano")
    manifest = write_manifest(
        tmp_path,
        run_name="test-run",
        phase="phase0",
        seed=1337,
        config=cfg,
        token_budget=cfg.train.token_budget,
        final_loss=1.23,
    )
    payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    for key in (
        "git_sha",
        "config_hash",
        "seed",
        "torch_version",
        "python_version",
        "hardware",
        "token_budget",
        "wall_clock_s",
    ):
        assert key in payload, f"manifest is missing the {key} field"
    assert payload["config_hash"] == cfg.config_hash()
    assert payload["metrics"]["final_loss"] == 1.23
    assert manifest.finished_at is not None


def test_manifest_accepts_plain_dicts(tmp_path: Path) -> None:
    m = Manifest(run_name="r", phase="p", seed=1, config_hash="", config={"a": 1})
    path = m.finish(x=2.0).write(tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["config"] == {"a": 1}


def test_git_sha_returns_something() -> None:
    assert isinstance(git_sha(), str)


def test_metric_logger_writes_jsonl_and_csv(tmp_path: Path) -> None:
    with MetricLogger(tmp_path) as logger:
        logger.log(step=0, loss=2.0)
        logger.log(step=1, loss=1.5)
        logger.log(step=2, loss=1.0, val_loss=1.1)  # a new column appears mid-run
        logger.summary(best=1.0)

    lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[-1])["val_loss"] == 1.1

    csv_lines = (tmp_path / "metrics.csv").read_text(encoding="utf-8").strip().splitlines()
    assert csv_lines[0].split(",") == ["step", "elapsed_s", "loss", "val_loss"]
    assert len(csv_lines) == 4, "the CSV is rewritten in full when a column is added"
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["best"] == 1.0


def test_metric_logger_format_row(tmp_path: Path) -> None:
    logger = MetricLogger(tmp_path)
    row = logger.log(step=5, loss=1.25)
    text = logger.format_row(row)
    assert "step" in text and "loss" in text
