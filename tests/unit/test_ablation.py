"""Tests for the ablation harness (spec Phase 5).

The harness's job is to make comparisons honest, so what is tested is exactly that:
variants differ from the baseline only where they say they do, the steps-to-target
metric is robust to a single lucky batch, and the reporting helper refuses to declare a
winner on a difference smaller than single-seed noise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nanoscale.bench.ablation import (
    NOISE_THRESHOLD,
    AblationResult,
    AblationSuite,
    Variant,
    describe_difference,
)
from nanoscale.bench.ablation import _first_crossing as first_crossing
from nanoscale.train import TrainResult


def make_suite(**kwargs: object) -> AblationSuite:
    base: dict[str, object] = {
        "name": "demo",
        "question": "Does X help?",
        "variants": [Variant("a"), Variant("b", overrides=("model.qk_norm=false",))],
        "base_overrides": ("train.device=cpu", "train.max_steps=3"),
    }
    base.update(kwargs)
    return AblationSuite(**base)  # type: ignore[arg-type]


def fake_result(
    loss: float, *, steps: int | None = 10, seconds: float | None = 1.0, name: str = "v"
) -> AblationResult:
    return AblationResult(
        variant=Variant(name, label=name),
        result=TrainResult(
            final_train_loss=loss,
            final_val_loss=loss,
            best_val_loss=loss,
            steps=100,
            tokens=1000,
            wall_clock_s=5.0,
            tokens_per_second=200.0,
        ),
        steps_to_target=steps,
        seconds_to_target=seconds,
        run_dir=Path("runs/demo"),
    )


# ------------------------------------------------------------------ variant configs


def test_variants_differ_only_where_declared() -> None:
    """The core honesty property: an ablation arm changes exactly one thing."""
    suite = make_suite()
    baseline = suite.config_for(suite.variants[0])
    challenger = suite.config_for(suite.variants[1])

    base_dump = baseline.dump_inputs(mode="json")
    other_dump = challenger.dump_inputs(mode="json")

    def flatten(payload: dict[str, object], prefix: str = "") -> dict[str, object]:
        out: dict[str, object] = {}
        for key, value in payload.items():
            path = f"{prefix}{key}"
            if isinstance(value, dict):
                out.update(flatten(value, f"{path}."))
            else:
                out[path] = value
        return out

    flat_base = flatten(base_dump)
    flat_other = flatten(other_dump)
    differing = {k for k in flat_base if flat_base[k] != flat_other[k]}
    assert differing == {"model.qk_norm"}


def test_every_variant_shares_the_seed_and_schedule() -> None:
    suite = make_suite()
    configs = [suite.config_for(v) for v in suite.variants]
    assert len({c.train.seed for c in configs}) == 1
    assert len({c.train.max_steps for c in configs}) == 1
    assert len({c.train.schedule.name for c in configs}) == 1
    assert len({c.data.seq_len for c in configs}) == 1


def test_variant_display_falls_back_to_the_name() -> None:
    assert Variant("x").display() == "x"
    assert Variant("x", label="Nice X").display() == "Nice X"


# ---------------------------------------------------------------- steps-to-target


def test_first_crossing_finds_the_step() -> None:
    history = [
        {"step": 10, "loss": 3.0, "elapsed_s": 1.0},
        {"step": 20, "loss": 2.0, "elapsed_s": 2.0},
        {"step": 30, "loss": 1.0, "elapsed_s": 3.0},
        {"step": 40, "loss": 0.5, "elapsed_s": 4.0},
    ]
    step, seconds = first_crossing(history, target=1.5)
    assert step == 40  # smoothed over 3 rows: (3+2+1)/3 = 2.0, then (2+1+0.5)/3 = 1.17
    assert seconds == 4.0


def test_first_crossing_ignores_a_single_lucky_batch() -> None:
    """One outlier dipping below target is not "reaching" it."""
    history = [
        {"step": 1, "loss": 5.0, "elapsed_s": 0.1},
        {"step": 2, "loss": 0.01, "elapsed_s": 0.2},  # a fluke
        {"step": 3, "loss": 5.0, "elapsed_s": 0.3},
        {"step": 4, "loss": 5.0, "elapsed_s": 0.4},
    ]
    step, _ = first_crossing(history, target=1.0)
    assert step is None


def test_first_crossing_returns_none_when_never_reached() -> None:
    history = [{"step": s, "loss": 9.0, "elapsed_s": float(s)} for s in range(5)]
    assert first_crossing(history, target=1.0) == (None, None)


def test_first_crossing_skips_rows_without_a_loss() -> None:
    history = [
        {"step": 1, "val_loss": 0.1},
        {"step": 2, "loss": 0.1, "elapsed_s": 1.0},
        {"step": 3, "loss": 0.1, "elapsed_s": 2.0},
    ]
    step, _ = first_crossing(history, target=0.5)
    assert step == 2


# --------------------------------------------------------------------- reporting


def test_small_differences_are_reported_as_noise() -> None:
    baseline = fake_result(1.000, name="base")
    challenger = fake_result(0.995, name="new")  # 0.5% better
    text = describe_difference(baseline, challenger)
    assert "No measurable difference" in text
    assert f"{NOISE_THRESHOLD * 100:.0f}%" in text


def test_large_differences_are_reported_with_a_direction() -> None:
    baseline = fake_result(1.0, name="base")
    better = describe_difference(baseline, fake_result(0.5, name="new"))
    worse = describe_difference(baseline, fake_result(1.5, name="new"))
    assert "50.0% better" in better
    assert "50.0% worse" in worse


def test_a_variant_that_never_reaches_the_target_is_called_out() -> None:
    baseline = fake_result(1.0, steps=10, name="base")
    challenger = fake_result(2.0, steps=None, seconds=None, name="new")
    assert "never reached the target loss" in describe_difference(baseline, challenger)


def test_describe_handles_a_degenerate_baseline() -> None:
    assert "meaningless" in describe_difference(fake_result(0.0), fake_result(1.0))


# ------------------------------------------------------------------------ output


def test_write_json_records_provenance(tmp_path: Path) -> None:
    suite = make_suite(out_dir=tmp_path)
    path = suite.write_json([fake_result(1.0, name="a"), fake_result(0.5, name="b")])
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["name"] == "demo"
    assert payload["question"]
    assert "git_sha" in payload and "hardware" in payload
    assert len(payload["rows"]) == 2
    assert payload["rows"][0]["variant"] == "a"


def test_result_row_is_json_serialisable() -> None:
    import json

    json.dumps(fake_result(1.0).row())


@pytest.mark.parametrize("suite_name", ["optimizer", "architecture"])
def test_the_shipped_suites_are_well_formed(suite_name: str) -> None:
    """The suites in scripts/ablate.py must build valid configs for every arm."""
    from scripts.ablate import architecture_suite, optimizer_suite

    suite = (optimizer_suite if suite_name == "optimizer" else architecture_suite)(20, 1.0)
    assert suite.variants
    assert suite.question
    for variant in suite.variants:
        cfg = suite.config_for(variant)
        assert cfg.train.max_steps == 20
        assert cfg.train.device == "cpu"
