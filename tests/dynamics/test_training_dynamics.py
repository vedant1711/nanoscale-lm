"""Training-dynamics tests (spec D3).

These are the tests that catch a training loop which *runs* but does not *learn*: the
failure mode that unit tests on individual components cannot see:

* **Overfit a batch.** Any correct trainer must drive the loss on one repeated batch to
  approximately zero. If it cannot, something between the data and the optimizer is
  broken, and no amount of component-level testing will tell you what.
* **Determinism.** Two seeded runs must produce identical loss trajectories.
* **Resume.** Continuing from a checkpoint must be indistinguishable from never having
  stopped.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from nanoscale.config import ExperimentConfig, TokenizerConfig, load_experiment
from nanoscale.data.toy import generate_corpus
from nanoscale.model import NanoScaleLM, build_model
from nanoscale.optim import build_optimizer
from nanoscale.tokenizer import BPETokenizer
from nanoscale.train import Trainer


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    corpus = generate_corpus(seed=1, n_stories=1200)
    return BPETokenizer.train(corpus, TokenizerConfig(vocab_size=512, max_train_bytes=300_000))


def tiny_config(**overrides: object) -> ExperimentConfig:
    """A `nano`-shaped but deliberately minuscule config, so tests stay quick."""
    base = [
        "tokenizer.vocab_size=512",
        "model.vocab_size=512",
        "model.n_layers=2",
        "model.d_model=64",
        "model.n_heads=2",
        "model.n_kv_heads=1",
        "model.max_seq_len=64",
        "data.seq_len=64",
        "data.val_fraction=0.1",
        "train.device=cpu",
        "train.batch_size=4",
        "train.max_steps=20",
        "train.token_budget=null",
        "train.eval_interval=10",
        "train.log_interval=5",
        "train.ckpt_interval=1000",
    ]
    extra = [f"{k}={v}" for k, v in overrides.items()]
    return load_experiment(tier="nano", overrides=base + extra)


# =====================================================================================
# Overfit a batch -- the canonical "is my training loop correct" test
# =====================================================================================


@pytest.mark.parametrize("optimizer", ["muon", "adamw"])
def test_pretrainer_overfits_a_single_batch(optimizer: str) -> None:
    """One batch, repeated: loss must collapse toward zero with either optimizer."""
    cfg = tiny_config(**{"train.optim.name": optimizer})
    torch.manual_seed(0)
    model = build_model(cfg.model)
    opt = build_optimizer(model, cfg.train.optim)

    ids = torch.randint(0, cfg.model.vocab_size, (4, 32))
    targets = torch.randint(0, cfg.model.vocab_size, (4, 32))

    initial = model(ids, targets=targets).loss
    assert initial is not None
    first = float(initial.detach())
    for _ in range(200):
        opt.zero_grad()
        loss = model(ids, targets=targets).loss
        assert loss is not None
        loss.backward()
        opt.step()
    last = float(loss.detach())

    assert first > 5.0, "a zero-init model should start near ln(vocab)"
    assert last < 0.15, f"{optimizer} failed to overfit one batch: {first:.3f} -> {last:.3f}"


def test_loss_starts_at_ln_vocab_and_decreases_over_a_real_run(
    tokenizer: BPETokenizer, tmp_path: Path
) -> None:
    cfg = tiny_config(**{"train.max_steps": 80, "train.eval_interval": 40})
    trainer = Trainer(cfg, tokenizer=tokenizer, out_dir=tmp_path / "run")
    result = trainer.train()

    losses = [row["loss"] for row in result.history if "loss" in row]
    assert losses[0] == pytest.approx(math.log(cfg.model.vocab_size), abs=0.05)
    assert losses[-1] < losses[0] * 0.6, f"loss barely moved: {losses}"
    assert result.final_val_loss < losses[0]
    assert math.isfinite(result.final_val_loss)


def test_loss_trajectory_is_roughly_monotone(tokenizer: BPETokenizer, tmp_path: Path) -> None:
    """Not strictly monotone -- stochastic batches bounce -- but the trend must be down."""
    cfg = tiny_config(**{"train.max_steps": 80, "train.log_interval": 1})
    result = Trainer(cfg, tokenizer=tokenizer, out_dir=tmp_path / "run").train()
    losses = [row["loss"] for row in result.history if "loss" in row]
    first_quarter = sum(losses[:20]) / 20
    last_quarter = sum(losses[-20:]) / 20
    assert last_quarter < first_quarter * 0.75


# =====================================================================================
# Determinism
# =====================================================================================


def test_two_seeded_runs_are_identical(tokenizer: BPETokenizer, tmp_path: Path) -> None:
    """Spec D3: seeded `nano` runs reproduce loss to floating-point tolerance."""
    cfg = tiny_config()
    a = Trainer(cfg, tokenizer=tokenizer, out_dir=tmp_path / "a").train()
    b = Trainer(cfg, tokenizer=tokenizer, out_dir=tmp_path / "b").train()

    losses_a = [row["loss"] for row in a.history if "loss" in row]
    losses_b = [row["loss"] for row in b.history if "loss" in row]
    assert losses_a == pytest.approx(losses_b, rel=1e-9, abs=1e-9)
    assert a.final_val_loss == pytest.approx(b.final_val_loss, rel=1e-9, abs=1e-9)
    assert a.tokens == b.tokens


def test_a_different_seed_gives_a_different_trajectory(
    tokenizer: BPETokenizer, tmp_path: Path
) -> None:
    """Guards against the determinism test passing because nothing is random at all."""
    a = Trainer(tiny_config(), tokenizer=tokenizer, out_dir=tmp_path / "a").train()
    b = Trainer(
        tiny_config(**{"train.seed": 4242}), tokenizer=tokenizer, out_dir=tmp_path / "b"
    ).train()
    assert a.final_val_loss != pytest.approx(b.final_val_loss, rel=1e-6)


# =====================================================================================
# Resume
# =====================================================================================


def test_resume_continues_identically(tokenizer: BPETokenizer, tmp_path: Path) -> None:
    """Stopping and resuming must be indistinguishable from running straight through."""
    cfg = tiny_config(**{"train.max_steps": 20})
    uninterrupted = Trainer(cfg, tokenizer=tokenizer, out_dir=tmp_path / "full").train()

    # Interrupt at step 10 *without* re-planning the schedule -- that is what a
    # pre-empted session does, and it is the only interruption a resume can match.
    first_half = Trainer(cfg, tokenizer=tokenizer, out_dir=tmp_path / "half")
    first_half.train(stop_at_step=10)
    checkpoint = tmp_path / "half" / "final.pt"
    assert checkpoint.exists()

    second_half = Trainer(cfg, tokenizer=tokenizer, out_dir=tmp_path / "resumed")
    second_half.maybe_resume(checkpoint)
    assert second_half.state.step == 10
    resumed = second_half.train()

    assert resumed.steps == uninterrupted.steps == 20
    assert resumed.tokens == uninterrupted.tokens
    assert resumed.final_val_loss == pytest.approx(uninterrupted.final_val_loss, rel=1e-6)


def test_resume_restores_optimizer_state_not_just_weights(
    tokenizer: BPETokenizer, tmp_path: Path
) -> None:
    """A weights-only resume shows up as a visible loss bump; this pins that it doesn't."""
    trainer = Trainer(
        tiny_config(**{"train.max_steps": 15}), tokenizer=tokenizer, out_dir=tmp_path / "a"
    )
    trainer.train()

    resumed = Trainer(
        tiny_config(**{"train.max_steps": 15}), tokenizer=tokenizer, out_dir=tmp_path / "b"
    )
    resumed.maybe_resume(tmp_path / "a" / "final.pt")
    muon_state = resumed.optimizer.optimizers["muon"].state
    assert muon_state, "optimizer state was not restored"
    buffers = [s["momentum_buffer"] for s in muon_state.values()]
    assert any(float(b.abs().sum()) > 0 for b in buffers), "momentum buffers are all zero"


# =====================================================================================
# Budgets and stopping
# =====================================================================================


def test_token_budget_stops_before_max_steps(tokenizer: BPETokenizer, tmp_path: Path) -> None:
    cfg = tiny_config(**{"train.max_steps": 1000, "train.token_budget": 4 * 64 * 5})
    result = Trainer(cfg, tokenizer=tokenizer, out_dir=tmp_path / "run").train()
    assert result.steps == 5
    assert result.tokens == 4 * 64 * 5


def test_manifest_and_metrics_are_written(tokenizer: BPETokenizer, tmp_path: Path) -> None:
    out = tmp_path / "run"
    Trainer(tiny_config(), tokenizer=tokenizer, out_dir=out).train()
    for name in ("manifest.json", "metrics.jsonl", "metrics.csv", "summary.json", "final.pt"):
        assert (out / name).exists(), f"{name} was not written"


def test_summary_records_the_chinchilla_fraction(tokenizer: BPETokenizer, tmp_path: Path) -> None:
    """Every nano manifest must state how far under the compute-optimal budget it is."""
    import json

    out = tmp_path / "run"
    Trainer(tiny_config(), tokenizer=tokenizer, out_dir=out).train()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert 0.0 < summary["chinchilla_fraction"] < 1.0
    assert summary["params"] > 0
    assert summary["muon_params"] > 0 and summary["adamw_params"] > 0


def test_gradient_clipping_bounds_the_update(tokenizer: BPETokenizer, tmp_path: Path) -> None:
    cfg = tiny_config(
        **{
            "train.max_steps": 5,
            "train.optim.grad_clip": 0.01,
            "train.optim.lr": 1.0,
            "train.optim.adamw_lr": 1.0,
        }
    )
    result = Trainer(cfg, tokenizer=tokenizer, out_dir=tmp_path / "run").train()
    assert math.isfinite(result.final_val_loss), "clipping should keep an absurd LR survivable"


def test_generation_after_training_is_coherent_enough_to_decode(
    tokenizer: BPETokenizer, tmp_path: Path
) -> None:
    cfg = tiny_config(**{"train.max_steps": 30})
    trainer = Trainer(cfg, tokenizer=tokenizer, out_dir=tmp_path / "run")
    trainer.train()
    model: NanoScaleLM = trainer.model
    prompt = torch.tensor([tokenizer.encode("Lily went to", add_bos=True)])
    out = model.generate(prompt, max_new_tokens=16, temperature=0.8, eos_id=tokenizer.eos_id)
    text = tokenizer.decode(out[0].tolist(), skip_special=True)
    assert text.startswith("Lily went to")
    assert len(text) > len("Lily went to")
