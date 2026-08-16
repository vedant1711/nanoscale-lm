"""Unit tests for the training pipeline: data, schedules, checkpoints (spec B4)."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import torch

from nanoscale.config import DataConfig, ModelConfig, ScheduleConfig, TokenizerConfig, get_preset
from nanoscale.data.toy import generate_corpus
from nanoscale.model import NanoScaleLM
from nanoscale.optim import build_optimizer
from nanoscale.tokenizer import BPETokenizer
from nanoscale.train import (
    CHECKPOINT_VERSION,
    Batch,
    TokenBatcher,
    TrainState,
    build_packed_tokens,
    evaluate_loss,
    grad_global_norm,
    iter_text_documents,
    load_checkpoint,
    load_config_from_checkpoint,
    lr_multiplier,
    make_schedule,
    save_checkpoint,
    tokenize_documents,
    weight_decay_multiplier,
)


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    corpus = generate_corpus(seed=1, n_stories=1500)
    return BPETokenizer.train(corpus, TokenizerConfig(vocab_size=512, max_train_bytes=400_000))


# =====================================================================================
# Schedules
# =====================================================================================


def test_warmup_ramps_from_near_zero_to_one() -> None:
    cfg = ScheduleConfig(name="cosine", warmup_frac=0.1, min_lr_frac=0.0)
    mults = [lr_multiplier(s, 100, cfg) for s in range(100)]
    assert mults[0] == pytest.approx(0.1)  # one warmup step's worth, not exactly zero
    assert mults[9] == pytest.approx(1.0)
    assert all(a < b for a, b in pairwise(mults[:10]))


def test_cosine_decays_monotonically_to_the_floor() -> None:
    cfg = ScheduleConfig(name="cosine", warmup_frac=0.1, min_lr_frac=0.1)
    mults = [lr_multiplier(s, 200, cfg) for s in range(200)]
    tail = mults[20:]
    assert all(a >= b - 1e-12 for a, b in pairwise(tail))
    assert mults[-1] == pytest.approx(0.1, abs=1e-3)
    assert max(mults) == pytest.approx(1.0)


def test_cosine_is_at_the_midpoint_halfway_through() -> None:
    cfg = ScheduleConfig(name="cosine", warmup_frac=0.0, min_lr_frac=0.0)
    assert lr_multiplier(500, 1000, cfg) == pytest.approx(0.5, abs=0.01)


def test_wsd_holds_a_stable_phase_then_decays() -> None:
    cfg = ScheduleConfig(name="wsd", warmup_frac=0.05, decay_frac=0.2, min_lr_frac=0.0)
    mults = [lr_multiplier(s, 100, cfg) for s in range(100)]
    assert all(m == pytest.approx(1.0) for m in mults[10:75]), "stable phase must be flat"
    assert mults[-1] < 0.1
    tail = mults[80:]
    assert all(a >= b - 1e-12 for a, b in pairwise(tail))


def test_constant_schedule_is_flat_after_warmup() -> None:
    cfg = ScheduleConfig(name="constant", warmup_frac=0.1)
    mults = [lr_multiplier(s, 50, cfg) for s in range(50)]
    assert all(m == pytest.approx(1.0) for m in mults[5:])


def test_schedule_clamps_out_of_range_steps() -> None:
    cfg = ScheduleConfig(name="cosine")
    assert 0.0 <= lr_multiplier(-5, 100, cfg) <= 1.0
    assert 0.0 <= lr_multiplier(500, 100, cfg) <= 1.0


def test_schedule_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="total_steps must be positive"):
        lr_multiplier(0, 0, ScheduleConfig())
    bogus = ScheduleConfig.model_construct(
        **{**ScheduleConfig(warmup_frac=0.0).dump_inputs(), "name": "quadratic"}
    )
    with pytest.raises(ValueError, match="Unknown schedule"):
        lr_multiplier(5, 10, bogus)  # past warmup, so the schedule name is consulted


def test_make_schedule_length() -> None:
    assert len(make_schedule(ScheduleConfig(), 37)) == 37


def test_cautious_weight_decay_schedule_decays_to_zero() -> None:
    assert weight_decay_multiplier(0, 100, enabled=True) == pytest.approx(1.0)
    assert weight_decay_multiplier(50, 100, enabled=True) == pytest.approx(0.5)
    assert weight_decay_multiplier(100, 100, enabled=True) == pytest.approx(0.0)
    assert weight_decay_multiplier(50, 100, enabled=False) == 1.0


# =====================================================================================
# Data: tokenization, packing, splitting
# =====================================================================================


def test_documents_are_separated_by_eos(tokenizer: BPETokenizer) -> None:
    """Without a separator the model never learns that text can end."""
    tokens = tokenize_documents(["hello", "world"], tokenizer)
    assert int(tokens[-1]) == tokenizer.eos_id
    assert list(tokens).count(tokenizer.eos_id) == 2


def test_tokenize_respects_max_tokens(tokenizer: BPETokenizer) -> None:
    tokens = tokenize_documents(["a b c d e f g"] * 100, tokenizer, max_tokens=50)
    assert tokens.size == 50


def test_tokenize_handles_an_empty_stream(tokenizer: BPETokenizer) -> None:
    assert tokenize_documents([], tokenizer).size == 0


def test_packed_split_is_contiguous_and_does_not_leak(tokenizer: BPETokenizer) -> None:
    """A random per-window split would put windows from one document on both sides."""
    cfg = DataConfig(source="toy", seq_len=64, val_fraction=0.1)
    packed = build_packed_tokens(cfg, tokenizer, toy_stories=800)
    assert packed.n_train_tokens > 0
    assert packed.n_val_tokens > 0
    assert packed.n_val_tokens % (cfg.seq_len + 1) == 0, "the split must land on a window edge"
    ratio = packed.n_val_tokens / (packed.n_train_tokens + packed.n_val_tokens)
    assert 0.05 < ratio < 0.15


def test_packed_tokens_summary(tokenizer: BPETokenizer) -> None:
    packed = build_packed_tokens(DataConfig(source="toy", seq_len=32), tokenizer, toy_stories=400)
    summary = packed.summary()
    assert summary["train_tokens"] == packed.n_train_tokens
    assert summary["val_tokens"] == packed.n_val_tokens


def test_packing_rejects_a_corpus_that_is_too_small(tokenizer: BPETokenizer) -> None:
    with pytest.raises(ValueError, match="not enough"):
        build_packed_tokens(
            DataConfig(source="toy", seq_len=250), tokenizer, toy_stories=1, max_tokens=100
        )


def test_textfile_source(tmp_path: Path, tokenizer: BPETokenizer) -> None:
    txt = tmp_path / "a.txt"
    txt.write_text("Lily went to the park. " * 200, encoding="utf-8")
    jsonl = tmp_path / "b.jsonl"
    jsonl.write_text(
        '{"text": "Tom found a key."}\n\n{"text": "Mia ran home."}\n', encoding="utf-8"
    )

    docs = list(iter_text_documents([txt, jsonl]))
    assert len(docs) == 3
    assert "Tom found a key." in docs

    cfg = DataConfig(source="textfile", paths=(str(txt),), seq_len=16, val_fraction=0.1)
    packed = build_packed_tokens(cfg, tokenizer)
    assert packed.n_train_tokens > 0


def test_textfile_source_requires_paths(tokenizer: BPETokenizer) -> None:
    with pytest.raises(ValueError, match=r"requires data\.paths"):
        build_packed_tokens(DataConfig(source="textfile"), tokenizer)


def test_missing_path_raises(tokenizer: BPETokenizer) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_text_documents(["/nonexistent/file.txt"]))


# =====================================================================================
# Batching
# =====================================================================================


def _tokens(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int32) % 100


def test_inputs_and_targets_are_shifted_by_exactly_one() -> None:
    batcher = TokenBatcher(_tokens(1000), seq_len=8, batch_size=2, shuffle=False)
    batch = next(iter(batcher.epoch_batches()))
    assert batch.inputs.shape == (2, 8)
    assert batch.targets.shape == (2, 8)
    # The target at position i is the input at position i+1 of the same window.
    assert torch.equal(batch.inputs[:, 1:], batch.targets[:, :-1])


def test_unshuffled_batches_walk_the_corpus_in_order() -> None:
    batcher = TokenBatcher(_tokens(90), seq_len=8, batch_size=1, shuffle=False)
    batches = list(batcher.epoch_batches())
    assert len(batches) == 10  # 90 // 9
    assert batches[0].inputs[0, 0].item() == 0
    assert batches[1].inputs[0, 0].item() == 9


def test_batch_order_is_a_pure_function_of_seed_and_epoch() -> None:
    def order(seed: int, epoch: int) -> list[int]:
        b = TokenBatcher(_tokens(900), seq_len=8, batch_size=4, seed=seed)
        b.set_epoch(epoch)
        return [int(x) for batch in b.epoch_batches() for x in batch.inputs[:, 0]]

    assert order(1337, 0) == order(1337, 0)
    assert order(1337, 0) != order(1337, 1), "each epoch must reshuffle"
    assert order(1337, 0) != order(7, 0), "the seed must matter"


def test_every_window_appears_exactly_once_per_epoch() -> None:
    batcher = TokenBatcher(_tokens(900), seq_len=8, batch_size=1, seed=5)
    starts = [int(b.inputs[0, 0]) for b in batcher.epoch_batches()]
    assert len(starts) == len(set(starts)) == batcher.n_windows


def test_infinite_iteration_advances_the_epoch() -> None:
    batcher = TokenBatcher(_tokens(90), seq_len=8, batch_size=1, seed=3)
    it = iter(batcher)
    for _ in range(len(batcher) + 2):
        next(it)
    assert batcher.epoch == 1


def test_drop_last_keeps_batch_shapes_uniform() -> None:
    dropped = TokenBatcher(_tokens(95), seq_len=8, batch_size=4, drop_last=True)
    kept = TokenBatcher(_tokens(95), seq_len=8, batch_size=4, drop_last=False)
    assert len(dropped) == 2
    assert len(kept) == 3
    assert all(b.inputs.shape[0] == 4 for b in dropped.epoch_batches())
    assert kept.take(3)[-1].inputs.shape[0] == 2


def test_batcher_rejects_a_corpus_shorter_than_one_window() -> None:
    with pytest.raises(ValueError, match="fewer than one window"):
        TokenBatcher(_tokens(5), seq_len=8, batch_size=1)


def test_batch_helpers() -> None:
    batch = Batch(
        inputs=torch.zeros(2, 4, dtype=torch.long), targets=torch.ones(2, 4, dtype=torch.long)
    )
    assert batch.n_tokens == 8
    assert batch.to(torch.device("cpu")).inputs.device.type == "cpu"


def test_take_returns_at_most_n_batches() -> None:
    batcher = TokenBatcher(_tokens(900), seq_len=8, batch_size=4)
    assert len(batcher.take(3)) == 3
    assert len(batcher.take(10_000)) == len(batcher)


# =====================================================================================
# Checkpoints
# =====================================================================================


def _tiny_model() -> NanoScaleLM:
    return NanoScaleLM(
        ModelConfig(vocab_size=32, n_layers=2, d_model=16, n_heads=2, n_kv_heads=1, max_seq_len=8)
    )


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = _tiny_model()
    opt = build_optimizer(model, get_preset("nano").train.optim)
    state = TrainState(step=17, tokens=1234, epoch=2, best_val_loss=1.5)
    path = save_checkpoint(
        tmp_path / "ck.pt", model=model, optimizer=opt, state=state, config=get_preset("nano")
    )

    fresh = _tiny_model()
    fresh_opt = build_optimizer(fresh, get_preset("nano").train.optim)
    loaded, payload = load_checkpoint(path, model=fresh, optimizer=fresh_opt)

    assert loaded.step == 17
    assert loaded.tokens == 1234
    assert loaded.epoch == 2
    assert loaded.best_val_loss == 1.5
    assert payload["version"] == CHECKPOINT_VERSION
    for a, b in zip(model.parameters(), fresh.parameters(), strict=True):
        torch.testing.assert_close(a, b)


def test_checkpoint_embeds_a_usable_config(tmp_path: Path) -> None:
    cfg = get_preset("nano")
    path = save_checkpoint(tmp_path / "ck.pt", model=_tiny_model(), config=cfg)
    assert load_config_from_checkpoint(path) == cfg


def test_checkpoint_without_a_config_raises(tmp_path: Path) -> None:
    path = save_checkpoint(tmp_path / "ck.pt", model=_tiny_model())
    with pytest.raises(KeyError, match="does not embed a config"):
        load_config_from_checkpoint(path)


def test_checkpoint_records_rng_state(tmp_path: Path) -> None:
    torch.manual_seed(0)
    path = save_checkpoint(tmp_path / "ck.pt", model=_tiny_model())
    expected = torch.rand(3)

    torch.manual_seed(999)
    load_checkpoint(path, restore_rng=True)
    torch.testing.assert_close(torch.rand(3), expected)


def test_loading_a_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")


def test_loading_a_wrong_version_raises(tmp_path: Path) -> None:
    path = save_checkpoint(tmp_path / "ck.pt", model=_tiny_model())
    payload = torch.load(path, weights_only=False)
    payload["version"] = 99
    torch.save(payload, path)
    with pytest.raises(ValueError, match="version"):
        load_checkpoint(path)


def test_train_state_tolerates_missing_fields() -> None:
    state = TrainState.from_dict({"step": 3})
    assert state.step == 3
    assert state.tokens == 0
    assert state.best_val_loss == float("inf")


# =====================================================================================
# Loop helpers
# =====================================================================================


def test_grad_global_norm_matches_a_hand_computation() -> None:
    a = torch.zeros(2, requires_grad=True)
    b = torch.zeros(2, requires_grad=True)
    a.grad = torch.tensor([3.0, 0.0])
    b.grad = torch.tensor([4.0, 0.0])
    assert grad_global_norm([a, b]) == pytest.approx(5.0)


def test_grad_global_norm_ignores_missing_grads() -> None:
    a = torch.zeros(2, requires_grad=True)
    assert grad_global_norm([a]) == 0.0


def test_evaluate_loss_is_token_weighted() -> None:
    model = _tiny_model().eval()
    batches = [
        Batch(
            inputs=torch.randint(0, 32, (2, 4)),
            targets=torch.randint(0, 32, (2, 4)),
        )
        for _ in range(3)
    ]
    loss = evaluate_loss(model, batches, device=torch.device("cpu"))
    assert loss > 0
    assert abs(loss - float(np.log(32))) < 0.1  # zero-init model is uniform


def test_evaluate_loss_restores_training_mode() -> None:
    model = _tiny_model()
    model.train()
    evaluate_loss(model, [], device=torch.device("cpu"))
    assert model.training
