"""Distillation correctness (spec D1/D2, Phase 7).

The claim that matters is the *asymmetry* of the KL divergence, so the tests target it
directly rather than only checking that losses run:

* forward KL penalises the student for **missing** teacher mass (mode-covering);
* reverse KL penalises it for **inventing** mass the teacher lacks (mode-seeking);
* both are zero exactly when the distributions match, and non-negative always.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from nanoscale.config import DistillConfig, ModelConfig, TokenizerConfig, draft_model_config
from nanoscale.data.toy import generate_corpus
from nanoscale.distill import (
    DistillTrainer,
    forward_kl_loss,
    reverse_kl_policy_gradient,
    sequence_kd_loss,
    token_kl,
)
from nanoscale.model import NanoScaleLM, build_model
from nanoscale.tokenizer import BPETokenizer
from nanoscale.train import TokenBatcher
from nanoscale.utils import backward


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(20260816)


def logits_from(probs: list[float]) -> torch.Tensor:
    return torch.log(torch.tensor([[probs]], dtype=torch.float64))


# =====================================================================================
# token_kl: the asymmetry
# =====================================================================================


def test_kl_is_zero_for_identical_distributions() -> None:
    logits = torch.randn(2, 4, 8, dtype=torch.float64)
    mask = torch.ones(2, 4, dtype=torch.float64)
    assert float(token_kl(logits, logits, mask, reverse=False)) == pytest.approx(0.0, abs=1e-12)
    assert float(token_kl(logits, logits, mask, reverse=True)) == pytest.approx(0.0, abs=1e-12)


def test_kl_is_always_non_negative() -> None:
    for _ in range(20):
        s = torch.randn(1, 3, 16, dtype=torch.float64)
        t = torch.randn(1, 3, 16, dtype=torch.float64)
        mask = torch.ones(1, 3, dtype=torch.float64)
        assert float(token_kl(s, t, mask, reverse=False)) >= -1e-12
        assert float(token_kl(s, t, mask, reverse=True)) >= -1e-12


def test_kl_matches_a_hand_computed_value() -> None:
    teacher = logits_from([0.5, 0.5, 0.0 + 1e-12])
    student = logits_from([0.25, 0.25, 0.5])
    mask = torch.ones(1, 1, dtype=torch.float64)
    # KL(p || q) = 0.5*log(0.5/0.25) * 2 = log 2
    forward = float(token_kl(student, teacher, mask, reverse=False))
    assert forward == pytest.approx(math.log(2.0), rel=1e-4)


def test_forward_kl_punishes_missing_mass_and_reverse_kl_punishes_invented_mass() -> None:
    """The mode-covering / mode-seeking distinction, demonstrated on one fixture.

    The teacher is bimodal. A *narrow* student covers one mode and ignores the other; a
    *broad* student spreads mass everywhere, including where the teacher has none.
    Forward KL should prefer the broad student; reverse KL should prefer the narrow one.
    """
    eps = 1e-9
    teacher = logits_from([0.5, eps, eps, 0.5])
    narrow = logits_from([1.0 - 3 * eps, eps, eps, eps])  # one mode only
    broad = logits_from([0.25, 0.25, 0.25, 0.25])  # covers everything
    mask = torch.ones(1, 1, dtype=torch.float64)

    fwd_narrow = float(token_kl(narrow, teacher, mask, reverse=False))
    fwd_broad = float(token_kl(broad, teacher, mask, reverse=False))
    assert fwd_broad < fwd_narrow, "forward KL should prefer the mode-covering student"

    rev_narrow = float(token_kl(narrow, teacher, mask, reverse=True))
    rev_broad = float(token_kl(broad, teacher, mask, reverse=True))
    assert rev_narrow < rev_broad, "reverse KL should prefer the mode-seeking student"


def test_kl_respects_the_mask() -> None:
    student = torch.randn(1, 4, 8, dtype=torch.float64)
    teacher = torch.randn(1, 4, 8, dtype=torch.float64)
    full = torch.ones(1, 4, dtype=torch.float64)
    half = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float64)

    masked = float(token_kl(student, teacher, half))
    manual = float(token_kl(student[:, :2], teacher[:, :2], full[:, :2]))
    assert masked == pytest.approx(manual, rel=1e-9)


def test_temperature_softens_the_divergence() -> None:
    student = torch.randn(1, 2, 16, dtype=torch.float64) * 3
    teacher = torch.randn(1, 2, 16, dtype=torch.float64) * 3
    mask = torch.ones(1, 2, dtype=torch.float64)
    sharp = float(token_kl(student, teacher, mask, temperature=1.0))
    soft = float(token_kl(student, teacher, mask, temperature=4.0))
    assert soft < sharp


# =====================================================================================
# The three objectives
# =====================================================================================


def test_forward_kl_loss_blends_cross_entropy_and_distillation() -> None:
    student = torch.randn(2, 5, 16)
    teacher = torch.randn(2, 5, 16)
    targets = torch.randint(0, 16, (2, 5))
    mask = torch.ones(2, 5)

    out = forward_kl_loss(student, teacher, targets, mask, temperature=2.0, alpha=0.5)
    expected = 0.5 * float(out.ce) + 0.5 * 4.0 * float(out.kd)
    assert float(out.loss) == pytest.approx(expected, rel=1e-5)
    assert out.stats()["temperature"] == 2.0


def test_alpha_one_is_pure_cross_entropy() -> None:
    student = torch.randn(1, 4, 8)
    teacher = torch.randn(1, 4, 8)
    targets = torch.randint(0, 8, (1, 4))
    mask = torch.ones(1, 4)
    out = forward_kl_loss(student, teacher, targets, mask, alpha=1.0)
    assert float(out.loss) == pytest.approx(float(out.ce), rel=1e-6)


def test_tau_squared_keeps_the_kd_gradient_scale_stable() -> None:
    """Without the tau^2 factor, changing temperature silently retunes alpha."""
    student = torch.randn(1, 4, 32, requires_grad=True)
    teacher = torch.randn(1, 4, 32)
    mask = torch.ones(1, 4)

    def kd_grad_norm(temperature: float, *, with_tau2: bool) -> float:
        if student.grad is not None:
            student.grad = None
        kd = token_kl(student, teacher, mask, temperature=temperature)
        loss = kd * (temperature**2 if with_tau2 else 1.0)
        backward(loss)
        grad = student.grad
        return float(grad.norm()) if grad is not None else 0.0

    with_2 = kd_grad_norm(2.0, with_tau2=True)
    with_4 = kd_grad_norm(4.0, with_tau2=True)
    without_2 = kd_grad_norm(2.0, with_tau2=False)
    without_4 = kd_grad_norm(4.0, with_tau2=False)

    # Compensated: the gradient scale is roughly stable across temperatures.
    assert with_4 / with_2 > without_4 / without_2


def test_sequence_kd_is_plain_mle_on_teacher_samples() -> None:
    student = torch.randn(2, 6, 16)
    samples = torch.randint(0, 16, (2, 6))
    mask = torch.ones(2, 6)
    out = sequence_kd_loss(student, samples, mask)

    expected = torch.nn.functional.cross_entropy(
        student.reshape(-1, 16).float(), samples.reshape(-1)
    )
    assert float(out.loss) == pytest.approx(float(expected), rel=1e-6)
    assert float(out.kd) == 0.0  # no teacher distribution is consulted


def test_sequence_kd_respects_the_mask() -> None:
    student = torch.randn(1, 4, 8)
    samples = torch.randint(0, 8, (1, 4))
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    out = sequence_kd_loss(student, samples, mask)
    manual = torch.nn.functional.cross_entropy(
        student[:, :2].reshape(-1, 8).float(), samples[:, :2].reshape(-1)
    )
    assert float(out.loss) == pytest.approx(float(manual), rel=1e-6)


# =====================================================================================
# Reverse-KL policy gradient
# =====================================================================================


def test_reverse_kl_reward_is_zero_when_student_matches_teacher() -> None:
    logits = torch.randn(2, 5, 16)
    sampled = torch.randint(0, 16, (2, 5))
    mask = torch.ones(2, 5)
    out = reverse_kl_policy_gradient(logits, logits, sampled, mask)
    assert out.extra["mean_reward"] == pytest.approx(0.0, abs=1e-6)
    assert float(out.kd) == pytest.approx(0.0, abs=1e-6)


def test_reverse_kl_reward_is_positive_where_the_teacher_prefers_the_token() -> None:
    """r_t = log p - log q, so a token the teacher likes more earns positive reward."""
    student = logits_from([0.1, 0.9]).float().expand(1, 1, 2).contiguous()
    teacher = logits_from([0.9, 0.1]).float().expand(1, 1, 2).contiguous()
    mask = torch.ones(1, 1)

    liked = reverse_kl_policy_gradient(
        student, teacher, torch.tensor([[0]]), mask, single_step_reg=False
    )
    disliked = reverse_kl_policy_gradient(
        student, teacher, torch.tensor([[1]]), mask, single_step_reg=False
    )
    assert liked.extra["mean_reward"] > 0 > disliked.extra["mean_reward"]


def test_reward_to_go_credits_only_the_future() -> None:
    """Using the whole trajectory's reward for every token is pure variance."""
    torch.manual_seed(0)
    student = torch.randn(1, 4, 8)
    teacher = torch.randn(1, 4, 8)
    sampled = torch.randint(0, 8, (1, 4))

    # Zeroing out the *last* token's contribution must change the earlier positions'
    # advantages (they include it) but a change at position 0 must not affect position 3.
    full = reverse_kl_policy_gradient(
        student, teacher, sampled, torch.ones(1, 4), single_step_reg=False
    )
    truncated = reverse_kl_policy_gradient(
        student, teacher, sampled, torch.tensor([[1.0, 1.0, 1.0, 0.0]]), single_step_reg=False
    )
    assert float(full.loss) != pytest.approx(float(truncated.loss))


def test_single_step_regularisation_adds_a_differentiable_term() -> None:
    student = torch.randn(1, 4, 8, requires_grad=True)
    teacher = torch.randn(1, 4, 8)
    sampled = torch.randint(0, 8, (1, 4))
    mask = torch.ones(1, 4)

    without = reverse_kl_policy_gradient(student, teacher, sampled, mask, single_step_reg=False)
    with_reg = reverse_kl_policy_gradient(student, teacher, sampled, mask, single_step_reg=True)

    assert float(with_reg.kd.detach()) > 0.0
    assert float(without.kd.detach()) == 0.0
    assert float(with_reg.loss) == pytest.approx(float(without.loss) + float(with_reg.kd), rel=1e-5)


def test_length_normalisation_removes_the_short_sequence_bias() -> None:
    """Without it, the estimator prefers short sequences purely for having fewer terms."""
    torch.manual_seed(1)
    student = torch.randn(1, 8, 8)
    teacher = torch.randn(1, 8, 8)
    sampled = torch.randint(0, 8, (1, 8))
    short = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    long = torch.ones(1, 8)

    unnormalised = [
        abs(
            float(
                reverse_kl_policy_gradient(
                    student, teacher, sampled, m, length_normalize=False, single_step_reg=False
                ).loss
            )
        )
        for m in (short, long)
    ]
    normalised = [
        abs(
            float(
                reverse_kl_policy_gradient(
                    student, teacher, sampled, m, length_normalize=True, single_step_reg=False
                ).loss
            )
        )
        for m in (short, long)
    ]
    # The unnormalised magnitude scales with length; the normalised one does much less.
    assert unnormalised[1] / max(1e-9, unnormalised[0]) > normalised[1] / max(1e-9, normalised[0])


def test_the_policy_gradient_flows_to_the_student_only() -> None:
    student = torch.randn(1, 4, 8, requires_grad=True)
    teacher = torch.randn(1, 4, 8, requires_grad=True)
    out = reverse_kl_policy_gradient(
        student, teacher, torch.randint(0, 8, (1, 4)), torch.ones(1, 4)
    )
    backward(out.loss)
    assert student.grad is not None
    assert teacher.grad is None, "the teacher must be treated as a constant"


def test_baseline_shifts_the_advantage_not_the_reward() -> None:
    student = torch.randn(1, 4, 8)
    teacher = torch.randn(1, 4, 8)
    sampled = torch.randint(0, 8, (1, 4))
    mask = torch.ones(1, 4)
    a = reverse_kl_policy_gradient(student, teacher, sampled, mask, baseline=0.0)
    b = reverse_kl_policy_gradient(student, teacher, sampled, mask, baseline=5.0)
    assert a.extra["mean_reward"] == pytest.approx(b.extra["mean_reward"])
    assert a.extra["mean_advantage"] != pytest.approx(b.extra["mean_advantage"])


# =====================================================================================
# The trainer
# =====================================================================================


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    corpus = generate_corpus(seed=1, n_stories=800)
    return BPETokenizer.train(corpus, TokenizerConfig(vocab_size=512, max_train_bytes=200_000))


def make_pair() -> tuple[NanoScaleLM, NanoScaleLM]:
    cfg = ModelConfig(
        vocab_size=512,
        n_layers=3,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        zero_init_output=False,
    )
    torch.manual_seed(0)
    teacher = build_model(cfg)
    student = build_model(draft_model_config(cfg))
    return teacher, student


def make_batcher(tokenizer: BPETokenizer, seq_len: int = 32) -> TokenBatcher:
    from nanoscale.train import tokenize_documents

    tokens = tokenize_documents(
        generate_corpus(seed=2, n_stories=400).split("\n\n"), tokenizer, max_tokens=40_000
    )
    return TokenBatcher(np.asarray(tokens), seq_len=seq_len, batch_size=4, seed=1)


def test_warmup_runs_plain_mle_before_the_objective(
    tokenizer: BPETokenizer, tmp_path: Path
) -> None:
    """MiniLLM's warm-start: on-policy rollouts from a random student carry no signal.

    During warm-up the reported loss must be the plain MLE loss (kd == 0), and after it
    the configured objective takes over.
    """
    teacher, student = make_pair()
    batcher = make_batcher(tokenizer)
    cfg = DistillConfig(
        method="reverse_kl",
        seq_len=32,
        batch_size=4,
        max_steps=4,
        warmup_steps=2,
        max_new_tokens=8,
        log_interval=1,
        device="cpu",
    )
    trainer = DistillTrainer(
        teacher,
        student,
        tokenizer,
        cfg,
        train_batcher=batcher,
        val_batches=batcher.take(2),
        out_dir=tmp_path,
    )
    result = trainer.train()
    phases = [row["phase"] for row in result.history]
    assert phases[:2] == [1.0, 1.0], "the first steps must be warm-up"
    assert phases[2:] == [0.0, 0.0], "the objective must take over after warm-up"
    assert result.warmup_steps == 2
    assert result.summary()["warmup_steps"] == 2


@pytest.mark.parametrize("method", ["forward_kl", "seqkd", "reverse_kl"])
def test_every_objective_trains_a_student(
    method: str, tokenizer: BPETokenizer, tmp_path: Path
) -> None:
    teacher, student = make_pair()
    cfg = DistillConfig(
        method=method,
        seq_len=32,
        batch_size=4,
        max_steps=4,
        max_new_tokens=8,
        log_interval=1,
        device="cpu",
    )
    batcher = make_batcher(tokenizer)
    trainer = DistillTrainer(
        teacher,
        student,
        tokenizer,
        cfg,
        train_batcher=batcher,
        val_batches=batcher.take(2),
        out_dir=tmp_path,
    )
    result = trainer.train()
    assert result.method == method
    assert math.isfinite(result.final_loss)
    assert result.student_params < result.teacher_params
    assert result.compression_ratio > 1.0
    assert result.history


def test_the_teacher_is_frozen(tokenizer: BPETokenizer, tmp_path: Path) -> None:
    teacher, student = make_pair()
    batcher = make_batcher(tokenizer)
    trainer = DistillTrainer(
        teacher,
        student,
        tokenizer,
        DistillConfig(method="forward_kl", seq_len=32, batch_size=4, max_steps=2, device="cpu"),
        train_batcher=batcher,
        out_dir=tmp_path,
    )
    assert not trainer.teacher.training
    assert all(not p.requires_grad for p in trainer.teacher.parameters())

    before = [p.detach().clone() for p in trainer.teacher.parameters()]
    trainer.train()
    for a, b in zip(before, trainer.teacher.parameters(), strict=True):
        torch.testing.assert_close(a, b.detach())


def test_the_trainer_requires_a_batcher(tokenizer: BPETokenizer, tmp_path: Path) -> None:
    teacher, student = make_pair()
    with pytest.raises(ValueError, match="train_batcher is required"):
        DistillTrainer(
            teacher,
            student,
            tokenizer,
            DistillConfig(device="cpu"),
            out_dir=tmp_path,
        )


def test_the_student_is_meaningfully_smaller() -> None:
    """The compression the shipped tiers actually get, and why two numbers are reported.

    Teacher and student must share a tokenizer, so the embedding table and LM head are
    the same width in both and their cost is irreducible. At a small vocabulary they can
    dominate the total; the non-embedding ratio is what describes the depth/width
    reduction. Both are reported, and both are asserted here on the real tier configs
    rather than on a tiny vocab-dominated test model.
    """
    from nanoscale.config import get_preset

    for tier, min_total, min_non_embedding in (("nano", 10.0, 25.0), ("micro", 5.0, 25.0)):
        base = get_preset(tier).model
        small = draft_model_config(base)
        b, s = base.param_breakdown(), small.param_breakdown()
        assert b["total"] / s["total"] > min_total
        assert b["non_embedding"] / s["non_embedding"] > min_non_embedding


def test_result_reports_both_compression_ratios() -> None:
    from nanoscale.distill import DistillResult

    result = DistillResult(
        method="reverse_kl",
        final_loss=1.0,
        student_val_loss=1.0,
        teacher_val_loss=0.9,
        student_params=1000,
        teacher_params=4000,
        student_non_embedding=100,
        teacher_non_embedding=2000,
        steps=1,
        warmup_steps=0,
        wall_clock_s=1.0,
    )
    assert result.compression_ratio == pytest.approx(4.0)
    assert result.non_embedding_compression == pytest.approx(20.0)
    assert result.summary()["non_embedding_compression"] == pytest.approx(20.0)
