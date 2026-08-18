"""Alignment correctness (spec D1 / Phase 6 acceptance).

The load-bearing tests:

* **SFT loss masking**: prompt tokens receive *exactly zero* gradient, verified by a
  gradient check rather than by inspecting the mask.
* **DPO and SimPO losses**: match hand-computed values on tiny fixtures, with no model
  involved, so a bug in the loss cannot hide behind a bug in the model.
* **The length-exploitation mechanism**: DPO's reward grows with response length and
  SimPO's does not, demonstrated directly on synthetic log-probabilities.
* **GRPO's group-relative advantage**: zero-mean within each group, and exactly zero
  for a unanimous group.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from nanoscale.align import (
    ArithmeticTask,
    GRPOTrainer,
    PreferenceTrainer,
    SFTTrainer,
    build_preference_batches,
    build_sft_batches,
    dpo_loss,
    encode_example,
    group_relative_advantages,
    make_arithmetic_tasks,
    sequence_logprobs,
    simpo_loss,
    verify_arithmetic,
)
from nanoscale.config import GRPOConfig, ModelConfig, PreferenceConfig, SFTConfig, TokenizerConfig
from nanoscale.data.instruct import InstructExample, iter_instructions, iter_preference_pairs
from nanoscale.data.toy import generate_corpus
from nanoscale.eval import head_to_head, repetition_rate, score_completion
from nanoscale.model import IGNORE_INDEX, NanoScaleLM
from nanoscale.tokenizer import BPETokenizer
from nanoscale.utils import backward


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    corpus = generate_corpus(seed=1, n_stories=1200)
    return BPETokenizer.train(corpus, TokenizerConfig(vocab_size=512, max_train_bytes=300_000))


@pytest.fixture
def model(tokenizer: BPETokenizer) -> NanoScaleLM:
    torch.manual_seed(0)
    return NanoScaleLM(
        ModelConfig(
            vocab_size=tokenizer.vocab_size,
            n_layers=2,
            d_model=32,
            n_heads=2,
            n_kv_heads=1,
            max_seq_len=128,
            zero_init_output=False,
        )
    )


# =====================================================================================
# Instruction / preference data
# =====================================================================================


def test_instruction_data_is_deterministic() -> None:
    a = list(iter_instructions(seed=5, n=20))
    b = list(iter_instructions(seed=5, n=20))
    c = list(iter_instructions(seed=6, n=20))
    assert a == b
    assert a != c


def test_every_instruction_has_a_nonempty_response() -> None:
    for ex in iter_instructions(seed=1, n=200):
        assert ex.instruction.strip()
        assert ex.response.strip()
        assert "{" not in ex.response


def test_preference_lengths_are_matched_by_construction() -> None:
    """The length diagnostic is only interpretable if the *data* is length-neutral."""
    pairs = list(iter_preference_pairs(seed=2, n=400))
    chosen = sum(len(p.chosen) for p in pairs) / len(pairs)
    rejected = sum(len(p.rejected) for p in pairs) / len(pairs)
    assert abs(chosen - rejected) / chosen < 0.05, (
        f"chosen {chosen:.1f} vs rejected {rejected:.1f} chars: the preference data itself "
        "correlates quality with length, which would confound the DPO length diagnostic"
    )


def test_every_rejection_kind_is_represented() -> None:
    kinds = {p.rejection_kind for p in iter_preference_pairs(seed=3, n=200)}
    assert kinds == {"off_topic", "truncated", "repetitive", "non_answer"}


def test_rejected_never_equals_chosen() -> None:
    """A pair labelled both preferred and dispreferred is contradictory training data.

    This caught a real bug: the "repetitive" rejection repeated the *whole first
    sentence* and then truncated to the chosen response's length, which for a
    single-sentence response reproduced it exactly.
    """
    for pair in iter_preference_pairs(seed=4, n=400):
        assert pair.rejected.strip()
        assert pair.chosen != pair.rejected, f"{pair.rejection_kind}: {pair.chosen!r}"


def test_repetitive_rejections_actually_repeat() -> None:
    from nanoscale.eval import repetition_rate

    repetitive = [
        p for p in iter_preference_pairs(seed=7, n=400) if p.rejection_kind == "repetitive"
    ]
    assert repetitive
    mean_rejected = sum(repetition_rate(p.rejected) for p in repetitive) / len(repetitive)
    mean_chosen = sum(repetition_rate(p.chosen) for p in repetitive) / len(repetitive)
    assert mean_rejected > mean_chosen + 0.2, (
        f"repetitive rejections are not degenerate: {mean_rejected:.3f} vs {mean_chosen:.3f}"
    )


# =====================================================================================
# SFT: completion-only masking
# =====================================================================================


def test_encode_example_masks_only_the_response(tokenizer: BPETokenizer) -> None:
    example = InstructExample(instruction="What did Lily find?", response="A shiny key.")
    ids, mask = encode_example(tokenizer, example, seq_len=128)
    supervised = [i for i, m in zip(ids, mask, strict=True) if m]
    assert supervised == [*tokenizer.encode("A shiny key."), tokenizer.eot_id]
    assert sum(mask) < len(mask), "the prompt must not be supervised"


def test_sft_batches_put_ignore_index_on_prompt_and_padding(tokenizer: BPETokenizer) -> None:
    examples = list(iter_instructions(seed=1, n=8))
    batches = build_sft_batches(tokenizer, examples, seq_len=96, batch_size=4)
    assert batches
    batch = batches[0]
    assert batch.inputs.shape == batch.targets.shape == batch.completion_mask.shape
    supervised = batch.completion_mask.bool()
    assert (batch.targets[~supervised] == IGNORE_INDEX).all()
    assert (batch.targets[supervised] != IGNORE_INDEX).all()
    assert batch.n_supervised > 0


def test_prompt_tokens_receive_no_gradient(tokenizer: BPETokenizer, model: NanoScaleLM) -> None:
    """The strongest form of the masking check: gradient w.r.t. prompt logits is zero.

    Spec Phase 6 asks for loss masking "verified by test". Inspecting the mask only
    proves the mask is what we built; this proves the *loss* ignores those positions.
    """
    examples = list(iter_instructions(seed=1, n=4))
    batch = build_sft_batches(tokenizer, examples, seq_len=96, batch_size=4)[0]

    logits = model(batch.inputs).logits
    logits.retain_grad()
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        batch.targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    backward(loss)

    assert logits.grad is not None
    per_position = logits.grad.abs().sum(dim=-1)
    supervised = batch.completion_mask.bool()
    assert float(per_position[~supervised].abs().max()) == 0.0, (
        "unsupervised positions received gradient: the loss is not completion-masked"
    )
    assert float(per_position[supervised].abs().max()) > 0.0, (
        "supervised positions received no gradient: the test is vacuous"
    )


def test_sft_trainer_reduces_loss(
    tokenizer: BPETokenizer, model: NanoScaleLM, tmp_path: Path
) -> None:
    cfg = SFTConfig(
        seq_len=96,
        batch_size=4,
        max_steps=40,
        lr=3e-4,
        log_interval=10,
        eval_interval=20,
        device="cpu",
    )
    trainer = SFTTrainer(model, tokenizer, cfg, out_dir=tmp_path, n_examples=200)
    before = trainer.evaluate()
    result = trainer.train()
    assert result.final_loss < before
    assert result.supervised_tokens > 0
    assert (tmp_path / "final.pt").exists()
    assert (tmp_path / "manifest.json").exists()


# =====================================================================================
# DPO / SimPO losses against hand-computed values
# =====================================================================================


def test_sequence_logprobs_sums_only_masked_positions() -> None:
    # Two positions, a vocabulary of two, uniform logits -> log(0.5) per token.
    logits = torch.zeros(1, 2, 2)
    targets = torch.tensor([[0, 1]])
    mask = torch.tensor([[1.0, 0.0]])
    total = sequence_logprobs(logits, targets, mask)
    assert float(total) == pytest.approx(math.log(0.5))

    both = sequence_logprobs(logits, targets, torch.ones(1, 2))
    assert float(both) == pytest.approx(2 * math.log(0.5))
    assert float(sequence_logprobs(logits, targets, torch.ones(1, 2), average=True)) == (
        pytest.approx(math.log(0.5))
    )


def test_dpo_loss_matches_a_hand_computed_value() -> None:
    # π_θ(y_w) = -1, π_ref(y_w) = -2 -> chosen reward = 0.1*(1) = 0.1
    # π_θ(y_l) = -3, π_ref(y_l) = -2 -> rejected reward = 0.1*(-1) = -0.1
    # margin = 0.2; loss = -log σ(0.2)
    out = dpo_loss(
        torch.tensor([-1.0]),
        torch.tensor([-3.0]),
        torch.tensor([-2.0]),
        torch.tensor([-2.0]),
        beta=0.1,
    )
    expected = -math.log(1.0 / (1.0 + math.exp(-0.2)))
    assert float(out.loss) == pytest.approx(expected, rel=1e-6)
    assert float(out.chosen_reward) == pytest.approx(0.1)
    assert float(out.rejected_reward) == pytest.approx(-0.1)
    assert float(out.margin) == pytest.approx(0.2)
    assert float(out.accuracy) == 1.0


def test_dpo_loss_is_zero_when_the_policy_equals_the_reference_up_to_a_large_margin() -> None:
    """A policy that has learned the preference perfectly drives the loss toward 0."""
    out = dpo_loss(
        torch.tensor([0.0]),
        torch.tensor([-100.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        beta=1.0,
    )
    assert float(out.loss) < 1e-6


def test_dpo_at_the_reference_gives_log_two() -> None:
    """With π_θ = π_ref both rewards are 0, so the loss is exactly -log σ(0) = log 2."""
    zeros = torch.zeros(4)
    out = dpo_loss(zeros, zeros, zeros, zeros, beta=0.1)
    assert float(out.loss) == pytest.approx(math.log(2.0), rel=1e-6)
    assert float(out.accuracy.mean()) == 0.0  # margin is exactly 0, so not > 0


def test_dpo_label_smoothing_symmetrises_the_objective() -> None:
    """At eps=0.5 the loss must be independent of the margin's sign."""
    a = dpo_loss(
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        torch.zeros(1),
        torch.zeros(1),
        beta=1.0,
        label_smoothing=0.5,
    )
    b = dpo_loss(
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.zeros(1),
        torch.zeros(1),
        beta=1.0,
        label_smoothing=0.5,
    )
    assert float(a.loss) == pytest.approx(float(b.loss), rel=1e-6)


def test_simpo_loss_matches_a_hand_computed_value() -> None:
    # chosen reward = 2.0 * -0.5 = -1.0; rejected = 2.0 * -1.5 = -3.0
    # logits = -1.0 - (-3.0) - gamma(0.5) = 1.5; loss = -log σ(1.5)
    out = simpo_loss(torch.tensor([-0.5]), torch.tensor([-1.5]), beta=2.0, gamma=0.5)
    expected = -math.log(1.0 / (1.0 + math.exp(-1.5)))
    assert float(out.loss) == pytest.approx(expected, rel=1e-6)
    assert float(out.margin) == pytest.approx(2.0)  # reported without the target offset


def test_simpo_target_margin_raises_the_bar() -> None:
    """A larger gamma must make the same policy score a higher loss."""
    small = simpo_loss(torch.tensor([-1.0]), torch.tensor([-1.2]), beta=2.0, gamma=0.0)
    large = simpo_loss(torch.tensor([-1.0]), torch.tensor([-1.2]), beta=2.0, gamma=2.0)
    assert float(large.loss) > float(small.loss)


def test_dpo_reward_grows_with_length_but_simpo_reward_does_not() -> None:
    """The mechanism behind DPO's length bias, demonstrated directly (spec E4).

    Take a response whose per-token log-probability is constant. DPO sums, so doubling
    the length doubles the reward; SimPO averages, so it does not move at all.
    """
    per_token = -0.5
    short_len, long_len = 10, 20
    short_sum = torch.tensor([per_token * short_len])
    long_sum = torch.tensor([per_token * long_len])
    ref = torch.tensor([0.0])

    dpo_short = dpo_loss(short_sum, ref, ref, ref, beta=1.0)
    dpo_long = dpo_loss(long_sum, ref, ref, ref, beta=1.0)
    assert float(dpo_long.chosen_reward) == pytest.approx(2 * float(dpo_short.chosen_reward))

    avg = torch.tensor([per_token])
    simpo_short = simpo_loss(avg, avg, beta=1.0, gamma=0.0)
    simpo_long = simpo_loss(avg, avg, beta=1.0, gamma=0.0)
    assert float(simpo_long.chosen_reward) == pytest.approx(float(simpo_short.chosen_reward))


def test_preference_loss_stats_are_scalars() -> None:
    out = dpo_loss(torch.randn(4), torch.randn(4), torch.randn(4), torch.randn(4))
    stats = out.stats()
    assert set(stats) == {
        "loss",
        "chosen_reward",
        "rejected_reward",
        "reward_margin",
        "reward_accuracy",
    }
    assert all(isinstance(v, float) for v in stats.values())


# =====================================================================================
# Preference trainer
# =====================================================================================


def test_preference_batches_mask_only_responses(tokenizer: BPETokenizer) -> None:
    pairs = list(iter_preference_pairs(seed=1, n=8))
    batches = build_preference_batches(tokenizer, pairs, seq_len=96, batch_size=4)
    assert batches
    batch = batches[0]
    assert batch.chosen_mask.sum() > 0
    assert batch.rejected_mask.sum() > 0
    assert float(batch.chosen_mask.max()) == 1.0
    assert batch.chosen_lengths.shape == (4,)


def test_dpo_freezes_the_reference_policy(
    tokenizer: BPETokenizer, model: NanoScaleLM, tmp_path: Path
) -> None:
    cfg = PreferenceConfig(method="dpo", seq_len=96, batch_size=2, max_steps=3, device="cpu")
    trainer = PreferenceTrainer(model, tokenizer, cfg, out_dir=tmp_path, n_pairs=16)
    assert trainer.reference is not None
    assert not trainer.reference.training
    assert all(not p.requires_grad for p in trainer.reference.parameters())

    before = [p.detach().clone() for p in trainer.reference.parameters()]
    trainer.train()
    for a, b in zip(before, trainer.reference.parameters(), strict=True):
        torch.testing.assert_close(a, b.detach()), "the reference policy drifted"


def test_simpo_allocates_no_reference_model(
    tokenizer: BPETokenizer, model: NanoScaleLM, tmp_path: Path
) -> None:
    """SimPO's headline practical benefit: half the memory."""
    cfg = PreferenceConfig(method="simpo", seq_len=96, batch_size=2, max_steps=2, device="cpu")
    trainer = PreferenceTrainer(model, tokenizer, cfg, out_dir=tmp_path, n_pairs=16)
    assert trainer.reference is None


def test_preference_training_raises_the_reward_margin(
    tokenizer: BPETokenizer, model: NanoScaleLM, tmp_path: Path
) -> None:
    cfg = PreferenceConfig(
        method="dpo",
        seq_len=96,
        batch_size=4,
        max_steps=30,
        lr=1e-4,
        log_interval=5,
        device="cpu",
    )
    trainer = PreferenceTrainer(model, tokenizer, cfg, out_dir=tmp_path, n_pairs=120)
    result = trainer.train()
    margins = [row["reward_margin"] for row in result.history if "reward_margin" in row]
    assert margins[-1] > margins[0], f"reward margin did not increase: {margins}"


# =====================================================================================
# GRPO
# =====================================================================================


def test_group_relative_advantages_are_zero_mean() -> None:
    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0]])
    adv = group_relative_advantages(rewards)
    torch.testing.assert_close(adv.mean(dim=-1), torch.zeros(2), atol=1e-6, rtol=0)
    assert float(adv[0, 0]) > 0 > float(adv[0, 1])


def test_a_unanimous_group_carries_no_signal() -> None:
    """All-right or all-wrong says nothing about which action to reinforce."""
    assert float(group_relative_advantages(torch.ones(1, 8)).abs().max()) == 0.0
    assert float(group_relative_advantages(torch.zeros(1, 8)).abs().max()) == 0.0


def test_verify_arithmetic_is_strict_about_which_number() -> None:
    assert verify_arithmetic("The answer is 7.", 7) == 1.0
    assert verify_arithmetic("7", 7) == 1.0
    assert verify_arithmetic("-3 is the answer", -3) == 1.0
    assert verify_arithmetic("The answer is 8.", 7) == 0.0
    assert verify_arithmetic("no numbers here", 7) == 0.0
    # A model listing every number must not be rewarded for containing the right one.
    assert verify_arithmetic("1 2 3 4 5 6 7 8 9", 7) == 0.0


def test_arithmetic_tasks_are_solvable_and_correct() -> None:
    for task in make_arithmetic_tasks(seed=1, n=100):
        assert verify_arithmetic(str(task.answer), task.answer) == 1.0
        assert task.answer >= 0


def test_grpo_runs_and_reports_accuracy(
    tokenizer: BPETokenizer, model: NanoScaleLM, tmp_path: Path
) -> None:
    cfg = GRPOConfig(
        group_size=4, n_prompts=2, max_steps=2, max_new_tokens=8, log_interval=1, device="cpu"
    )
    tasks = [ArithmeticTask("What is 1 plus 1?", 2)] * 4
    trainer = GRPOTrainer(model, tokenizer, cfg, out_dir=tmp_path, tasks=tasks)
    result = trainer.train()
    assert 0.0 <= result.accuracy_before <= 1.0
    assert 0.0 <= result.accuracy_after <= 1.0
    assert (tmp_path / "manifest.json").exists()


# =====================================================================================
# The scripted preference eval
# =====================================================================================


def test_repetition_rate_detects_looping() -> None:
    assert repetition_rate("the cat sat on the mat quietly today") == 0.0
    assert repetition_rate("a b c a b c a b c a b c") > 0.5
    assert repetition_rate("short") == 0.0


def test_score_completion_rewards_the_stated_criteria() -> None:
    prompt = "What did Lily find at the park?"
    good = score_completion(prompt, "Lily found a shiny key at the park.", terminated=True)
    off_topic = score_completion(prompt, "The weather was cold and grey.", terminated=True)
    looping = score_completion(prompt, "Lily found. " * 8, terminated=True)
    unterminated = score_completion(prompt, "Lily found a key at the park.", terminated=False)

    assert good.total > off_topic.total
    assert good.non_degenerate > looping.non_degenerate
    assert good.terminated > unterminated.terminated


def test_the_judge_is_not_length_sensitive() -> None:
    """A model that gamed DPO's length bias must gain nothing here."""
    prompt = "What did Lily find at the park?"
    short = score_completion(prompt, "Lily found a key at the park.", terminated=True)
    padded = score_completion(
        prompt,
        "Lily found a key at the park. It was there. Then more happened after that.",
        terminated=True,
    )
    assert short.total >= padded.total - 1e-9


def test_head_to_head_is_paired_and_symmetric(tokenizer: BPETokenizer, model: NanoScaleLM) -> None:
    import copy

    other = copy.deepcopy(model)
    pairs = list(iter_preference_pairs(seed=1, n=6))
    result = head_to_head(model, other, tokenizer, pairs, n_prompts=6, max_new_tokens=8)
    assert result.n_prompts == 6
    assert result.wins_a + result.wins_b + result.ties == 6
    # Identical models under identical seeds must tie on every prompt.
    assert result.ties == 6
    assert result.win_rate_b == 0.5
    assert result.summary()["n_prompts"] == 6


def test_sft_loss_weight_adds_an_anchoring_nll_term(
    tokenizer: BPETokenizer, model: NanoScaleLM, tmp_path: Path
) -> None:
    """The RPO-style fix for DPO's likelihood collapse must actually change the loss."""
    pairs = list(iter_preference_pairs(seed=1, n=16))
    plain_cfg = PreferenceConfig(
        method="dpo", seq_len=96, batch_size=4, max_steps=1, sft_loss_weight=0.0, device="cpu"
    )
    anchored_cfg = plain_cfg.merged(sft_loss_weight=1.0)

    plain = PreferenceTrainer(model, tokenizer, plain_cfg, out_dir=tmp_path / "a", pairs=pairs)
    anchored = PreferenceTrainer(
        model, tokenizer, anchored_cfg, out_dir=tmp_path / "b", pairs=pairs
    )
    batch = plain.batches[0]

    plain_out, plain_aux = plain.compute_loss(batch)
    anchored_out, anchored_aux = anchored.compute_loss(batch)

    assert "sft_nll" not in plain_aux
    assert "sft_nll" in anchored_aux
    assert float(anchored_out.loss.detach()) > float(plain_out.loss.detach())
    # The preference part is untouched -- only an extra term is added.
    torch.testing.assert_close(plain_out.margin, anchored_out.margin)
    assert float(anchored_out.loss.detach()) == pytest.approx(
        float(plain_out.loss.detach()) + anchored_aux["sft_nll"], rel=1e-5
    )


def test_preference_trainer_reports_absolute_logprobs(
    tokenizer: BPETokenizer, model: NanoScaleLM, tmp_path: Path
) -> None:
    """Without these, a run can show a rising margin while collapsing both likelihoods."""
    cfg = PreferenceConfig(
        method="dpo", seq_len=96, batch_size=4, max_steps=4, log_interval=1, device="cpu"
    )
    trainer = PreferenceTrainer(model, tokenizer, cfg, out_dir=tmp_path, n_pairs=32)
    result = trainer.train()
    assert result.history
    for row in result.history:
        assert "chosen_logp" in row and "rejected_logp" in row
        assert row["chosen_logp"] < 0.0  # per-token log-probabilities are negative
