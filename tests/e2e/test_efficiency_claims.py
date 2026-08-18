"""Efficiency-claim direction tests (spec D4).

Every efficiency number this project publishes comes from a committed benchmark script.
Those scripts are too slow for CI and their absolute numbers are hardware-dependent, so
they cannot be gates. What *can* be a gate is the **direction** of each claim, re-derived
at `nano` scale on a CPU runner. If a refactor inverts one of these, the README becomes
false, and this file is what notices.

One claim in the spec is deliberately **not** asserted here:

    "speculative tokens/sec > autoregressive"

Because at this scale it is false, and measured to be false, 0.79x at gamma=6 (see
`results/speculative/speculative.md`). Speculation trades compute for memory bandwidth,
and a 5M-parameter forward pass on a CPU is bound by Python dispatch, not by streaming
weights. Asserting a claim the project does not make would be worse than useless. What is
asserted instead is the hardware-independent quantity the method actually delivers, and
which the docs actually claim: **fewer target forward passes for the same output**.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from nanoscale.bench import model_memory_bytes
from nanoscale.config import (
    ExperimentConfig,
    TokenizerConfig,
    draft_model_config,
    load_experiment,
)
from nanoscale.data.toy import generate_corpus
from nanoscale.distill import DistillTrainer
from nanoscale.eval import perplexity, run_tiny_bench
from nanoscale.model import NanoScaleLM, build_model
from nanoscale.quantize import GPTQQuantizer, effective_bits, quantize_rtn
from nanoscale.specdec import SpeculativeSampler, autoregressive_baseline
from nanoscale.tokenizer import BPETokenizer
from nanoscale.train import Batch, TokenBatcher, Trainer, build_packed_tokens

pytestmark = pytest.mark.slow


def _config() -> ExperimentConfig:
    return load_experiment(
        tier="nano",
        overrides=[
            "train.device=cpu",
            "train.max_steps=150",
            "train.token_budget=null",
            "train.eval_interval=150",
            "train.log_interval=50",
            "train.ckpt_interval=100000",
        ],
    )


@pytest.fixture(scope="module")
def trained(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[NanoScaleLM, BPETokenizer, ExperimentConfig, list[Batch], list[Batch]]:
    """One trained `nano` model, shared by every claim below."""
    tok = BPETokenizer.train(
        generate_corpus(seed=1337, n_stories=4000),
        TokenizerConfig(vocab_size=1024, max_train_bytes=1_200_000),
    )
    cfg = _config()
    out: Path = tmp_path_factory.mktemp("claims")
    trainer = Trainer(cfg, tokenizer=tok, out_dir=out / "pretrain")
    trainer.train()

    data = build_packed_tokens(cfg.data, tok)
    val = TokenBatcher(data.val, seq_len=cfg.data.seq_len, batch_size=4, shuffle=False).take(8)
    calib = TokenBatcher(data.train, seq_len=cfg.data.seq_len, batch_size=4, seed=1337).take(4)
    return trainer.model, tok, cfg, val, calib


# =====================================================================================
# Claim: GPTQ is never worse than RTN, and is clearly better where bits are scarce
# =====================================================================================


@pytest.mark.parametrize("bits", [4, 2])
def test_gptq_is_at_least_as_good_as_rtn(
    trained: tuple[NanoScaleLM, BPETokenizer, ExperimentConfig, list[Batch], list[Batch]],
    bits: int,
) -> None:
    """README claim: 2-bit GPTQ beats RTN; 4-bit is reported as a tie, not a win."""
    model, _tok, _cfg, val, calib = trained

    rtn = copy.deepcopy(model)
    quantize_rtn(rtn, bits=bits, group_size=64)
    rtn_ppl = perplexity(rtn, val).perplexity

    gptq = copy.deepcopy(model)
    quantizer = GPTQQuantizer(gptq, bits=bits, group_size=64, act_order=True)
    quantizer.collect([b.inputs for b in calib])
    quantizer.apply()
    gptq_ppl = perplexity(gptq, val).perplexity

    assert gptq_ppl <= rtn_ppl * 1.01, f"{bits}-bit: GPTQ {gptq_ppl:.4f} lost to RTN {rtn_ppl:.4f}"
    if bits == 2:
        # This is the claim the docs actually make a point of, so it gets a real margin.
        assert gptq_ppl < rtn_ppl, (
            f"2-bit GPTQ {gptq_ppl:.4f} did not beat RTN {rtn_ppl:.4f}; the whole argument "
            "for Hessian-guided error compensation is that it wins when bits are scarce"
        )


def test_four_bit_quantization_shrinks_weights_without_hurting_quality(
    trained: tuple[NanoScaleLM, BPETokenizer, ExperimentConfig, list[Batch], list[Batch]],
) -> None:
    """README claim: 4.3x smaller weights at no measurable perplexity or benchmark cost."""
    model, tok, _cfg, val, calib = trained

    gptq = copy.deepcopy(model)
    quantizer = GPTQQuantizer(gptq, bits=4, group_size=64, act_order=True)
    quantizer.collect([b.inputs for b in calib])
    quantizer.apply()

    fp32_bytes = model_memory_bytes(model)
    q4_bytes = model_memory_bytes(gptq, weight_bits=effective_bits(4, group_size=64))
    assert fp32_bytes / q4_bytes > 3.5, f"only {fp32_bytes / q4_bytes:.2f}x smaller"

    assert perplexity(gptq, val).perplexity <= perplexity(model, val).perplexity * 1.01
    assert run_tiny_bench(gptq, tok).accuracy >= run_tiny_bench(model, tok).accuracy - 0.1


# =====================================================================================
# Claim: distillation makes a much smaller model with a bounded quality drop
# =====================================================================================


def test_distilled_student_is_much_smaller_with_bounded_quality_loss(
    trained: tuple[NanoScaleLM, BPETokenizer, ExperimentConfig, list[Batch], list[Batch]],
    tmp_path: Path,
) -> None:
    """README claim: 17.7x fewer parameters, still a usable model."""
    teacher, tok, cfg, val, _calib = trained

    data = build_packed_tokens(cfg.data, tok)
    batcher = TokenBatcher(data.train, seq_len=128, batch_size=8, seed=1337)
    dcfg = cfg.distill.merged(
        method="forward_kl", max_steps=300, warmup_steps=200, lr=1e-3, seq_len=128, device="cpu"
    )
    student = build_model(draft_model_config(cfg.model))
    trainer = DistillTrainer(
        teacher, student, tok, dcfg, train_batcher=batcher, out_dir=tmp_path / "distill"
    )
    result = trainer.train()

    assert result.teacher_params / result.student_params > 10.0
    # "Bounded" is the operative word: the student is genuinely worse, and the claim is
    # only that it stays in the same order of magnitude rather than collapsing.
    teacher_ppl = perplexity(teacher, val).perplexity
    student_ppl = perplexity(trainer.student, val).perplexity
    assert student_ppl < teacher_ppl * 4.0, (
        f"student ppl {student_ppl:.3f} vs teacher {teacher_ppl:.3f}: distillation collapsed"
    )


# =====================================================================================
# Claim: speculation reduces target forward passes, exactly
# =====================================================================================


def test_speculation_cuts_target_forward_passes(
    trained: tuple[NanoScaleLM, BPETokenizer, ExperimentConfig, list[Batch], list[Batch]],
    tmp_path: Path,
) -> None:
    """README claim: ~3x fewer target passes at gamma=6, output distribution unchanged.

    The draft is distilled here rather than random, because that is the configuration the
    claim is about; an untrained draft has an acceptance rate near zero and speculation
    correctly degenerates to autoregressive decoding.
    """
    target, tok, cfg, _val, _calib = trained

    data = build_packed_tokens(cfg.data, tok)
    batcher = TokenBatcher(data.train, seq_len=128, batch_size=8, seed=1337)
    dcfg = cfg.distill.merged(
        method="forward_kl", max_steps=300, warmup_steps=200, lr=1e-3, seq_len=128, device="cpu"
    )
    draft_trainer = DistillTrainer(
        target,
        build_model(draft_model_config(cfg.model)),
        tok,
        dcfg,
        train_batcher=batcher,
        out_dir=tmp_path / "draft",
    )
    draft_trainer.train()
    draft = draft_trainer.student.eval()

    prompt = torch.tensor([tok.encode("It was a sunny day. Lily went to", add_bos=True)])
    n = 48

    baseline = autoregressive_baseline(
        target, prompt, max_new_tokens=n, generator=torch.Generator().manual_seed(0)
    )
    spec = SpeculativeSampler(target, draft, gamma=6).generate(
        prompt, max_new_tokens=n, generator=torch.Generator().manual_seed(0)
    )

    assert spec.target_calls < baseline.target_calls, (
        f"speculation used {spec.target_calls} target passes vs {baseline.target_calls} "
        "autoregressive: it bought nothing"
    )
    assert spec.mean_accepted_length > 1.5, (
        f"only {spec.mean_accepted_length:.2f} tokens per target pass with a distilled draft"
    )


def test_greedy_speculation_is_exactly_lossless(
    trained: tuple[NanoScaleLM, BPETokenizer, ExperimentConfig, list[Batch], list[Batch]],
) -> None:
    """README claim: the output distribution is provably unchanged.

    Greedy is the case where "unchanged distribution" collapses to a checkable equality,
    so it is the one that belongs in a gate. The distributional version lives in the
    unit tests, where it can afford the sample count it needs.
    """
    target, tok, cfg, _val, _calib = trained
    draft = build_model(draft_model_config(cfg.model))  # even a useless draft must be lossless
    prompt = torch.tensor([tok.encode("It was a sunny day. Lily went to", add_bos=True)])

    for gamma in (1, 4, 7):
        spec = SpeculativeSampler(target, draft, gamma=gamma, temperature=0.0).generate(
            prompt, max_new_tokens=24
        )
        base = autoregressive_baseline(target, prompt, max_new_tokens=24, temperature=0.0)
        assert torch.equal(spec.tokens, base.tokens), f"gamma={gamma} changed the output"
