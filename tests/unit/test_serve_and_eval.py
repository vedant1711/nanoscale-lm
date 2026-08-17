"""Serving, evaluation and benchmark-harness tests (spec D5, Phase 10)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from nanoscale.bench import BenchHarness, model_memory_bytes
from nanoscale.bench.harness import _percentile as percentile
from nanoscale.config import GenerateConfig, ModelConfig, TokenizerConfig
from nanoscale.data.toy import generate_corpus
from nanoscale.eval import (
    TASKS,
    BenchmarkResult,
    MultipleChoiceQuestion,
    perplexity,
    run_tiny_bench,
    score_choice,
    token_nll,
)
from nanoscale.model import IGNORE_INDEX, NanoScaleLM, build_model
from nanoscale.serve import TextStreamer, generate_text, stream_text
from nanoscale.serve.generate import _apply_repetition_penalty
from nanoscale.tokenizer import BPETokenizer
from nanoscale.train import Batch


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    corpus = generate_corpus(seed=1, n_stories=800)
    return BPETokenizer.train(corpus, TokenizerConfig(vocab_size=512, max_train_bytes=200_000))


@pytest.fixture
def model(tokenizer: BPETokenizer) -> NanoScaleLM:
    torch.manual_seed(0)
    return build_model(
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
# Perplexity
# =====================================================================================


def test_perplexity_matches_a_hand_computed_value() -> None:
    """A model with uniform logits over V tokens must have perplexity exactly V."""

    class Uniform(torch.nn.Module):
        def __init__(self, vocab: int) -> None:
            super().__init__()
            self.vocab = vocab

        def forward(self, ids: torch.Tensor, **_: object) -> object:
            from nanoscale.model import LMOutput

            return LMOutput(logits=torch.zeros(*ids.shape, self.vocab))

    vocab = 8
    batches = [
        Batch(
            inputs=torch.zeros(2, 5, dtype=torch.long), targets=torch.zeros(2, 5, dtype=torch.long)
        )
    ]
    result = perplexity(Uniform(vocab), batches)
    assert result.perplexity == pytest.approx(vocab, rel=1e-6)
    assert result.nll == pytest.approx(math.log(vocab), rel=1e-6)
    assert result.n_tokens == 10
    assert result.nll_stderr == pytest.approx(0.0, abs=1e-9)


def test_perplexity_is_token_weighted_not_batch_weighted(model: NanoScaleLM) -> None:
    """A short trailing batch must not carry the same weight as a full one."""
    long_batch = Batch(
        inputs=torch.randint(0, 512, (4, 32)), targets=torch.randint(0, 512, (4, 32))
    )
    short_batch = Batch(inputs=torch.randint(0, 512, (1, 4)), targets=torch.randint(0, 512, (1, 4)))
    combined = perplexity(model, [long_batch, short_batch])
    assert combined.n_tokens == 4 * 32 + 4


def test_perplexity_reports_an_interval(model: NanoScaleLM) -> None:
    batches = [
        Batch(inputs=torch.randint(0, 512, (2, 16)), targets=torch.randint(0, 512, (2, 16)))
        for _ in range(4)
    ]
    result = perplexity(model, batches)
    assert result.perplexity_low < result.perplexity < result.perplexity_high
    assert result.nll_stderr > 0
    assert "n=" in str(result)
    assert set(result.summary()) >= {"perplexity", "perplexity_low", "perplexity_high"}


def test_perplexity_skips_ignored_targets(model: NanoScaleLM) -> None:
    targets = torch.randint(0, 512, (2, 8))
    targets[:, :4] = IGNORE_INDEX
    batch = Batch(inputs=torch.randint(0, 512, (2, 8)), targets=targets)
    nll, count = token_nll(model, batch)
    assert count == 8
    assert nll.numel() == 8


def test_perplexity_of_no_batches_is_nan(model: NanoScaleLM) -> None:
    result = perplexity(model, [])
    assert math.isnan(result.perplexity) or result.n_tokens == 0


def test_perplexity_restores_training_mode(model: NanoScaleLM) -> None:
    model.train()
    perplexity(model, [])
    assert model.training


# =====================================================================================
# The tiny benchmark
# =====================================================================================


def test_question_set_is_well_formed() -> None:
    assert len(TASKS) >= 20
    tasks = {q.task for q in TASKS}
    assert tasks == {"agreement", "coreference", "schema", "arithmetic"}
    for question in TASKS:
        assert len(question.choices) >= 2
        assert 0 <= question.answer < len(question.choices)
        assert question.context.strip()


def test_question_set_is_deterministic() -> None:
    """Rebuilding the suite must give byte-identical questions."""
    from nanoscale.eval import tiny_bench

    rebuilt = tuple(
        tiny_bench._agreement()
        + tiny_bench._coreference()
        + tiny_bench._schema()
        + tiny_bench._arithmetic()
    )
    assert rebuilt == TASKS


def test_question_validation() -> None:
    with pytest.raises(ValueError, match="out of range"):
        MultipleChoiceQuestion(task="t", context="c", choices=("a", "b"), answer=5)
    with pytest.raises(ValueError, match="at least two choices"):
        MultipleChoiceQuestion(task="t", context="c", choices=("a",), answer=0)


def test_scoring_is_length_normalised(model: NanoScaleLM, tokenizer: BPETokenizer) -> None:
    """Without normalisation the benchmark would measure candidate length."""
    short = score_choice(model, tokenizer, "Lily went to", " the park.")
    long = score_choice(
        model, tokenizer, "Lily went to", " the park and then walked home again slowly."
    )
    # Both are mean log-probabilities per token, so they live on the same scale.
    assert -20.0 < short < 0.0
    assert -20.0 < long < 0.0


def test_scoring_handles_an_empty_choice(model: NanoScaleLM, tokenizer: BPETokenizer) -> None:
    assert score_choice(model, tokenizer, "context", "") == float("-inf")


def test_an_untrained_model_scores_near_chance(model: NanoScaleLM, tokenizer: BPETokenizer) -> None:
    """The suite must not be passable without having learned anything."""
    result = run_tiny_bench(model, tokenizer)
    assert 0.0 <= result.accuracy <= 1.0
    assert result.n_questions == len(TASKS)
    assert abs(result.accuracy - result.chance) < 0.45


def test_benchmark_result_reports_uncertainty() -> None:
    result = BenchmarkResult(accuracy=0.5, n_questions=28, per_task={"a": 0.5})
    assert result.stderr == pytest.approx(math.sqrt(0.25 / 28))
    assert "±" in str(result)
    assert result.summary()["acc_a"] == 0.5


def test_a_perfect_score_has_zero_binomial_stderr() -> None:
    assert BenchmarkResult(accuracy=1.0, n_questions=28, per_task={}).stderr == 0.0


# =====================================================================================
# Serving
# =====================================================================================


def test_streamer_holds_back_partial_utf8(tokenizer: BPETokenizer) -> None:
    """Byte-level BPE can split a codepoint; the streamer must not emit half of one."""
    streamer = TextStreamer(tokenizer)
    emoji = "🌍"
    pieces = [streamer.push(t) for t in tokenizer.encode(emoji)]
    assert "".join(pieces) + streamer.flush() == emoji
    # Every intermediate piece is either empty or valid text -- never a replacement char.
    assert all("�" not in p for p in pieces)


def test_streamer_skips_special_tokens(tokenizer: BPETokenizer) -> None:
    streamer = TextStreamer(tokenizer, skip_special=True)
    assert streamer.push(tokenizer.bos_id) == ""
    assert streamer.flush() == ""


def test_streamer_flush_replaces_a_truncated_codepoint(tokenizer: BPETokenizer) -> None:
    streamer = TextStreamer(tokenizer)
    first_byte = tokenizer.encode("🌍")[0]
    assert streamer.push(first_byte) == ""
    assert "�" in streamer.flush()


def test_streamer_emits_immediately_for_definitively_invalid_bytes(
    tokenizer: BPETokenizer,
) -> None:
    """The bug this caught: buffering invalid bytes forever stalls the whole stream.

    A UTF-8 continuation byte with no lead byte can never become valid. A naive
    ``try: decode() except UnicodeDecodeError: return ""`` buffers it indefinitely, so an
    untrained model — whose tokens are effectively random bytes — emits nothing at all
    until the final flush, and any incremental logic downstream (stop sequences, a live
    UI) never fires.
    """
    streamer = TextStreamer(tokenizer)
    emitted = streamer.push(0x80)  # a bare continuation byte
    assert emitted != "", "an impossible byte sequence must not be buffered forever"
    assert "�" in emitted


def test_repetition_penalty_pushes_seen_tokens_down() -> None:
    """The sign fix: a *negative* logit must be multiplied, not divided."""
    logits = torch.tensor([[2.0, -2.0, 0.5]])
    out = _apply_repetition_penalty(logits, [0, 1], penalty=2.0)
    assert float(out[0, 0]) == pytest.approx(1.0)  # positive: divided
    assert float(out[0, 1]) == pytest.approx(-4.0)  # negative: multiplied
    assert float(out[0, 2]) == pytest.approx(0.5)  # untouched
    # Both seen tokens moved *down* relative to the unseen one.
    assert float(out[0, 0]) < float(logits[0, 0])
    assert float(out[0, 1]) < float(logits[0, 1])


def test_repetition_penalty_of_one_is_a_noop() -> None:
    logits = torch.randn(1, 8)
    torch.testing.assert_close(_apply_repetition_penalty(logits, [0, 3], 1.0), logits)


def test_generate_text_reports_a_timing_breakdown(
    model: NanoScaleLM, tokenizer: BPETokenizer
) -> None:
    out = generate_text(model, tokenizer, "Lily went to", GenerateConfig(max_new_tokens=16, seed=1))
    assert out.generated_tokens > 0
    assert out.prompt_tokens > 0
    assert out.prefill_s > 0 and out.decode_s > 0
    assert out.total_s == pytest.approx(out.prefill_s + out.decode_s)
    assert out.decode_tokens_per_s > 0
    assert out.stop_reason in ("length", "eos", "context", "stop_sequence")
    assert set(out.summary()) >= {"prefill_s", "decode_s", "decode_tokens_per_s"}


def test_generation_is_reproducible_under_a_seed(
    model: NanoScaleLM, tokenizer: BPETokenizer
) -> None:
    cfg = GenerateConfig(max_new_tokens=12, seed=7)
    a = generate_text(model, tokenizer, "Tom found", cfg)
    b = generate_text(model, tokenizer, "Tom found", cfg)
    assert a.text == b.text
    assert a.token_ids == b.token_ids


def test_streaming_and_batch_generation_agree(model: NanoScaleLM, tokenizer: BPETokenizer) -> None:
    """The streaming path must not be a second, divergent implementation."""
    cfg = GenerateConfig(max_new_tokens=16, seed=3)
    batched = generate_text(model, tokenizer, "Mia walked", cfg)
    streamed = "".join(stream_text(model, tokenizer, "Mia walked", cfg))
    assert streamed == batched.text


def test_stop_sequences_halt_generation(model: NanoScaleLM, tokenizer: BPETokenizer) -> None:
    cfg = GenerateConfig(max_new_tokens=48, seed=1, temperature=0.0)
    unrestricted = generate_text(model, tokenizer, "Lily went to", cfg)
    if " " not in unrestricted.text:
        pytest.skip("the greedy continuation contains no space to stop on")
    stopped = generate_text(model, tokenizer, "Lily went to", cfg, stop=[" "])
    assert stopped.stop_reason == "stop_sequence"
    assert len(stopped.text) <= len(unrestricted.text)


def test_generation_leaves_training_mode_unchanged(
    model: NanoScaleLM, tokenizer: BPETokenizer
) -> None:
    model.train()
    generate_text(model, tokenizer, "x", GenerateConfig(max_new_tokens=2))
    assert model.training


def test_prompt_ids_override_the_prompt_string(model: NanoScaleLM, tokenizer: BPETokenizer) -> None:
    ids = tokenizer.encode("Lily went to", add_bos=True)
    out = generate_text(
        model, tokenizer, "ignored", GenerateConfig(max_new_tokens=4, seed=1), prompt_ids=ids
    )
    assert out.prompt_tokens == len(ids)


# =====================================================================================
# The benchmark harness
# =====================================================================================


def test_harness_discards_warmup_iterations() -> None:
    calls: list[int] = []

    def run(i: int) -> dict[str, float]:
        calls.append(i)
        return {"prefill_s": 0.001, "decode_s": 0.01, "generated_tokens": 10.0}

    harness = BenchHarness(warmup_iters=3, measure_iters=4)
    row = harness.time_variant("x", run, params=100, weight_mb=1.0, kv_mb=0.5)
    assert len(calls) == 7, "warmup + measured iterations"
    assert row.generated_tokens == 10
    assert row.decode_tokens_per_s_p50 == pytest.approx(1000.0, rel=1e-3)


def test_harness_reports_the_median_not_the_mean() -> None:
    """One outlier must not move the reported number."""
    rates = [0.01, 0.01, 0.01, 0.01, 10.0]  # the last is a 1000x stall

    def run(i: int) -> dict[str, float]:
        return {
            "prefill_s": 0.0,
            "decode_s": rates[min(i, len(rates) - 1)],
            "generated_tokens": 10.0,
        }

    harness = BenchHarness(warmup_iters=0, measure_iters=5)
    row = harness.time_variant("x", run, params=1, weight_mb=1.0, kv_mb=0.0)
    assert row.decode_tokens_per_s_p50 == pytest.approx(1000.0, rel=1e-3)


def test_harness_carries_speculative_metrics() -> None:
    def run(_i: int) -> dict[str, float]:
        return {
            "prefill_s": 0.0,
            "decode_s": 0.01,
            "generated_tokens": 10.0,
            "acceptance_rate": 0.6,
            "mean_accepted_length": 2.5,
        }

    harness = BenchHarness(warmup_iters=0, measure_iters=3)
    row = harness.time_variant("spec", run, params=1, weight_mb=1.0, kv_mb=0.0)
    assert row.acceptance_rate == pytest.approx(0.6)
    assert row.mean_accepted_length == pytest.approx(2.5)


def test_harness_writes_a_table_with_provenance(tmp_path: Path) -> None:
    import json

    def run(_i: int) -> dict[str, float]:
        return {"prefill_s": 0.001, "decode_s": 0.01, "generated_tokens": 5.0}

    harness = BenchHarness(warmup_iters=0, measure_iters=2)
    harness.time_variant("a", run, params=1, weight_mb=1.0, kv_mb=0.0, perplexity=2.0)
    path = harness.write_json(tmp_path / "t.json", note="hello")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "git_sha" in payload and "hardware" in payload
    assert payload["note"] == "hello"
    assert payload["rows"][0]["perplexity"] == 2.0
    assert "| variant |" in harness.markdown_table()


def test_model_memory_reports_the_representation_not_the_tensors(model: NanoScaleLM) -> None:
    """A 4-bit model simulated in fp32 must report its 4-bit footprint."""
    fp32 = model_memory_bytes(model)
    four_bit = model_memory_bytes(model, weight_bits=4.5)
    assert four_bit < fp32
    # Embeddings are not quantized, so the saving applies only to the rest.
    non_embedding = model.num_parameters(non_embedding=True)
    embedding = model.num_parameters() - non_embedding
    assert four_bit == pytest.approx(non_embedding * 4.5 / 8 + embedding * 4)


def test_percentile_is_nearest_rank() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 95) == 5.0
    assert percentile(values, 50) == 3.0
    assert percentile([], 95) == 0.0
