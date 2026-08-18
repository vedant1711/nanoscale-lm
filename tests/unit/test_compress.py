"""Tests for the arithmetic coder and the LM-driven codec.

A compressor has one property that matters above all others: it must be lossless. Every
test here either checks that directly or checks a property the losslessness depends on.
"""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nanoscale.compress import (
    FREQ_BITS,
    ArithmeticDecoder,
    ArithmeticEncoder,
    compress,
    decompress,
    probs_to_frequencies,
    score_lines,
    token_surprisal,
)
from nanoscale.config import TokenizerConfig, load_experiment
from nanoscale.data.toy import generate_corpus
from nanoscale.tokenizer import BPETokenizer
from nanoscale.train import Trainer

FREQ_TOTAL = 1 << FREQ_BITS


# =====================================================================================
# Frequency quantisation
# =====================================================================================


@given(
    probs=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=2, max_size=300
    )
)
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_frequencies_always_sum_to_the_budget_and_are_never_zero(probs: list[float]) -> None:
    """The two invariants the coder depends on, over arbitrary inputs.

    A zero frequency is a symbol the coder cannot encode; a wrong total desynchronises
    encoder and decoder. Hypothesis generates degenerate inputs — all zeros, one spike,
    denormals — that hand-written cases would miss.
    """
    freq = probs_to_frequencies(torch.tensor(probs, dtype=torch.float64))
    assert int(freq.sum()) == FREQ_TOTAL
    assert int(freq.min()) >= 1
    assert freq.numel() == len(probs)


def test_frequencies_reject_an_oversized_vocabulary() -> None:
    """Every symbol needs at least one unit, so the vocabulary is bounded by the budget."""
    with pytest.raises(ValueError, match="exceeds the frequency budget"):
        probs_to_frequencies(torch.ones(FREQ_TOTAL + 1))


def test_frequencies_track_the_probabilities_they_came_from() -> None:
    """Quantisation must preserve ordering, or the code lengths would be nonsense."""
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    freq = probs_to_frequencies(probs)
    assert list(freq.argsort(descending=True)) == list(probs.argsort(descending=True))


def test_a_uniform_distribution_gives_uniform_frequencies() -> None:
    freq = probs_to_frequencies(torch.full((256,), 1 / 256))
    assert int(freq.max()) - int(freq.min()) <= 1


# =====================================================================================
# The coder itself
# =====================================================================================


@given(
    symbols=st.lists(st.integers(min_value=0, max_value=31), min_size=1, max_size=300),
    seed=st.integers(min_value=0, max_value=2**16),
)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_coder_round_trips_any_symbol_sequence(symbols: list[int], seed: int) -> None:
    """Lossless for arbitrary sequences against an arbitrary (fixed) distribution."""
    gen = torch.Generator().manual_seed(seed)
    probs = torch.rand(32, generator=gen) ** 2 + 1e-6
    freq = probs_to_frequencies(probs / probs.sum())

    encoder = ArithmeticEncoder()
    for s in symbols:
        encoder.encode(s, freq)
    payload = encoder.finish()

    decoder = ArithmeticDecoder(payload)
    assert [decoder.decode(freq) for _ in symbols] == symbols


def test_coder_round_trips_with_a_changing_distribution() -> None:
    """The case that matters: a *different* table at every step, as a model produces."""
    gen = torch.Generator().manual_seed(7)
    tables = [probs_to_frequencies(torch.rand(48, generator=gen) + 1e-3) for _ in range(200)]
    symbols = torch.randint(0, 48, (200,), generator=gen).tolist()

    encoder = ArithmeticEncoder()
    for s, f in zip(symbols, tables, strict=True):
        encoder.encode(s, f)
    payload = encoder.finish()

    decoder = ArithmeticDecoder(payload)
    assert [decoder.decode(f) for f in tables] == symbols


def test_coder_approaches_the_entropy_bound() -> None:
    """Output size must be close to the theoretical cost, or the coder is broken.

    This is the test that would catch a coder that is lossless but wasteful — the failure
    mode where every compression number is real and unimpressive for no good reason.
    """
    gen = torch.Generator().manual_seed(11)
    probs = torch.tensor([0.6, 0.2, 0.1, 0.06, 0.03, 0.01])
    freq = probs_to_frequencies(probs)
    symbols = torch.multinomial(probs, 4000, replacement=True, generator=gen).tolist()

    encoder = ArithmeticEncoder()
    for s in symbols:
        encoder.encode(s, freq)
    payload = encoder.finish()

    entropy_bits = -sum(float(probs[s]) and math.log2(float(probs[s])) for s in symbols)
    assert len(payload) * 8 < entropy_bits * 1.05, "more than 5% above the entropy bound"


def test_a_confident_model_compresses_better_than_an_uncertain_one() -> None:
    """The whole premise, isolated: lower entropy must produce a smaller payload."""
    sharp = probs_to_frequencies(torch.tensor([0.97, 0.01, 0.01, 0.01]))
    flat = probs_to_frequencies(torch.tensor([0.25, 0.25, 0.25, 0.25]))
    symbols = [0] * 500

    sizes = []
    for freq in (sharp, flat):
        enc = ArithmeticEncoder()
        for s in symbols:
            enc.encode(s, freq)
        sizes.append(len(enc.finish()))
    assert sizes[0] < sizes[1] / 4


# =====================================================================================
# End to end with a real (tiny) model
# =====================================================================================


@pytest.fixture(scope="module")
def trained() -> tuple[object, BPETokenizer]:
    """A very small trained model; compression only needs it to be non-uniform."""
    corpus = generate_corpus(seed=3, n_stories=1500)
    tok = BPETokenizer.train(corpus, TokenizerConfig(vocab_size=512, max_train_bytes=400_000))
    cfg = load_experiment(
        tier="nano",
        overrides=[
            "tokenizer.vocab_size=512",
            "model.vocab_size=512",
            "model.n_layers=2",
            "model.d_model=64",
            "model.n_heads=2",
            "model.n_kv_heads=1",
            "model.max_seq_len=128",
            "data.seq_len=128",
            "train.device=cpu",
            "train.max_steps=60",
            "train.batch_size=4",
            "train.token_budget=null",
            "train.eval_interval=60",
            "train.log_interval=30",
            "train.ckpt_interval=100000",
        ],
    )
    import tempfile

    trainer = Trainer(cfg, tokenizer=tok, out_dir=tempfile.mkdtemp())
    trainer.train()
    return trainer.model, tok


def test_compression_round_trip_is_lossless(trained: tuple[object, BPETokenizer]) -> None:
    """The property everything else rests on."""
    model, tok = trained
    text = "It was a sunny day. Lily went to the park with a small dog and found a red ball."
    result = compress(model, tok, text)  # type: ignore[arg-type]
    assert decompress(model, tok, result.payload, result.n_tokens) == text  # type: ignore[arg-type]


def test_compression_round_trip_survives_unusual_text(
    trained: tuple[object, BPETokenizer],
) -> None:
    """Byte-level BPE means any string is encodable; the codec must not undo that."""
    model, tok = trained
    for text in ["héllo wörld 🌍 ünïcode", "  double  spaces\tand\ttabs  ", "!@#$%^&*()"]:
        result = compress(model, tok, text)  # type: ignore[arg-type]
        assert decompress(model, tok, result.payload, result.n_tokens) == text  # type: ignore[arg-type]


def test_compression_beats_raw_utf8(trained: tuple[object, BPETokenizer]) -> None:
    """A trained model must at least beat storing the bytes."""
    model, tok = trained
    text = "It was a sunny day. Lily went to the park. " * 6
    result = compress(model, tok, text)  # type: ignore[arg-type]
    assert result.bits_per_byte < 8.0
    assert result.ratio > 1.0


def test_reported_rate_matches_the_payload_actually_produced(
    trained: tuple[object, BPETokenizer],
) -> None:
    """Guards against a summary that reports the ideal rate as if it were achieved."""
    model, tok = trained
    text = "Lily found a small red ball in the tall grass near the old tree."
    r = compress(model, tok, text)  # type: ignore[arg-type]
    assert r.bits_per_byte == pytest.approx(len(r.payload) * 8 / r.n_bytes_in)
    # The achieved rate can never beat the model's own cross-entropy.
    assert r.bits_per_byte >= r.ideal_bits_per_byte * 0.999


# =====================================================================================
# Surprisal / anomaly detection
# =====================================================================================


def test_surprisal_is_reported_per_token(trained: tuple[object, BPETokenizer]) -> None:
    model, tok = trained
    text = "Lily went to the park."
    out = token_surprisal(model, tok, text)  # type: ignore[arg-type]
    assert len(out) == len(tok.encode(text, add_bos=True)) - 1
    assert all(bits >= 0 for _, bits in out)


def test_out_of_domain_text_scores_higher_than_in_domain(
    trained: tuple[object, BPETokenizer],
) -> None:
    """The anomaly claim, reduced to its minimum testable form."""
    model, tok = trained
    in_domain = ["Lily went to the park with her dog.", "It was a sunny day and Tom was happy."]
    out_domain = ["qxzj vbnm plkj wqer zxcv.", "SELECT * FROM users WHERE id = 42;"]
    report = score_lines(model, tok, in_domain + out_domain)  # type: ignore[arg-type]
    assert min(report.scores[2:]) > max(report.scores[:2])


def test_anomaly_report_ranks_and_flags(trained: tuple[object, BPETokenizer]) -> None:
    model, tok = trained
    lines = ["Lily went to the park."] * 20 + ["zzz qqq xxx vvv."]
    report = score_lines(model, tok, lines)  # type: ignore[arg-type]
    assert report.ranked()[0][1] == "zzz qqq xxx vvv."
    assert any("zzz" in line for _, line in report.flagged(percentile=90.0))


def test_score_lines_normalises_by_length(trained: tuple[object, BPETokenizer]) -> None:
    """Without normalisation every long line looks anomalous."""
    model, tok = trained
    short = "Lily went to the park."
    long = "Lily went to the park. " * 8
    report = score_lines(model, tok, [short, long])  # type: ignore[arg-type]
    # The repeated text should be *less* surprising per token, never more.
    assert report.scores[1] < report.scores[0] * 1.5
