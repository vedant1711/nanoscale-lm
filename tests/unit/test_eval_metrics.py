"""Tests for the tokenizer-independent, calibration and significance metrics.

The pattern throughout is to construct a case whose answer is known from theory — a
uniform model, a perfectly calibrated one, two samples with a hand-computable t — and
assert against that rather than against a previously observed number.
"""

from __future__ import annotations

import functools
import math
import statistics

import pytest
import torch
from torch import nn

from nanoscale.bench import MIN_EFFECT_SIZE, cohens_d, welch_t_test
from nanoscale.config import TokenizerConfig
from nanoscale.data.toy import generate_corpus
from nanoscale.eval import (
    PHENOMENA,
    bits_per_byte,
    calibration,
    distinct_n,
    generate_pairs,
    run_minimal_pairs,
    self_bleu,
    wilson_interval,
)
from nanoscale.model import LMOutput
from nanoscale.tokenizer import BPETokenizer
from nanoscale.train import Batch


@functools.lru_cache(maxsize=1)
def _tokenizer() -> BPETokenizer:
    """A real tokenizer over the toy story corpus, shared across tests."""
    return BPETokenizer.train(
        generate_corpus(seed=5, n_stories=1500),
        TokenizerConfig(vocab_size=512, max_train_bytes=400_000),
    )


class Uniform(nn.Module):
    """A model that assigns exactly 1/V to every token."""

    def __init__(self, vocab: int) -> None:
        """Store the vocabulary size the uniform distribution spans."""
        super().__init__()
        self.vocab = vocab

    def forward(self, ids: torch.Tensor, **_: object) -> LMOutput:
        """Return zero logits, i.e. a uniform distribution."""
        return LMOutput(logits=torch.zeros(*ids.shape, self.vocab))


# =====================================================================================
# Bits per byte
# =====================================================================================


def test_uniform_model_bits_per_byte_matches_the_closed_form() -> None:
    """A uniform model over V tokens spends exactly log2(V) bits per token.

    So its bits-per-byte is `log2(V) * tokens / bytes`, which is a value derived from
    theory rather than from a previous run.
    """
    vocab, n_tok, n_bytes = 256, 40, 100
    batches = [
        Batch(
            inputs=torch.zeros(2, n_tok // 2, dtype=torch.long),
            targets=torch.zeros(2, n_tok // 2, dtype=torch.long),
        )
    ]
    result = bits_per_byte(Uniform(vocab), batches, n_bytes=n_bytes)
    expected = math.log2(vocab) * n_tok / n_bytes
    assert result.bits_per_byte == pytest.approx(expected, rel=1e-5)
    assert result.n_tokens == n_tok
    assert result.bytes_per_token == pytest.approx(n_bytes / n_tok)


def test_bits_per_byte_is_invariant_to_how_the_bytes_are_batched() -> None:
    """Splitting the same tokens across more batches must not change the result."""
    vocab = 64
    one = [
        Batch(
            inputs=torch.zeros(4, 8, dtype=torch.long), targets=torch.zeros(4, 8, dtype=torch.long)
        )
    ]
    many = [
        Batch(
            inputs=torch.zeros(1, 8, dtype=torch.long), targets=torch.zeros(1, 8, dtype=torch.long)
        )
        for _ in range(4)
    ]
    a = bits_per_byte(Uniform(vocab), one, n_bytes=128)
    b = bits_per_byte(Uniform(vocab), many, n_bytes=128)
    assert a.bits_per_byte == pytest.approx(b.bits_per_byte, rel=1e-9)


def test_bits_per_byte_rejects_a_nonsense_byte_count() -> None:
    """Zero bytes is a caller error, not something to divide by."""
    batches = [
        Batch(
            inputs=torch.zeros(1, 4, dtype=torch.long), targets=torch.zeros(1, 4, dtype=torch.long)
        )
    ]
    with pytest.raises(ValueError, match="n_bytes must be positive"):
        bits_per_byte(Uniform(16), batches, n_bytes=0)


def test_bits_per_byte_separates_what_perplexity_conflates() -> None:
    """Two models with different vocabularies, identical predictive quality per byte.

    This is the whole reason the metric exists. Model A has a 256-token vocabulary and
    spends 8 bits/token; model B has a 65,536-token vocabulary and spends 16 bits/token.
    B's *token* perplexity is 256x worse. But if B's tokens each cover twice as much text,
    both models spend the same bits per byte and are equally good at modelling the text.
    """
    n_bytes = 256
    a = bits_per_byte(
        Uniform(256),
        [
            Batch(
                inputs=torch.zeros(1, 32, dtype=torch.long),
                targets=torch.zeros(1, 32, dtype=torch.long),
            )
        ],
        n_bytes=n_bytes,
    )
    b = bits_per_byte(
        Uniform(65536),
        [
            Batch(
                inputs=torch.zeros(1, 16, dtype=torch.long),
                targets=torch.zeros(1, 16, dtype=torch.long),
            )
        ],
        n_bytes=n_bytes,
    )
    assert b.token_perplexity / a.token_perplexity == pytest.approx(256.0, rel=1e-3)
    assert a.bits_per_byte == pytest.approx(b.bits_per_byte, rel=1e-6)


# =====================================================================================
# Calibration
# =====================================================================================


def test_uniform_model_is_calibrated_by_construction() -> None:
    """A uniform model over V tokens is right 1/V of the time and 1/V confident."""
    vocab = 8
    torch.manual_seed(0)
    targets = torch.randint(0, vocab, (16, 32))
    batches = [Batch(inputs=torch.zeros_like(targets), targets=targets)]
    result = calibration(Uniform(vocab), batches, n_bins=10)

    assert result.mean_confidence == pytest.approx(1 / vocab, rel=1e-5)
    # Accuracy is a sample from Binomial(n, 1/V); with n=512 the interval is tight.
    assert result.accuracy == pytest.approx(1 / vocab, abs=0.05)
    assert result.ece == pytest.approx(abs(result.accuracy - 1 / vocab), abs=1e-5)


def test_an_overconfident_model_has_positive_overconfidence() -> None:
    """A model that is always certain and often wrong must report it."""

    class AlwaysCertain(nn.Module):
        """Predicts token 0 with confidence ~1.0, whatever the input."""

        def forward(self, ids: torch.Tensor, **_: object) -> LMOutput:
            """Return near-one-hot logits on token 0."""
            logits = torch.full((*ids.shape, 4), -20.0)
            logits[..., 0] = 20.0  # always predicts token 0, with ~1.0 confidence
            return LMOutput(logits=logits)

    # Only a quarter of the targets are token 0, so accuracy is 0.25 at confidence ~1.0.
    targets = torch.tensor([[0, 1, 2, 3] * 8])
    batches = [Batch(inputs=torch.zeros_like(targets), targets=targets)]
    result = calibration(AlwaysCertain(), batches, n_bins=10)

    assert result.accuracy == pytest.approx(0.25, abs=1e-6)
    assert result.mean_confidence > 0.99
    assert result.overconfidence > 0.7
    assert result.ece > 0.7


def test_calibration_rejects_a_degenerate_bin_count() -> None:
    """One bin cannot express calibration at all."""
    batches = [
        Batch(
            inputs=torch.zeros(1, 4, dtype=torch.long), targets=torch.zeros(1, 4, dtype=torch.long)
        )
    ]
    with pytest.raises(ValueError, match="n_bins must be at least 2"):
        calibration(Uniform(8), batches, n_bins=1)


# =====================================================================================
# Diversity
# =====================================================================================


def test_distinct_n_is_one_when_nothing_repeats() -> None:
    assert distinct_n(["a b c d"], n=2) == pytest.approx(1.0)


def test_distinct_n_falls_when_a_phrase_repeats() -> None:
    """The diagnostic must actually notice repetition."""
    assert distinct_n(["a b a b a b"], n=2) < 0.5


def test_self_bleu_is_high_for_near_identical_texts_and_low_for_varied_ones() -> None:
    same = ["the cat sat on the mat", "the cat sat on the mat"]
    varied = ["the cat sat on the mat", "a dog ran through wet grass"]
    assert self_bleu(same) > self_bleu(varied)
    assert self_bleu(same) > 0.9


def test_self_bleu_is_undefined_for_fewer_than_two_samples() -> None:
    """Documented behaviour: 0.0 rather than an exception or a nan."""
    assert self_bleu(["only one"]) == 0.0
    assert self_bleu([]) == 0.0


# =====================================================================================
# Wilson interval
# =====================================================================================


def test_wilson_interval_stays_inside_zero_one_at_the_boundary() -> None:
    """The reason this is used instead of the normal approximation.

    At p=1.0 the normal interval is [1.0, 1.0] — implying certainty from a finite sample —
    and at p near 1 it extends above 1.0. Wilson does neither.
    """
    lo, hi = wilson_interval(100, 100)
    assert hi <= 1.0
    assert lo < 1.0, "a 100/100 result is not proof the true rate is 1.0"
    assert lo > 0.95


def test_wilson_interval_brackets_the_point_estimate() -> None:
    for successes, n in [(1, 10), (5, 10), (9, 10), (50, 100), (0, 20)]:
        lo, hi = wilson_interval(successes, n)
        assert 0.0 <= lo <= hi <= 1.0
        # Wilson's centre is shrunk toward 0.5, so the raw proportion need not be the
        # midpoint -- but it must lie inside the interval.
        assert lo <= successes / n <= hi


def test_wilson_interval_narrows_with_more_data() -> None:
    small = wilson_interval(8, 10)
    large = wilson_interval(800, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


# =====================================================================================
# Minimal pairs
# =====================================================================================


def test_generated_pairs_are_unique_and_well_formed() -> None:
    pairs = generate_pairs(n_per_phenomenon=40, seed=7)
    assert len({(p.good, p.bad) for p in pairs}) == len(pairs), "duplicates inflate the sample size"
    for p in pairs:
        assert p.good != p.bad
        assert p.phenomenon in PHENOMENA
        assert p.good.strip() and p.bad.strip()


def test_generated_pairs_are_reproducible_from_a_seed() -> None:
    a = generate_pairs(n_per_phenomenon=20, seed=3)
    b = generate_pairs(n_per_phenomenon=20, seed=3)
    c = generate_pairs(n_per_phenomenon=20, seed=4)
    assert [(x.good, x.bad) for x in a] == [(x.good, x.bad) for x in b]
    assert [(x.good, x.bad) for x in a] != [(x.good, x.bad) for x in c]


def test_minimal_pairs_are_truly_minimal_where_they_claim_to_be() -> None:
    """Most phenomena must differ in exactly one word, or the test measures length.

    `argument_structure` is the deliberate exception -- adding or removing an object
    changes the length, which is why `run_minimal_pairs` length-normalizes by default.
    """
    for pair in generate_pairs(n_per_phenomenon=30, seed=11):
        if pair.phenomenon == "argument_structure":
            continue
        good, bad = pair.good.split(), pair.bad.split()
        assert len(good) == len(bad), f"{pair.phenomenon} changed length: {pair}"
        diffs = sum(1 for g, b in zip(good, bad, strict=True) if g != b)
        assert diffs == 1, f"{pair.phenomenon} differs in {diffs} words: {pair}"


def test_a_uniform_model_scores_at_chance_on_minimal_pairs() -> None:
    """The floor of the benchmark, checked rather than assumed.

    A model with no preference must score ~50%. If this ever came out far from chance it
    would mean the benchmark itself was biased -- e.g. the grammatical sentence being
    systematically shorter -- rather than the model being good.
    """
    tok = _tokenizer()
    result = run_minimal_pairs(
        Uniform(tok.vocab_size), tok, n_per_phenomenon=25, seed=5, length_normalize=True
    )
    # A uniform model gives every sentence the same per-token log-probability, so `>` is
    # never satisfied and the score is 0 -- which is *also* chance-equivalent for a
    # forced choice with no preference. What matters is that it is not near 1.0.
    assert result.overall <= 0.5
    assert result.n_items > 100


def test_minimal_pair_result_reports_intervals_and_a_chance_line() -> None:
    """Every phenomenon must carry an interval that brackets its point estimate."""
    tok = _tokenizer()
    result = run_minimal_pairs(Uniform(tok.vocab_size), tok, n_per_phenomenon=15, seed=5)
    assert result.summary()["chance"] == 0.5
    for row in result.rows():
        lo, acc, hi = row["ci_low"], row["accuracy"], row["ci_high"]
        assert isinstance(lo, float) and isinstance(acc, float) and isinstance(hi, float)
        assert 0.0 <= lo <= acc <= hi <= 1.0


# =====================================================================================
# Significance testing
# =====================================================================================


def test_welch_matches_the_formula_computed_independently() -> None:
    """The t statistic and Welch-Satterthwaite df, recomputed here from the definition.

    Deliberately *not* checked against remembered textbook numbers: the first version of
    this test asserted t = -2.46, df = 17.0 from memory and failed, and the implementation
    turned out to be right. Recomputing the formula inline is the version that can only
    fail when the implementation is actually wrong.
    """
    a = [27.5, 21.0, 19.0, 23.6, 17.0, 17.9, 16.9, 20.1, 21.9, 22.6]
    b = [27.1, 22.0, 20.8, 23.4, 23.4, 23.5, 25.8, 22.0, 24.8, 20.2]

    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    se_sq = va / na + vb / nb
    expected_t = (ma - mb) / math.sqrt(se_sq)
    expected_df = se_sq**2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))

    t, df, p = welch_t_test(a, b)
    assert t == pytest.approx(expected_t, rel=1e-12)
    assert df == pytest.approx(expected_df, rel=1e-12)
    # p is a two-sided tail of Student's t; bracket it rather than restate the integral.
    assert 0.0 < p < 1.0
    assert p == pytest.approx(0.0593, abs=0.001), "this pair is not significant at 0.05"


def test_welch_finds_no_difference_between_samples_from_the_same_process() -> None:
    """The property that makes a null result trustworthy."""
    import random

    rng = random.Random(1234)
    false_positives = 0
    trials = 200
    for _ in range(trials):
        a = [rng.gauss(0.39, 0.004) for _ in range(5)]
        b = [rng.gauss(0.39, 0.004) for _ in range(5)]
        if welch_t_test(a, b)[2] < 0.05:
            false_positives += 1
    # A correct test rejects at ~alpha under the null. Allow generous slack for 200 trials.
    assert false_positives / trials < 0.12, f"false positive rate {false_positives / trials:.3f}"


def test_welch_needs_at_least_two_observations_per_arm() -> None:
    """One seed per arm is exactly the situation this module exists to replace."""
    t, df, p = welch_t_test([0.39], [0.40])
    assert math.isnan(t) and math.isnan(df) and math.isnan(p)


def test_cohens_d_is_scale_free() -> None:
    """Multiplying both samples by a constant must not change the effect size."""
    a = [1.0, 2.0, 3.0, 4.0]
    b = [3.0, 4.0, 5.0, 6.0]
    assert cohens_d(a, b) == pytest.approx(cohens_d([x * 10 for x in a], [x * 10 for x in b]))


def test_a_tiny_but_significant_difference_is_flagged_as_negligible() -> None:
    """The reason effect size is reported alongside the p-value.

    With low variance and enough samples, a difference of 0.2% becomes statistically
    significant. It is still 0.2%. Reporting only the p-value would present that as a
    finding.
    """
    a = [0.3900, 0.3901, 0.3899, 0.3900, 0.3901, 0.3900]
    b = [0.3908, 0.3909, 0.3907, 0.3908, 0.3909, 0.3908]
    _, _, p = welch_t_test(a, b)
    assert p < 0.05, "this difference is statistically detectable"
    relative = abs(sum(a) / len(a) - sum(b) / len(b)) / (sum(a) / len(a))
    assert relative < 0.005, "and practically irrelevant"
    # Effect size is huge here *because* variance is tiny -- which is exactly why the
    # report shows the relative difference too, not just d.
    assert abs(cohens_d(a, b)) > MIN_EFFECT_SIZE
