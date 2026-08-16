"""Cross-check the from-scratch BPE against ``tiktoken`` (spec B1 / A6).

``tiktoken`` is allowed *only* as a cross-check, never as an implementation.

What is and is not being claimed
--------------------------------
Our ``nano`` tokenizer is trained on a 3 MB synthetic story corpus with a 1k
vocabulary; GPT-2's was trained on ~40 GB of web text with a 50k vocabulary. Claiming
we match it on arbitrary English would be dishonest, and asserting it would be a test
that only passes by luck. So the parity checks are split into three sharp claims:

1. **Pre-tokenization is exactly GPT-2's.** Our split regex must chunk text
   character-for-character the way ``tiktoken``'s does. This is an exact equality, and
   it isolates the one component that *is* directly comparable.
2. **In-domain length parity.** On a fixed passage drawn from the distribution our
   tokenizer was trained on, our encoded length must be within tolerance of GPT-2's
   (and in practice beats it, because a 1k in-domain vocabulary is a good fit).
3. **Out-of-domain behaviour is measured, not hidden.** On unseen literary prose we
   are clearly *worse* than GPT-2, and the test pins how much worse. It also pins that
   we are nowhere near the ~1 byte/token of a broken merge loop.
"""

from __future__ import annotations

from typing import Any

import pytest

from nanoscale.config import TokenizerConfig
from nanoscale.data.toy import generate_corpus, iter_stories
from nanoscale.tokenizer import BPETokenizer

tiktoken = pytest.importorskip("tiktoken", reason="install the 'compare' extra for parity tests")

#: A fixed passage of unseen literary prose (public domain: Orwell, *1984*, 1949).
OUT_OF_DOMAIN_PASSAGE = (
    "It was a bright cold day in April, and the clocks were striking thirteen. "
    "Winston Smith, his chin nuzzled into his breast in an effort to escape the "
    "vile wind, slipped quickly through the glass doors of Victory Mansions, "
    "though not quickly enough to prevent a swirl of gritty dust from entering "
    "along with him. The hallway smelt of boiled cabbage and old rag mats."
)

#: A fixed in-domain passage: a held-out story from a seed the tokenizer never saw.
IN_DOMAIN_PASSAGE = next(iter_stories(seed=987_654, n_stories=1))

#: In-domain tolerance: we must be within 25% of GPT-2's bytes/token.
IN_DOMAIN_TOLERANCE = 0.75


@pytest.fixture(scope="module")
def trained() -> BPETokenizer:
    corpus = generate_corpus(seed=3, n_stories=8000)
    return BPETokenizer.train(corpus, TokenizerConfig(vocab_size=1024, max_train_bytes=3_000_000))


@pytest.fixture(scope="module")
def gpt2() -> Any:
    return tiktoken.get_encoding("gpt2")


def _ratio(n_bytes: int, n_tokens: int) -> float:
    return n_bytes / n_tokens


@pytest.mark.network
def test_pretokenization_is_exactly_gpt2s(gpt2: Any) -> None:
    """The one component that is directly comparable must match character-for-character."""
    import regex as re

    from nanoscale.tokenizer import GPT2_SPLIT_PATTERN

    for text in (OUT_OF_DOMAIN_PASSAGE, IN_DOMAIN_PASSAGE, "don't 42 things!!  \n\n x"):
        ours = re.findall(GPT2_SPLIT_PATTERN, text)
        theirs = re.findall(gpt2._pat_str, text)
        assert ours == theirs, f"pre-tokenization diverged on {text[:40]!r}"


@pytest.mark.network
def test_in_domain_length_parity_on_a_fixed_passage(trained: BPETokenizer, gpt2: Any) -> None:
    n_bytes = len(IN_DOMAIN_PASSAGE.encode("utf-8"))
    ref = _ratio(n_bytes, len(gpt2.encode(IN_DOMAIN_PASSAGE)))
    ours = _ratio(n_bytes, len(trained.encode(IN_DOMAIN_PASSAGE)))

    assert ours >= ref * IN_DOMAIN_TOLERANCE, (
        f"in-domain: ours {ours:.2f} B/tok vs gpt2 {ref:.2f} B/tok, outside tolerance"
    )
    # A 1k in-domain vocabulary should not be beating a 50k one by an implausible margin.
    assert ours <= ref * 2.5


@pytest.mark.network
def test_out_of_domain_degradation_is_bounded_and_documented(
    trained: BPETokenizer, gpt2: Any
) -> None:
    """We are worse than GPT-2 on unseen prose. The test pins *how much* worse."""
    n_bytes = len(OUT_OF_DOMAIN_PASSAGE.encode("utf-8"))
    ref = _ratio(n_bytes, len(gpt2.encode(OUT_OF_DOMAIN_PASSAGE)))
    ours = _ratio(n_bytes, len(trained.encode(OUT_OF_DOMAIN_PASSAGE)))

    # Not degenerate: a broken merge loop sits at ~1.0 bytes/token.
    assert ours > 2.0, f"only {ours:.2f} bytes/token -- the merge loop looks broken"
    # But honestly worse than a web-scale tokenizer, which is the expected result.
    assert ours < ref, "a 1k synthetic-corpus vocabulary should not beat GPT-2 on Orwell"
    assert ours >= ref * 0.4, f"degradation worse than expected: {ours:.2f} vs {ref:.2f}"


@pytest.mark.network
def test_round_trip_holds_wherever_tiktokens_does(trained: BPETokenizer, gpt2: Any) -> None:
    for text in (OUT_OF_DOMAIN_PASSAGE, IN_DOMAIN_PASSAGE, "héllo wörld", "日本語", "🌍🚀"):
        assert gpt2.decode(gpt2.encode(text)) == text
        assert trained.decode(trained.encode(text)) == text
