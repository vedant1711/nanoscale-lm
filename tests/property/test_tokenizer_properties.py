"""Hypothesis property tests for the tokenizer (spec D2).

The headline property is the one the spec names explicitly: ``decode(encode(x)) == x``
for *arbitrary* UTF-8. Byte-level BPE makes this a theorem rather than a hope; there
is no unknown token and no lossy normalisation anywhere in the pipeline, so a
counter-example here would be a genuine bug, not a tuning issue.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from nanoscale.config import TokenizerConfig
from nanoscale.data.toy import generate_corpus
from nanoscale.tokenizer import BPETokenizer, Message, render_chat

_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    corpus = generate_corpus(seed=11, n_stories=2000)
    return BPETokenizer.train(corpus, TokenizerConfig(vocab_size=1024, max_train_bytes=500_000))


@pytest.fixture(scope="module")
def byte_tokenizer() -> BPETokenizer:
    """An untrained tokenizer: pure byte-level, so its behaviour is fully determined."""
    return BPETokenizer(config=TokenizerConfig(vocab_size=512))


@_SETTINGS
@given(text=st.text())
def test_round_trip_arbitrary_text(tokenizer: BPETokenizer, text: str) -> None:
    assert tokenizer.decode(tokenizer.encode(text)) == text


@_SETTINGS
@given(text=st.text(alphabet=st.characters(), min_size=0, max_size=200))
def test_round_trip_full_unicode_range(tokenizer: BPETokenizer, text: str) -> None:
    # Surrogates cannot be encoded to UTF-8 at all, so they are out of scope by
    # construction: `str` values containing them are not valid text.
    assume(all(not (0xD800 <= ord(c) <= 0xDFFF) for c in text))
    assert tokenizer.decode(tokenizer.encode(text)) == text


@_SETTINGS
@given(data=st.binary(max_size=400))
def test_arbitrary_bytes_round_trip_through_the_byte_level_path(
    byte_tokenizer: BPETokenizer, data: bytes
) -> None:
    """Byte-level means *every* byte string is representable, valid UTF-8 or not."""
    ids = list(data)
    assert byte_tokenizer.decode_bytes(ids) == data


@_SETTINGS
@given(text=st.text(min_size=1))
def test_encoding_never_exceeds_the_byte_length(tokenizer: BPETokenizer, text: str) -> None:
    """Merging can only shorten a sequence; it can never make it longer."""
    assert len(tokenizer.encode(text)) <= len(text.encode("utf-8"))


@_SETTINGS
@given(text=st.text())
def test_ids_are_always_in_range(tokenizer: BPETokenizer, text: str) -> None:
    ids = tokenizer.encode(text)
    assert all(0 <= i < tokenizer.vocab_size for i in ids)


@_SETTINGS
@given(text=st.text())
def test_ordinary_encoding_never_emits_a_special_token(tokenizer: BPETokenizer, text: str) -> None:
    ids = tokenizer.encode_ordinary(text)
    assert not (set(ids) & set(tokenizer.id_to_special))


def test_bpe_token_counts_are_not_subadditive(tokenizer: BPETokenizer) -> None:
    """A counterexample, pinned: ``len(encode(a+b))`` can *exceed* the sum of the parts.

    This test exists because the obvious-looking property is false, and hypothesis found
    the counterexample. With the committed `nano` vocabulary::

        encode("eps")   -> 1 token
        encode("ep")    -> 1 token
        encode("epsep") -> 3 tokens

    Greedy rank-ordered merging is **path-dependent**: on the concatenated string an
    earlier-ranked merge straddling the join fires first and consumes characters that
    would otherwise have formed the longer tokens, and the greedy walk cannot backtrack.
    So BPE token counts are neither subadditive nor monotone under concatenation.

    Practical consequence, and the reason this is worth a test rather than a comment:
    you cannot assume a prompt's token count is the sum of its parts' token counts, and
    you cannot cache a tokenization by concatenating cached pieces -- unless the join
    lands on a pre-token boundary, which is exactly the next test.
    """
    assert len(tokenizer.encode("eps")) == 1
    assert len(tokenizer.encode("ep")) == 1
    assert len(tokenizer.encode("epsep")) > 2


@_SETTINGS
@given(a=st.text(max_size=100), b=st.text(max_size=100))
def test_encoding_splits_exactly_at_pretoken_boundaries(
    tokenizer: BPETokenizer, a: str, b: str
) -> None:
    """When the join is a pre-token boundary, encoding *is* exactly concatenative.

    The GPT-2 regex starts a new pre-token at a leading space, and merges never cross a
    pre-token boundary. So ``encode(a + " " + b) == encode(a) + encode(" " + b)``
    exactly -- which is what makes incremental/streaming tokenization sound.
    """
    assume(a and not a[-1].isspace())
    left = tokenizer.encode(a)
    right = tokenizer.encode(" " + b)
    assert tokenizer.encode(a + " " + b) == left + right


@_SETTINGS
@given(text=st.text(max_size=200))
def test_encoding_is_deterministic(tokenizer: BPETokenizer, text: str) -> None:
    assert tokenizer.encode(text) == tokenizer.encode(text)


@_SETTINGS
@given(
    prompt=st.text(max_size=120),
    reply=st.text(min_size=1, max_size=120),
)
def test_chat_mask_is_binary_and_aligned(tokenizer: BPETokenizer, prompt: str, reply: str) -> None:
    example = render_chat(tokenizer, [Message("user", prompt), Message("assistant", reply)])
    assert len(example.ids) == len(example.completion_mask)
    assert set(example.completion_mask) <= {0, 1}
    # At minimum the terminating <eot> of the assistant turn is supervised.
    assert example.n_completion_tokens >= 1
    # The prompt half is never supervised.
    assistant_pos = example.ids.index(tokenizer.special_to_id["<assistant>"])
    assert sum(example.completion_mask[: assistant_pos + 1]) == 0
