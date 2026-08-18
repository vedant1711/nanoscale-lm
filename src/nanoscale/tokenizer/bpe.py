"""Byte-level Byte-Pair Encoding, implemented from scratch (spec B1).

Algorithm
---------
BPE (Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units*,
arXiv:1508.07909) in its byte-level form (Radford et al., GPT-2):

1. Initialise the vocabulary with the 256 possible byte values, so **every** UTF-8
   string is representable and the tokenizer can never emit ``<unk>``.
2. Split the corpus with a pre-tokenization regex, so that merges never cross a
   word/punctuation boundary. This is what stops BPE from learning tokens like
   ``"dog."`` or ``". The"`` and materially improves compression.
3. Repeatedly find the most frequent adjacent pair of symbols across the corpus,
   mint a new token for it, and replace every occurrence. Stop at the target vocab.
4. Reserve the top of the ID range for special tokens.

Complexity
----------
The naive implementation re-counts every pair after every merge, which is
``O(merges x corpus)``. This implementation instead keeps

* the corpus as a list of *unique* pre-token symbol sequences with multiplicities,
* an incremental ``Counter`` of pair frequencies, and
* an inverted index ``pair -> {word indices containing it}``,

so a merge only touches the words that actually contain the merged pair, and pair
counts are updated by diffing the affected neighbourhoods. That is the difference
between "trains in seconds" and "trains in an hour" at a 32k vocabulary.

Encoding
--------
Encoding applies the learned merges in rank order to each pre-token: repeatedly find
the adjacent pair with the lowest merge rank and apply it. Results are memoised per
pre-token string, which is what makes encoding a large corpus practical.
"""

from __future__ import annotations

import base64
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Final

import regex as re

from nanoscale.config import TokenizerConfig
from nanoscale.utils.logging import get_logger

__all__ = [
    "GPT2_SPLIT_PATTERN",
    "GPT4_SPLIT_PATTERN",
    "BPETokenizer",
    "Merge",
]

log = get_logger("nanoscale.tokenizer")

#: GPT-2's pre-tokenization regex (Radford et al., 2019).
GPT2_SPLIT_PATTERN: Final[str] = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)

#: GPT-4 / ``cl100k_base``'s pre-tokenization regex: case-insensitive contractions and
#: digits capped at runs of three, which keeps number tokens from exploding the vocab.
GPT4_SPLIT_PATTERN: Final[str] = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

_PATTERNS: Final[dict[str, str]] = {
    "gpt2": GPT2_SPLIT_PATTERN,
    "gpt4": GPT4_SPLIT_PATTERN,
    "none": "",
}

#: A learned merge: ``(left_id, right_id) -> new_id``.
Merge = tuple[tuple[int, int], int]

_FILE_VERSION: Final[int] = 1


class BPETokenizer:
    """A byte-level BPE tokenizer: train, encode, decode, specials, chat template.

    Vocabulary layout (IDs are assigned in this order, which the tests pin):

    * ``[0, 256)``: the raw byte tokens.
    * ``[256, 256 + n_merges)``: learned merges, in the order they were learned, so a
      token's ID is also its merge rank offset.
    * ``[vocab_size - n_specials, vocab_size)``: special tokens.

    Special tokens are never produced by merging, so they can only enter a token
    stream when the caller explicitly asks for them. That is a safety property: user
    text can never forge a ``<assistant>`` turn boundary.
    """

    def __init__(
        self,
        merges: Sequence[Merge] | None = None,
        *,
        config: TokenizerConfig | None = None,
    ) -> None:
        """Build a tokenizer from learned merges (or an untrained byte-level one)."""
        self.config = config or TokenizerConfig()
        self.pattern = _PATTERNS[self.config.split_pattern]
        self._compiled = re.compile(self.pattern) if self.pattern else None

        self.merges: list[Merge] = list(merges or [])
        self.merge_ranks: dict[tuple[int, int], int] = {
            pair: rank for rank, (pair, _) in enumerate(self.merges)
        }

        # id -> the byte string that token expands to.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for pair, new_id in self.merges:
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

        n_specials = len(self.config.special_tokens)
        first_special = self.config.vocab_size - n_specials
        self.special_to_id: dict[str, int] = {
            tok: first_special + i for i, tok in enumerate(self.config.special_tokens)
        }
        self.id_to_special: dict[int, str] = {v: k for k, v in self.special_to_id.items()}
        for tok, tid in self.special_to_id.items():
            self.vocab[tid] = tok.encode("utf-8")

        self._special_re = (
            re.compile("(" + "|".join(re.escape(t) for t in self.special_to_id) + ")")
            if self.special_to_id
            else None
        )
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ properties

    @property
    def vocab_size(self) -> int:
        """Configured vocabulary size (byte tokens + merges + specials)."""
        return self.config.vocab_size

    @property
    def n_merges(self) -> int:
        """Number of merges actually learned."""
        return len(self.merges)

    @property
    def bos_id(self) -> int:
        """ID of the ``<bos>`` token."""
        return self.special_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        """ID of the ``<eos>`` token."""
        return self.special_to_id["<eos>"]

    @property
    def pad_id(self) -> int:
        """ID of the ``<pad>`` token."""
        return self.special_to_id["<pad>"]

    @property
    def eot_id(self) -> int:
        """ID of the ``<eot>`` (end-of-turn) token."""
        return self.special_to_id["<eot>"]

    def __len__(self) -> int:
        """Configured vocabulary size."""
        return self.vocab_size

    def __repr__(self) -> str:
        """Compact description including how much of the vocab is actually learned."""
        return (
            f"BPETokenizer(vocab_size={self.vocab_size}, merges={self.n_merges}, "
            f"specials={len(self.special_to_id)}, pattern={self.config.split_pattern!r})"
        )

    # -------------------------------------------------------------------- training

    @classmethod
    def train(
        cls,
        corpus: str | Iterable[str],
        config: TokenizerConfig,
        *,
        verbose: bool = False,
    ) -> BPETokenizer:
        """Learn merges from a corpus and return a trained tokenizer.

        Args:
            corpus: A single string, or an iterable of documents.
            config: Tokenizer configuration; ``config.n_merges`` merges are learned.
            verbose: Log progress every 512 merges.

        Returns:
            A trained :class:`BPETokenizer`.

        The corpus is truncated to ``config.max_train_bytes`` so that training the
        tokenizer never becomes the slow part of the pipeline.
        """
        text = corpus if isinstance(corpus, str) else "".join(corpus)
        data = text.encode("utf-8")[: config.max_train_bytes]
        # Truncation can land mid-codepoint; back off to a valid boundary.
        while data and len(data) > 1:
            try:
                text = data.decode("utf-8")
                break
            except UnicodeDecodeError:
                data = data[:-1]
        else:
            text = data.decode("utf-8", errors="ignore")

        pattern = _PATTERNS[config.split_pattern]
        chunks = re.findall(pattern, text) if pattern else [text]

        # Work on unique pre-tokens with multiplicities, not on the raw byte stream.
        chunk_counts = Counter(chunks)
        words: list[list[int]] = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        counts: list[int] = list(chunk_counts.values())

        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_to_words: dict[tuple[int, int], set[int]] = {}
        for widx, word in enumerate(words):
            weight = counts[widx]
            for pair in pairwise(word):
                pair_counts[pair] += weight
                pair_to_words.setdefault(pair, set()).add(widx)

        merges: list[Merge] = []
        target = config.n_merges
        next_id = 256
        for step in range(target):
            if not pair_counts:
                log.warning(
                    "BPE training exhausted all pairs after %d merges (target %d); the "
                    "corpus is too small or too uniform for the requested vocab size.",
                    step,
                    target,
                )
                break
            best_pair, best_count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
            if best_count <= 0:
                break
            merges.append((best_pair, next_id))
            cls._apply_merge(best_pair, next_id, words, counts, pair_counts, pair_to_words)
            next_id += 1
            if verbose and (step + 1) % 512 == 0:
                log.info("  merge %5d/%d  %-24s count=%d", step + 1, target, best_pair, best_count)

        return cls(merges, config=config)

    @staticmethod
    def _apply_merge(
        pair: tuple[int, int],
        new_id: int,
        words: list[list[int]],
        counts: list[int],
        pair_counts: Counter[tuple[int, int]],
        pair_to_words: dict[tuple[int, int], set[int]],
    ) -> None:
        """Replace ``pair`` with ``new_id`` everywhere, updating counts incrementally.

        Only the words in the inverted index for ``pair`` are touched, and within each
        word only the pairs adjacent to a replacement change, so the cost of a merge is
        proportional to the number of occurrences rather than to the corpus size.
        """
        affected = pair_to_words.pop(pair, set())
        pair_counts.pop(pair, None)
        left, right = pair

        for widx in affected:
            word = words[widx]
            weight = counts[widx]

            # Remove this word's current pair contributions.
            for old in pairwise(word):
                pair_counts[old] -= weight
                if pair_counts[old] <= 0:
                    pair_counts.pop(old, None)
                bucket = pair_to_words.get(old)
                if bucket is not None:
                    bucket.discard(widx)
                    if not bucket:
                        pair_to_words.pop(old, None)

            merged: list[int] = []
            i = 0
            n = len(word)
            while i < n:
                if i < n - 1 and word[i] == left and word[i + 1] == right:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            words[widx] = merged

            for new in pairwise(merged):
                pair_counts[new] += weight
                pair_to_words.setdefault(new, set()).add(widx)

    # -------------------------------------------------------------------- encoding

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Encode one pre-token by greedily applying the lowest-rank merge available."""
        cached = self._cache.get(chunk)
        if cached is not None:
            return list(cached)

        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            best_rank = None
            best_i = -1
            for i in range(len(ids) - 1):
                rank = self.merge_ranks.get((ids[i], ids[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_i = rank, i
            if best_rank is None:
                break
            new_id = self.merges[best_rank][1]
            ids[best_i : best_i + 2] = [new_id]

        if len(self._cache) < 500_000:
            self._cache[chunk] = list(ids)
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating any special-token *string* as ordinary characters."""
        if self._compiled is None:
            return self._encode_chunk(text)
        out: list[int] = []
        for chunk in self._compiled.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(
        self,
        text: str,
        *,
        allowed_special: bool | Iterable[str] = False,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input string.
            allowed_special: ``False`` (default) treats special-token strings in the
                input as ordinary text, untrusted input can never forge a control
                token. ``True`` recognises all specials; an iterable recognises only
                those named.
            add_bos: Prepend ``<bos>``.
            add_eos: Append ``<eos>``.
        """
        if allowed_special is False or self._special_re is None:
            ids = self.encode_ordinary(text)
        else:
            allowed = (
                set(self.special_to_id)
                if allowed_special is True
                else {t for t in allowed_special if t in self.special_to_id}
            )
            ids = []
            for piece in self._special_re.split(text):
                if not piece:
                    continue
                if piece in allowed:
                    ids.append(self.special_to_id[piece])
                else:
                    ids.extend(self.encode_ordinary(piece))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        allowed_special: bool | Iterable[str] = False,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[list[int]]:
        """Encode a batch of strings with the same options."""
        return [
            self.encode(t, allowed_special=allowed_special, add_bos=add_bos, add_eos=add_eos)
            for t in texts
        ]

    # -------------------------------------------------------------------- decoding

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        """Decode token IDs to raw bytes (exact; never lossy)."""
        parts: list[bytes] = []
        for tid in ids:
            piece = self.vocab.get(tid)
            if piece is None:
                raise KeyError(f"Token id {tid} is not in the vocabulary.")
            parts.append(piece)
        return b"".join(parts)

    def decode(
        self,
        ids: Iterable[int],
        *,
        errors: str = "replace",
        skip_special: bool = False,
    ) -> str:
        """Decode token IDs to text.

        Args:
            ids: Token IDs.
            errors: Passed to ``bytes.decode``. ``"replace"`` is the default because
                streaming decoders legitimately hand us a partial multi-byte codepoint.
            skip_special: Drop special tokens instead of rendering their literal text.
        """
        seq = [i for i in ids if not (skip_special and i in self.id_to_special)]
        return self.decode_bytes(seq).decode("utf-8", errors=errors)

    def decode_batch(
        self,
        batch: Iterable[Iterable[int]],
        *,
        errors: str = "replace",
        skip_special: bool = False,
    ) -> list[str]:
        """Decode a batch of ID sequences."""
        return [self.decode(ids, errors=errors, skip_special=skip_special) for ids in batch]

    # -------------------------------------------------------------- serialisation

    def save(self, path: str | Path) -> Path:
        """Write the tokenizer to JSON and return the path.

        The format is plain JSON so the artifact is reviewable in a diff: merges are
        stored as ``[left, right, new_id]`` triples in learned order, and the config is
        stored alongside so a loaded tokenizer cannot silently disagree with the vocab.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _FILE_VERSION,
            "config": self.config.dump_inputs(mode="json"),
            "merges": [[pair[0], pair[1], new_id] for pair, new_id in self.merges],
        }
        dest.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        return dest

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Load a tokenizer previously written by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = payload.get("version")
        if version != _FILE_VERSION:
            raise ValueError(f"Unsupported tokenizer file version {version!r}.")
        config = TokenizerConfig.model_validate(payload["config"])
        merges: list[Merge] = [((int(a), int(b)), int(c)) for a, b, c in payload["merges"]]
        return cls(merges, config=config)

    # ------------------------------------------------------------------ inspection

    def token_repr(self, tid: int) -> str:
        """Human-readable rendering of one token, for vocabulary dumps."""
        if tid in self.id_to_special:
            return self.id_to_special[tid]
        piece = self.vocab.get(tid)
        if piece is None:
            return f"<unused:{tid}>"
        return piece.decode("utf-8", errors="backslashreplace")

    def iter_vocab(self) -> Iterator[tuple[int, str]]:
        """Yield ``(id, printable_token)`` for every defined token, in ID order."""
        for tid in sorted(self.vocab):
            yield tid, self.token_repr(tid)

    def compression_ratio(self, text: str) -> float:
        """Bytes per token on ``text``: the headline quality metric for a tokenizer."""
        ids = self.encode(text)
        if not ids:
            return 0.0
        return len(text.encode("utf-8")) / len(ids)

    def to_base64_vocab(self) -> dict[str, int]:
        """Export a ``{base64(token_bytes): id}`` mapping (tiktoken-style interchange)."""
        return {
            base64.b64encode(piece).decode("ascii"): tid
            for tid, piece in sorted(self.vocab.items())
            if tid not in self.id_to_special
        }
