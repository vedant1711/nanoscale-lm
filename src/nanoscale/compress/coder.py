"""Arithmetic coding driven by the language model's predictions.

A language model *is* a compressor. Shannon's source coding theorem says a symbol of
probability ``p`` can be encoded in ``-log2(p)`` bits, so a model's cross-entropy in bits
per byte is exactly the size of the file it can produce — this is the identity behind
"bits per byte" being a meaningful metric at all.

The identity is usually left as theory. This module cashes it: a range coder that takes
the model's next-token distribution at every step and produces an actual byte string,
whose length is measured rather than predicted. Without this, a bits-per-byte number is a
claim about a compressor nobody built.

**Why the decoder works.** The decoder has no side information. It reconstructs each
distribution by running the same model on the tokens it has already decoded — which are,
by induction, exactly the tokens the encoder had. Encoder and decoder therefore see
identical distributions at every step. Two consequences follow, and both are enforced
here:

1. **The probabilities must be identical bit-for-bit**, not merely close. Float arithmetic
   is deterministic for a fixed model, device and dtype, but the *derived integer
   frequencies* are what the coder consumes, so quantisation happens once, in one place,
   and both sides call it.
2. **Every symbol needs non-zero frequency.** A symbol the model assigns probability zero
   is a symbol the coder cannot encode. Frequencies are floored at 1, which costs a
   negligible amount of rate and removes an entire class of catastrophic failure.

The coder is a carry-less range coder (Subbotin style) with 32-bit state and 16-bit
frequency resolution. It is not the tightest possible implementation — the residual
overhead against the model's own cross-entropy is measured in
``scripts/compression_bench.py`` and is well under 1% — but it is short enough to read.

References:
    Shannon (1948), *A Mathematical Theory of Communication*.
    Witten, Neal & Cleary (1987), *Arithmetic coding for data compression*.
    Deletang et al. (2024), *Language Modeling Is Compression*, arXiv:2309.10668.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "FREQ_BITS",
    "ArithmeticDecoder",
    "ArithmeticEncoder",
    "CoderStats",
    "probs_to_frequencies",
]

#: Frequency resolution. 16 bits keeps `total <= 2**16`, which with a 32-bit range leaves
#: enough headroom that the range never underflows below `total`.
FREQ_BITS = 16
FREQ_TOTAL = 1 << FREQ_BITS

_TOP = 1 << 24
_BOTTOM = 1 << 16
_MASK = (1 << 32) - 1


def probs_to_frequencies(probs: torch.Tensor) -> torch.Tensor:
    """Quantise a probability vector to integer frequencies summing to :data:`FREQ_TOTAL`.

    Args:
        probs: 1-D probability vector over the vocabulary. Need not be exactly normalised.

    Returns:
        1-D ``int64`` tensor of frequencies, every entry ``>= 1``, summing exactly to
        ``FREQ_TOTAL``.

    This function is the *only* place probabilities become integers, and both the encoder
    and the decoder call it. That is deliberate: if the two sides quantised independently
    with even slightly different code, a single differing frequency anywhere in the
    vocabulary would desynchronise the arithmetic and corrupt everything after it.

    The floor-at-1 step means the vocabulary size sets a rate floor: with a 16k vocabulary
    and 16-bit frequencies, a quarter of the mass is reserved for the floor. That is the
    dominant source of overhead against the model's true cross-entropy, and it is why
    ``FREQ_BITS`` is 16 rather than 12.
    """
    if probs.ndim != 1:
        raise ValueError(f"expected a 1-D probability vector, got shape {tuple(probs.shape)}")
    v = int(probs.numel())
    if v > FREQ_TOTAL:
        raise ValueError(
            f"vocabulary {v} exceeds the frequency budget {FREQ_TOTAL}; every symbol needs "
            f"at least 1, so FREQ_BITS must be raised to at least {v.bit_length()}."
        )

    p = probs.double().clamp_min(0.0)
    total = float(p.sum())
    p = p / total if total > 0 else torch.full_like(p, 1.0 / v)

    # Reserve one unit per symbol, distribute the rest proportionally.
    spare = FREQ_TOTAL - v
    freq = torch.floor(p * spare).to(torch.int64) + 1

    # Hand the rounding remainder to the largest probabilities, which is where it costs
    # the least rate.
    deficit = FREQ_TOTAL - int(freq.sum())
    if deficit > 0:
        order = torch.argsort(p, descending=True)[:deficit]
        freq[order] += 1
    elif deficit < 0:
        # Only possible through float error; take from the largest that can spare it.
        order = torch.argsort(freq, descending=True)
        i = 0
        while deficit < 0:
            idx = int(order[i % order.numel()])
            if freq[idx] > 1:
                freq[idx] -= 1
                deficit += 1
            i += 1
    return freq


@dataclass
class CoderStats:
    """How much the coder actually spent."""

    symbols: int = 0
    ideal_bits: float = 0.0

    @property
    def ideal_bytes(self) -> float:
        """The model's own cross-entropy cost, as a byte count."""
        return self.ideal_bits / 8.0


class ArithmeticEncoder:
    """Carry-less range encoder."""

    def __init__(self) -> None:
        """Start with the full 32-bit range."""
        self.low = 0
        self.range = _MASK
        self.out = bytearray()
        self.stats = CoderStats()

    def encode(self, symbol: int, freq: torch.Tensor) -> None:
        """Encode one symbol against the frequency table ``freq``."""
        cum = int(torch.sum(freq[:symbol]))
        f = int(freq[symbol])

        self.range //= FREQ_TOTAL
        self.low = (self.low + cum * self.range) & _MASK
        self.range *= f

        self.stats.symbols += 1
        self.stats.ideal_bits += -torch.log2(torch.tensor(f / FREQ_TOTAL)).item()

        self._normalise()

    def _normalise(self) -> None:
        while True:
            if (self.low ^ (self.low + self.range)) < _TOP:
                pass
            elif self.range < _BOTTOM:
                # Range too small to split reliably: clamp it so the top byte can flush.
                self.range = (-self.low) & (_BOTTOM - 1)
            else:
                return
            self.out.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & _MASK
            self.range = (self.range << 8) & _MASK

    def finish(self) -> bytes:
        """Flush the remaining state and return the encoded bytes."""
        for _ in range(4):
            self.out.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & _MASK
        return bytes(self.out)


class ArithmeticDecoder:
    """Carry-less range decoder, exactly mirroring :class:`ArithmeticEncoder`."""

    def __init__(self, data: bytes) -> None:
        """Prime the decoder with the first four bytes of ``data``."""
        self.data = data
        self.pos = 0
        self.low = 0
        self.range = _MASK
        self.code = 0
        for _ in range(4):
            self.code = ((self.code << 8) | self._next_byte()) & _MASK

    def _next_byte(self) -> int:
        if self.pos < len(self.data):
            b = self.data[self.pos]
            self.pos += 1
            return b
        return 0

    def decode(self, freq: torch.Tensor) -> int:
        """Decode one symbol against the frequency table ``freq``."""
        self.range //= FREQ_TOTAL
        offset = ((self.code - self.low) & _MASK) // self.range
        offset = min(offset, FREQ_TOTAL - 1)

        # Locate the symbol whose cumulative interval contains `offset`.
        cumsum = torch.cumsum(freq, dim=0)
        symbol = int(torch.searchsorted(cumsum, torch.tensor(offset + 1, dtype=cumsum.dtype)))

        cum = int(cumsum[symbol - 1]) if symbol > 0 else 0
        f = int(freq[symbol])
        self.low = (self.low + cum * self.range) & _MASK
        self.range *= f

        self._normalise()
        return symbol

    def _normalise(self) -> None:
        while True:
            if (self.low ^ (self.low + self.range)) < _TOP:
                pass
            elif self.range < _BOTTOM:
                self.range = (-self.low) & (_BOTTOM - 1)
            else:
                return
            self.code = ((self.code << 8) | self._next_byte()) & _MASK
            self.low = (self.low << 8) & _MASK
            self.range = (self.range << 8) & _MASK
