"""Tokenized data pipeline: packing, splitting and deterministic batching (spec B4).

Packing
-------
Language models train on fixed-length windows, but documents have arbitrary lengths.
Padding to the longest document wastes compute on ``<pad>``; truncating throws away
text. **Packing** concatenates the token stream end to end (with ``<eos>`` between
documents so the model learns document boundaries) and slices fixed-length windows out
of it. Every position in every batch carries a real training signal.

Each window is ``seq_len + 1`` tokens long: inputs are ``w[:-1]``, targets are ``w[1:]``.
Slicing that way rather than shifting inside the model keeps the loss definition
obvious and makes off-by-one errors visible in a test rather than as a mysterious
one-token lag.

Determinism
-----------
Spec A3.4 makes reproducibility a graded feature, and D3 requires two seeded ``nano``
runs to produce identical loss trajectories. So batching does **not** use
``torch.utils.data.DataLoader``'s sampler: the order of windows within an epoch is a
seeded permutation derived from ``(global_seed, "data", epoch)``, which makes the
``n``-th batch of the ``k``-th epoch a pure function of the seed. Resuming from a
checkpoint replays exactly the same order.

Sources
-------
``toy`` (the offline synthetic corpus), ``textfile`` (local ``.txt``/``.jsonl``) and
``hf`` (streaming Hugging Face datasets, for the ``micro``/``small`` tiers). The first
two are materialised in memory as a single token array — at ``nano`` scale that is a
few megabytes and makes the whole pipeline trivially reproducible.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from nanoscale.config import DataConfig
from nanoscale.data.toy import generate_corpus
from nanoscale.tokenizer import BPETokenizer
from nanoscale.utils.logging import get_logger
from nanoscale.utils.seed import derive_seed

__all__ = [
    "Batch",
    "PackedTokens",
    "TokenBatcher",
    "build_packed_tokens",
    "iter_hf_documents",
    "iter_text_documents",
    "tokenize_documents",
]

log = get_logger("nanoscale.train.data")


@dataclass(frozen=True, slots=True)
class Batch:
    """One training batch: inputs and next-token targets."""

    inputs: Tensor  # (B, T) int64
    targets: Tensor  # (B, T) int64

    @property
    def n_tokens(self) -> int:
        """Number of supervised tokens in this batch."""
        return int(self.targets.numel())

    def to(self, device: torch.device) -> Batch:
        """Move both tensors to ``device``."""
        return Batch(
            self.inputs.to(device, non_blocking=True), self.targets.to(device, non_blocking=True)
        )


@dataclass(slots=True)
class PackedTokens:
    """A corpus tokenized and split into train/validation token arrays."""

    train: np.ndarray
    val: np.ndarray
    vocab_size: int

    @property
    def n_train_tokens(self) -> int:
        """Training tokens available."""
        return int(self.train.size)

    @property
    def n_val_tokens(self) -> int:
        """Validation tokens available."""
        return int(self.val.size)

    def summary(self) -> dict[str, int]:
        """Token counts, recorded in the run manifest."""
        return {"train_tokens": self.n_train_tokens, "val_tokens": self.n_val_tokens}


# --------------------------------------------------------------------------------------
# Document sources
# --------------------------------------------------------------------------------------


def iter_text_documents(paths: Iterable[str | Path]) -> Iterator[str]:
    """Yield documents from local ``.txt`` (whole file) or ``.jsonl`` (``text`` field)."""
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"data path {path} does not exist.")
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    text = record.get("text")
                    if isinstance(text, str) and text:
                        yield text
        else:
            yield path.read_text(encoding="utf-8")


def iter_hf_documents(config: DataConfig, limit: int | None = None) -> Iterator[str]:
    """Stream documents from a Hugging Face dataset.

    Streaming rather than downloading is what keeps the ``micro``/``small`` tiers inside
    a free Colab/Kaggle disk quota: FineWeb-Edu's smallest sample is 10B tokens, which
    does not fit anywhere free.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "source='hf' needs the 'data' extra: uv pip install -e '.[data]'"
        ) from exc

    stream = load_dataset(
        config.hf_dataset, name=config.hf_config, split=config.hf_split, streaming=True
    )
    if config.shuffle_buffer > 1:
        stream = stream.shuffle(seed=0, buffer_size=config.shuffle_buffer)
    for i, record in enumerate(stream):
        if limit is not None and i >= limit:
            return
        text = record.get(config.hf_text_field)
        if isinstance(text, str) and text:
            yield text


# --------------------------------------------------------------------------------------
# Tokenization and packing
# --------------------------------------------------------------------------------------


def tokenize_documents(
    documents: Iterable[str],
    tokenizer: BPETokenizer,
    *,
    max_tokens: int | None = None,
) -> np.ndarray:
    """Tokenize documents into one flat array, separated by ``<eos>``.

    The separator matters: without it the model learns to run one document straight
    into the next, and never learns that text can end.
    """
    chunks: list[np.ndarray] = []
    total = 0
    dtype = np.uint16 if tokenizer.vocab_size <= 65_536 else np.int32
    for doc in documents:
        ids = tokenizer.encode(doc)
        ids.append(tokenizer.eos_id)
        chunks.append(np.asarray(ids, dtype=dtype))
        total += len(ids)
        if max_tokens is not None and total >= max_tokens:
            break
    if not chunks:
        return np.zeros(0, dtype=dtype)
    packed = np.concatenate(chunks)
    if max_tokens is not None:
        packed = packed[:max_tokens]
    return packed


def build_packed_tokens(
    config: DataConfig,
    tokenizer: BPETokenizer,
    *,
    max_tokens: int | None = None,
    toy_stories: int = 8000,
    toy_seed: int = 1337,
) -> PackedTokens:
    """Materialise a corpus as train/validation token arrays.

    The validation split is taken from the **end** of the stream, contiguously. Random
    per-window splitting would leak: packed windows from the same document would land
    on both sides, and validation loss would flatter the model.
    """
    if config.source == "toy":
        documents: Iterable[str] = generate_corpus(seed=toy_seed, n_stories=toy_stories).split(
            "\n\n"
        )
    elif config.source == "textfile":
        if not config.paths:
            raise ValueError("data.source='textfile' requires data.paths.")
        documents = iter_text_documents(config.paths)
    else:
        documents = iter_hf_documents(config)

    tokens = tokenize_documents(documents, tokenizer, max_tokens=max_tokens)
    if tokens.size < 2 * (config.seq_len + 1):
        raise ValueError(
            f"corpus has only {tokens.size} tokens, which is not enough for even two "
            f"windows of {config.seq_len + 1}. Increase the corpus or lower seq_len."
        )

    n_val = int(tokens.size * config.val_fraction)
    # Round the split to a whole number of windows so no window straddles it.
    window = config.seq_len + 1
    n_val = max(window, (n_val // window) * window) if config.val_fraction > 0 else 0
    n_val = min(n_val, tokens.size - window)

    train = tokens[: tokens.size - n_val]
    val = tokens[tokens.size - n_val :] if n_val else tokens[-window:]
    log.info(
        "packed corpus: %s train tokens, %s val tokens (%.1f%% held out)",
        f"{train.size:,}",
        f"{val.size:,}",
        100.0 * val.size / max(1, tokens.size),
    )
    return PackedTokens(train=train, val=val, vocab_size=tokenizer.vocab_size)


# --------------------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------------------


class TokenBatcher:
    """Deterministic batch iterator over a packed token array.

    The corpus is cut into non-overlapping windows of ``seq_len + 1``. Each epoch
    permutes the windows with a seed derived from ``(seed, "data", epoch)``, so the
    ``n``-th batch of the ``k``-th epoch is a pure function of the global seed — which
    is what makes checkpoint resume bit-reproducible.

    Args:
        tokens: Flat array of token IDs.
        seq_len: Model context length.
        batch_size: Windows per batch.
        seed: Global seed.
        shuffle: Permute windows within an epoch. False gives a fixed sequential order,
            which is what validation wants.
        drop_last: Drop a trailing partial batch, keeping every batch the same shape.
    """

    def __init__(
        self,
        tokens: np.ndarray,
        *,
        seq_len: int,
        batch_size: int,
        seed: int = 1337,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        """Create a batcher over ``tokens``."""
        self.window = seq_len + 1
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.n_windows = tokens.size // self.window
        if self.n_windows == 0:
            raise ValueError(f"{tokens.size} tokens is fewer than one window of {self.window}.")
        usable = self.n_windows * self.window
        # int64 for embedding lookup; uint16 storage keeps the corpus small on disk.
        self.windows = torch.from_numpy(tokens[:usable].astype(np.int64, copy=False)).view(
            self.n_windows, self.window
        )
        self.epoch = 0

    def __len__(self) -> int:
        """Number of batches per epoch."""
        if self.drop_last:
            return self.n_windows // self.batch_size
        return (self.n_windows + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch, which selects the permutation."""
        self.epoch = epoch

    def _order(self) -> Tensor:
        if not self.shuffle:
            return torch.arange(self.n_windows)
        gen = torch.Generator().manual_seed(derive_seed(self.seed, "data", self.epoch))
        return torch.randperm(self.n_windows, generator=gen)

    def epoch_batches(self) -> Iterator[Batch]:
        """Yield one epoch of batches."""
        order = self._order()
        limit = len(self) * self.batch_size if self.drop_last else self.n_windows
        for start in range(0, limit, self.batch_size):
            idx = order[start : start + self.batch_size]
            if idx.numel() == 0:
                continue
            chunk = self.windows[idx]
            yield Batch(inputs=chunk[:, :-1].contiguous(), targets=chunk[:, 1:].contiguous())

    def stream(self, start_batch: int = 0) -> Iterator[Batch]:
        """Yield batches forever, starting from the ``start_batch``-th batch overall.

        The offset is what makes checkpoint resume exact. Position in the data is a pure
        function of how many batches have been consumed: epoch ``n // len(self)`` selects
        the permutation, and ``n % len(self)`` selects the position within it. Resuming
        by only restoring the *epoch* — and thus restarting that epoch from its first
        batch — silently re-trains on data the run already saw, and shows up as a
        resumed run that diverges from an uninterrupted one. That was a real bug here,
        and ``tests/dynamics`` now pins it.
        """
        per_epoch = max(1, len(self))
        epoch = start_batch // per_epoch
        offset = start_batch % per_epoch
        while True:
            self.set_epoch(epoch)
            for i, batch in enumerate(self.epoch_batches()):
                if i < offset:
                    continue
                yield batch
            offset = 0
            epoch += 1

    def __iter__(self) -> Iterator[Batch]:
        """Yield batches forever from the current epoch."""
        return self.stream(self.epoch * max(1, len(self)))

    def take(self, n: int) -> list[Batch]:
        """Materialise the first ``n`` batches — used by evaluation and by tests."""
        out: list[Batch] = []
        for batch in self.epoch_batches():
            out.append(batch)
            if len(out) >= n:
                break
        return out
