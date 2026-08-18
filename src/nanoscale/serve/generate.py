"""The serving generation loop (spec Phase 10).

A thin, explicit layer over :meth:`NanoScaleLM.generate` that adds the things a server
needs and a research loop does not:

* **streaming**, yielding decoded text incrementally rather than one block at the end;
* **stop sequences** beyond a single EOS token;
* **repetition penalty**, the one sampling control the core loop deliberately omits;
* a **timing breakdown** separating prefill from decode, because they are different
  costs with different scaling and reporting one number hides that.

Partial-token streaming
-----------------------
Byte-level BPE means a single token can be part of a multi-byte UTF-8 codepoint, so
decoding token-by-token can produce replacement characters mid-emoji. :class:`TextStreamer`
uses Python's **incremental UTF-8 decoder**, which distinguishes the two cases that a
naive try/except cannot:

* bytes that are *incomplete but still potentially valid*, buffer them and emit nothing;
* bytes that are *definitively invalid*: emit a replacement character and move on.

A hand-rolled ``try: buffer.decode() except UnicodeDecodeError: return ""`` conflates
them, and on an untrained model (whose tokens are effectively random bytes) it buffers
the entire output and emits nothing until the final flush. That is not a hypothetical:
it broke the stop-sequence check, which only sees text that has actually been emitted.
"""

from __future__ import annotations

import codecs
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from nanoscale.config import GenerateConfig
from nanoscale.model import KVCache, NanoScaleLM, sample_next_token
from nanoscale.tokenizer import BPETokenizer
from nanoscale.utils.logging import get_logger

__all__ = ["GenerationOutput", "TextStreamer", "generate_text", "stream_text"]

log = get_logger("nanoscale.serve")


class TextStreamer:
    """Decodes tokens to text incrementally, holding back partial UTF-8 sequences."""

    def __init__(self, tokenizer: BPETokenizer, *, skip_special: bool = True) -> None:
        """Create a streamer over ``tokenizer``."""
        self.tokenizer = tokenizer
        self.skip_special = skip_special
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def push(self, token: int) -> str:
        """Add one token; return whatever text is now safely decodable (often ``""``)."""
        if self.skip_special and token in self.tokenizer.id_to_special:
            return ""
        return self._decoder.decode(self.tokenizer.vocab[token])

    def flush(self) -> str:
        """Emit any remaining buffered bytes, replacing an unfinished codepoint."""
        tail = self._decoder.decode(b"", final=True)
        self._decoder.reset()
        return tail


@dataclass(slots=True)
class GenerationOutput:
    """A completed generation with its timing breakdown."""

    text: str
    prompt_tokens: int
    generated_tokens: int
    prefill_s: float
    decode_s: float
    stop_reason: str = "length"
    token_ids: list[int] = field(default_factory=list)

    @property
    def total_s(self) -> float:
        """Wall clock for the whole request."""
        return self.prefill_s + self.decode_s

    @property
    def decode_tokens_per_s(self) -> float:
        """Decode throughput, excluding prefill.

        Reported separately because prefill is parallel over the prompt while decode is
        sequential: a single tokens/second figure over both is dominated by whichever
        phase the prompt length happens to favour.
        """
        return self.generated_tokens / max(1e-9, self.decode_s)

    def summary(self) -> dict[str, float | int | str]:
        """Flat numbers for logging and for the bench table."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "prefill_s": round(self.prefill_s, 5),
            "decode_s": round(self.decode_s, 5),
            "total_s": round(self.total_s, 5),
            "decode_tokens_per_s": round(self.decode_tokens_per_s, 2),
            "stop_reason": self.stop_reason,
        }


def _apply_repetition_penalty(logits: Tensor, generated: Sequence[int], penalty: float) -> Tensor:
    """Divide the logit of every already-generated token by ``penalty``.

    Following Keskar et al. (CTRL): for a *negative* logit the penalty must **multiply**
    rather than divide, otherwise dividing a negative number by a number greater than one
    moves it closer to zero and makes the token *more* likely, the opposite of the
    intent. Getting this backwards is a common bug.
    """
    if penalty == 1.0 or not generated:
        return logits
    out = logits.clone()
    unique = torch.tensor(sorted(set(generated)), device=logits.device, dtype=torch.long)
    values = out[..., unique]
    out[..., unique] = torch.where(values > 0, values / penalty, values * penalty)
    return out


@torch.no_grad()
def stream_text(
    model: NanoScaleLM,
    tokenizer: BPETokenizer,
    prompt: str,
    config: GenerateConfig | None = None,
    *,
    stop: Sequence[str] = (),
    prompt_ids: Sequence[int] | None = None,
) -> Iterator[str]:
    """Yield decoded text incrementally as the model generates.

    Args:
        model: The model to sample from.
        tokenizer: Its tokenizer.
        prompt: The prompt text (ignored if ``prompt_ids`` is given).
        config: Sampling configuration.
        stop: Stop sequences; generation halts once one appears in the output.
        prompt_ids: Pre-rendered prompt tokens, e.g. from a chat template.
    """
    cfg = config or GenerateConfig()
    was_training = model.training
    model.eval()
    try:
        ids = list(prompt_ids) if prompt_ids is not None else tokenizer.encode(prompt, add_bos=True)
        tokens = torch.tensor([ids], device=model.device)
        budget = min(model.config.max_seq_len, tokens.shape[1] + cfg.max_new_tokens)
        cache: KVCache = model.make_cache(1, max_seq_len=budget)

        logits = model(tokens, cache=cache).logits[:, -1]
        generator = torch.Generator().manual_seed(cfg.seed)
        streamer = TextStreamer(tokenizer)
        produced: list[int] = []
        emitted = ""

        for _ in range(cfg.max_new_tokens):
            if tokens.shape[1] >= budget:
                break
            penalised = _apply_repetition_penalty(logits, produced, cfg.repetition_penalty)
            token = sample_next_token(
                penalised,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                generator=generator,
            )
            token_id = int(token)
            if cfg.stop_on_eos and token_id in (tokenizer.eos_id, tokenizer.eot_id):
                break

            produced.append(token_id)
            piece = streamer.push(token_id)
            if piece:
                emitted += piece
                yield piece
            if stop and any(s in emitted for s in stop):
                break

            step = token.view(1, 1)
            tokens = torch.cat([tokens, step], dim=1)
            logits = model(step, cache=cache).logits[:, -1]

        tail = streamer.flush()
        if tail:
            yield tail
    finally:
        model.train(was_training)


@torch.no_grad()
def generate_text(
    model: NanoScaleLM,
    tokenizer: BPETokenizer,
    prompt: str,
    config: GenerateConfig | None = None,
    *,
    stop: Sequence[str] = (),
    prompt_ids: Sequence[int] | None = None,
) -> GenerationOutput:
    """Generate a completion, measuring prefill and decode separately."""
    cfg = config or GenerateConfig()
    was_training = model.training
    model.eval()
    try:
        ids = list(prompt_ids) if prompt_ids is not None else tokenizer.encode(prompt, add_bos=True)
        tokens = torch.tensor([ids], device=model.device)
        budget = min(model.config.max_seq_len, tokens.shape[1] + cfg.max_new_tokens)
        cache: KVCache = model.make_cache(1, max_seq_len=budget)

        start = time.perf_counter()
        logits = model(tokens, cache=cache).logits[:, -1]
        prefill_s = time.perf_counter() - start

        generator = torch.Generator().manual_seed(cfg.seed)
        streamer = TextStreamer(tokenizer)
        produced: list[int] = []
        text = ""
        stop_reason = "length"

        decode_start = time.perf_counter()
        for _ in range(cfg.max_new_tokens):
            if tokens.shape[1] >= budget:
                stop_reason = "context"
                break
            penalised = _apply_repetition_penalty(logits, produced, cfg.repetition_penalty)
            token = sample_next_token(
                penalised,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                generator=generator,
            )
            token_id = int(token)
            if cfg.stop_on_eos and token_id in (tokenizer.eos_id, tokenizer.eot_id):
                stop_reason = "eos"
                break

            produced.append(token_id)
            text += streamer.push(token_id)
            if stop and any(s in text for s in stop):
                stop_reason = "stop_sequence"
                break

            step = token.view(1, 1)
            tokens = torch.cat([tokens, step], dim=1)
            logits = model(step, cache=cache).logits[:, -1]
        text += streamer.flush()
        decode_s = time.perf_counter() - decode_start

        return GenerationOutput(
            text=text,
            prompt_tokens=len(ids),
            generated_tokens=len(produced),
            prefill_s=prefill_s,
            decode_s=decode_s,
            stop_reason=stop_reason,
            token_ids=produced,
        )
    finally:
        model.train(was_training)
