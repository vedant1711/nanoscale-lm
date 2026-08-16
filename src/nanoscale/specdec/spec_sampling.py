"""Draft–target speculative decoding (spec B8).

The loop
--------
Per round:

1. The **draft** model autoregressively proposes ``γ`` tokens — ``γ`` sequential forward
   passes, but of a model several times smaller.
2. The **target** model scores all ``γ`` proposals **in one forward pass**. This is the
   whole trick: verifying ``γ`` tokens costs the target one pass, because attention over
   an already-known sequence is parallel across positions — the same reason training is
   parallel and decoding is not.
3. The acceptance rule (:mod:`nanoscale.specdec.accept_rule`) walks the proposals left to
   right, accepting each with probability ``min(1, p/q)`` and stopping at the first
   rejection, where it resamples from the normalised residual.
4. If **all** ``γ`` are accepted, one extra token comes free from the target's own
   distribution at position ``γ+1``, which the target already computed in step 2. So a
   perfect draft yields ``γ+1`` tokens per target pass.

Cache invariant
---------------
Both caches hold exactly ``len(tokens) - 1`` positions at the top of every round: every
token except the last. A round then feeds ``[last_token] + proposals`` to the target, so
one pass produces the distribution for proposal 0 (conditioned on the prefix), for each
subsequent proposal, *and* for the bonus position. Getting this off by one is the classic
way to build a speculative decoder that runs, produces fluent text, and silently samples
from the wrong distribution — which is why the distributional-equivalence test in
``tests/unit/test_specdec.py`` is the most important test in the repository.

After the accept/reject walk, both caches are truncated back to ``new_len - 1``,
discarding the speculative positions that were rejected.

Why this is a win
-----------------
Single-stream decoding is memory-bandwidth-bound: each step reads every weight to produce
one token. Verifying ``γ`` tokens reads the weights once instead of ``γ`` times, so the
speedup is bounded above by the mean accepted length. The draft model's cost is what you
pay for it, which is why the draft must be several times smaller.

**Losslessness.** The output distribution is exactly the target's — proven in
:mod:`nanoscale.specdec.accept_rule`, verified statistically over 100k samples in the
tests. This is not an approximation with a quality knob.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from torch import Tensor

from nanoscale.model import NanoScaleLM
from nanoscale.specdec.accept_rule import sample_accept_reject
from nanoscale.utils.logging import get_logger

__all__ = [
    "SpeculativeResult",
    "SpeculativeSampler",
    "apply_sampling_transforms",
    "autoregressive_baseline",
]

log = get_logger("nanoscale.specdec")


def apply_sampling_transforms(
    logits: Tensor, *, temperature: float = 1.0, top_p: float = 1.0
) -> Tensor:
    """Turn logits into the distribution the accept/reject rule operates on.

    The **same** transform must be applied to both models. The correctness proof is about
    the distributions actually sampled from, so if the draft is nucleus-filtered and the
    target is not, the emitted distribution is neither model's.
    """
    if temperature <= 0.0:
        probs = torch.zeros_like(logits, dtype=torch.float32)
        probs.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return probs

    scaled = logits.float() / temperature
    if top_p < 1.0:
        ordered, order = torch.sort(scaled, descending=True, dim=-1)
        probs = torch.softmax(ordered, dim=-1)
        cumulative = probs.cumsum(dim=-1) - probs
        drop = cumulative >= top_p
        drop[..., 0] = False
        ordered = ordered.masked_fill(drop, float("-inf"))
        scaled = torch.empty_like(scaled).scatter_(-1, order, ordered)
    return torch.softmax(scaled, dim=-1)


@dataclass(slots=True)
class SpeculativeResult:
    """The outcome of a generation run, speculative or autoregressive."""

    tokens: Tensor
    generated: int
    target_calls: int
    draft_calls: int
    accepted_tokens: int
    proposed_tokens: int
    wall_clock_s: float
    per_round_accepted: list[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens that were accepted."""
        return self.accepted_tokens / max(1, self.proposed_tokens)

    @property
    def mean_accepted_length(self) -> float:
        """Tokens emitted per target forward pass — the theoretical speedup bound."""
        return self.generated / max(1, self.target_calls)

    @property
    def tokens_per_second(self) -> float:
        """Measured decode throughput."""
        return self.generated / max(1e-9, self.wall_clock_s)

    def summary(self) -> dict[str, float | int]:
        """Headline numbers for the benchmark table."""
        return {
            "generated": self.generated,
            "target_calls": self.target_calls,
            "draft_calls": self.draft_calls,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "mean_accepted_length": round(self.mean_accepted_length, 4),
            "wall_clock_s": round(self.wall_clock_s, 4),
            "tokens_per_s": round(self.tokens_per_second, 2),
        }


class SpeculativeSampler:
    """Draft–target speculative decoding with the exact modified-rejection rule."""

    def __init__(
        self,
        target: NanoScaleLM,
        draft: NanoScaleLM,
        *,
        gamma: int = 4,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> None:
        """Pair a target model with a smaller draft model."""
        if target.config.vocab_size != draft.config.vocab_size:
            raise ValueError(
                f"target vocab {target.config.vocab_size} != draft vocab "
                f"{draft.config.vocab_size}; the two models must share a tokenizer."
            )
        if gamma < 1:
            raise ValueError(f"gamma must be at least 1, got {gamma}.")
        self.target = target.eval()
        self.draft = draft.eval()
        self.gamma = gamma
        self.temperature = temperature
        self.top_p = top_p

    def _probs(self, logits: Tensor) -> Tensor:
        return apply_sampling_transforms(logits, temperature=self.temperature, top_p=self.top_p)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int = 64,
        eos_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> SpeculativeResult:
        """Generate speculatively; returns the tokens and the acceptance statistics.

        Only batch size 1 is supported. Batched speculation needs per-row bookkeeping of
        divergent accepted lengths and a ragged cache — a serving-engine concern rather
        than an algorithmic one. The correctness argument is unchanged by it.
        """
        if input_ids.shape[0] != 1:
            raise ValueError(
                f"speculative decoding here is single-sequence; got batch {input_ids.shape[0]}."
            )

        prompt_len = input_ids.shape[1]
        budget = min(self.target.config.max_seq_len, prompt_len + max_new_tokens + self.gamma + 1)
        target_cache = self.target.make_cache(1, max_seq_len=budget)
        draft_cache = self.draft.make_cache(1, max_seq_len=budget)

        tokens = input_ids
        target_calls = draft_calls = 0
        accepted_total = proposed_total = 0
        per_round: list[int] = []
        start = time.perf_counter()

        # Establish the invariant: both caches hold everything except the last token.
        if prompt_len > 1:
            self.target(tokens[:, :-1], cache=target_cache)
            self.draft(tokens[:, :-1], cache=draft_cache)
            target_calls += 1
            draft_calls += 1

        finished = False
        while tokens.shape[1] - prompt_len < max_new_tokens and not finished:
            if tokens.shape[1] + self.gamma + 1 > budget:
                break
            remaining = max_new_tokens - (tokens.shape[1] - prompt_len)
            gamma = min(self.gamma, remaining)
            base_len = tokens.shape[1]

            # --- 1. draft gamma tokens autoregressively -------------------------------
            draft_tokens: list[Tensor] = []
            draft_probs: list[Tensor] = []
            draft_input = tokens[:, -1:]
            for _ in range(gamma):
                logits = self.draft(draft_input, cache=draft_cache).logits[:, -1]
                draft_calls += 1
                q = self._probs(logits)
                token = torch.multinomial(q, 1, generator=generator).squeeze(-1)
                draft_tokens.append(token)
                draft_probs.append(q)
                draft_input = token.unsqueeze(1)
            proposed_total += gamma

            # --- 2. target verifies [last] + proposals in ONE pass ---------------------
            verify_input = torch.cat(
                [tokens[:, -1:], *[t.unsqueeze(1) for t in draft_tokens]], dim=1
            )
            verify_logits = self.target(verify_input, cache=target_cache).logits
            target_calls += 1
            # verify_logits[:, j] is the target distribution for the token at position
            # base_len + j, i.e. exactly the distribution proposal j must be judged against.
            target_dists = [self._probs(verify_logits[:, j]) for j in range(gamma + 1)]

            # --- 3. walk the proposals, stopping at the first rejection ----------------
            emitted: list[Tensor] = []
            n_accepted = 0
            for j in range(gamma):
                token, accepted = sample_accept_reject(
                    target_dists[j], draft_probs[j], draft_tokens[j], generator=generator
                )
                emitted.append(token)
                if not bool(accepted):
                    break
                n_accepted += 1

            accepted_total += n_accepted
            per_round.append(n_accepted)

            # --- 4. all accepted: one bonus token, free, from the target ---------------
            if n_accepted == gamma:
                bonus = torch.multinomial(target_dists[gamma], 1, generator=generator).squeeze(-1)
                emitted.append(bonus)

            new_tokens = torch.cat([t.unsqueeze(1) for t in emitted], dim=1)
            tokens = torch.cat([tokens, new_tokens], dim=1)

            # --- restore the invariant: caches hold all but the last token -------------
            keep = tokens.shape[1] - 1
            target_cache.truncate(keep)
            draft_cache.truncate(keep)
            if draft_cache.length < keep:
                # The draft only ever advanced over its own proposals; after a rejection
                # it must catch up on the tokens that were actually emitted.
                self.draft(tokens[:, draft_cache.length : keep], cache=draft_cache)
                draft_calls += 1

            if eos_id is not None and bool((new_tokens == eos_id).any()):
                finished = True
            del base_len

        wall = time.perf_counter() - start
        tokens = tokens[:, : prompt_len + max_new_tokens]
        return SpeculativeResult(
            tokens=tokens,
            generated=tokens.shape[1] - prompt_len,
            target_calls=target_calls,
            draft_calls=draft_calls,
            accepted_tokens=accepted_total,
            proposed_tokens=proposed_total,
            wall_clock_s=wall,
            per_round_accepted=per_round,
        )


@torch.no_grad()
def autoregressive_baseline(
    model: NanoScaleLM,
    input_ids: Tensor,
    *,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> SpeculativeResult:
    """Plain cached autoregressive decoding, reported in the same shape for comparison."""
    was_training = model.training
    model.eval()
    try:
        prompt_len = input_ids.shape[1]
        budget = min(model.config.max_seq_len, prompt_len + max_new_tokens)
        cache = model.make_cache(input_ids.shape[0], max_seq_len=budget)

        tokens = input_ids
        start = time.perf_counter()
        logits = model(tokens, cache=cache).logits[:, -1]
        calls = 1
        for _ in range(max_new_tokens):
            if tokens.shape[1] >= budget:
                break
            probs = apply_sampling_transforms(logits, temperature=temperature, top_p=top_p)
            token = torch.multinomial(probs, 1, generator=generator)
            tokens = torch.cat([tokens, token], dim=1)
            if eos_id is not None and bool((token == eos_id).any()):
                break
            logits = model(token, cache=cache).logits[:, -1]
            calls += 1
        wall = time.perf_counter() - start

        return SpeculativeResult(
            tokens=tokens,
            generated=tokens.shape[1] - prompt_len,
            target_calls=calls,
            draft_calls=0,
            accepted_tokens=0,
            proposed_tokens=0,
            wall_clock_s=wall,
        )
    finally:
        model.train(was_training)
