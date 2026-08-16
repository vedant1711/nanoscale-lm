"""A scripted head-to-head preference evaluation (spec Phase 6 acceptance).

Spec Phase 6 requires the aligned model to "win a scripted head-to-head vs the SFT
model on a small preference eval". At this scale there is no LLM judge available and no
budget for a human one, so the judge has to be **programmatic and stated**.

The judge scores a completion on three criteria drawn from the failure modes the
preference data actually contains (see :mod:`nanoscale.data.instruct`):

1. **On-topic** — does the completion mention the entities named in the prompt?
2. **Non-degenerate** — is it free of the immediate n-gram repetition that the
   ``repetitive`` rejection mode exhibits?
3. **Terminated** — did the model emit ``<eot>`` rather than run to the token cap?

These are the properties the preference labels encode, so a model that learned the
labels should score higher. That is a real, checkable claim — and a much weaker one than
"the aligned model is better", which nothing at this scale could support. The write-up
says so.

The judge is deliberately **not** length-sensitive, so that a model which learned to
game DPO's length bias gains nothing here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from nanoscale.data.instruct import PreferencePair
from nanoscale.model import NanoScaleLM
from nanoscale.tokenizer import BPETokenizer, Message, render_prompt

__all__ = [
    "CompletionScore",
    "HeadToHeadResult",
    "head_to_head",
    "repetition_rate",
    "score_completion",
]

_WORD = re.compile(r"[A-Za-z']+")


def repetition_rate(text: str, *, n: int = 3) -> float:
    """Fraction of ``n``-grams that are repeats — the degeneracy diagnostic.

    ``0.0`` means every n-gram is unique; values above ~0.3 indicate the looping that
    small models fall into when they have nothing left to say.
    """
    words = _WORD.findall(text.lower())
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


@dataclass(frozen=True, slots=True)
class CompletionScore:
    """A judged completion."""

    on_topic: float
    non_degenerate: float
    terminated: float
    n_tokens: int

    @property
    def total(self) -> float:
        """Unweighted sum of the three criteria, in ``[0, 3]``."""
        return self.on_topic + self.non_degenerate + self.terminated


def score_completion(prompt: str, completion: str, *, terminated: bool) -> CompletionScore:
    """Score one completion against the three stated criteria."""
    prompt_words = {w for w in _WORD.findall(prompt.lower()) if len(w) > 3}
    completion_words = set(_WORD.findall(completion.lower()))
    overlap = len(prompt_words & completion_words) / max(1, len(prompt_words))

    return CompletionScore(
        on_topic=min(1.0, overlap * 2.0),
        non_degenerate=1.0 - min(1.0, repetition_rate(completion) * 2.0),
        terminated=1.0 if terminated else 0.0,
        n_tokens=len(_WORD.findall(completion)),
    )


@dataclass(frozen=True, slots=True)
class HeadToHeadResult:
    """Outcome of a scripted head-to-head between two checkpoints."""

    n_prompts: int
    wins_a: int
    wins_b: int
    ties: int
    mean_score_a: float
    mean_score_b: float
    mean_len_a: float
    mean_len_b: float
    label_a: str = "A"
    label_b: str = "B"

    @property
    def win_rate_b(self) -> float:
        """Fraction of decided comparisons that ``B`` won."""
        decided = self.wins_a + self.wins_b
        return self.wins_b / decided if decided else 0.5

    def summary(self) -> dict[str, float | int | str]:
        """Flat numbers for the manifest and the comparison table."""
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "n_prompts": self.n_prompts,
            "wins_a": self.wins_a,
            "wins_b": self.wins_b,
            "ties": self.ties,
            "win_rate_b": round(self.win_rate_b, 4),
            "mean_score_a": round(self.mean_score_a, 4),
            "mean_score_b": round(self.mean_score_b, 4),
            "mean_len_a": round(self.mean_len_a, 2),
            "mean_len_b": round(self.mean_len_b, 2),
        }


@torch.no_grad()
def _generate(
    model: NanoScaleLM,
    tokenizer: BPETokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    seed: int,
    temperature: float,
) -> tuple[str, bool, int]:
    ids = torch.tensor([render_prompt(tokenizer, [Message("user", prompt)])], device=model.device)
    gen = torch.Generator().manual_seed(seed)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        eos_id=tokenizer.eot_id,
        generator=gen,
    )
    produced = out[0, ids.shape[1] :].tolist()
    terminated = tokenizer.eot_id in produced or tokenizer.eos_id in produced
    return tokenizer.decode(produced, skip_special=True), terminated, len(produced)


def head_to_head(
    model_a: NanoScaleLM,
    model_b: NanoScaleLM,
    tokenizer: BPETokenizer,
    pairs: Sequence[PreferencePair],
    *,
    n_prompts: int = 40,
    max_new_tokens: int = 48,
    temperature: float = 0.7,
    seed: int = 1337,
    label_a: str = "sft",
    label_b: str = "aligned",
) -> HeadToHeadResult:
    """Compare two models on the same prompts with the same sampling seed.

    Both models see identical prompts and identical RNG seeds per prompt, so the
    comparison is paired rather than two independent samples — which is what makes 40
    prompts informative at all.
    """
    was_a, was_b = model_a.training, model_b.training
    model_a.eval()
    model_b.eval()
    try:
        wins_a = wins_b = ties = 0
        total_a = total_b = 0.0
        len_a = len_b = 0.0
        prompts = [p.prompt for p in pairs[:n_prompts]]

        for i, prompt in enumerate(prompts):
            text_a, term_a, tok_a = _generate(
                model_a,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                seed=seed + i,
                temperature=temperature,
            )
            text_b, term_b, tok_b = _generate(
                model_b,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                seed=seed + i,
                temperature=temperature,
            )
            score_a = score_completion(prompt, text_a, terminated=term_a)
            score_b = score_completion(prompt, text_b, terminated=term_b)
            total_a += score_a.total
            total_b += score_b.total
            len_a += tok_a
            len_b += tok_b
            if score_b.total > score_a.total + 1e-9:
                wins_b += 1
            elif score_a.total > score_b.total + 1e-9:
                wins_a += 1
            else:
                ties += 1

        n = max(1, len(prompts))
        return HeadToHeadResult(
            n_prompts=len(prompts),
            wins_a=wins_a,
            wins_b=wins_b,
            ties=ties,
            mean_score_a=total_a / n,
            mean_score_b=total_b / n,
            mean_len_a=len_a / n,
            mean_len_b=len_b / n,
            label_a=label_a,
            label_b=label_b,
        )
    finally:
        model_a.train(was_a)
        model_b.train(was_b)
