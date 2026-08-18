"""Offline instruction and preference datasets for the ``nano`` alignment track.

Why these are synthetic
-----------------------
Alignment data is the one part of the pipeline where a real corpus (UltraFeedback,
HH-RLHF) would need the network and a licence check, and where a 5M-parameter model
trained on a 400-word synthetic vocabulary could not use it anyway. So the ``nano``
tier gets instruction and preference data generated from the *same* story grammar the
model was pretrained on: the model has the vocabulary to follow these instructions, and
the preference signal is something it can actually learn in a hundred CPU steps.

The preference axis is deliberately **not** "one response is longer". It is:

* **chosen**: answers the question, on-topic, ends properly with ``<eot>``;
* **rejected**: one of a small set of concrete failure modes (off-topic, truncated,
  degenerate repetition, or a non-answer).

That matters for the Phase-6 length-exploitation diagnostic (spec E4). If the preference
data itself correlated quality with length, DPO's known length bias would be
indistinguishable from it having learned the labels. Here the rejected responses are
generated with a *length distribution deliberately matched* to the chosen ones; a test
asserts the two are within a few percent, so any length drift after DPO comes from the
objective, not from the data.

``micro``/``small`` point at real preference data instead; see ``configs/align/``.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final, Literal

__all__ = [
    "InstructExample",
    "PreferencePair",
    "RejectionKind",
    "iter_instructions",
    "iter_preference_pairs",
]

RejectionKind = Literal["off_topic", "truncated", "repetitive", "non_answer"]


@dataclass(frozen=True, slots=True)
class InstructExample:
    """One instruction-following example."""

    instruction: str
    response: str
    system: str | None = None


@dataclass(frozen=True, slots=True)
class PreferencePair:
    """A prompt with a preferred and a dispreferred response."""

    prompt: str
    chosen: str
    rejected: str
    rejection_kind: RejectionKind

    @property
    def length_delta(self) -> int:
        """``len(chosen) - len(rejected)`` in characters, for the length diagnostic."""
        return len(self.chosen) - len(self.rejected)


_NAMES: Final[tuple[str, ...]] = (
    "Lily",
    "Tom",
    "Mia",
    "Ben",
    "Anna",
    "Sam",
    "Nora",
    "Max",
    "Ella",
    "Leo",
    "Ruby",
    "Finn",
    "Clara",
    "Oscar",
    "Poppy",
    "Hugo",
    "Iris",
    "Jonah",
)
_PLACES: Final[tuple[str, ...]] = (
    "the park",
    "the garden",
    "the river",
    "the quiet forest",
    "the sunny beach",
    "the green field",
    "the little bridge",
    "the old house",
    "the busy market",
)
_OBJECTS: Final[tuple[str, ...]] = (
    "a red ball",
    "a blue box",
    "a shiny key",
    "a paper boat",
    "a warm hat",
    "a big book",
    "a long rope",
    "a silver spoon",
    "a glass jar",
    "a torn map",
)
_ANIMALS: Final[tuple[str, ...]] = (
    "a small dog",
    "a grey cat",
    "a brown bird",
    "a tiny frog",
    "a clever fox",
    "a sleepy owl",
    "a quiet mouse",
    "a happy duck",
)
_FEELINGS: Final[tuple[str, ...]] = (
    "happy",
    "tired",
    "excited",
    "curious",
    "brave",
    "quiet",
    "proud",
    "grateful",
)

_SYSTEM: Final[str] = "You are a helpful assistant who answers with short clear stories."


def _templates(rng: random.Random) -> tuple[str, str]:
    """Draw one (instruction, response) pair from the instruction grammar."""
    name = rng.choice(_NAMES)
    place = rng.choice(_PLACES)
    obj = rng.choice(_OBJECTS)
    animal = rng.choice(_ANIMALS)
    feeling = rng.choice(_FEELINGS)

    kind = rng.randrange(6)
    if kind == 0:
        return (
            f"Tell me a short story about {name} at {place}.",
            f"{name} went to {place} with {animal}. They found {obj} near the path. "
            f"{name} was very {feeling}.",
        )
    if kind == 1:
        return (
            f"What did {name} find at {place}?",
            f"{name} found {obj} at {place}. It was small and easy to carry home.",
        )
    if kind == 2:
        return (
            f"Who went with {name} to {place}?",
            f"{animal} went with {name} to {place}. They walked together all morning.",
        )
    if kind == 3:
        return (
            f"How did {name} feel after finding {obj}?",
            f"{name} felt very {feeling} after finding {obj}. It had been a long search.",
        )
    if kind == 4:
        return (
            f"Write one sentence about {animal} at {place}.",
            f"{animal} sat quietly at {place} and watched the water move.",
        )
    return (
        f"Give me a short ending for a story about {name} and {obj}.",
        f"At last {name} carried {obj} home and set it by the window. {name} was {feeling}.",
    )


def iter_instructions(*, seed: int = 1337, n: int = 2000) -> Iterator[InstructExample]:
    """Yield instruction-following examples deterministically."""
    rng = random.Random(seed)
    for _ in range(n):
        instruction, response = _templates(rng)
        system = _SYSTEM if rng.random() < 0.3 else None
        yield InstructExample(instruction=instruction, response=response, system=system)


def _reject(chosen: str, kind: RejectionKind, rng: random.Random) -> str:
    """Produce a dispreferred response of roughly the same length as ``chosen``.

    Length matching is the point: see the module docstring. Each failure mode is padded
    or trimmed toward the chosen response's length so that "which is longer" carries no
    information about which is preferred.
    """
    target = len(chosen)
    if kind == "off_topic":
        # Grammatical and fluent, but about something else entirely.
        filler = (
            f"{rng.choice(_NAMES)} counted to three and looked at the sky. "
            f"The wind moved {rng.choice(_OBJECTS)} across {rng.choice(_PLACES)}. "
        )
        text = (filler * 3)[:target].rstrip()
    elif kind == "truncated":
        # Starts correctly and then stops mid-thought, then pads with a stray clause so
        # the length still matches.
        cut = max(8, len(chosen) // 3)
        text = chosen[:cut] + " and then " + ("the " * ((target - cut) // 4))
        text = text[:target].rstrip()
    elif kind == "repetitive":
        # The repeated unit must be short enough that the padded-to-length result
        # actually contains a repeat. Using the whole first sentence fails on
        # single-sentence responses: the truncation lands exactly at its end and the
        # "rejected" text comes out identical to the chosen one, which would feed DPO a
        # pair labelled both preferred and dispreferred. A test pins this.
        first = chosen.split(".")[0].strip()
        unit = " ".join(first.split()[: max(2, len(first.split()) // 2)]) + ". "
        text = (unit * 10)[:target].rstrip()
    else:  # non_answer
        stub = "I am not sure about that. Maybe someone else knows. "
        text = (stub * 4)[:target].rstrip()

    if not text or text == chosen:  # pragma: no cover - defensive
        text = "I am not sure about that."
    return text


def iter_preference_pairs(*, seed: int = 1337, n: int = 800) -> Iterator[PreferencePair]:
    """Yield preference pairs whose chosen/rejected lengths are matched by construction."""
    rng = random.Random(seed)
    kinds: tuple[RejectionKind, ...] = ("off_topic", "truncated", "repetitive", "non_answer")
    for _ in range(n):
        instruction, response = _templates(rng)
        kind = kinds[rng.randrange(len(kinds))]
        yield PreferencePair(
            prompt=instruction,
            chosen=response,
            rejected=_reject(response, kind, rng),
            rejection_kind=kind,
        )
