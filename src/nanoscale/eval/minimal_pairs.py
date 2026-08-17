"""A BLiMP-style minimal-pair benchmark, generated rather than hand-written.

The hand-written 28-question benchmark in :mod:`nanoscale.eval.tiny_bench` saturated: the
base model scores 100%, so it can only detect *damage* from compression and cannot measure
capability at all. This module replaces it as the primary quality metric.

The method is Warstadt et al.'s BLiMP (2020). Each item is a **minimal pair**: two
sentences differing in exactly one place, one grammatical and one not.

.. code-block:: text

    good:  The boy near the cats runs quickly.
    bad:   The boy near the cats run quickly.

The model scores each sentence's total log-probability, and the item is correct if it
assigns more probability to the grammatical one. Three properties make this the right
measurement for this project:

1. **Chance is exactly 50%**, and it is a *forced choice*, so there is no prompt-format
   sensitivity of the kind that made the old agreement probe read as chance
   (see ``docs/limitations.md``).
2. **It is tokenizer-independent.** Both sentences are scored by the same model with the
   same tokenizer, and only their ordering matters — so scores are comparable across
   models that tokenize differently, unlike perplexity.
3. **Difficulty is controllable.** An agreement item with an intervening attractor noun
   (``the boy near the cats``) is far harder than one without, because the model must
   track the true subject across a closer, disagreeing noun. That gives headroom, which a
   saturated benchmark does not.

Every phenomenon here is generated from templates over a lexicon, so the suite scales to
thousands of items with a seed rather than being capped by what a person will type out.
Items are deduplicated and the lexicon is partitioned so that evaluation items never reuse
a noun/verb combination from the training corpus generator.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from nanoscale.tokenizer import BPETokenizer

__all__ = [
    "PHENOMENA",
    "MinimalPair",
    "MinimalPairResult",
    "PhenomenonScore",
    "generate_pairs",
    "run_minimal_pairs",
    "wilson_interval",
]

# ---------------------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------------------

#: ``(singular, plural)`` animate nouns, all of which appear in the story distribution.
ANIMATE = (
    ("boy", "boys"),
    ("girl", "girls"),
    ("cat", "cats"),
    ("dog", "dogs"),
    ("bird", "birds"),
    ("rabbit", "rabbits"),
    ("bear", "bears"),
    ("frog", "frogs"),
    ("mouse", "mice"),
    ("duck", "ducks"),
    ("fox", "foxes"),
    ("child", "children"),
)

#: ``(singular, plural)`` inanimate nouns.
INANIMATE = (
    ("ball", "balls"),
    ("box", "boxes"),
    ("hat", "hats"),
    ("cup", "cups"),
    ("toy", "toys"),
    ("book", "books"),
    ("stone", "stones"),
    ("flower", "flowers"),
    ("key", "keys"),
    ("apple", "apples"),
    ("leaf", "leaves"),
    ("chair", "chairs"),
)

#: ``(third-person-singular, bare-plural)`` intransitive verbs.
INTRANSITIVE = (
    ("runs", "run"),
    ("jumps", "jump"),
    ("sleeps", "sleep"),
    ("plays", "play"),
    ("sings", "sing"),
    ("walks", "walk"),
    ("smiles", "smile"),
    ("laughs", "laugh"),
)

#: ``(third-person-singular, bare-plural)`` transitive verbs.
TRANSITIVE = (
    ("finds", "find"),
    ("holds", "hold"),
    ("wants", "want"),
    ("sees", "see"),
    ("likes", "like"),
    ("takes", "take"),
    ("drops", "drop"),
    ("carries", "carry"),
)

MALE = ("Tom", "Ben", "Sam", "Max", "Jack", "Leo")
FEMALE = ("Mia", "Lily", "Anna", "Sue", "Emma", "Zoe")


def article(word: str) -> str:
    """``a`` or ``an``, by the following word's initial sound.

    Without this the generator produced "a apple", which is ungrammatical for a reason
    that has nothing to do with the phenomenon under test — the model could then get the
    item right by noticing the article, not the target contrast. Approximating "sound" by
    "starts with a vowel letter" is wrong for words like *hour*, none of which are in this
    lexicon.
    """
    return "an" if word[:1].lower() in "aeiou" else "a"


@dataclass(frozen=True)
class MinimalPair:
    """One forced-choice item: the grammatical sentence and its minimal corruption."""

    phenomenon: str
    good: str
    bad: str

    def __str__(self) -> str:
        """Show the pair as ``good / bad``."""
        return f"{self.good!r} / {self.bad!r}"


# ---------------------------------------------------------------------------------------
# Generators, one per phenomenon
# ---------------------------------------------------------------------------------------


def _agreement_simple(rng: random.Random, n: int) -> Iterator[MinimalPair]:
    """Subject-verb agreement with no intervening material. The easy control condition."""
    for _ in range(n):
        sg, pl = rng.choice(ANIMATE)
        vs, vp = rng.choice(INTRANSITIVE)
        if rng.random() < 0.5:
            yield MinimalPair("agreement_simple", f"The {sg} {vs}.", f"The {sg} {vp}.")
        else:
            yield MinimalPair("agreement_simple", f"The {pl} {vp}.", f"The {pl} {vs}.")


def _agreement_attractor(rng: random.Random, n: int) -> Iterator[MinimalPair]:
    """Agreement across an intervening noun of the *opposite* number.

    This is the diagnostic condition. A model doing agreement by "match the nearest noun"
    scores at or below chance here while scoring perfectly on the simple condition, so the
    gap between the two is the measurement that matters.
    """
    for _ in range(n):
        sg, pl = rng.choice(ANIMATE)
        other_sg, other_pl = rng.choice(INANIMATE)
        vs, vp = rng.choice(INTRANSITIVE)
        prep = rng.choice(("near", "beside", "behind", "next to"))
        if rng.random() < 0.5:
            # Singular subject, plural attractor.
            stem = f"The {sg} {prep} the {other_pl}"
            yield MinimalPair("agreement_attractor", f"{stem} {vs}.", f"{stem} {vp}.")
        else:
            stem = f"The {pl} {prep} the {other_sg}"
            yield MinimalPair("agreement_attractor", f"{stem} {vp}.", f"{stem} {vs}.")


def _determiner_noun(rng: random.Random, n: int) -> Iterator[MinimalPair]:
    """Determiner-noun number agreement: ``this cat`` vs ``these cat``."""
    for _ in range(n):
        sg, pl = rng.choice(ANIMATE + INANIMATE)
        vs, vp = rng.choice(TRANSITIVE)
        obj_sg, _ = rng.choice(INANIMATE)
        if rng.random() < 0.5:
            det_good, det_bad = rng.choice((("this", "these"), ("that", "those")))
            yield MinimalPair(
                "determiner_noun",
                f"{det_good.capitalize()} {sg} {vs} the {obj_sg}.",
                f"{det_bad.capitalize()} {sg} {vs} the {obj_sg}.",
            )
        else:
            det_good, det_bad = rng.choice((("these", "this"), ("those", "that")))
            yield MinimalPair(
                "determiner_noun",
                f"{det_good.capitalize()} {pl} {vp} the {obj_sg}.",
                f"{det_bad.capitalize()} {pl} {vp} the {obj_sg}.",
            )


def _reflexive(rng: random.Random, n: int) -> Iterator[MinimalPair]:
    """Reflexive anaphora must agree with its antecedent in gender and number."""
    for _ in range(n):
        verb = rng.choice(("hurt", "washed", "saw", "helped", "dried"))
        if rng.random() < 0.5:
            name = rng.choice(MALE)
            yield MinimalPair("reflexive", f"{name} {verb} himself.", f"{name} {verb} herself.")
        else:
            name = rng.choice(FEMALE)
            yield MinimalPair("reflexive", f"{name} {verb} herself.", f"{name} {verb} himself.")


def _pronoun_gender(rng: random.Random, n: int) -> Iterator[MinimalPair]:
    """Cross-sentence pronoun resolution: the pronoun must match the named antecedent."""
    for _ in range(n):
        obj, _ = rng.choice(INANIMATE)
        verb = rng.choice(("found", "wanted", "held", "dropped"))
        if rng.random() < 0.5:
            name = rng.choice(MALE)
            good, bad = "He", "She"
        else:
            name = rng.choice(FEMALE)
            good, bad = "She", "He"
        stem = f"{name} {verb} {article(obj)} {obj}."
        yield MinimalPair(
            "pronoun_gender", f"{stem} {good} was very happy.", f"{stem} {bad} was very happy."
        )


def _entity_tracking(rng: random.Random, n: int) -> Iterator[MinimalPair]:
    """Which character is holding which object, after both have been introduced?

    The hardest phenomenon in the suite, and the one that separates a model maintaining
    state from a model doing local statistics.

    **Both candidate objects appear in the context**, each bound to a different character,
    so the model must resolve the binding rather than notice which word it has seen
    before. An earlier version of this template introduced only the correct object
    ("X picked up the key ... X looked down at the key/book"), which any model with a
    copying bias scores 100% on without tracking anything — and this model duly scored
    100%. Making both objects present dropped the score to where it belongs. The lesson
    generalises: a benchmark item is only measuring the thing you named if the *wrong*
    answer is equally available.
    """
    for _ in range(n):
        first, second = rng.sample(list(MALE + FEMALE), 2)
        (a_obj, _), (b_obj, _) = rng.sample(list(INANIMATE), 2)
        stem = (
            f"{first} picked up the {a_obj}. "
            f"{second} picked up the {b_obj}. "
            f"Then {first} looked down at the"
        )
        yield MinimalPair("entity_tracking", f"{stem} {a_obj}.", f"{stem} {b_obj}.")


def _negation(rng: random.Random, n: int) -> Iterator[MinimalPair]:
    """A negated premise must be followed by the consistent continuation.

    The two continuations are ``nothing`` and ``something`` — same length, same syntactic
    frame, differing in exactly one word. An earlier version contrasted ``nothing inside``
    with ``a key inside``, which differs in *length* as well as in meaning, so a model
    could have scored well by preferring shorter sentences. A test comparing word counts
    across every phenomenon caught it.
    """
    for _ in range(n):
        name = rng.choice(MALE + FEMALE)
        container = rng.choice(("box", "bag", "cup", "jar"))
        if rng.random() < 0.5:
            stem = f"{name} looked in the {container}. The {container} was empty. There was"
            yield MinimalPair("negation", f"{stem} nothing inside.", f"{stem} something inside.")
        else:
            stem = f"{name} looked in the {container}. The {container} was not empty. There was"
            yield MinimalPair("negation", f"{stem} something inside.", f"{stem} nothing inside.")


def _argument_structure(rng: random.Random, n: int) -> Iterator[MinimalPair]:
    """A transitive verb needs its object; an intransitive one cannot take one."""
    for _ in range(n):
        sg, _ = rng.choice(ANIMATE)
        obj, _ = rng.choice(INANIMATE)
        if rng.random() < 0.5:
            vs, _ = rng.choice(TRANSITIVE)
            yield MinimalPair("argument_structure", f"The {sg} {vs} the {obj}.", f"The {sg} {vs}.")
        else:
            vs, _ = rng.choice(INTRANSITIVE)
            yield MinimalPair("argument_structure", f"The {sg} {vs}.", f"The {sg} {vs} the {obj}.")


def _tense_consistency(rng: random.Random, n: int) -> Iterator[MinimalPair]:
    """A past-tense narrative must not switch to present in the next clause."""
    past = {
        "runs": "ran",
        "jumps": "jumped",
        "sleeps": "slept",
        "plays": "played",
        "sings": "sang",
        "walks": "walked",
        "smiles": "smiled",
        "laughs": "laughed",
    }
    for _ in range(n):
        name = rng.choice(MALE + FEMALE)
        vs, _ = rng.choice(INTRANSITIVE)
        obj, _ = rng.choice(INANIMATE)
        stem = f"Yesterday {name} found {article(obj)} {obj} and"
        yield MinimalPair("tense_consistency", f"{stem} {past[vs]}.", f"{stem} {vs}.")


#: ``phenomenon -> generator``. Ordered easiest-first so a results table reads as a ladder.
PHENOMENA = {
    "agreement_simple": _agreement_simple,
    "determiner_noun": _determiner_noun,
    "reflexive": _reflexive,
    "argument_structure": _argument_structure,
    "tense_consistency": _tense_consistency,
    "pronoun_gender": _pronoun_gender,
    "negation": _negation,
    "agreement_attractor": _agreement_attractor,
    "entity_tracking": _entity_tracking,
}


def generate_pairs(
    *, n_per_phenomenon: int = 100, seed: int = 1337, phenomena: Sequence[str] | None = None
) -> list[MinimalPair]:
    """Generate a deduplicated minimal-pair suite.

    Args:
        n_per_phenomenon: Target number of distinct items per phenomenon.
        seed: Seed for the lexical sampling, so a suite is reproducible.
        phenomena: Optional subset of :data:`PHENOMENA` keys.

    Returns:
        The generated items. Duplicates are dropped, so a phenomenon whose template space
        is smaller than ``n_per_phenomenon`` simply yields fewer items rather than
        repeating them and inflating its apparent sample size.
    """
    rng = random.Random(seed)
    names = list(phenomena) if phenomena is not None else list(PHENOMENA)
    out: list[MinimalPair] = []
    for name in names:
        if name not in PHENOMENA:
            raise KeyError(f"unknown phenomenon {name!r}; known: {sorted(PHENOMENA)}")
        seen: set[tuple[str, str]] = set()
        # Oversample, then dedupe: the template space is large but sampling collides.
        for pair in PHENOMENA[name](rng, n_per_phenomenon * 6):
            key = (pair.good, pair.bad)
            if key in seen:
                continue
            seen.add(key)
            out.append(pair)
            if len(seen) >= n_per_phenomenon:
                break
    return out


# ---------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------


def wilson_interval(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because accuracies here run close to 0.5 and
    to 1.0, where the normal interval misbehaves badly — it can extend past 1.0, and it is
    far too narrow near the boundary. At n=100 and p=0.99 the normal interval is
    ±0.019 and the Wilson interval is correctly asymmetric.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass(frozen=True)
class PhenomenonScore:
    """Accuracy on one phenomenon, with a Wilson interval."""

    phenomenon: str
    correct: int
    n: int

    @property
    def accuracy(self) -> float:
        """Fraction of items where the grammatical sentence scored higher."""
        return self.correct / max(1, self.n)

    @property
    def interval(self) -> tuple[float, float]:
        """95% Wilson score interval."""
        return wilson_interval(self.correct, self.n)

    @property
    def above_chance(self) -> bool:
        """True when the interval's lower bound clears the 50% chance line."""
        return self.interval[0] > 0.5

    def summary(self) -> dict[str, float | int | str | bool]:
        """Flat numbers for the results table."""
        lo, hi = self.interval
        return {
            "phenomenon": self.phenomenon,
            "accuracy": round(self.accuracy, 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "correct": self.correct,
            "n": self.n,
            "above_chance": self.above_chance,
        }


@dataclass(frozen=True)
class MinimalPairResult:
    """The whole suite's outcome."""

    scores: tuple[PhenomenonScore, ...]
    length_controlled: bool = True
    per_phenomenon: dict[str, PhenomenonScore] = field(default_factory=dict)

    @property
    def overall(self) -> float:
        """Macro-average over phenomena, so a large phenomenon cannot dominate."""
        return sum(s.accuracy for s in self.scores) / max(1, len(self.scores))

    @property
    def n_items(self) -> int:
        """Total item count across phenomena."""
        return sum(s.n for s in self.scores)

    @property
    def n_above_chance(self) -> int:
        """How many phenomena are significantly above chance."""
        return sum(1 for s in self.scores if s.above_chance)

    def rows(self) -> list[dict[str, float | int | str | bool]]:
        """Per-phenomenon rows, typed, for tables and assertions."""
        return [s.summary() for s in self.scores]

    def summary(self) -> dict[str, object]:
        """Flat numbers plus the per-phenomenon breakdown."""
        return {
            "overall_macro_accuracy": round(self.overall, 4),
            "n_items": self.n_items,
            "n_phenomena": len(self.scores),
            "n_above_chance": self.n_above_chance,
            "chance": 0.5,
            "phenomena": self.rows(),
        }


@torch.no_grad()
def _sentence_logprob(
    model: nn.Module, tokenizer: BPETokenizer, text: str, device: torch.device
) -> tuple[float, int]:
    """Total log-probability of ``text`` and the number of scored tokens."""
    ids = tokenizer.encode(text, add_bos=True)
    inp = torch.tensor([ids[:-1]], device=device)
    tgt = torch.tensor([ids[1:]], device=device)
    logits = model(inp).logits
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    gathered = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return float(gathered.sum()), int(tgt.numel())


@torch.no_grad()
def run_minimal_pairs(
    model: nn.Module,
    tokenizer: BPETokenizer,
    *,
    pairs: Sequence[MinimalPair] | None = None,
    n_per_phenomenon: int = 100,
    seed: int = 1337,
    length_normalize: bool = True,
    device: torch.device | None = None,
) -> MinimalPairResult:
    """Score a model on the minimal-pair suite.

    Args:
        model: The model to score.
        tokenizer: Its tokenizer.
        pairs: Optional explicit item list; generated from templates when omitted.
        n_per_phenomenon: Items per phenomenon when generating.
        seed: Generation seed.
        length_normalize: Divide each sentence's log-probability by its token count.
            This matters for :func:`_argument_structure`, where the two sentences differ
            in length and an un-normalized comparison would reward the shorter one for
            being shorter rather than for being grammatical. Every other phenomenon has
            equal-length alternatives, so normalization is a no-op there.
        device: Optional device.

    Returns:
        A :class:`MinimalPairResult` with per-phenomenon accuracies and Wilson intervals.
    """
    items = (
        list(pairs)
        if pairs is not None
        else generate_pairs(n_per_phenomenon=n_per_phenomenon, seed=seed)
    )
    # `next(model.parameters())` raises StopIteration on a parameterless module, which is
    # exactly what the uniform reference model used to establish this benchmark's chance
    # floor is. Falling back to CPU keeps the floor measurable.
    if device is not None:
        dev = device
    else:
        first = next(model.parameters(), None)
        dev = first.device if first is not None else torch.device("cpu")
    was_training = model.training
    model.eval()

    tally: dict[str, list[int]] = {}
    for item in items:
        gl, gn = _sentence_logprob(model, tokenizer, item.good, dev)
        bl, bn = _sentence_logprob(model, tokenizer, item.bad, dev)
        if length_normalize:
            gl, bl = gl / max(1, gn), bl / max(1, bn)
        bucket = tally.setdefault(item.phenomenon, [0, 0])
        bucket[0] += int(gl > bl)
        bucket[1] += 1

    if was_training:
        model.train()

    order = [p for p in PHENOMENA if p in tally]
    scores = tuple(PhenomenonScore(p, tally[p][0], tally[p][1]) for p in order)
    return MinimalPairResult(
        scores=scores,
        length_controlled=length_normalize,
        per_phenomenon={s.phenomenon: s for s in scores},
    )
