"""A deterministic, offline, TinyStories-style corpus for the ``nano`` tier.

Why this exists
---------------
Spec constraint A3.1 requires a laptop tier that runs end-to-end with no GPU, and
Part D3 requires ``make smoke`` and CI to be green without network access. Streaming
FineWeb-Edu satisfies neither: it needs the network, it is not byte-reproducible, and
a 6M-parameter model trained for 400 CPU steps on open-web text produces noise.

So the ``nano`` tier trains on a synthetic corpus generated from a small story grammar,
in the spirit of TinyStories (Eldan & Li, arXiv:2305.07759): a deliberately narrow
vocabulary and a strong, repeated narrative structure, which is exactly the regime in
which a very small model can learn something genuinely coherent.

Properties this generator guarantees:

* **Deterministic.** Output is a pure function of ``(seed, n_stories)``. Two runs of
  the smoke test see byte-identical data.
* **Offline.** No downloads, no cached artifacts, nothing committed to git.
* **Structured.** Each story keeps one protagonist, uses consistent pronouns for that
  protagonist, and follows a setup -> problem -> action -> resolution arc. There is
  real signal to learn beyond unigram statistics: agreement, coreference and structure.

``micro`` and ``small`` use real streaming data instead; see
:mod:`nanoscale.train.data`. Every result reported from the ``nano`` tier is labelled
as coming from this synthetic corpus.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = ["TOY_VOCABULARY_SIZE_HINT", "generate_corpus", "iter_stories", "write_corpus"]

# --------------------------------------------------------------------------------------
# The grammar. Small, concrete, and heavy on repetition -- that is the point.
# --------------------------------------------------------------------------------------

_CHARACTERS: Final[tuple[tuple[str, str, str], ...]] = (
    # (name, subject pronoun, possessive pronoun)
    ("Lily", "she", "her"),
    ("Tom", "he", "his"),
    ("Mia", "she", "her"),
    ("Ben", "he", "his"),
    ("Anna", "she", "her"),
    ("Sam", "he", "his"),
    ("Nora", "she", "her"),
    ("Max", "he", "his"),
    ("Ella", "she", "her"),
    ("Leo", "he", "his"),
    ("Ruby", "she", "her"),
    ("Finn", "he", "his"),
    ("Clara", "she", "her"),
    ("Oscar", "he", "his"),
    ("Poppy", "she", "her"),
    ("Hugo", "he", "his"),
    ("Iris", "she", "her"),
    ("Jonah", "he", "his"),
    ("Willa", "she", "her"),
    ("Silas", "he", "his"),
    ("Maya", "she", "her"),
    ("Rowan", "he", "his"),
    ("Delia", "she", "her"),
    ("Caspar", "he", "his"),
)

_COMPANIONS: Final[tuple[str, ...]] = (
    "a small dog",
    "a grey cat",
    "a brown bird",
    "a tiny frog",
    "an old turtle",
    "a friendly bee",
    "a quiet mouse",
    "a happy duck",
    "a clever fox",
    "a sleepy owl",
    "a spotted rabbit",
    "a curious squirrel",
    "a gentle donkey",
    "a striped kitten",
    "a chatty parrot",
    "a patient hedgehog",
)

_PLACES: Final[tuple[str, ...]] = (
    "the garden",
    "the park",
    "the old house",
    "the river",
    "the small hill",
    "the quiet forest",
    "the sunny beach",
    "the busy market",
    "the green field",
    "the little bridge",
    "the stone wall",
    "the empty barn",
    "the narrow path",
    "the wooden pier",
    "the flower meadow",
    "the shady orchard",
    "the village square",
    "the dusty attic",
    "the winter pond",
    "the lighthouse steps",
)

_OBJECTS: Final[tuple[str, ...]] = (
    "a red ball",
    "a blue box",
    "a shiny key",
    "a paper boat",
    "a warm hat",
    "a soft blanket",
    "a big book",
    "a bright lamp",
    "a round stone",
    "a long rope",
    "a silver spoon",
    "a wooden chair",
    "a glass jar",
    "a torn map",
    "a copper bell",
    "a striped scarf",
    "a heavy basket",
    "a folded letter",
    "a broken kite",
    "a painted drum",
    "a golden ring",
    "a leather bag",
    "a crooked stick",
    "a tiny mirror",
)

_ADJECTIVES: Final[tuple[str, ...]] = (
    "happy",
    "sad",
    "tired",
    "excited",
    "curious",
    "brave",
    "quiet",
    "hungry",
    "proud",
    "gentle",
    "cheerful",
    "thoughtful",
    "nervous",
    "grateful",
    "patient",
    "restless",
)

_WEATHER: Final[tuple[str, ...]] = (
    "It was a sunny day.",
    "It was raining softly.",
    "The wind was cold.",
    "The sky was full of clouds.",
    "The morning was very quiet.",
    "The evening was warm.",
    "A thin mist covered everything.",
    "Snow was falling in slow circles.",
    "The afternoon smelled like fresh bread.",
    "Thunder rumbled far away behind the hills.",
    "Bright sunlight filled the whole street.",
    "A soft rain had washed the roofs clean.",
)

_PROBLEMS: Final[tuple[str, ...]] = (
    "But {obj} was stuck under a tree.",
    "But {obj} was too heavy to lift.",
    "But {obj} fell into the water.",
    "But {obj} was lost in the tall grass.",
    "But {obj} would not open.",
    "But {obj} was very far away.",
    "But {obj} had rolled beneath a fence.",
    "But {obj} was buried under a pile of leaves.",
    "But {obj} sat high on a narrow shelf.",
    "But {obj} was tangled in a knot of string.",
    "But {obj} had cracked along one side.",
    "But {obj} belonged to somebody else.",
)

_ACTIONS: Final[tuple[str, ...]] = (
    "{pronoun} pulled it slowly with both hands.",
    "{pronoun} asked {companion} for help.",
    "{pronoun} looked at it for a long time and made a plan.",
    "{pronoun} pushed it with all {possessive} strength.",
    "{pronoun} ran home to get a rope.",
    "{pronoun} counted to three and tried again.",
    "{pronoun} climbed onto a low branch to reach further.",
    "{pronoun} borrowed a ladder from the neighbour.",
    "{pronoun} dug carefully around the edges.",
    "{pronoun} tried a different way instead of forcing it.",
    "{pronoun} sat down and thought about the problem.",
    "{pronoun} tied {possessive} scarf around it for grip.",
    "{pronoun} waited until the wind stopped blowing.",
    "{pronoun} drew a picture of the plan in the dirt.",
)

_HELPERS: Final[tuple[str, ...]] = (
    "{companion} came to help.",
    "{companion} watched from a safe place.",
    "{companion} made a small happy sound.",
    "A kind neighbour walked by and smiled.",
    "{companion} pushed from the other side.",
    "An old woman lent {name} a pair of gloves.",
    "{companion} showed {name} a shortcut through the grass.",
    "Two children stopped to watch and then joined in.",
)

_RESOLUTIONS: Final[tuple[str, ...]] = (
    "At last {obj} was free.",
    "Then {obj} moved just a little.",
    "Soon {obj} was safe again.",
    "Finally {obj} was back where it belonged.",
    "With one more try {obj} came loose.",
    "After a while {obj} slipped out easily.",
    "In the end {obj} was clean and whole again.",
)

_ENDINGS: Final[tuple[str, ...]] = (
    "{name} was very {adj}.",
    "{name} smiled and went home.",
    "{name} said thank you to {companion}.",
    "{name} told everyone about the day.",
    "{name} sat down and had a rest.",
    "{name} carried it back to {place} with care.",
    "{name} laughed until {possessive} sides hurt.",
    "{name} promised to come back tomorrow.",
)

_MORALS: Final[tuple[str, ...]] = (
    "{name} learned that help makes hard things easy.",
    "{name} learned that a good plan is better than hurry.",
    "{name} learned that friends are always worth asking.",
    "{name} learned that patience wins in the end.",
    "{name} learned that small steps still reach the top.",
    "{name} learned that asking twice is not a shame.",
    "{name} learned that kindness travels further than noise.",
    "",
)

#: Rough number of distinct whitespace-delimited word types the grammar can emit.
#: Used by tests as a sanity bound on the corpus, not as a hard contract.
TOY_VOCABULARY_SIZE_HINT: Final[int] = 500


@dataclass(frozen=True, slots=True)
class _Cast:
    """The fixed entities of one story, so pronouns and references stay consistent."""

    name: str
    pronoun: str
    possessive: str
    companion: str
    place: str
    obj: str
    adj: str


def _draw_cast(rng: random.Random) -> _Cast:
    name, pronoun, possessive = rng.choice(_CHARACTERS)
    return _Cast(
        name=name,
        pronoun=pronoun,
        possessive=possessive,
        companion=rng.choice(_COMPANIONS),
        place=rng.choice(_PLACES),
        obj=rng.choice(_OBJECTS),
        adj=rng.choice(_ADJECTIVES),
    )


def _fill(template: str, cast: _Cast, *, capitalize_pronoun: bool = True) -> str:
    """Substitute the cast into a template, fixing sentence-initial capitalisation."""
    text = template.format(
        name=cast.name,
        pronoun=cast.pronoun,
        possessive=cast.possessive,
        companion=cast.companion,
        place=cast.place,
        obj=cast.obj,
        adj=cast.adj,
    )
    if capitalize_pronoun and text:
        text = text[0].upper() + text[1:]
    return text


def _story(rng: random.Random) -> str:
    """Generate one story: setup -> problem -> action -> help -> resolution -> ending."""
    cast = _draw_cast(rng)
    sentences: list[str] = [
        rng.choice(_WEATHER),
        f"{cast.name} went to {cast.place} with {cast.companion}.",
        f"{cast.name} wanted to find {cast.obj}.",
        _fill(rng.choice(_PROBLEMS), cast),
        _fill(rng.choice(_ACTIONS), cast),
    ]
    if rng.random() < 0.7:
        sentences.append(_fill(rng.choice(_HELPERS), cast))
    if rng.random() < 0.5:
        sentences.append(_fill(rng.choice(_ACTIONS), cast))
    sentences.append(_fill(rng.choice(_RESOLUTIONS), cast))
    sentences.append(_fill(rng.choice(_ENDINGS), cast))
    moral = _fill(rng.choice(_MORALS), cast)
    if moral:
        sentences.append(moral)
    return " ".join(sentences)


def iter_stories(*, seed: int = 1337, n_stories: int = 20_000) -> Iterator[str]:
    """Yield ``n_stories`` stories deterministically from ``seed``."""
    rng = random.Random(seed)
    for _ in range(n_stories):
        yield _story(rng)


def generate_corpus(*, seed: int = 1337, n_stories: int = 20_000) -> str:
    """Return the whole corpus as one string, stories separated by blank lines."""
    return "\n\n".join(iter_stories(seed=seed, n_stories=n_stories)) + "\n"


def write_corpus(
    path: str | Path,
    *,
    seed: int = 1337,
    n_stories: int = 20_000,
) -> Path:
    """Write the toy corpus to ``path`` (creating parents) and return the path."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(generate_corpus(seed=seed, n_stories=n_stories), encoding="utf-8")
    return dest
