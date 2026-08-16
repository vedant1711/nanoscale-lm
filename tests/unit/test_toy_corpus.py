"""Tests for the offline toy corpus.

The corpus is a dependency of the smoke test and of CI, so its two load-bearing
properties -- determinism and structure -- are tested rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

from nanoscale.data.toy import (
    TOY_VOCABULARY_SIZE_HINT,
    generate_corpus,
    iter_stories,
    write_corpus,
)


def test_generation_is_deterministic_in_the_seed() -> None:
    a = generate_corpus(seed=42, n_stories=50)
    b = generate_corpus(seed=42, n_stories=50)
    c = generate_corpus(seed=43, n_stories=50)
    assert a == b, "the same seed must produce byte-identical corpora"
    assert a != c


def test_prefix_stability_across_sizes() -> None:
    """A larger corpus must extend the smaller one, not reshuffle it."""
    small = list(iter_stories(seed=5, n_stories=10))
    large = list(iter_stories(seed=5, n_stories=40))
    assert large[:10] == small


def test_stories_have_narrative_structure() -> None:
    stories = list(iter_stories(seed=1, n_stories=200))
    assert len(stories) == 200
    for story in stories:
        sentences = [s for s in story.split(". ") if s]
        assert len(sentences) >= 6, "setup -> problem -> action -> resolution -> ending"
        assert story.endswith("."), "stories end with a full stop"
        assert "{" not in story and "}" not in story, "no unsubstituted template slots"


def test_protagonist_is_referred_to_consistently() -> None:
    """Each story keeps one protagonist, so there is coreference signal to learn."""
    names = {
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
        "Willa",
        "Silas",
        "Maya",
        "Rowan",
        "Delia",
        "Caspar",
    }
    for story in iter_stories(seed=2, n_stories=300):
        present = {n for n in names if n in story}
        assert len(present) == 1, f"expected exactly one protagonist, found {present}"


def test_pronouns_agree_with_the_protagonist() -> None:
    female = {
        "Lily",
        "Mia",
        "Anna",
        "Nora",
        "Ella",
        "Ruby",
        "Clara",
        "Poppy",
        "Iris",
        "Willa",
        "Maya",
        "Delia",
    }
    male = {
        "Tom",
        "Ben",
        "Sam",
        "Max",
        "Leo",
        "Finn",
        "Oscar",
        "Hugo",
        "Jonah",
        "Silas",
        "Rowan",
        "Caspar",
    }
    for story in iter_stories(seed=4, n_stories=300):
        name = next(n for n in female | male if n in story)
        words = story.replace(".", "").replace(",", "").split()
        if name in female:
            assert "he" not in words and "his" not in words, story
        else:
            assert "she" not in words and "her" not in words, story


def test_vocabulary_is_small_and_bounded() -> None:
    corpus = generate_corpus(seed=6, n_stories=3000)
    types = {w.strip(".,!?").lower() for w in corpus.split()}
    assert 100 < len(types) <= TOY_VOCABULARY_SIZE_HINT, len(types)


def test_corpus_size_scales_with_story_count() -> None:
    small = generate_corpus(seed=8, n_stories=100)
    large = generate_corpus(seed=8, n_stories=1000)
    assert 8 < len(large) / len(small) < 12


def test_write_corpus_creates_parents(tmp_path: Path) -> None:
    dest = write_corpus(tmp_path / "nested" / "dir" / "corpus.txt", seed=9, n_stories=20)
    assert dest.exists()
    assert dest.read_text(encoding="utf-8") == generate_corpus(seed=9, n_stories=20)
