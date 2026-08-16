"""Datasets: the offline toy corpus plus loaders for real streaming corpora."""

from __future__ import annotations

from nanoscale.data.toy import generate_corpus, iter_stories, write_corpus

__all__ = ["generate_corpus", "iter_stories", "write_corpus"]
