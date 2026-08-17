"""Serving: the generation loop, streaming, and a minimal chat interface."""

from __future__ import annotations

from nanoscale.serve.generate import (
    GenerationOutput,
    TextStreamer,
    generate_text,
    stream_text,
)

__all__ = ["GenerationOutput", "TextStreamer", "generate_text", "stream_text"]
