"""Typed shapes for LLM summarization output."""

import msgspec


class SummaryResult(msgspec.Struct):
    """Result of an LLM-generated prediction summary."""

    text: str
    voice: str
    is_fallback: bool
