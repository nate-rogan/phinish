"""Typed shapes for the GitHub Actions process layer."""

import msgspec


class Usage(msgspec.Struct):
    """Daily prediction-counter (``state/usage.json``); empty on a fresh day."""

    date: str = ""
    total: int = 0
    by_user: dict[str, int] = msgspec.field(default_factory=dict)
