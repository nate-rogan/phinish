"""Typed shapes for the feature store (``state/features/*.json``)."""

import msgspec

TransitionDist = dict[str, float]


class SongStats(msgspec.Struct):
    """Per-song statistics (``state/features/song_stats.json``)."""

    total_plays: int
    lifetime_frequency: float
    recent_frequency_50: float
    recent_frequency_20: float
    avg_set_position: float
    typical_set: int
    set_distribution: dict[str, float]
    opener_frequency: float
    closer_frequency: float
    is_cover: bool
    debut_year: int


class SongGap(msgspec.Struct):
    """Per-song gap snapshot (``state/features/song_gaps.json``)."""

    gap: int
    last_played: str


class VenueHistory(msgspec.Struct):
    """Per-venue history (``state/features/venue_history.json``)."""

    venue_id: str
    name: str
    total_shows: int
    song_freq: dict[str, float]
    common_openers: list[str]
    common_closers: list[str]


class TransitionMatrix(msgspec.Struct):
    """Order-1 / order-2 transitions plus opener / closer marginals.

    Persisted to ``state/features/transition_matrix.json`` and to
    ``models/markov_order2.json`` (with ``trained_on_shows``).
    """

    order_1: dict[str, TransitionDist]
    order_2: dict[str, TransitionDist]
    set_openers: dict[str, TransitionDist]
    set_closers: dict[str, TransitionDist]
    vocab_size: int
    laplace_k: float
    trained_on_shows: int = 0
