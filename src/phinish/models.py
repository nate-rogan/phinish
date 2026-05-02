"""Typed shapes for everything persisted, exchanged, or returned by Phinish.

Each ``TypedDict`` here mirrors the schema of a JSON file or in-memory
record so that constructors and consumers share a single source of truth
for field names and types. Group by stage of the pipeline:

- **Setlist domain** (``data/processed/``): raw scraped data — ``Show``,
  ``SongEntry``, ``SongCatalogEntry``, ``VenueRecord``.
- **Feature store** (``state/features/``): aggregate stats —
  ``SongStats``, ``SongGap``, ``VenueHistory``, ``TransitionMatrix``.
- **Model artifacts** (``models/``): trained-model outputs —
  ``EnsembleWeights``, ``Manifest``.
- **Prediction output**: ``Prediction``, ``PredictionItem``,
  ``PredictionWeights``.
- **Mutable run-state**: ``Usage`` for the rate limiter.

TypedDict construction keyword args are type-checked, so prefer
``Show(show_id=..., date=..., ...)`` over raw ``{"show_id": ...}`` dict
literals at every construction site — that turns the magic string keys
into named parameters that ruff / pyright / mypy will validate.
"""

from typing import NotRequired, TypedDict

# --- Setlist domain (data/processed/) ---


class SongEntry(TypedDict):
    """One song slot inside a set: position, transition mark, jam/reprise flags."""

    song: str
    song_id: str
    position: int
    transition: str
    is_jam: bool
    is_reprise: bool


class Show(TypedDict):
    """One Phish show with date, venue, tour metadata, and all sets played.

    Persisted as one entry in ``data/processed/setlists.json``. Also
    produced in-memory by ``predict.synthetic_show`` for inference
    (without ``total_songs``).
    """

    show_id: str
    date: str
    year: int
    month: int
    day: int
    day_of_week: str
    venue_id: str
    venue_name: str
    city: str
    state: str
    country: str
    tour: str
    tour_id: str
    sets: dict[str, list[SongEntry]]
    is_nye: bool
    is_halloween: bool
    is_festival: bool
    total_songs: NotRequired[int]


class SongCatalogEntry(TypedDict):
    """One row from the Phish.net song catalog (``data/processed/songs.json``)."""

    song_id: str
    name: str
    slug: str
    artist: str
    is_original: bool
    debut: str
    last_played: str
    times_played: int


class VenueRecord(TypedDict):
    """One venue from ``data/processed/venues.json``, keyed by ``venue_id``."""

    venue_id: str
    name: str
    city: str
    state: str
    country: str
    phishnet_id: NotRequired[str]


# --- Feature store (state/features/) ---


class SongStats(TypedDict):
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


class SongGap(TypedDict):
    """Per-song gap snapshot (``state/features/song_gaps.json``)."""

    gap: int
    last_played: str


class VenueHistory(TypedDict):
    """Per-venue history (``state/features/venue_history.json``)."""

    venue_id: str
    name: str
    total_shows: int
    song_freq: dict[str, float]
    common_openers: list[str]
    common_closers: list[str]


# Mapping of song -> probability used inside transition matrices.
TransitionDist = dict[str, float]


class TransitionMatrix(TypedDict):
    """Order-1 / order-2 transitions plus opener / closer marginals.

    Persisted to ``state/features/transition_matrix.json`` (no
    ``trained_on_shows``) and to ``models/markov_order2.json`` (with
    ``trained_on_shows``).
    """

    order_1: dict[str, TransitionDist]
    order_2: dict[str, TransitionDist]
    set_openers: dict[str, TransitionDist]
    set_closers: dict[str, TransitionDist]
    vocab_size: int
    laplace_k: float
    trained_on_shows: NotRequired[int]


# --- Model artifacts (models/) ---


class EnsembleWeights(TypedDict):
    """Optimal ensemble weights (``models/ensemble_weights.json``)."""

    w_xgboost: float
    w_markov: float
    w_gap: float
    w_venue: float
    val_precision_at_25: float | None
    val_year: int
    n_val_shows: int


# --- Prediction output ---


class PredictionItem(TypedDict):
    """One song slot in a prediction setlist with calibrated confidence."""

    song: str
    confidence: float
    gap: int


class PredictionWeights(TypedDict):
    """Ensemble weights echoed back in the prediction payload."""

    xgboost: float
    markov: float
    gap: float
    venue: float


class Prediction(TypedDict):
    """Output of ``predict.predict()`` — structured 3-set forecast."""

    date: str
    venue: str
    venue_id: str
    city: str
    setlist: dict[str, list[PredictionItem]]
    avg_confidence: float
    model_version: int | str
    weights: PredictionWeights


# --- Mutable run-state ---


class Usage(TypedDict, total=False):
    """Daily prediction-counter (``state/usage.json``); empty on a fresh day."""

    date: str
    total: int
    by_user: dict[str, int]
