"""Typed artifact loaders, shared scoring helpers, and computation utilities.

Centralizes the repeated ``load_json`` + ``msgspec.convert`` pattern that
appears across predict, train, and evaluate modules. Also owns cross-cutting
scoring functions (``markov_score_per_song``, ``gap_score_per_song``) that
multiple stages consume.
"""

import math
import pickle

import msgspec
import numpy as np

from phinish.features.types import SongGap, SongStats, TransitionMatrix, VenueHistory
from phinish.scrape.types import Show, SongCatalogEntry, VenueRecord
from phinish.train.types import EnsembleWeights
from phinish.utils import (
    FEATURES_DIR,
    MODELS_DIR,
    SETLISTS_PATH,
    SONGS_PATH,
    TOP_K,
    VENUES_PATH,
    load_json,
)

# ---------------------------------------------------------------------------
# JSON artifact loaders
# ---------------------------------------------------------------------------


def load_shows() -> list[Show]:
    """Load and convert ``data/processed/setlists.json`` to typed ``Show`` structs.

    Raises
    ------
    FileNotFoundError
        If the setlists file does not exist.
    """
    raw = load_json(SETLISTS_PATH)
    return [msgspec.convert(s, Show, strict=False) for s in raw]


def load_songs_catalog() -> list[SongCatalogEntry]:
    """Load the song catalog; returns ``[]`` if the file is missing."""
    if not SONGS_PATH.exists():
        return []
    raw = load_json(SONGS_PATH)
    return [msgspec.convert(s, SongCatalogEntry, strict=False) for s in raw]


def load_venues() -> dict[str, VenueRecord]:
    """Load the venue catalog; returns ``{}`` if the file is missing."""
    if not VENUES_PATH.exists():
        return {}
    raw = load_json(VENUES_PATH)
    return {k: msgspec.convert(v, VenueRecord, strict=False) for k, v in raw.items()}


def load_song_stats() -> dict[str, SongStats]:
    """Load per-song aggregate statistics from the feature store."""
    raw = load_json(FEATURES_DIR / "song_stats.json")
    return {k: msgspec.convert(v, SongStats, strict=False) for k, v in raw.items()}


def load_song_gaps() -> dict[str, SongGap]:
    """Load per-song rotation gaps from the feature store."""
    raw = load_json(FEATURES_DIR / "song_gaps.json")
    return {k: msgspec.convert(v, SongGap, strict=False) for k, v in raw.items()}


def load_transition_matrix() -> TransitionMatrix:
    """Load the Markov transition matrix from the feature store."""
    raw = load_json(FEATURES_DIR / "transition_matrix.json")
    return msgspec.convert(raw, TransitionMatrix, strict=False)


def load_venue_history() -> dict[str, VenueHistory]:
    """Load per-venue song frequency history from the feature store."""
    raw = load_json(FEATURES_DIR / "venue_history.json")
    return {k: msgspec.convert(v, VenueHistory, strict=False) for k, v in raw.items()}


def load_ensemble_weights() -> EnsembleWeights:
    """Load optimized ensemble weights from ``models/ensemble_weights.json``."""
    raw = load_json(MODELS_DIR / "ensemble_weights.json")
    return msgspec.convert(raw, EnsembleWeights, strict=False)


def load_cover_set() -> set[str]:
    """Load the song catalog and return the set of non-original (cover) song names."""
    return {s.name for s in load_songs_catalog() if not s.is_original}


# ---------------------------------------------------------------------------
# Pickle model loaders
# ---------------------------------------------------------------------------


def load_xgb_model():
    """Load the trained XGBoost classifier from ``models/``."""
    return pickle.loads((MODELS_DIR / "xgboost_song_selector.pkl").read_bytes())


def load_calibrator():
    """Load the Platt calibrator if it exists, otherwise return ``None``."""
    path = MODELS_DIR / "calibrator.pkl"
    return pickle.loads(path.read_bytes()) if path.exists() else None


# ---------------------------------------------------------------------------
# Shared scoring helpers (moved from train/ensemble.py)
# ---------------------------------------------------------------------------


def markov_score_per_song(matrix: TransitionMatrix) -> dict[str, float]:
    """Compute a marginal "popularity as a set opener" score per song.

    Parameters
    ----------
    matrix
        The trained transition matrix.

    Returns
    -------
    dict[str, float]
        Song -> mean opener probability across set keys.
    """
    openers = matrix.set_openers
    if not openers:
        return {}
    scores: dict[str, float] = {}
    for dist in openers.values():
        for song, p in dist.items():
            scores[song] = scores.get(song, 0.0) + p
    return {song: s / len(openers) for song, s in scores.items()}


def gap_score_per_song(
    stats: dict[str, SongStats],
    gaps: dict[str, SongGap],
) -> dict[str, float]:
    """Compute the gap-weighted ensemble component per song.

    Parameters
    ----------
    stats
        Per-song statistics from the feature store.
    gaps
        Per-song current-gap snapshot.

    Returns
    -------
    dict[str, float]
        Song -> ``recent_frequency_50 * log(gap + 2)``.
    """
    return {
        song: s.recent_frequency_50 * math.log((gaps[song].gap if song in gaps else 1) + 2)
        for song, s in stats.items()
    }


def precision_at_k(predicted: list[str], actual: set[str], k: int = TOP_K) -> float:
    """Fraction of the top-k predicted songs that were actually played."""
    return sum(1 for s in predicted[:k] if s in actual) / k if predicted else 0.0


# ---------------------------------------------------------------------------
# Shared computation helpers
# ---------------------------------------------------------------------------


def calibrated_predict_proba(model, calibrator, X: np.ndarray) -> np.ndarray:
    """Run XGBoost prediction with optional Platt calibration.

    Parameters
    ----------
    model
        Trained XGBoost classifier.
    calibrator
        Platt scaling calibrator, or ``None`` to skip calibration.
    X
        Feature matrix.

    Returns
    -------
    np.ndarray
        Calibrated (or raw) probability estimates for the positive class.
    """
    probs = model.predict_proba(X)[:, 1]
    if calibrator is not None:
        probs = calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
    return probs


def venue_freq_for_show(
    venue_history: dict[str, VenueHistory],
    venue_id: str,
) -> dict[str, float]:
    """Look up per-song frequency for a venue, defaulting to empty."""
    vh = venue_history.get(venue_id)
    return vh.song_freq if vh else {}
