"""Typed shapes for trained-model artifacts (``models/*.json``)."""

import msgspec


class EnsembleWeights(msgspec.Struct):
    """Optimal ensemble weights (``models/ensemble_weights.json``)."""

    w_xgboost: float
    w_markov: float
    w_gap: float
    w_venue: float
    val_precision_at_25: float | None
    val_year: int
    n_val_shows: int
