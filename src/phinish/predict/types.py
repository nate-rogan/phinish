"""Typed shapes for prediction output."""

import msgspec


class PredictionItem(msgspec.Struct):
    """One song slot in a prediction setlist with calibrated confidence."""

    song: str
    confidence: float
    gap: int


class PredictionWeights(msgspec.Struct):
    """Ensemble weights echoed back in the prediction payload."""

    xgboost: float
    markov: float
    gap: float
    venue: float


class Prediction(msgspec.Struct):
    """Output of ``predict.pipeline.predict()`` — structured 3-set forecast."""

    date: str
    venue: str
    venue_id: str
    city: str
    setlist: dict[str, list[PredictionItem]]
    avg_confidence: float
    model_version: int | str
    weights: PredictionWeights
