"""Train stage: fit baseline + Markov + XGBoost + ensemble models."""

from phinish.train.state import (
    MIN_PLAYS_FOR_CANDIDATE,
    TEST_START_YEAR,
    TRAIN_END_YEAR,
    VAL_YEAR,
    StreamingState,
    feature_names,
    featurize,
)
from phinish.train.types import EnsembleWeights

__all__ = (
    "MIN_PLAYS_FOR_CANDIDATE",
    "TEST_START_YEAR",
    "TRAIN_END_YEAR",
    "VAL_YEAR",
    "EnsembleWeights",
    "StreamingState",
    "feature_names",
    "featurize",
)
