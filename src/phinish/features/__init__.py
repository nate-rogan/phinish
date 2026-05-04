"""Features stage: aggregate raw setlists into per-song / per-venue stats."""

from phinish.features.build import build_features, main
from phinish.features.types import (
    SongGap,
    SongStats,
    TransitionDist,
    TransitionMatrix,
    VenueHistory,
)

__all__ = (
    "SongGap",
    "SongStats",
    "TransitionDist",
    "TransitionMatrix",
    "VenueHistory",
    "build_features",
    "main",
)
