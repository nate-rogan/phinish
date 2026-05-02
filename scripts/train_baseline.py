"""Compute baseline reference scores. Writes models/baselines.json.

Baselines:
- frequency: lifetime_frequency (top songs by lifetime play rate)
- gap_weighted: recent_frequency_50 * log(gap + 2)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.utils import (
    FEATURES_DIR,
    MODELS_DIR,
    SongGap,
    SongStats,
    load_json,
    save_json,
)


def gap_weighted_score(
    stats: dict[str, SongStats], gaps: dict[str, SongGap],
) -> dict[str, float]:
    """Score = recent play frequency boosted by log(gap), reflecting overdue songs."""
    out: dict[str, float] = {}
    for song, s in stats.items():
        gap = gaps.get(song, {}).get("gap", 1)
        out[song] = s.get("recent_frequency_50", 0.0) * math.log(gap + 2)
    return out


def frequency_score(stats: dict[str, SongStats]) -> dict[str, float]:
    """Baseline score: lifetime play frequency per song."""
    return {song: s.get("lifetime_frequency", 0.0) for song, s in stats.items()}


def main() -> None:
    """Compute baseline reference scores and write models/baselines.json."""
    stats = load_json(FEATURES_DIR / "song_stats.json")
    gaps = load_json(FEATURES_DIR / "song_gaps.json")
    out = {
        "frequency": frequency_score(stats),
        "gap_weighted": gap_weighted_score(stats, gaps),
        "config": {"recent_window": 50, "log_offset": 2},
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(MODELS_DIR / "baselines.json", out)
    print(f"wrote baselines -> {MODELS_DIR / 'baselines.json'}")


if __name__ == "__main__":
    main()
