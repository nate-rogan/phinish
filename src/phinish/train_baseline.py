"""Compute baseline reference scores. Writes models/baselines.json.

Baselines:
- frequency: lifetime_frequency (top songs by lifetime play rate)
- gap_weighted: recent_frequency_50 * log(gap + 2)
"""

import math

from phinish.utils import (
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
    """Compute the gap-weighted baseline score for each song.

    Score is recent play frequency (50-show window) boosted by the log
    of current gap, so songs that are *both* common in recent rotation
    *and* overdue rank highest. Adding 2 inside the log keeps the
    boost finite at gap 0 and monotone increasing.

    Parameters
    ----------
    stats
        Per-song statistics from the feature store.
    gaps
        Per-song current-gap snapshot from the feature store. Songs
        missing from ``gaps`` get a default gap of 1.

    Returns
    -------
    dict[str, float]
        Mapping of song -> score; higher means more likely to appear.
    """
    out: dict[str, float] = {}
    for song, s in stats.items():
        gap = gaps.get(song, {}).get("gap", 1)
        out[song] = s.get("recent_frequency_50", 0.0) * math.log(gap + 2)
    return out


def frequency_score(stats: dict[str, SongStats]) -> dict[str, float]:
    """Compute the lifetime-frequency baseline score for each song.

    Parameters
    ----------
    stats
        Per-song statistics from the feature store.

    Returns
    -------
    dict[str, float]
        Mapping of song -> ``lifetime_frequency``; higher means the
        song is played in a larger fraction of all shows historically.
    """
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
