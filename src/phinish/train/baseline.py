"""Compute baseline reference scores. Writes models/baselines.json.

Baselines:
- frequency: lifetime_frequency (top songs by lifetime play rate)
- gap_weighted: recent_frequency_50 * log(gap + 2)
"""

import structlog

from phinish.artifacts import gap_score_per_song, load_song_gaps, load_song_stats
from phinish.features.types import SongStats
from phinish.utils import MODELS_DIR, save_json

log = structlog.get_logger()


def frequency_score(stats: dict[str, SongStats]) -> dict[str, float]:
    """Compute the lifetime-frequency baseline score for each song."""
    return {song: s.lifetime_frequency for song, s in stats.items()}


def train_baseline(stats: dict[str, SongStats], gaps: dict) -> dict:
    """Compute baseline reference scores.

    Parameters
    ----------
    stats
        Per-song statistics from ``song_stats.json``.
    gaps
        Per-song gap data from ``song_gaps.json``.

    Returns
    -------
    dict
        Baseline scores with keys ``frequency``, ``gap_weighted``, ``config``.
    """
    return {
        "frequency": frequency_score(stats),
        "gap_weighted": gap_score_per_song(stats, gaps),
        "config": {"recent_window": 50, "log_offset": 2},
    }


def main() -> None:
    """Console entry point: load artifacts, train baselines, write to disk."""
    result = train_baseline(load_song_stats(), load_song_gaps())
    save_json(MODELS_DIR / "baselines.json", result)
    log.info("wrote_baselines", path=str(MODELS_DIR / "baselines.json"))


if __name__ == "__main__":
    main()
