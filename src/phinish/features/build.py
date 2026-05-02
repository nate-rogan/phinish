"""Feature engineering: gaps, song stats, transitions, venue history.

Outputs to state/features/:
  song_gaps.json
  song_stats.json
  transition_matrix.json
  venue_history.json
"""

from collections import Counter, defaultdict

import structlog

from phinish.features.types import SongGap, SongStats, TransitionMatrix, VenueHistory
from phinish.scrape.types import Show, SongCatalogEntry
from phinish.utils import (
    FEATURES_DIR,
    SET_KEYS,
    SET_TO_INT,
    SETLISTS_PATH,
    save_json,
    show_song_set,
)

log = structlog.get_logger()

LAPLACE_K = 0.01  # small smoothing constant; avoids zero probs without distorting counts
RECENT_WINDOWS = (50, 20)  # ~2 tours and ~1 tour of recent history


def build_song_gaps(shows: list[Show]) -> dict[str, SongGap]:
    """Compute current rotation gap (shows since last play) for each song.

    Parameters
    ----------
    shows
        Chronological list of shows.

    Returns
    -------
    dict[str, SongGap]
        ``{song -> SongGap}``.
    """
    last_played: dict[str, str] = {}
    last_shown_idx: dict[str, int] = {}
    for idx, show in enumerate(shows):
        for song in show_song_set(show):
            last_played[song] = show.date
            last_shown_idx[song] = idx
    total = len(shows)
    return {
        song: SongGap(gap=total - 1 - idx, last_played=last_played[song])
        for song, idx in last_shown_idx.items()
    }


def build_song_stats(
    shows: list[Show], songs_catalog: list[SongCatalogEntry],
) -> dict[str, SongStats]:
    """Aggregate per-song play statistics from the full setlist history.

    Parameters
    ----------
    shows
        Full list of historical shows in chronological order.
    songs_catalog
        Phish.net song catalog used to flag covers.

    Returns
    -------
    dict[str, SongStats]
        Mapping ``song_name -> SongStats``.
    """
    total_shows = len(shows)
    if total_shows == 0:
        return {}

    cover_lookup = {s.name: not s.is_original for s in songs_catalog}

    plays: Counter[str] = Counter()
    debut_year: dict[str, int] = {}
    set_counts: dict[str, Counter[str]] = defaultdict(Counter)
    pos_sum: dict[str, float] = defaultdict(float)
    pos_n: dict[str, int] = defaultdict(int)
    opener_counts: Counter[str] = Counter()
    closer_counts: Counter[str] = Counter()
    recent_appearances: dict[int, Counter[str]] = {w: Counter() for w in RECENT_WINDOWS}

    for show in shows:
        played = show_song_set(show)
        year = show.year or 9999
        for song in played:
            plays[song] += 1
            debut_year[song] = min(debut_year.get(song, year), year)
        for set_key in SET_KEYS:
            set_songs = show.sets.get(set_key, [])
            n = len(set_songs)
            if n == 0:
                continue
            for i, s in enumerate(set_songs):
                set_counts[s.song][set_key] += 1
                pos_sum[s.song] += i / max(1, n - 1)
                pos_n[s.song] += 1
            opener_counts[set_songs[0].song] += 1
            closer_counts[set_songs[-1].song] += 1

    for w in RECENT_WINDOWS:
        for show in shows[-w:]:
            for song in show_song_set(show):
                recent_appearances[w][song] += 1

    out: dict[str, SongStats] = {}
    for song, count in plays.items():
        sets_played = set_counts[song]
        typical_key = max(sets_played.items(), key=lambda kv: kv[1])[0] if sets_played else "1"
        out[song] = SongStats(
            total_plays=count,
            lifetime_frequency=count / total_shows,
            recent_frequency_50=recent_appearances[50][song] / min(50, total_shows),
            recent_frequency_20=recent_appearances[20][song] / min(20, total_shows),
            avg_set_position=pos_sum[song] / pos_n[song] if pos_n[song] else 0.5,
            typical_set=SET_TO_INT.get(typical_key, 1),
            set_distribution={k: sets_played[k] / count for k in sets_played},
            opener_frequency=opener_counts[song] / count,
            closer_frequency=closer_counts[song] / count,
            is_cover=cover_lookup.get(song, False),
            debut_year=debut_year.get(song, 0) if debut_year.get(song, 9999) != 9999 else 0,
        )
    return out


def _normalize(counts: Counter[str], vocab_size: int) -> dict[str, float]:
    """Laplace-smooth raw counts into a probability distribution."""
    if not counts:
        return {}
    total = sum(counts.values()) + LAPLACE_K * vocab_size
    return {k: (v + LAPLACE_K) / total for k, v in counts.items()}


def build_transition_matrix(shows: list[Show]) -> TransitionMatrix:
    """Build order-1 and order-2 song-transition probabilities, Laplace smoothed.

    Parameters
    ----------
    shows
        Chronological list of shows.

    Returns
    -------
    TransitionMatrix
        Transition probabilities with ``order_1`` / ``order_2`` /
        ``set_openers`` / ``set_closers``, plus ``vocab_size`` and
        ``laplace_k``.
    """
    order1: dict[str, Counter[str]] = defaultdict(Counter)
    order2: dict[str, Counter[str]] = defaultdict(Counter)
    set_openers: dict[str, Counter[str]] = {k: Counter() for k in SET_KEYS}
    set_closers: dict[str, Counter[str]] = {k: Counter() for k in SET_KEYS}
    vocab: set[str] = set()

    for show in shows:
        for set_key in SET_KEYS:
            songs = [s.song for s in show.sets.get(set_key, [])]
            if not songs:
                continue
            set_openers[set_key][songs[0]] += 1
            set_closers[set_key][songs[-1]] += 1
            vocab.update(songs)
            for i in range(len(songs) - 1):
                order1[songs[i]][songs[i + 1]] += 1
            for i in range(len(songs) - 2):
                key = f"{songs[i]}|{songs[i + 1]}"
                order2[key][songs[i + 2]] += 1

    V = max(1, len(vocab))
    return TransitionMatrix(
        order_1={k: _normalize(v, V) for k, v in order1.items()},
        order_2={k: _normalize(v, V) for k, v in order2.items()},
        set_openers={k: _normalize(v, V) for k, v in set_openers.items()},
        set_closers={k: _normalize(v, V) for k, v in set_closers.items()},
        vocab_size=V,
        laplace_k=LAPLACE_K,
    )


def build_venue_history(shows: list[Show]) -> dict[str, VenueHistory]:
    """Aggregate per-venue song frequencies and common opener / closer picks.

    Parameters
    ----------
    shows
        Chronological list of shows.

    Returns
    -------
    dict[str, VenueHistory]
        Mapping of canonical ``venue_id`` to a ``VenueHistory`` record.
    """
    per_venue: dict[str, dict] = {}
    for show in shows:
        vid = show.venue_id
        if not vid:
            continue
        v = per_venue.setdefault(vid, {
            "venue_id": vid,
            "name": show.venue_name,
            "total_shows": 0,
            "song_counts": Counter(),
            "openers": Counter(),
            "closers": Counter(),
        })
        v["total_shows"] += 1
        v["song_counts"].update(show_song_set(show))
        set1 = show.sets.get("1", [])
        if set1:
            v["openers"][set1[0].song] += 1
        for k in ("encore", "3", "2", "1"):
            songs = show.sets.get(k, [])
            if songs:
                v["closers"][songs[-1].song] += 1
                break

    out: dict[str, VenueHistory] = {}
    for vid, v in per_venue.items():
        n = v["total_shows"]
        out[vid] = VenueHistory(
            venue_id=vid,
            name=v["name"],
            total_shows=n,
            song_freq={s: c / n for s, c in v["song_counts"].most_common()},
            common_openers=[s for s, _ in v["openers"].most_common(10)],
            common_closers=[s for s, _ in v["closers"].most_common(10)],
        )
    return out


def main() -> None:
    """Build all feature artifacts from the canonical setlist data."""
    # Deferred import to avoid circular dependency: artifacts -> features.types
    # -> features.__init__ -> features.build -> artifacts.
    from phinish.artifacts import load_shows, load_songs_catalog

    if not SETLISTS_PATH.exists():
        raise SystemExit(f"Missing {SETLISTS_PATH}; run phinish-scrape first.")
    shows = load_shows()
    songs_catalog = load_songs_catalog()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    log.info("building_features", shows=len(shows))
    save_json(FEATURES_DIR / "song_gaps.json", build_song_gaps(shows))
    save_json(FEATURES_DIR / "song_stats.json", build_song_stats(shows, songs_catalog))
    save_json(FEATURES_DIR / "transition_matrix.json", build_transition_matrix(shows))
    save_json(FEATURES_DIR / "venue_history.json", build_venue_history(shows))
    log.info("wrote_features", path=str(FEATURES_DIR))


if __name__ == "__main__":
    main()
