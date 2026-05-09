"""Streaming state and feature vector — shared by training and inference.

``StreamingState`` accumulates per-song / per-venue / per-tour statistics
as historical shows are replayed in chronological order. ``featurize``
turns a (state, show, song) triple into the XGBoost feature row aligned
with ``feature_names()``.

Lives outside ``train/xgboost.py`` because both the training pipeline
and the prediction pipeline use it; keeping it here avoids circular
imports between ``train.xgboost`` and ``predict.pipeline``.
"""

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import date

from phinish.scrape.types import Show
from phinish.utils import (
    SET_KEYS,
    SET_TO_INT,
    show_song_set,
)

DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
BUSTOUT_GAP = 50  # ~1 full tour without playing; empirically strong signal
MIN_PLAYS_FOR_CANDIDATE = 5  # filters one-off teases/debuts with too little history
NEVER_PLAYED_GAP_PENALTY = 100  # large synthetic gap for songs never seen; exceeds any real gap
TYPICAL_TOUR_LENGTH = 30.0  # average number of shows in a Phish tour


def split_years(max_year: int) -> tuple[int, int, int]:
    """Compute train/val/test split years relative to available data.

    Parameters
    ----------
    max_year
        The most recent year present in the dataset.

    Returns
    -------
    tuple[int, int, int]
        ``(train_end, val_year, test_start)`` where train covers up to
        ``max_year - 2``, val is ``max_year - 1``, and test is ``max_year``.
    """
    return max_year - 2, max_year - 1, max_year


def feature_names() -> list[str]:
    """Return the ordered list of feature names produced by ``featurize()``.

    Returns
    -------
    list[str]
        Ordered feature names; positional alignment with rows produced
        by ``featurize()``.
    """
    base = [
        "gap",
        "log_gap",
        "is_bustout",
        "lifetime_freq",
        "recent_freq_50",
        "recent_freq_20",
        "avg_set_position",
        "typical_set",
        "is_cover",
        "opener_freq",
        "closer_freq",
        "played_last_show",
        "played_last_3",
        "times_played_this_tour",
        "venue_total_shows",
        "venue_song_freq",
        "venue_is_new",
        "days_since_last_show",
        "tour_position_pct",
        "month",
        "is_weekend",
        "is_nye",
        "is_halloween",
        "is_festival",
    ]
    return base + [f"dow_{d}" for d in DAYS]


@dataclass(slots=True)
class StreamingState:
    """Incremental statistics accumulated as shows are replayed in order."""

    n: int = 0
    plays: Counter[str] = field(default_factory=Counter)
    last_played_idx: dict[str, int] = field(default_factory=dict)
    last_played_date: dict[str, str] = field(default_factory=dict)
    set_pos_sum: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    set_pos_n: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    typical_set_counts: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    opener_counts: Counter[str] = field(default_factory=Counter)
    closer_counts: Counter[str] = field(default_factory=Counter)
    recent_50: deque[set[str]] = field(default_factory=lambda: deque(maxlen=50))
    recent_20: deque[set[str]] = field(default_factory=lambda: deque(maxlen=20))
    last_show_set: set[str] = field(default_factory=set)
    last_3_shows: deque[set[str]] = field(default_factory=lambda: deque(maxlen=3))
    last_show_date: str = ""
    venue_shows: Counter[str] = field(default_factory=Counter)
    venue_song_counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    tour_song_plays: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    tour_show_count: Counter[str] = field(default_factory=Counter)

    def update(self, show: Show) -> None:
        played = show_song_set(show)
        self.plays.update(played)
        for song in played:
            self.last_played_idx[song] = self.n
            self.last_played_date[song] = show.date
        for set_key in SET_KEYS:
            songs = show.sets.get(set_key, [])
            n = len(songs)
            if n == 0:
                continue
            for i, s in enumerate(songs):
                self.typical_set_counts[s.song][set_key] += 1
                self.set_pos_sum[s.song] += i / max(1, n - 1)
                self.set_pos_n[s.song] += 1
            self.opener_counts[songs[0].song] += 1
            self.closer_counts[songs[-1].song] += 1
        self.recent_50.append(played)
        self.recent_20.append(played)
        self.last_show_set = played
        self.last_3_shows.append(played)
        self.last_show_date = show.date
        vid = show.venue_id
        if vid:
            self.venue_shows[vid] += 1
            self.venue_song_counts[vid].update(played)
        tour = show.tour
        if tour:
            self.tour_song_plays[tour].update(played)
            self.tour_show_count[tour] += 1
        self.n += 1


def days_between(d1: str, d2: str) -> int:
    """Absolute day count between two ISO dates; 0 if either is empty.

    Parameters
    ----------
    d1, d2
        ISO date strings (``YYYY-MM-DD``).

    Returns
    -------
    int
        Absolute day difference.
    """
    if not d1 or not d2:
        return 0
    return abs((date.fromisoformat(d2) - date.fromisoformat(d1)).days)


def featurize(state: StreamingState, show: Show, song: str, cover_set: set[str]) -> list[float]:
    """Build the XGBoost feature vector for one (state, show, song) triple.

    Parameters
    ----------
    state
        Streaming statistics accumulated from all shows strictly before ``show``.
    show
        The target show being scored.
    song
        Candidate song name.
    cover_set
        Set of song names considered covers.

    Returns
    -------
    list[float]
        Feature vector aligned positionally with ``feature_names()``.
    """
    last_idx = state.last_played_idx.get(song)
    gap = state.n - last_idx if last_idx is not None else state.n + NEVER_PLAYED_GAP_PENALTY
    plays = state.plays.get(song, 0)
    n = max(1, state.n)
    recent_50_n = max(1, len(state.recent_50))
    recent_20_n = max(1, len(state.recent_20))

    typical_set_counts = state.typical_set_counts.get(song, Counter())
    if typical_set_counts:
        typical_key = max(typical_set_counts.items(), key=lambda kv: kv[1])[0]
    else:
        typical_key = "1"
    typical_set = SET_TO_INT.get(typical_key, 1)
    avg_pos = state.set_pos_sum[song] / state.set_pos_n[song] if state.set_pos_n[song] else 0.5

    vid = show.venue_id
    venue_total = state.venue_shows.get(vid, 0)
    venue_song = state.venue_song_counts.get(vid, Counter()).get(song, 0)
    venue_song_freq = venue_song / venue_total if venue_total else 0.0

    tour = show.tour
    tour_plays = state.tour_song_plays.get(tour, Counter()).get(song, 0)
    tour_n = state.tour_show_count.get(tour, 0)
    tour_pos_pct = tour_n / TYPICAL_TOUR_LENGTH if tour_n else 0.0

    dow = show.day_of_week or "monday"
    month = show.month

    row = [
        gap,
        math.log(gap + 2),
        1.0 if gap >= BUSTOUT_GAP else 0.0,
        plays / n,
        sum(1 for s in state.recent_50 if song in s) / recent_50_n,
        sum(1 for s in state.recent_20 if song in s) / recent_20_n,
        avg_pos,
        typical_set,
        1.0 if song in cover_set else 0.0,
        state.opener_counts.get(song, 0) / max(1, plays),
        state.closer_counts.get(song, 0) / max(1, plays),
        1.0 if song in state.last_show_set else 0.0,
        1.0 if any(song in s for s in state.last_3_shows) else 0.0,
        tour_plays,
        venue_total,
        venue_song_freq,
        1.0 if venue_total == 0 else 0.0,
        days_between(state.last_show_date, show.date or ""),
        tour_pos_pct,
        month,
        1.0 if dow in ("saturday", "sunday") else 0.0,
        1.0 if show.is_nye else 0.0,
        1.0 if show.is_halloween else 0.0,
        1.0 if show.is_festival else 0.0,
    ]
    row.extend(1.0 if dow == d else 0.0 for d in DAYS)
    return row
