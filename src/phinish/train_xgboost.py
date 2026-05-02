"""Train XGBoost song selector with temporal split + Platt calibration.

For each historical show, generate one row per candidate song using only data
strictly before that show (streaming-state approach for temporal correctness).

Writes:
  models/xgboost_song_selector.pkl
  models/calibrator.pkl
  models/xgboost_meta.json
"""

import math
import pickle
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from phinish.utils import (
    MODELS_DIR,
    SET_KEYS,
    SET_TO_INT,
    SETLISTS_PATH,
    SONGS_PATH,
    STATE_SNAPSHOT_PATH,
    Show,
    SongCatalogEntry,
    day_of_week,
    load_json,
    save_json,
    show_song_set,
    special_show_flags,
)

DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
TRAIN_END_YEAR = 2023
VAL_YEAR = 2024
TEST_START_YEAR = 2025
BUSTOUT_GAP = 50
MIN_PLAYS_FOR_CANDIDATE = 5

XGB_PARAMS = dict(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=300,
    scale_pos_weight=12,
    objective="binary:logistic",
    tree_method="hist",
    n_jobs=-1,
    verbosity=0,
)


def feature_names() -> list[str]:
    """Return the ordered list of feature names produced by ``featurize()``.

    Kept in lockstep with ``featurize()`` so the model card and debug
    tooling can map a feature index back to a meaningful name.
    Day-of-week is one-hot expanded as ``dow_<day>`` over all seven
    weekdays.

    Returns
    -------
    list[str]
        Ordered feature names; positional alignment with rows produced
        by ``featurize()``.
    """
    base = [
        "gap", "log_gap", "is_bustout", "lifetime_freq",
        "recent_freq_50", "recent_freq_20",
        "avg_set_position", "typical_set",
        "is_cover", "opener_freq", "closer_freq",
        "played_last_show", "played_last_3", "times_played_this_tour",
        "venue_total_shows", "venue_song_freq", "venue_is_new",
        "days_since_last_show", "tour_position_pct",
        "month", "is_weekend", "is_nye", "is_halloween", "is_festival",
    ]
    return base + [f"dow_{d}" for d in DAYS]


@dataclass(slots=True)
class StreamingState:
    """Incremental statistics accumulated as shows are replayed in order.

    Holds every signal ``featurize()`` needs to score a candidate song:
    cumulative play counts, set-position averages, opener / closer rates,
    sliding 50- and 20-show windows for recent-frequency features, the
    last show's song set (for hard exclusion), and per-venue / per-tour
    play counters. Designed to be ``update()``-d once per historical show
    in chronological order, then read by ``featurize()`` *before* the
    next ``update()`` — this ordering is what guarantees temporal
    correctness during training and at inference.

    Fields are intentionally exposed (no encapsulation) because
    ``featurize()`` reads many of them and benefits from direct access.
    Use ``slots=True`` to keep memory tight; a full backtest replays
    ~2,100 shows.

    Attributes
    ----------
    n : int
        Number of shows already merged into the state.
    plays : Counter[str]
        Lifetime play count per song.
    last_played_idx : dict[str, int]
        Most recent show index (in replay order) at which each song was
        played; used to compute gap.
    last_show_set : set[str]
        Songs played in the most recent show (``state.n - 1``); used for
        hard exclusion at inference time.
    last_3_shows : deque[set[str]]
        Sliding window of the most recent three shows' song sets, used
        for the soft "played recently" penalty.
    """

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
    venue_song_counts: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    tour_song_plays: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    tour_show_count: Counter[str] = field(default_factory=Counter)

    def update(self, show: Show) -> None:
        played = show_song_set(show)
        self.plays.update(played)
        for song in played:
            self.last_played_idx[song] = self.n
            self.last_played_date[song] = show.get("date", "")
        for set_key in SET_KEYS:
            songs = show.get("sets", {}).get(set_key, [])
            n = len(songs)
            if n == 0:
                continue
            for i, s in enumerate(songs):
                song = s["song"]
                self.typical_set_counts[song][set_key] += 1
                self.set_pos_sum[song] += i / max(1, n - 1)
                self.set_pos_n[song] += 1
            self.opener_counts[songs[0]["song"]] += 1
            self.closer_counts[songs[-1]["song"]] += 1
        self.recent_50.append(played)
        self.recent_20.append(played)
        self.last_show_set = played
        self.last_3_shows.append(played)
        self.last_show_date = show.get("date", "")
        vid = show.get("venue_id", "")
        if vid:
            self.venue_shows[vid] += 1
            self.venue_song_counts[vid].update(played)
        tour = show.get("tour", "")
        if tour:
            self.tour_song_plays[tour].update(played)
            self.tour_show_count[tour] += 1
        self.n += 1


def days_between(d1: str, d2: str) -> int:
    """Absolute day count between two ISO dates; 0 if either is empty.

    Parameters
    ----------
    d1, d2
        ISO date strings (``YYYY-MM-DD``). Either may be empty, which
        triggers the zero fallback (used for the very first show in
        the dataset, where ``state.last_show_date`` is still empty).

    Returns
    -------
    int
        Absolute day difference. Order of arguments doesn't matter.
    """
    if not d1 or not d2:
        return 0
    return abs((date.fromisoformat(d2) - date.fromisoformat(d1)).days)


def featurize(state: StreamingState, show: Show, song: str, cover_set: set[str]) -> list[float]:
    """Build the XGBoost feature vector for one (state, show, song) triple.

    Order matches ``feature_names()`` exactly. Mixes rotation features
    (gap, log_gap, is_bustout), play-rate features (lifetime, recent_50,
    recent_20), set-placement (typical_set, opener/closer rates),
    venue/tour history, and calendar flags (NYE, Halloween, weekend, dow).

    Parameters
    ----------
    state
        Streaming statistics accumulated from all shows strictly before ``show``.
        Reading from ``state`` here must not include any data from ``show``
        itself, to avoid temporal leakage.
    show
        The target show being scored; only its date / venue / tour fields
        are read.
    song
        Candidate song name.
    cover_set
        Set of song names considered covers (artist != Phish).

    Returns
    -------
    list[float]
        Feature vector aligned positionally with ``feature_names()``.
    """
    last_idx = state.last_played_idx.get(song)
    gap = state.n - last_idx if last_idx is not None else state.n + 100
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

    vid = show.get("venue_id", "")
    venue_total = state.venue_shows.get(vid, 0)
    venue_song = state.venue_song_counts.get(vid, Counter()).get(song, 0)
    venue_song_freq = venue_song / venue_total if venue_total else 0.0

    tour = show.get("tour", "")
    tour_plays = state.tour_song_plays.get(tour, Counter()).get(song, 0)
    tour_n = state.tour_show_count.get(tour, 0)
    tour_pos_pct = tour_n / 30.0 if tour_n else 0.0  # rough; tours are typically ~20-40 shows

    flags = special_show_flags(show.get("date", "0000-01-01"), show.get("tour", ""))
    dow = day_of_week(show["date"]) if show.get("date") else "monday"
    month = show.get("month", 1)

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
        days_between(state.last_show_date, show.get("date", "")),
        tour_pos_pct,
        month,
        1.0 if dow in ("saturday", "sunday") else 0.0,
        1.0 if flags["is_nye"] else 0.0,
        1.0 if flags["is_halloween"] else 0.0,
        1.0 if flags["is_festival"] else 0.0,
    ]
    row.extend(1.0 if dow == d else 0.0 for d in DAYS)
    return row


def build_training_matrix(
    shows: list[Show], cover_set: set[str], min_year: int, max_year: int,
) -> tuple[np.ndarray, np.ndarray, list[str], StreamingState]:
    """Replay history and emit (X, y, dates) for shows in ``[min_year, max_year]``.

    Walks every show in chronological order. When a show falls inside the
    target year window, generates one training row per candidate song
    (any song with at least ``MIN_PLAYS_FOR_CANDIDATE`` lifetime plays as
    of the prior history) — feature vector via ``featurize()``, label of
    1 if that song was actually played at the show. State is then
    advanced to include the show, so the next iteration's features
    reflect everything up to (but not including) that show.

    Parameters
    ----------
    shows
        Full chronological list of shows. The function reads every show
        for state replay, but only emits training rows for shows in the
        ``[min_year, max_year]`` window.
    cover_set
        Set of song names considered covers, passed through to
        ``featurize()`` for the ``is_cover`` flag.
    min_year, max_year
        Inclusive year window for which training rows are generated.

    Returns
    -------
    X : np.ndarray, shape (n_rows, n_features), dtype float32
        Feature matrix aligned with ``feature_names()``.
    y : np.ndarray, shape (n_rows,), dtype int8
        Binary labels (1 if song was played at the show, else 0).
    show_dates : list[str]
        ISO date string per row, useful for diagnostic slicing /
        per-show evaluation.
    final_state : StreamingState
        The streaming state after replaying all shows (including any past
        ``max_year``). Used by inference to skip the full historical
        replay — the snapshot is saved alongside the model.
    """
    state = StreamingState()
    rows: list[list[float]] = []
    labels: list[int] = []
    show_dates: list[str] = []

    for show in shows:
        year = show.get("year", 0)
        played = show_song_set(show)
        if min_year <= year <= max_year:
            candidates = [s for s, c in state.plays.items() if c >= MIN_PLAYS_FOR_CANDIDATE]
            for song in candidates:
                rows.append(featurize(state, show, song, cover_set))
                labels.append(1 if song in played else 0)
                show_dates.append(show["date"])
        state.update(show)

    return np.array(rows, dtype=np.float32), np.array(labels, dtype=np.int8), show_dates, state


def main() -> None:
    """Train XGBoost song selector with temporal split and Platt calibration."""
    if not SETLISTS_PATH.exists():
        raise SystemExit(f"Missing {SETLISTS_PATH}; run scrape.py first.")
    shows = load_json(SETLISTS_PATH)
    songs_catalog: list[SongCatalogEntry] = (
        load_json(SONGS_PATH) if SONGS_PATH.exists() else []
    )
    cover_set = {s["name"] for s in songs_catalog if not s.get("is_original", True)}

    print("Building training matrix (train < 2024)...", flush=True)
    X_train, y_train, _, _ = build_training_matrix(shows, cover_set, 1983, TRAIN_END_YEAR)
    print(f"  train: {X_train.shape}, positives: {int(y_train.sum())}")

    print(f"Building validation matrix ({VAL_YEAR})...", flush=True)
    X_val, y_val, _, final_state = build_training_matrix(shows, cover_set, VAL_YEAR, VAL_YEAR)
    print(f"  val: {X_val.shape}, positives: {int(y_val.sum())}")

    if X_train.size == 0:
        raise SystemExit("No training rows generated. Check data/processed/setlists.json.")

    print("Fitting XGBoost...", flush=True)
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    if X_val.size > 0:
        print("Fitting Platt calibration...", flush=True)
        raw_val = model.predict_proba(X_val)[:, 1]
        calibrator = LogisticRegression()
        calibrator.fit(raw_val.reshape(-1, 1), y_val)
    else:
        calibrator = None
        print("(skipping calibration: no validation rows)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "xgboost_song_selector.pkl").write_bytes(pickle.dumps(model))
    (MODELS_DIR / "calibrator.pkl").write_bytes(pickle.dumps(calibrator))
    # Snapshot the streaming state so inference can skip the historical replay.
    # The snapshot encodes aggregated stats + recent-window song bags; raw
    # setlist data is not stored here (see .gitignore).
    STATE_SNAPSHOT_PATH.write_bytes(pickle.dumps(final_state))
    save_json(MODELS_DIR / "xgboost_meta.json", {
        "feature_names": feature_names(),
        "n_train_rows": int(X_train.shape[0]),
        "n_val_rows": int(X_val.shape[0]),
        "params": {k: v for k, v in XGB_PARAMS.items() if isinstance(v, (int, float, str, bool))},
        "min_plays_for_candidate": MIN_PLAYS_FOR_CANDIDATE,
        "train_end_year": TRAIN_END_YEAR,
        "val_year": VAL_YEAR,
    })
    print(f"wrote model -> {MODELS_DIR / 'xgboost_song_selector.pkl'}")


if __name__ == "__main__":
    main()
