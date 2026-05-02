"""Train XGBoost song selector with temporal split + Platt calibration.

For each historical show, generate one row per candidate song using only data
strictly before that show (streaming-state approach for temporal correctness).

Writes:
  models/xgboost_song_selector.pkl
  models/calibrator.pkl
  models/xgboost_meta.json
"""
from __future__ import annotations

import math
import pickle
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.utils import (
    MODELS_DIR,
    SETLISTS_PATH,
    SONGS_PATH,
    day_of_week,
    load_json,
    save_json,
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


def show_song_set(show: dict) -> set[str]:
    return {s["song"] for songs in show.get("sets", {}).values() for s in songs}


def feature_names() -> list[str]:
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


class StreamingState:
    """Incremental statistics over shows seen so far."""

    def __init__(self) -> None:
        self.n = 0
        self.plays: Counter[str] = Counter()
        self.last_played_idx: dict[str, int] = {}
        self.last_played_date: dict[str, str] = {}
        self.set_pos_sum: dict[str, float] = defaultdict(float)
        self.set_pos_n: dict[str, int] = defaultdict(int)
        self.typical_set_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.opener_counts: Counter[str] = Counter()
        self.closer_counts: Counter[str] = Counter()
        self.recent_50: deque[set[str]] = deque(maxlen=50)
        self.recent_20: deque[set[str]] = deque(maxlen=20)
        self.last_show_set: set[str] = set()
        self.last_3_shows: deque[set[str]] = deque(maxlen=3)
        self.last_show_date: str = ""
        self.venue_shows: Counter[str] = Counter()
        self.venue_song_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.tour_song_plays: dict[str, Counter[str]] = defaultdict(Counter)
        self.tour_show_count: Counter[str] = Counter()

    def update(self, show: dict) -> None:
        played = show_song_set(show)
        self.plays.update(played)
        for song in played:
            self.last_played_idx[song] = self.n
            self.last_played_date[song] = show.get("date", "")
        for set_key in ("1", "2", "3", "encore"):
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
    if not d1 or not d2:
        return 0
    from datetime import date as _date
    a = _date(*(int(x) for x in d1.split("-")))
    b = _date(*(int(x) for x in d2.split("-")))
    return abs((b - a).days)


def featurize(state: StreamingState, show: dict, song: str, cover_set: set[str]) -> list[float]:
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
    typical_set = {"1": 1, "2": 2, "3": 3, "encore": 3}.get(typical_key, 1)
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
    shows: list[dict], cover_set: set[str], min_year: int, max_year: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
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

    return np.array(rows, dtype=np.float32), np.array(labels, dtype=np.int8), show_dates


def main() -> None:
    if not SETLISTS_PATH.exists():
        raise SystemExit(f"Missing {SETLISTS_PATH}; run scrape.py first.")
    shows = load_json(SETLISTS_PATH)
    songs_catalog = load_json(SONGS_PATH) if SONGS_PATH.exists() else []
    cover_set = {s["name"] for s in songs_catalog if not s.get("is_original", True)}

    print("Building training matrix (train < 2024)...", flush=True)
    X_train, y_train, _ = build_training_matrix(shows, cover_set, 1983, TRAIN_END_YEAR)
    print(f"  train: {X_train.shape}, positives: {int(y_train.sum())}")

    print(f"Building validation matrix ({VAL_YEAR})...", flush=True)
    X_val, y_val, _ = build_training_matrix(shows, cover_set, VAL_YEAR, VAL_YEAR)
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
    with open(MODELS_DIR / "xgboost_song_selector.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODELS_DIR / "calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)
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
