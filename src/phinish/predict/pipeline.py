"""Inference pipeline: produce a structured setlist for a given date + venue.

CLI:
  pixi run phinish-predict --date 2026-12-31 --venue "MSG"
"""

import argparse
import pickle
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from phinish.artifacts import (
    calibrated_predict_proba,
    gap_score_per_song,
    load_calibrator,
    load_cover_set,
    load_ensemble_weights,
    load_song_gaps,
    load_song_stats,
    load_transition_matrix,
    load_venue_history,
    load_venues,
    load_xgb_model,
    markov_score_per_song,
    venue_freq_for_show,
)
from phinish.features.types import SongGap, SongStats, TransitionMatrix, VenueHistory
from phinish.predict.types import Prediction, PredictionItem, PredictionWeights
from phinish.scrape.types import Show, VenueRecord
from phinish.train import MIN_PLAYS_FOR_CANDIDATE, StreamingState, featurize
from phinish.train.types import EnsembleWeights
from phinish.utils import (
    SET_DISPLAY,
    STATE_SNAPSHOT_PATH,
    VALID_DATE,
    day_of_week,
    fuzzy_venue_match,
    min_max_normalize,
    special_show_flags,
    venue_id,
)

SET_TARGETS = {"1": 10, "2": 8, "encore": 2}  # typical Phish set lengths
LAST_3_PENALTY = 0.3  # strongly discount songs played in last 3 shows
MARKOV_BLEND = 0.5  # equal weight to transition probs vs. ensemble score during sequencing
OPENER_BLEND = 0.5  # equal weight to opener/closer history vs. base score


@dataclass(slots=True)
class _Artifacts:
    """All loaded model artifacts and precomputed scores needed for inference."""

    venues: dict[str, VenueRecord]
    stats: dict[str, SongStats]
    gaps: dict[str, SongGap]
    transition: TransitionMatrix
    venue_history: dict[str, VenueHistory]
    weights: EnsembleWeights
    cover_set: set[str]
    markov: dict[str, float]
    gap_scores: dict[str, float]
    xgb: Any
    calibrator: Any | None


def synthetic_show(date_str: str, venue_name: str, venues: dict[str, VenueRecord]) -> Show:
    """Build a stand-in ``Show`` for an upcoming (not-yet-played) show.

    Parameters
    ----------
    date_str
        Target show date as ``YYYY-MM-DD``.
    venue_name
        Venue as the user typed it.
    venues
        The catalog from ``data/processed/venues.json``.

    Returns
    -------
    Show
        A populated ``Show`` suitable for passing to ``featurize()``.
    """
    vid = fuzzy_venue_match(venue_name, venues) or venue_id(venue_name)
    flags = special_show_flags(date_str)
    d = date.fromisoformat(date_str)
    matched_venue = venues.get(vid)
    return Show(
        show_id="predict",
        date=date_str,
        year=d.year,
        month=d.month,
        day=d.day,
        day_of_week=day_of_week(d),
        venue_id=vid,
        venue_name=venue_name,
        city=matched_venue.city if matched_venue else "",
        state=matched_venue.state if matched_venue else "",
        country=matched_venue.country if matched_venue else "",
        tour="",
        tour_id="",
        sets={},
        is_nye=flags.is_nye,
        is_halloween=flags.is_halloween,
        is_festival=flags.is_festival,
    )


def _build_state(shows: list[Show]) -> StreamingState:
    """Replay all shows through a fresh StreamingState and return it."""
    state = StreamingState()
    for show in shows:
        state.update(show)
    return state


def _load_or_replay_state() -> StreamingState:
    """Load the pickled state snapshot, falling back to full replay on failure."""
    if STATE_SNAPSHOT_PATH.exists():
        try:
            return pickle.loads(STATE_SNAPSHOT_PATH.read_bytes())
        except Exception as e:
            print(f"warning: state snapshot unreadable ({e!r}); replaying", flush=True)
    from phinish.artifacts import load_shows
    return _build_state(load_shows())


def _split_by_typical_set(
    stats: dict[str, SongStats], candidates: list[str],
) -> dict[str, list[str]]:
    """Bucket candidate songs into set 1 / set 2 / encore by historical tendency."""
    buckets: dict[str, list[str]] = {"1": [], "2": [], "encore": []}
    for song in candidates:
        s = stats.get(song)
        ts = s.typical_set if s else 1
        if ts == 1:
            buckets["1"].append(song)
        elif ts == 2:
            buckets["2"].append(song)
        else:
            buckets["encore"].append(song)
    return buckets


def _sequence(
    pool_scores: list[tuple[str, float]],
    target_len: int,
    transition: TransitionMatrix,
    set_key: str,
) -> list[str]:
    """Order songs within a set using Markov transitions blended with ensemble scores."""
    pool = dict(pool_scores)
    set_openers = transition.set_openers.get(set_key, {})
    set_closers = transition.set_closers.get(set_key, {})

    if not pool:
        return []

    opener = max(pool, key=lambda s: OPENER_BLEND * pool[s]
                                     + (1 - OPENER_BLEND) * set_openers.get(s, 0.0))
    sequence = [opener]
    pool.pop(opener)

    while len(sequence) < target_len - 1 and pool:
        prev1 = sequence[-1]
        prev2 = sequence[-2] if len(sequence) >= 2 else None
        order2 = transition.order_2.get(f"{prev2}|{prev1}", {}) if prev2 else {}
        order1 = transition.order_1.get(prev1, {})

        scored = [
            (song, MARKOV_BLEND * order2.get(song, order1.get(song, 0.0))
                   + (1 - MARKOV_BLEND) * pool_score)
            for song, pool_score in pool.items()
        ]
        best, _ = max(scored, key=lambda pair: pair[1])
        sequence.append(best)
        pool.pop(best)

    if pool and target_len >= 2:
        closer = max(pool, key=lambda s: OPENER_BLEND * pool[s]
                                         + (1 - OPENER_BLEND) * set_closers.get(s, 0.0))
        sequence.append(closer)

    return sequence[:target_len]


def _load_artifacts() -> _Artifacts:
    """Load all features and model files required for inference."""
    venues = load_venues()
    stats = load_song_stats()
    gaps = load_song_gaps()
    transition = load_transition_matrix()
    venue_history = load_venue_history()
    weights = load_ensemble_weights()

    return _Artifacts(
        venues=venues,
        stats=stats,
        gaps=gaps,
        transition=transition,
        venue_history=venue_history,
        weights=weights,
        cover_set=load_cover_set(),
        markov=markov_score_per_song(transition),
        gap_scores=gap_score_per_song(stats, gaps),
        xgb=load_xgb_model(),
        calibrator=load_calibrator(),
    )


def _score_candidates(
    state: StreamingState, show: Show, candidates: list[str], art: _Artifacts,
) -> list[dict]:
    """Score every candidate song with the weighted ensemble, sorted descending."""
    X = np.array(
        [featurize(state, show, song, art.cover_set) for song in candidates],
        dtype=np.float32,
    )
    xgb_probs = calibrated_predict_proba(art.xgb, art.calibrator, X)
    venue_freq = venue_freq_for_show(art.venue_history, show.venue_id)
    xgb_n = min_max_normalize(list(xgb_probs))
    m_n = min_max_normalize([art.markov.get(s, 0.0) for s in candidates])
    g_n = min_max_normalize([art.gap_scores.get(s, 0.0) for s in candidates])
    v_n = min_max_normalize([venue_freq.get(s, 0.0) for s in candidates])

    w = art.weights
    wx, wm, wg, wv = w.w_xgboost, w.w_markov, w.w_gap, w.w_venue
    last_show = state.last_show_set
    last_3 = {s for show_set in state.last_3_shows for s in show_set}

    scored: list[dict] = []
    for i, song in enumerate(candidates):
        if song in last_show:
            continue
        ensemble = wx * xgb_n[i] + wm * m_n[i] + wg * g_n[i] + wv * v_n[i]
        # Keep predictions musically plausible by down-weighting very recent repeats.
        if song in last_3:
            ensemble *= LAST_3_PENALTY
        gap_obj = art.gaps.get(song)
        scored.append({
            "song": song,
            "ensemble": ensemble,
            "confidence": float(xgb_probs[i]),
            "gap": gap_obj.gap if gap_obj else 0,
        })
    scored.sort(key=lambda s: s["ensemble"], reverse=True)
    return scored


def _assemble_setlist(
    scored: list[dict], stats: dict[str, SongStats], transition: TransitionMatrix,
) -> dict[str, list[PredictionItem]]:
    """Fill set 1 / set 2 / encore from scored candidates with Markov sequencing."""
    buckets = _split_by_typical_set(stats, [s["song"] for s in scored])
    score_lookup = {s["song"]: s["ensemble"] for s in scored}
    confidence_lookup = {s["song"]: s["confidence"] for s in scored}
    gap_lookup = {s["song"]: s["gap"] for s in scored}

    used: set[str] = set()
    setlist: dict[str, list[PredictionItem]] = {}
    for set_key, target in SET_TARGETS.items():
        pool = [(s, score_lookup[s]) for s in buckets.get(set_key, []) if s not in used]
        if len(pool) < target * 2:
            in_pool = {p[0] for p in pool}
            extras = [
                (s["song"], s["ensemble"]) for s in scored
                if s["song"] not in used and s["song"] not in in_pool
            ]
            pool.extend(extras[: target * 3])
        sequenced = _sequence(pool, target, transition, set_key)
        used.update(sequenced)
        setlist[set_key] = [
            PredictionItem(
                song=song,
                confidence=confidence_lookup.get(song, 0.0),
                gap=gap_lookup.get(song, 0),
            )
            for song in sequenced
        ]
    return setlist


def predict(show_date: str, venue: str, city: str | None = None) -> Prediction:
    """Produce a structured 3-set prediction for a given date and venue.

    Parameters
    ----------
    show_date
        Target show date in YYYY-MM-DD format.
    venue
        Venue name (fuzzy-matched against the venue catalog).
    city
        Optional city override.

    Returns
    -------
    Prediction
        Prediction payload with setlist, confidence, and weights.

    Raises
    ------
    ValueError
        If ``show_date`` is not a valid YYYY-MM-DD string.
    """
    if not VALID_DATE.match(show_date):
        raise ValueError(f"Invalid date format: {show_date!r} (expected YYYY-MM-DD)")

    art = _load_artifacts()
    state = _load_or_replay_state()
    show = synthetic_show(show_date, venue, art.venues)

    candidates = [s for s, c in state.plays.items() if c >= MIN_PLAYS_FOR_CANDIDATE]
    if not candidates:
        raise RuntimeError("No song candidates available; have models been trained?")

    scored = _score_candidates(state, show, candidates, art)
    setlist = _assemble_setlist(scored, art.stats, art.transition)

    avg_conf = (
        sum(item.confidence for items in setlist.values() for item in items)
        / max(1, sum(len(items) for items in setlist.values()))
    )
    w = art.weights
    return Prediction(
        date=show_date,
        venue=venue,
        venue_id=show.venue_id,
        city=show.city or (city or ""),
        setlist=setlist,
        avg_confidence=avg_conf,
        model_version=w.val_year,
        weights=PredictionWeights(
            xgboost=w.w_xgboost,
            markov=w.w_markov,
            gap=w.w_gap,
            venue=w.w_venue,
        ),
    )


def _format_text(prediction: Prediction) -> str:
    """Render a prediction as a plain-text table for CLI output."""
    lines = [
        f"Phinish Prediction — {prediction.venue} — {prediction.date}",
        f"Avg confidence: {prediction.avg_confidence:.1%}",
        "",
    ]
    for set_key, label in SET_DISPLAY:
        items = prediction.setlist.get(set_key, [])
        if not items:
            continue
        lines.append(f"{label}:")
        for i, item in enumerate(items, 1):
            lines.append(
                f"  {i:>2}. {item.song:<35s}  "
                f"{item.confidence:>5.1%}  gap={item.gap}"
            )
        lines.append("")
    return "\n".join(lines)


def cli() -> None:
    """Console-script entry point: parse args, run ``predict``, and print."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--venue", required=True)
    parser.add_argument("--city", default=None)
    args = parser.parse_args()

    result = predict(args.date, args.venue, args.city)
    print(_format_text(result))


if __name__ == "__main__":
    cli()
