"""Inference pipeline: produce a structured setlist for a given date + venue.

CLI:
  pixi run python scripts/predict.py --date 2026-12-31 --venue "MSG"
"""

import argparse
import pickle
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_ensemble import gap_score_per_song, markov_score_per_song
from scripts.train_xgboost import (
    MIN_PLAYS_FOR_CANDIDATE,
    StreamingState,
    featurize,
)
from scripts.utils import (
    FEATURES_DIR,
    MODELS_DIR,
    SET_DISPLAY,
    SETLISTS_PATH,
    SONGS_PATH,
    STATE_SNAPSHOT_PATH,
    VALID_DATE,
    VENUES_PATH,
    EnsembleWeights,
    Prediction,
    PredictionItem,
    Show,
    SongGap,
    SongStats,
    TransitionMatrix,
    VenueHistory,
    VenueRecord,
    day_of_week,
    fuzzy_venue_match,
    load_json,
    min_max_normalize,
    special_show_flags,
    venue_id,
)

SET_TARGETS = {"1": 10, "2": 8, "encore": 2}
LAST_3_PENALTY = 0.3
MARKOV_BLEND = 0.5
OPENER_BLEND = 0.5


@dataclass(slots=True)
class _Artifacts:
    shows: list[Show]
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

    Inference reuses the same ``featurize()`` machinery as training,
    which expects a ``Show``-shaped dict. This factory constructs that
    shape for a hypothetical show: it resolves the venue (via
    ``fuzzy_venue_match`` + alias table) so venue-history features can
    fire, derives calendar fields from ``date_str``, and merges the
    NYE / Halloween / festival flags. The ``sets`` field is empty (the
    point is to predict it) and ``total_songs`` is omitted — both are
    consistent with the ``Show`` TypedDict (``total_songs`` is
    ``NotRequired``).

    Parameters
    ----------
    date_str
        Target show date as ``YYYY-MM-DD``. Parsed with
        ``date.fromisoformat`` so invalid dates raise ``ValueError``
        before any work is done downstream.
    venue_name
        Venue as the user typed it; resolved via fuzzy matching against
        ``venues`` and falls back to the canonical alias slug.
    venues
        The catalog from ``data/processed/venues.json``, used both for
        venue-id resolution and for prefilling ``city`` / ``state`` /
        ``country`` when the venue is known.

    Returns
    -------
    Show
        A populated ``Show`` dict suitable for passing to
        ``featurize(state, show, ...)``. ``tour`` and ``tour_id`` are
        empty since upcoming-show tour metadata is not knowable from a
        date + venue alone.
    """
    vid = fuzzy_venue_match(venue_name, venues) or venue_id(venue_name)
    flags = special_show_flags(date_str)
    d = date.fromisoformat(date_str)
    return {
        "show_id": "predict",
        "date": date_str,
        "year": d.year,
        "month": d.month,
        "day": d.day,
        "day_of_week": day_of_week(date_str),
        "venue_id": vid,
        "venue_name": venue_name,
        "city": venues.get(vid, {}).get("city", ""),
        "state": venues.get(vid, {}).get("state", ""),
        "country": venues.get(vid, {}).get("country", ""),
        "tour": "",
        "tour_id": "",
        **flags,
        "sets": {},
    }


def _build_state(shows: list[Show]) -> StreamingState:
    """Return a StreamingState built by replaying ``shows`` in order."""
    state = StreamingState()
    for show in shows:
        state.update(show)
    return state


def _load_or_replay_state(shows: list[Show]) -> StreamingState:
    """Prefer the committed state snapshot; fall back to a full replay.

    The snapshot at ``models/state.pkl`` is written by ``train_xgboost.main``
    and lets prediction skip rebuilding StreamingState from raw setlists —
    which matters because raw setlists are gitignored per the API ToS and
    aren't present in a fresh checkout.
    """
    if STATE_SNAPSHOT_PATH.exists():
        try:
            return pickle.loads(STATE_SNAPSHOT_PATH.read_bytes())
        except Exception as e:
            print(f"warning: state snapshot unreadable ({e!r}); replaying", flush=True)
    return _build_state(shows)


def _split_by_typical_set(
    stats: dict[str, SongStats], candidates: list[str],
) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {"1": [], "2": [], "encore": []}
    for song in candidates:
        ts = stats.get(song, {}).get("typical_set", 1)
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
    pool = dict(pool_scores)
    set_openers = transition.get("set_openers", {}).get(set_key, {})
    set_closers = transition.get("set_closers", {}).get(set_key, {})

    if not pool:
        return []

    opener = max(pool, key=lambda s: OPENER_BLEND * pool[s]
                                     + (1 - OPENER_BLEND) * set_openers.get(s, 0.0))
    sequence = [opener]
    pool.pop(opener)

    while len(sequence) < target_len - 1 and pool:
        prev1 = sequence[-1]
        prev2 = sequence[-2] if len(sequence) >= 2 else None
        order2 = transition.get("order_2", {}).get(f"{prev2}|{prev1}", {}) if prev2 else {}
        order1 = transition.get("order_1", {}).get(prev1, {})

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
    # Raw setlists are only needed if the state snapshot is missing
    # (e.g., a fresh checkout where data/processed/ hasn't been populated).
    shows = load_json(SETLISTS_PATH) if SETLISTS_PATH.exists() else []
    songs_catalog = load_json(SONGS_PATH) if SONGS_PATH.exists() else []
    venues = load_json(VENUES_PATH) if VENUES_PATH.exists() else {}
    stats = load_json(FEATURES_DIR / "song_stats.json")
    gaps = load_json(FEATURES_DIR / "song_gaps.json")
    transition = load_json(FEATURES_DIR / "transition_matrix.json")
    venue_history = load_json(FEATURES_DIR / "venue_history.json")
    weights = load_json(MODELS_DIR / "ensemble_weights.json")

    xgb = pickle.loads((MODELS_DIR / "xgboost_song_selector.pkl").read_bytes())
    cal_path = MODELS_DIR / "calibrator.pkl"
    calibrator = pickle.loads(cal_path.read_bytes()) if cal_path.exists() else None

    return _Artifacts(
        shows=shows,
        venues=venues,
        stats=stats,
        gaps=gaps,
        transition=transition,
        venue_history=venue_history,
        weights=weights,
        cover_set={s["name"] for s in songs_catalog if not s.get("is_original", True)},
        markov=markov_score_per_song(transition),
        gap_scores=gap_score_per_song(stats, gaps),
        xgb=xgb,
        calibrator=calibrator,
    )


def _score_candidates(
    state: StreamingState, show: Show, candidates: list[str], art: _Artifacts,
) -> list[dict]:
    X = np.array(
        [featurize(state, show, song, art.cover_set) for song in candidates],
        dtype=np.float32,
    )
    xgb_probs = art.xgb.predict_proba(X)[:, 1]
    if art.calibrator is not None:
        xgb_probs = art.calibrator.predict_proba(xgb_probs.reshape(-1, 1))[:, 1]

    venue_freq = art.venue_history.get(show["venue_id"], {}).get("song_freq", {})
    xgb_n = min_max_normalize(list(xgb_probs))
    m_n = min_max_normalize([art.markov.get(s, 0.0) for s in candidates])
    g_n = min_max_normalize([art.gap_scores.get(s, 0.0) for s in candidates])
    v_n = min_max_normalize([venue_freq.get(s, 0.0) for s in candidates])

    w = art.weights
    wx, wm, wg, wv = w["w_xgboost"], w["w_markov"], w["w_gap"], w["w_venue"]
    last_show = state.last_show_set
    last_3 = {s for show_set in state.last_3_shows for s in show_set}

    scored: list[dict] = []
    for i, song in enumerate(candidates):
        if song in last_show:
            continue
        ensemble = wx * xgb_n[i] + wm * m_n[i] + wg * g_n[i] + wv * v_n[i]
        if song in last_3:
            ensemble *= LAST_3_PENALTY
        scored.append({
            "song": song,
            "ensemble": ensemble,
            "confidence": float(xgb_probs[i]),
            "gap": art.gaps.get(song, {}).get("gap", 0),
        })
    scored.sort(key=lambda s: s["ensemble"], reverse=True)
    return scored


def _assemble_setlist(
    scored: list[dict], stats: dict[str, SongStats], transition: TransitionMatrix,
) -> dict[str, list[PredictionItem]]:
    buckets = _split_by_typical_set(stats, [s["song"] for s in scored])
    score_lookup = {s["song"]: s["ensemble"] for s in scored}
    confidence_lookup = {s["song"]: s["confidence"] for s in scored}
    gap_lookup = {s["song"]: s["gap"] for s in scored}

    used: set[str] = set()
    setlist: dict[str, list[dict]] = {}
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
            {
                "song": song,
                "confidence": confidence_lookup.get(song, 0.0),
                "gap": gap_lookup.get(song, 0),
            }
            for song in sequenced
        ]
    return setlist


def predict(show_date: str, venue: str, city: str | None = None) -> Prediction:
    """Produce a structured 3-set prediction for a given date and venue.

    Loads all model artifacts, replays history through `StreamingState`,
    scores every candidate song with the calibrated XGBoost model blended
    with Markov / gap / venue signals, then assembles a Set 1 / Set 2 /
    Encore lineup ordered by transition probabilities.

    Parameters
    ----------
    show_date
        Target show date in YYYY-MM-DD format.
    venue
        Venue name (fuzzy-matched against the venue catalog).
    city
        Optional city; only used if the venue catalog has no city for the match.

    Returns
    -------
    dict
        Prediction payload: ``date``, ``venue``, ``venue_id``, ``city``,
        ``setlist`` (per-set lists of ``{song, confidence, gap}``),
        ``avg_confidence``, ``model_version``, and ``weights``.

    Raises
    ------
    ValueError
        If ``show_date`` is not a valid YYYY-MM-DD string.
    RuntimeError
        If no candidate songs are available — typically means models
        haven't been trained yet.
    """
    if not VALID_DATE.match(show_date):
        raise ValueError(f"Invalid date format: {show_date!r} (expected YYYY-MM-DD)")

    art = _load_artifacts()
    state = _load_or_replay_state(art.shows)
    show = synthetic_show(show_date, venue, art.venues)
    if city and not show["city"]:
        show["city"] = city

    candidates = [s for s, c in state.plays.items() if c >= MIN_PLAYS_FOR_CANDIDATE]
    if not candidates:
        raise RuntimeError("No song candidates available; have models been trained?")

    scored = _score_candidates(state, show, candidates, art)
    setlist = _assemble_setlist(scored, art.stats, art.transition)

    avg_conf = (
        sum(item["confidence"] for items in setlist.values() for item in items)
        / max(1, sum(len(items) for items in setlist.values()))
    )
    w = art.weights
    return {
        "date": show_date,
        "venue": venue,
        "venue_id": show["venue_id"],
        "city": show.get("city", "") or (city or ""),
        "setlist": setlist,
        "avg_confidence": avg_conf,
        "model_version": w.get("val_year", "unknown"),
        "weights": {
            "xgboost": w["w_xgboost"],
            "markov": w["w_markov"],
            "gap": w["w_gap"],
            "venue": w["w_venue"],
        },
    }


def _format_text(prediction: Prediction) -> str:
    lines = [
        f"Phinish Prediction — {prediction['venue']} — {prediction['date']}",
        f"Avg confidence: {prediction['avg_confidence']:.1%}",
        "",
    ]
    for set_key, label in SET_DISPLAY:
        items = prediction["setlist"].get(set_key, [])
        if not items:
            continue
        lines.append(f"{label}:")
        for i, item in enumerate(items, 1):
            lines.append(
                f"  {i:>2}. {item['song']:<35s}  "
                f"{item['confidence']:>5.1%}  gap={item['gap']}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point: parse args and print a formatted prediction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--venue", required=True)
    parser.add_argument("--city", default=None)
    args = parser.parse_args()

    result = predict(args.date, args.venue, args.city)
    print(_format_text(result))


if __name__ == "__main__":
    main()
