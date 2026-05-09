"""Optimize ensemble weights via grid search on validation set.

Loads xgb + calibrator, replays history through StreamingState up to validation
year, computes per-song scores from each component model, and grid-searches
weights to maximize average Precision@25 on the validation shows.

Writes models/ensemble_weights.json.
"""

from itertools import product

import numpy as np
import structlog

from phinish.artifacts import (
    calibrated_predict_proba,
    gap_score_per_song,
    load_calibrator,
    load_cover_set,
    load_shows,
    load_song_gaps,
    load_song_stats,
    load_transition_matrix,
    load_venue_history,
    load_xgb_model,
    markov_score_per_song,
    precision_at_k,
    venue_freq_for_show,
)
from phinish.features.types import VenueHistory
from phinish.scrape.types import Show
from phinish.train.state import MIN_PLAYS_FOR_CANDIDATE, VAL_YEAR, StreamingState, featurize
from phinish.utils import MODELS_DIR, min_max_normalize, save_json, show_song_set

log = structlog.get_logger()

GRID = (0.0, 0.25, 0.5, 0.75, 1.0)  # coarse grid; finer adds 625->3125 combos for <0.01 gain
SCORE_DIMS = ("xgb", "markov", "gap", "venue")
DEFAULT_WEIGHTS = {"w_xgboost": 0.5, "w_markov": 0.1, "w_gap": 0.3, "w_venue": 0.1}


def _normalize_per_record(records: list[dict]) -> None:
    """Min-max normalize each score dimension within every validation record."""
    for r in records:
        for dim in SCORE_DIMS:
            normed = min_max_normalize([s[dim] for s in r["scores"]])
            for s, n in zip(r["scores"], normed, strict=True):
                s[dim + "_n"] = n


def _build_val_records(
    shows: list[Show],
    cover_set: set[str],
    xgb,
    calibrator,
    venue_history: dict[str, VenueHistory],
    markov: dict[str, float],
    gap_scores: dict[str, float],
) -> list[dict]:
    """Replay history and collect candidate score vectors for validation shows.

    Parameters
    ----------
    shows
        Chronologically ordered show history used to build streaming state.
    cover_set
        Song names treated as covers during feature construction.
    xgb
        Trained XGBoost classifier used to produce per-song base probabilities.
    calibrator
        Optional probability calibrator fit on validation probabilities.
    venue_history
        Per-venue historical song frequencies keyed by venue ID.
    markov
        Global Markov transition prior score per song.
    gap_scores
        Global gap-based prior score per song.

    Returns
    -------
    list[dict]
        Validation records. Each record contains ``actual`` (songs played at the
        show) and ``scores`` (per-candidate component model scores).
    """
    state = StreamingState()
    val_records: list[dict] = []
    for show in shows:
        played = show_song_set(show)
        if show.year == VAL_YEAR:
            candidates = [s for s, c in state.plays.items() if c >= MIN_PLAYS_FOR_CANDIDATE]
            if candidates:
                X = np.array(
                    [featurize(state, show, song, cover_set) for song in candidates],
                    dtype=np.float32,
                )
                raw = calibrated_predict_proba(xgb, calibrator, X)
                venue_freq = venue_freq_for_show(venue_history, show.venue_id)
                scores = [
                    {
                        "song": song,
                        "xgb": float(raw[i]),
                        "markov": markov.get(song, 0.0),
                        "gap": gap_scores.get(song, 0.0),
                        "venue": venue_freq.get(song, 0.0),
                    }
                    for i, song in enumerate(candidates)
                ]
                val_records.append({"actual": list(played), "scores": scores})
        state.update(show)
    return val_records


def _grid_search_weights(
    val_records: list[dict],
) -> tuple[tuple[float, float, float, float], float]:
    """Search the weight grid for the best average Precision@25.

    Parameters
    ----------
    val_records
        Output from ``_build_val_records`` after normalization, with one score
        vector per candidate song and one ``actual`` song set per show.

    Returns
    -------
    tuple[tuple[float, float, float, float], float]
        The best normalized ``(w_xgboost, w_markov, w_gap, w_venue)`` tuple and
        its corresponding average Precision@25 over validation shows.
    """
    best_score = -1.0
    best_weights = (1.0, 0.0, 0.0, 0.0)
    for w in product(GRID, repeat=4):
        total = sum(w)
        if total == 0:
            continue
        wx, wm, wg, wv = (x / total for x in w)
        precisions = []
        for r in val_records:
            scored = [
                (
                    s["song"],
                    wx * s["xgb_n"] + wm * s["markov_n"] + wg * s["gap_n"] + wv * s["venue_n"],
                )
                for s in r["scores"]
            ]
            scored.sort(key=lambda pair: pair[1], reverse=True)
            precisions.append(precision_at_k([song for song, _ in scored], set(r["actual"])))
        avg = sum(precisions) / len(precisions)
        if avg > best_score:
            best_score = avg
            best_weights = (wx, wm, wg, wv)
    return best_weights, best_score


def train_ensemble(
    shows: list[Show],
    cover_set: set[str],
    xgb,
    calibrator,
    venue_history: dict[str, VenueHistory],
    markov: dict[str, float],
    gap_scores: dict[str, float],
) -> dict:
    """Fit ensemble blending weights via grid search on validation shows.

    Parameters
    ----------
    shows
        Full historical show list (chronological).
    cover_set
        Set of song names considered covers.
    xgb
        Trained XGBoost classifier.
    calibrator
        Optional Platt calibrator (or ``None``).
    venue_history
        Per-venue historical song frequencies.
    markov
        Global Markov transition prior score per song.
    gap_scores
        Global gap-based prior score per song.

    Returns
    -------
    dict
        Ensemble weights payload with keys ``w_xgboost``, ``w_markov``,
        ``w_gap``, ``w_venue``, ``val_precision_at_25``, ``val_year``,
        ``n_val_shows``.
    """
    val_records = _build_val_records(
        shows,
        cover_set,
        xgb,
        calibrator,
        venue_history,
        markov,
        gap_scores,
    )

    if not val_records:
        log.warning("no_val_shows", year=VAL_YEAR, action="using default weights")
        wx, wm, wg, wv = (
            DEFAULT_WEIGHTS["w_xgboost"],
            DEFAULT_WEIGHTS["w_markov"],
            DEFAULT_WEIGHTS["w_gap"],
            DEFAULT_WEIGHTS["w_venue"],
        )
        return {
            "w_xgboost": wx,
            "w_markov": wm,
            "w_gap": wg,
            "w_venue": wv,
            "val_precision_at_25": None,
            "val_year": VAL_YEAR,
            "n_val_shows": 0,
        }

    _normalize_per_record(val_records)
    best_weights, best_score = _grid_search_weights(val_records)
    wx, wm, wg, wv = best_weights
    log.info(
        "best_weights",
        xgb=f"{wx:.2f}",
        markov=f"{wm:.2f}",
        gap=f"{wg:.2f}",
        venue=f"{wv:.2f}",
        precision_at_25=f"{best_score:.3f}",
    )
    return {
        "w_xgboost": wx,
        "w_markov": wm,
        "w_gap": wg,
        "w_venue": wv,
        "val_precision_at_25": best_score,
        "val_year": VAL_YEAR,
        "n_val_shows": len(val_records),
    }


def main() -> None:
    """Console entry point: load artifacts, fit ensemble weights, write to disk."""
    shows = load_shows()
    cover_set = load_cover_set()
    xgb = load_xgb_model()
    calibrator = load_calibrator()
    transition = load_transition_matrix()
    venue_history = load_venue_history()
    stats = load_song_stats()
    gaps = load_song_gaps()
    markov = markov_score_per_song(transition)
    gap_scores = gap_score_per_song(stats, gaps)

    result = train_ensemble(
        shows,
        cover_set,
        xgb,
        calibrator,
        venue_history,
        markov,
        gap_scores,
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(MODELS_DIR / "ensemble_weights.json", result)
    log.info("wrote_ensemble_weights", path=str(MODELS_DIR / "ensemble_weights.json"))


if __name__ == "__main__":
    main()
