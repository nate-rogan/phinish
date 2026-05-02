"""Optimize ensemble weights via grid search on validation set.

Loads xgb + calibrator, replays history through StreamingState up to validation
year, computes per-song scores from each component model, and grid-searches
weights to maximize average Precision@25 on the validation shows.

Writes models/ensemble_weights.json.
"""
from __future__ import annotations

import math
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_xgboost import (
    MIN_PLAYS_FOR_CANDIDATE,
    StreamingState,
    featurize,
    show_song_set,
)
from scripts.utils import (
    FEATURES_DIR,
    MODELS_DIR,
    SETLISTS_PATH,
    SONGS_PATH,
    load_json,
    save_json,
)

VAL_YEAR = 2024
TOP_K = 25
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def markov_score_per_song(matrix: dict) -> dict[str, float]:
    """Marginal song score = mean opener probability across set keys."""
    openers = matrix.get("set_openers", {})
    if not openers:
        return {}
    scores: dict[str, float] = {}
    for dist in openers.values():
        for song, p in dist.items():
            scores[song] = scores.get(song, 0.0) + p
    return {song: s / len(openers) for song, s in scores.items()}


def gap_score_per_song(stats: dict, gaps: dict) -> dict[str, float]:
    return {
        song: s.get("recent_frequency_50", 0.0) * math.log(gaps.get(song, {}).get("gap", 1) + 2)
        for song, s in stats.items()
    }


def precision_at_k(predicted: list[str], actual: set[str], k: int = TOP_K) -> float:
    return sum(1 for s in predicted[:k] if s in actual) / k if predicted else 0.0


def _normalize_per_record(records: list[dict]) -> None:
    for r in records:
        for dim in ("xgb", "markov", "gap", "venue"):
            vals = [s[dim] for s in r["scores"]]
            if not vals:
                continue
            lo, hi = min(vals), max(vals)
            rng = (hi - lo) or 1.0
            for s in r["scores"]:
                s[dim + "_n"] = (s[dim] - lo) / rng


def main() -> None:
    shows = load_json(SETLISTS_PATH)
    songs_catalog = load_json(SONGS_PATH) if SONGS_PATH.exists() else []
    cover_set = {s["name"] for s in songs_catalog if not s.get("is_original", True)}

    xgb = pickle.loads((MODELS_DIR / "xgboost_song_selector.pkl").read_bytes())
    calibrator_path = MODELS_DIR / "calibrator.pkl"
    calibrator = pickle.loads(calibrator_path.read_bytes()) if calibrator_path.exists() else None

    transition = load_json(FEATURES_DIR / "transition_matrix.json")
    venue_history = load_json(FEATURES_DIR / "venue_history.json")
    stats = load_json(FEATURES_DIR / "song_stats.json")
    gaps = load_json(FEATURES_DIR / "song_gaps.json")
    markov = markov_score_per_song(transition)
    gap_scores = gap_score_per_song(stats, gaps)

    state = StreamingState()
    val_records: list[dict] = []
    for show in shows:
        played = show_song_set(show)
        if show.get("year", 0) == VAL_YEAR:
            candidates = [s for s, c in state.plays.items() if c >= MIN_PLAYS_FOR_CANDIDATE]
            if candidates:
                X = np.array(
                    [featurize(state, show, song, cover_set) for song in candidates],
                    dtype=np.float32,
                )
                raw = xgb.predict_proba(X)[:, 1]
                if calibrator is not None:
                    raw = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
                vid = show.get("venue_id", "")
                venue_freq = venue_history.get(vid, {}).get("song_freq", {})
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

    if not val_records:
        print(f"No validation shows for year {VAL_YEAR}; using default weights.")
        save_json(MODELS_DIR / "ensemble_weights.json", {
            "w_xgboost": 0.5, "w_markov": 0.1, "w_gap": 0.3, "w_venue": 0.1,
            "val_precision_at_25": None, "val_year": VAL_YEAR, "n_val_shows": 0,
        })
        return

    _normalize_per_record(val_records)

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
                (s["song"], wx * s["xgb_n"] + wm * s["markov_n"]
                            + wg * s["gap_n"] + wv * s["venue_n"])
                for s in r["scores"]
            ]
            scored.sort(key=lambda pair: pair[1], reverse=True)
            precisions.append(precision_at_k([song for song, _ in scored], set(r["actual"])))
        avg = sum(precisions) / len(precisions)
        if avg > best_score:
            best_score = avg
            best_weights = (wx, wm, wg, wv)

    save_json(MODELS_DIR / "ensemble_weights.json", {
        "w_xgboost": best_weights[0],
        "w_markov": best_weights[1],
        "w_gap": best_weights[2],
        "w_venue": best_weights[3],
        "val_precision_at_25": best_score,
        "val_year": VAL_YEAR,
        "n_val_shows": len(val_records),
    })
    print(f"best weights: xgb={best_weights[0]:.2f} markov={best_weights[1]:.2f} "
          f"gap={best_weights[2]:.2f} venue={best_weights[3]:.2f} -> P@25={best_score:.3f}")


if __name__ == "__main__":
    main()
