"""Backtest each model on the test holdout (>= TEST_START_YEAR).

Writes models/evaluation.json and prints a comparison table.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_ensemble import gap_score_per_song, markov_score_per_song
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

TOP_K = 25
TEST_START_YEAR = 2025


def metrics(predicted: list[str], actual: set[str], k: int = TOP_K) -> dict[str, float]:
    top = predicted[:k]
    if not top:
        return {"precision_at_25": 0.0, "recall": 0.0, "f1": 0.0}
    hits = sum(1 for s in top if s in actual)
    precision = hits / k
    recall = hits / len(actual) if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision_at_25": precision, "recall": recall, "f1": f1}


def _norm(values: list[float]) -> list[float]:
    if not values:
        return values
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1.0
    return [(v - lo) / rng for v in values]


def main() -> None:
    shows = load_json(SETLISTS_PATH)
    songs_catalog = load_json(SONGS_PATH) if SONGS_PATH.exists() else []
    cover_set = {s["name"] for s in songs_catalog if not s.get("is_original", True)}

    with open(MODELS_DIR / "xgboost_song_selector.pkl", "rb") as f:
        xgb = pickle.load(f)
    try:
        with open(MODELS_DIR / "calibrator.pkl", "rb") as f:
            calibrator = pickle.load(f)
    except FileNotFoundError:
        calibrator = None

    transition = load_json(FEATURES_DIR / "transition_matrix.json")
    venue_history = load_json(FEATURES_DIR / "venue_history.json")
    stats = load_json(FEATURES_DIR / "song_stats.json")
    gaps = load_json(FEATURES_DIR / "song_gaps.json")
    markov = markov_score_per_song(transition)
    gap_scores = gap_score_per_song(stats, gaps)
    set_openers_1 = transition.get("set_openers", {}).get("1", {})

    weights = load_json(MODELS_DIR / "ensemble_weights.json")
    wx, wm, wg, wv = weights["w_xgboost"], weights["w_markov"], weights["w_gap"], weights["w_venue"]

    state = StreamingState()
    model_names = ("frequency", "gap_weighted", "xgboost", "markov", "ensemble")
    results: dict[str, list[dict]] = {m: [] for m in model_names}
    opener_correct = {m: 0 for m in model_names}
    opener_total = 0

    for show in shows:
        played = show_song_set(show)
        actual_opener = None
        set1 = show.get("sets", {}).get("1", [])
        if set1:
            actual_opener = set1[0]["song"]

        if show.get("year", 0) >= TEST_START_YEAR:
            candidates = [s for s, c in state.plays.items() if c >= MIN_PLAYS_FOR_CANDIDATE]
            if candidates:
                X = np.array(
                    [featurize(state, show, s, cover_set) for s in candidates],
                    dtype=np.float32,
                )
                xgb_probs = xgb.predict_proba(X)[:, 1]
                if calibrator is not None:
                    xgb_probs = calibrator.predict_proba(xgb_probs.reshape(-1, 1))[:, 1]

                vid = show.get("venue_id", "")
                venue_freq = venue_history.get(vid, {}).get("song_freq", {})

                lifetime_freq = lambda s: stats.get(s, {}).get("lifetime_frequency", 0.0)  # noqa: E731
                xgb_pairs = sorted(
                    zip(xgb_probs, candidates, strict=True),
                    reverse=True, key=lambda t: t[0],
                )
                rankings = {
                    "frequency": sorted(candidates, key=lifetime_freq, reverse=True),
                    "gap_weighted": sorted(
                        candidates, key=lambda s: gap_scores.get(s, 0.0), reverse=True,
                    ),
                    "xgboost": [c for _, c in xgb_pairs],
                    "markov": sorted(candidates, key=lambda s: markov.get(s, 0.0), reverse=True),
                }
                xgb_n = _norm(list(xgb_probs))
                m_n = _norm([markov.get(s, 0.0) for s in candidates])
                g_n = _norm([gap_scores.get(s, 0.0) for s in candidates])
                v_n = _norm([venue_freq.get(s, 0.0) for s in candidates])
                ens_scores = [
                    wx * xgb_n[i] + wm * m_n[i] + wg * g_n[i] + wv * v_n[i]
                    for i in range(len(candidates))
                ]
                ens_pairs = sorted(
                    zip(ens_scores, candidates, strict=True),
                    reverse=True, key=lambda t: t[0],
                )
                rankings["ensemble"] = [c for _, c in ens_pairs]

                for name, ranked in rankings.items():
                    results[name].append(metrics(ranked, played))
                    if actual_opener:
                        pred_opener = max(ranked[:TOP_K],
                                          key=lambda s: set_openers_1.get(s, 0.0),
                                          default=None)
                        if pred_opener == actual_opener:
                            opener_correct[name] += 1

                if actual_opener:
                    opener_total += 1
        state.update(show)

    summary: dict[str, dict] = {}
    for name in model_names:
        rs = results[name]
        if not rs:
            summary[name] = {"n_shows": 0}
            continue
        avg = {k: sum(r[k] for r in rs) / len(rs) for k in rs[0]}
        avg["opener_accuracy"] = opener_correct[name] / opener_total if opener_total else 0.0
        avg["n_shows"] = len(rs)
        summary[name] = avg

    save_json(MODELS_DIR / "evaluation.json", {
        "test_start_year": TEST_START_YEAR,
        "summary": summary,
    })

    n = summary.get("ensemble", {}).get("n_shows", 0)
    print(f"\nEvaluation on {n} shows since {TEST_START_YEAR}:\n")
    for name in model_names:
        m = summary[name]
        if m.get("n_shows", 0) > 0:
            print(f"  {name:14s}  P@25={m['precision_at_25']:.3f}  R={m['recall']:.3f}  "
                  f"F1={m['f1']:.3f}  Opener={m['opener_accuracy']:.3f}")


if __name__ == "__main__":
    main()
