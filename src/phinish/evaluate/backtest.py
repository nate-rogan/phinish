"""Backtest each model on the test holdout (>= TEST_START_YEAR).

Writes models/evaluation.json and prints a comparison table.
"""

import numpy as np

from phinish.artifacts import (
    calibrated_predict_proba,
    gap_score_per_song,
    load_calibrator,
    load_cover_set,
    load_ensemble_weights,
    load_shows,
    load_song_gaps,
    load_song_stats,
    load_transition_matrix,
    load_venue_history,
    load_xgb_model,
    markov_score_per_song,
    venue_freq_for_show,
)
from phinish.features.types import SongStats, VenueHistory
from phinish.scrape.types import Show
from phinish.train import MIN_PLAYS_FOR_CANDIDATE, TEST_START_YEAR, StreamingState, featurize
from phinish.utils import (
    MODELS_DIR,
    TOP_K,
    min_max_normalize,
    save_json,
    show_song_set,
)

MODEL_NAMES = ("frequency", "gap_weighted", "xgboost", "markov", "ensemble")


def metrics(predicted: list[str], actual: set[str], k: int = TOP_K) -> dict[str, float]:
    """Compute precision@k, recall, and F1 for one show's top-k prediction."""
    top = predicted[:k]
    if not top:
        return {"precision_at_25": 0.0, "recall": 0.0, "f1": 0.0}
    hits = sum(1 for s in top if s in actual)
    precision = hits / k
    recall = hits / len(actual) if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision_at_25": precision, "recall": recall, "f1": f1}


def _rank_all_models(
    state: StreamingState,
    show: Show,
    candidates: list[str],
    cover_set: set[str],
    xgb,
    calibrator,
    stats: dict[str, SongStats],
    markov: dict[str, float],
    gap_scores: dict[str, float],
    venue_history: dict[str, VenueHistory],
    weights: tuple[float, float, float, float],
) -> dict[str, list[str]]:
    """Rank candidates by each model and return ``{model_name: ranked_songs}``."""
    X = np.array(
        [featurize(state, show, s, cover_set) for s in candidates],
        dtype=np.float32,
    )
    xgb_probs = calibrated_predict_proba(xgb, calibrator, X)
    venue_freq = venue_freq_for_show(venue_history, show.venue_id)
    xgb_pairs = sorted(
        zip(xgb_probs, candidates, strict=True),
        reverse=True, key=lambda t: t[0],
    )
    xgb_n = min_max_normalize(list(xgb_probs))
    m_n = min_max_normalize([markov.get(s, 0.0) for s in candidates])
    g_n = min_max_normalize([gap_scores.get(s, 0.0) for s in candidates])
    v_n = min_max_normalize([venue_freq.get(s, 0.0) for s in candidates])
    wx, wm, wg, wv = weights
    ens_pairs = sorted(
        zip(
            (wx * xgb_n[i] + wm * m_n[i] + wg * g_n[i] + wv * v_n[i]
             for i in range(len(candidates))),
            candidates,
            strict=True,
        ),
        reverse=True, key=lambda t: t[0],
    )
    return {
        "frequency": sorted(
            candidates,
            key=lambda s: stats[s].lifetime_frequency if s in stats else 0.0,
            reverse=True,
        ),
        "gap_weighted": sorted(
            candidates, key=lambda s: gap_scores.get(s, 0.0), reverse=True,
        ),
        "xgboost": [c for _, c in xgb_pairs],
        "markov": sorted(candidates, key=lambda s: markov.get(s, 0.0), reverse=True),
        "ensemble": [c for _, c in ens_pairs],
    }


def _summarize(
    results: dict[str, list[dict]],
    opener_correct: dict[str, int],
    opener_total: int,
) -> dict[str, dict]:
    """Average per-show metrics into a summary dict keyed by model name."""
    summary: dict[str, dict] = {}
    for name in MODEL_NAMES:
        rs = results[name]
        if not rs:
            summary[name] = {"n_shows": 0}
            continue
        avg = {k: sum(r[k] for r in rs) / len(rs) for k in rs[0]}
        avg["opener_accuracy"] = opener_correct[name] / opener_total if opener_total else 0.0
        avg["n_shows"] = len(rs)
        summary[name] = avg
    return summary


def _print_summary(summary: dict[str, dict]) -> None:
    """Print a compact comparison table of all model metrics to stdout."""
    n = summary.get("ensemble", {}).get("n_shows", 0)
    print(f"\nEvaluation on {n} shows since {TEST_START_YEAR}:\n")
    for name in MODEL_NAMES:
        m = summary[name]
        if m.get("n_shows", 0) > 0:
            print(f"  {name:14s}  P@25={m['precision_at_25']:.3f}  R={m['recall']:.3f}  "
                  f"F1={m['f1']:.3f}  Opener={m['opener_accuracy']:.3f}")


def main() -> None:
    """Backtest each model on the test holdout and write models/evaluation.json."""
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
    set_openers_1 = transition.set_openers.get("1", {})
    weights = load_ensemble_weights()
    w = (weights.w_xgboost, weights.w_markov, weights.w_gap, weights.w_venue)

    state = StreamingState()
    results: dict[str, list[dict]] = {m: [] for m in MODEL_NAMES}
    opener_correct = dict.fromkeys(MODEL_NAMES, 0)
    opener_total = 0

    for show in shows:
        played = show_song_set(show)
        set1 = show.sets.get("1", [])
        actual_opener = set1[0].song if set1 else None

        if show.year >= TEST_START_YEAR:
            candidates = [s for s, c in state.plays.items() if c >= MIN_PLAYS_FOR_CANDIDATE]
            if candidates:
                rankings = _rank_all_models(
                    state, show, candidates, cover_set,
                    xgb, calibrator, stats, markov, gap_scores, venue_history, w,
                )
                for name, ranked in rankings.items():
                    results[name].append(metrics(ranked, played))
                    if actual_opener:
                        pred_opener = max(
                            ranked[:TOP_K],
                            key=lambda s: set_openers_1.get(s, 0.0),
                            default=None,
                        )
                        if pred_opener == actual_opener:
                            opener_correct[name] += 1
                if actual_opener:
                    opener_total += 1
        state.update(show)

    summary = _summarize(results, opener_correct, opener_total)
    save_json(MODELS_DIR / "evaluation.json", {
        "test_start_year": TEST_START_YEAR,
        "summary": summary,
    })
    _print_summary(summary)


if __name__ == "__main__":
    main()
