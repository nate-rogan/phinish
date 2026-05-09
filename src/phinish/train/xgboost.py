"""Train the XGBoost song selector with temporal split + Platt calibration.

Writes:
  models/xgboost_song_selector.pkl
  models/calibrator.pkl
  models/xgboost_meta.json
  models/state.pkl  (final StreamingState, used by inference)
"""

import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from phinish.artifacts import load_cover_set, load_shows
from phinish.scrape.types import Show
from phinish.train.state import (
    MIN_PLAYS_FOR_CANDIDATE,
    StreamingState,
    feature_names,
    featurize,
    split_years,
)
from phinish.utils import (
    MODELS_DIR,
    SETLISTS_PATH,
    STATE_SNAPSHOT_PATH,
    save_json,
    show_song_set,
)

log = structlog.get_logger()

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


def build_training_matrix(
    shows: list[Show],
    cover_set: set[str],
    min_year: int,
    max_year: int,
) -> tuple[np.ndarray, np.ndarray, list[str], StreamingState]:
    """Replay history and emit (X, y, dates) for shows in ``[min_year, max_year]``."""
    state = StreamingState()
    rows: list[list[float]] = []
    labels: list[int] = []
    show_dates: list[str] = []

    for show in shows:
        year = show.year
        played = show_song_set(show)
        if min_year <= year <= max_year:
            candidates = [s for s, c in state.plays.items() if c >= MIN_PLAYS_FOR_CANDIDATE]
            for song in candidates:
                rows.append(featurize(state, show, song, cover_set))
                labels.append(1 if song in played else 0)
                show_dates.append(show.date)
        state.update(show)

    return np.array(rows, dtype=np.float32), np.array(labels, dtype=np.int8), show_dates, state


@dataclass(slots=True)
class XGBoostResult:
    """All outputs of XGBoost training."""

    model: Any
    calibrator: Any | None
    state: StreamingState
    meta: dict


def train_xgboost(shows: list[Show], cover_set: set[str]) -> XGBoostResult:
    """Train XGBoost song selector with temporal split and Platt calibration.

    Parameters
    ----------
    shows
        Full historical show list.
    cover_set
        Set of song names considered covers.

    Returns
    -------
    XGBoostResult
        Trained model, calibrator, final streaming state, and metadata.
    """
    max_year = max(s.year for s in shows)
    train_end, val_year, _ = split_years(max_year)

    log.info("building_train_matrix", end_year=train_end)
    X_train, y_train, _, _ = build_training_matrix(shows, cover_set, 1983, train_end)
    log.info("train_matrix_built", shape=X_train.shape, positives=int(y_train.sum()))

    log.info("building_val_matrix", year=val_year)
    X_val, y_val, _, final_state = build_training_matrix(shows, cover_set, val_year, val_year)
    log.info("val_matrix_built", shape=X_val.shape, positives=int(y_val.sum()))

    if X_train.size == 0:
        raise RuntimeError("No training rows generated. Check setlists data.")

    log.info("fitting_xgboost")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    if X_val.size > 0:
        log.info("fitting_platt_calibration")
        raw_val = model.predict_proba(X_val)[:, 1]
        calibrator = LogisticRegression()
        calibrator.fit(raw_val.reshape(-1, 1), y_val)
    else:
        calibrator = None
        log.warning("skipping_calibration", reason="no validation rows")

    meta = {
        "feature_names": feature_names(),
        "n_train_rows": int(X_train.shape[0]),
        "n_val_rows": int(X_val.shape[0]),
        "params": {k: v for k, v in XGB_PARAMS.items() if isinstance(v, (int, float, str, bool))},
        "min_plays_for_candidate": MIN_PLAYS_FOR_CANDIDATE,
        "train_end_year": train_end,
        "val_year": val_year,
    }
    return XGBoostResult(model=model, calibrator=calibrator, state=final_state, meta=meta)


def main() -> None:
    """Console entry point: load data, train XGBoost, write artifacts to disk."""
    if not SETLISTS_PATH.exists():
        raise SystemExit(f"Missing {SETLISTS_PATH}; run phinish-scrape first.")
    result = train_xgboost(load_shows(), load_cover_set())
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "xgboost_song_selector.pkl").write_bytes(pickle.dumps(result.model))
    (MODELS_DIR / "calibrator.pkl").write_bytes(pickle.dumps(result.calibrator))
    STATE_SNAPSHOT_PATH.write_bytes(pickle.dumps(result.state))
    save_json(MODELS_DIR / "xgboost_meta.json", result.meta)
    log.info("wrote_xgboost_model", path=str(MODELS_DIR / "xgboost_song_selector.pkl"))


if __name__ == "__main__":
    main()
