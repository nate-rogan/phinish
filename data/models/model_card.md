# Phinish Model Card

**Status:** Trained — last retrained on 2026-05-09T11:27:40 (year 2001).

## Dataset

484 shows scraped fresh per retrain. Raw setlist data is not committed
to the repo (see `.gitignore`); only trained model artifacts are persisted.

## Metrics

Evaluated on temporal holdout (test ≥ 2025).

| Model | Precision@25 | Recall | F1 | Opener Acc | Pair Match |
|---|---|---|---|---|---|
| Frequency baseline | — | — | — | — | — |
| Gap-weighted baseline | — | — | — | — | — |
| XGBoost (solo) | — | — | — | — | — |
| Markov (solo) | — | — | — | — | — |
| **Ensemble** | — | — | — | — | — |

## Hyperparameters

- **XGBoost:** `max_depth=6`, `learning_rate=0.1`, `n_estimators=300`, `scale_pos_weight=12`
- **Markov:** order 2, Laplace smoothing `k=0.01`
- **Calibration:** Platt scaling on validation set (2024)
- **Ensemble:** grid search over `(w_xgb, w_markov, w_gap, w_venue)` optimizing Precision@25

See `models/manifest.json` for the full retrain history.
