# Phinish Model Card

**Status:** Trained — last retrained on 2026-05-04 (full scrape, local).

## Dataset

3,840 shows scraped fresh per retrain. Raw setlist data is not committed
to the repo (see `.gitignore`); only trained model artifacts are persisted.

## Metrics

Evaluated on temporal holdout (test ≥ 2025).

| Model | Precision@25 | Recall | F1 | Opener Acc | Pair Match |
|---|---|---|---|---|---|
| Frequency baseline | 12.2% | 14.7% | 13.2% | 0.9% | — |
| Gap-weighted baseline | 12.5% | 15.5% | 13.7% | 3.7% | — |
| XGBoost (solo) | 22.5% | 27.6% | 24.1% | 3.7% | — |
| Markov (solo) | 10.4% | 14.2% | 11.5% | 0.9% | — |
| **Ensemble** | 43.3% | 54.7% | 46.5% | 6.5% | — |

## Hyperparameters

- **XGBoost:** `max_depth=6`, `learning_rate=0.1`, `n_estimators=300`, `scale_pos_weight=12`
- **Markov:** order 2, Laplace smoothing `k=0.01`
- **Calibration:** Platt scaling on validation set (2024)
- **Ensemble:** grid search over `(w_xgb, w_markov, w_gap, w_venue)` optimizing Precision@25

See `models/manifest.json` for the full retrain history.
