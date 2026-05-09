# Phinish Model Card

**Status:** Trained — last retrained on 2026-05-09T22:36:45 (bootstrap 1983-2000).

## Dataset

1464 shows. Raw setlist data is not committed (see `.gitignore`);
only trained model artifacts are persisted.

## Training

- **Training examples:** 227,260
- **Validation shows:** 115 (year 1999)
- **Val Precision@25:** 35.3%
- **Ensemble weights:** xgboost=0.33 / markov=0.00 / gap=0.00 / venue=0.67

## Test Metrics

Evaluated on temporal holdout (test >= 2000).

| Model | Precision@25 | Recall | F1 | Opener Acc | Pair Match |
|---|---|---|---|---|---|
| Frequency baseline | 6.5% | 9.2% | 7.6% | 0.0% | — |
| Gap-weighted baseline | 12.5% | 22.6% | 15.0% | 2.8% | — |
| XGBoost (solo) | 16.9% | 29.5% | 20.8% | 4.6% | — |
| Markov (solo) | 4.8% | 7.3% | 5.6% | 0.0% | — |
| **Ensemble** | 31.9% | 56.0% | 39.0% | 3.7% | — |

## Hyperparameters

- **XGBoost:** `max_depth=6`, `learning_rate=0.1`, `n_estimators=300`, `scale_pos_weight=12`
- **Markov:** order 2, Laplace smoothing `k=0.01`
- **Calibration:** Platt scaling on validation set (2024)
- **Ensemble:** grid search over `(w_xgb, w_markov, w_gap, w_venue)` optimizing Precision@25

See `models/manifest.json` for the full retrain history.
