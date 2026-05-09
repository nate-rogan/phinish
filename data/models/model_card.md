# Phinish Model Card

**Status:** Trained — last retrained on 2026-05-09T23:50:26 (bootstrap 1983-2015).

## Dataset

1609 shows. Setlist data is committed so incremental retrains
can merge new years into the existing dataset.

## Training

- **Training examples:** 308,796
- **Validation shows:** 42 (year 2014)
- **Val Precision@25:** 51.9%
- **Ensemble weights:** xgboost=0.40 / markov=0.00 / gap=0.00 / venue=0.60

## Test Metrics

Evaluated on temporal holdout (test >= 2015).

| Model | Precision@25 | Recall | F1 | Opener Acc | Pair Match |
|---|---|---|---|---|---|
| Frequency baseline | 13.4% | 16.9% | 14.9% | 0.0% | — |
| Gap-weighted baseline | 19.5% | 24.6% | 21.6% | 0.0% | — |
| XGBoost (solo) | 31.7% | 40.1% | 35.2% | 0.0% | — |
| Markov (solo) | 9.8% | 15.1% | 10.9% | 0.0% | — |
| **Ensemble** | 44.5% | 59.3% | 49.5% | 0.0% | — |

## Hyperparameters

- **XGBoost:** `max_depth=6`, `learning_rate=0.1`, `n_estimators=300`, `scale_pos_weight=12`
- **Markov:** order 2, Laplace smoothing `k=0.01`
- **Calibration:** Platt scaling on validation set (2014)
- **Ensemble:** grid search over `(w_xgb, w_markov, w_gap, w_venue)` optimizing Precision@25

See `models/manifest.json` for the full retrain history.
