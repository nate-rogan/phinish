# Phinish Model Card

**Status:** Trained — last retrained on 2026-05-09T13:43:59 (year 2002).

## Dataset

1246 shows. Setlist data is committed so incremental retrains
can merge new years into the existing dataset.

## Training

- **Training examples:** 215,357
- **Validation shows:** 1 (year 2001)
- **Val Precision@25:** 12.0%
- **Ensemble weights:** xgboost=0.00 / markov=0.00 / gap=0.00 / venue=1.00

## Test Metrics

Evaluated on temporal holdout (test >= 2002).

| Model | Precision@25 | Recall | F1 | Opener Acc | Pair Match |
|---|---|---|---|---|---|
| Frequency baseline | 6.7% | 22.2% | 7.9% | 0.0% | — |
| Gap-weighted baseline | 1.3% | 16.7% | 2.5% | 0.0% | — |
| XGBoost (solo) | 10.7% | 26.4% | 12.0% | 0.0% | — |
| Markov (solo) | 6.7% | 22.2% | 7.9% | 0.0% | — |
| **Ensemble** | 13.3% | 29.2% | 14.7% | 0.0% | — |

## Hyperparameters

- **XGBoost:** `max_depth=6`, `learning_rate=0.1`, `n_estimators=300`, `scale_pos_weight=12`
- **Markov:** order 2, Laplace smoothing `k=0.01`
- **Calibration:** Platt scaling on validation set (2001)
- **Ensemble:** grid search over `(w_xgb, w_markov, w_gap, w_venue)` optimizing Precision@25

See `models/manifest.json` for the full retrain history.
