# Phinish Model Card

**Status:** Trained — last retrained on 2026-05-09T13:45:04 (year 2003).

## Dataset

1293 shows. Setlist data is committed so incremental retrains
can merge new years into the existing dataset.

## Training

- **Training examples:** 215,642
- **Validation shows:** 3 (year 2002)
- **Val Precision@25:** 17.3%
- **Ensemble weights:** xgboost=0.00 / markov=0.50 / gap=0.50 / venue=0.00

## Test Metrics

Evaluated on temporal holdout (test >= 2003).

| Model | Precision@25 | Recall | F1 | Opener Acc | Pair Match |
|---|---|---|---|---|---|
| Frequency baseline | 14.3% | 20.9% | 16.7% | 6.4% | — |
| Gap-weighted baseline | 16.9% | 24.7% | 20.0% | 0.0% | — |
| XGBoost (solo) | 24.0% | 34.5% | 28.2% | 10.6% | — |
| Markov (solo) | 10.6% | 16.0% | 12.4% | 6.4% | — |
| **Ensemble** | 17.4% | 25.1% | 20.5% | 6.4% | — |

## Hyperparameters

- **XGBoost:** `max_depth=6`, `learning_rate=0.1`, `n_estimators=300`, `scale_pos_weight=12`
- **Markov:** order 2, Laplace smoothing `k=0.01`
- **Calibration:** Platt scaling on validation set (2002)
- **Ensemble:** grid search over `(w_xgb, w_markov, w_gap, w_venue)` optimizing Precision@25

See `models/manifest.json` for the full retrain history.
