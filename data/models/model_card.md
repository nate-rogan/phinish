# Phinish Model Card

**Status:** Trained — last retrained on 2026-05-09T13:31:44 (year 2001).

## Dataset

1243 shows. Setlist data is committed so incremental retrains
can merge new years into the existing dataset.

## Training

- **Training examples:** 198,828
- **Validation shows:** 58 (year 2000)
- **Val Precision@25:** 47.3%
- **Ensemble weights:** xgboost=0.33 / markov=0.00 / gap=0.00 / venue=0.67

## Test Metrics

Evaluated on temporal holdout (test >= 2001).

| Model | Precision@25 | Recall | F1 | Opener Acc | Pair Match |
|---|---|---|---|---|---|
| Frequency baseline | 0.0% | 0.0% | 0.0% | 0.0% | — |
| Gap-weighted baseline | 0.0% | 0.0% | 0.0% | 0.0% | — |
| XGBoost (solo) | 4.0% | 33.3% | 7.1% | 0.0% | — |
| Markov (solo) | 0.0% | 0.0% | 0.0% | 0.0% | — |
| **Ensemble** | 12.0% | 100.0% | 21.4% | 0.0% | — |

## Hyperparameters

- **XGBoost:** `max_depth=6`, `learning_rate=0.1`, `n_estimators=300`, `scale_pos_weight=12`
- **Markov:** order 2, Laplace smoothing `k=0.01`
- **Calibration:** Platt scaling on validation set (2000)
- **Ensemble:** grid search over `(w_xgb, w_markov, w_gap, w_venue)` optimizing Precision@25

See `models/manifest.json` for the full retrain history.
