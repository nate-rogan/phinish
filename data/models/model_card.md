# Phinish Model Card

**Status:** Trained — last retrained on 2026-05-10T04:48:07 (year 2017).

## Dataset

1638 shows. Setlist data is committed so incremental retrains
can merge new years into the existing dataset.

## Training

- **Training examples:** 334,303
- **Validation:** _no 2016 data yet_
- **Ensemble weights:** xgboost=0.50 / markov=0.10 / gap=0.30 / venue=0.10

## Test Metrics

Evaluated on temporal holdout (test >= 2017).

| Model | Precision@25 | Recall | F1 | Opener Acc | Pair Match |
|---|---|---|---|---|---|
| Frequency baseline | 9.4% | 12.8% | 10.8% | 0.0% | — |
| Gap-weighted baseline | 11.6% | 16.0% | 13.4% | 0.0% | — |
| XGBoost (solo) | 16.8% | 22.9% | 19.3% | 0.0% | — |
| Markov (solo) | 7.9% | 11.5% | 9.1% | 0.0% | — |
| **Ensemble** | 18.9% | 25.8% | 21.7% | 0.0% | — |

## Hyperparameters

- **XGBoost:** `max_depth=6`, `learning_rate=0.1`, `n_estimators=300`, `scale_pos_weight=12`
- **Markov:** order 2, Laplace smoothing `k=0.01`
- **Calibration:** Platt scaling on validation set (2016)
- **Ensemble:** grid search over `(w_xgb, w_markov, w_gap, w_venue)` optimizing Precision@25

See `models/manifest.json` for the full retrain history.
