# Phinish Model Card

**Status:** Trained — last retrained on 2026-05-09T12:44:39 (year 2002).

## Dataset

829 shows. Setlist data is committed so incremental retrains
can merge new years into the existing dataset.

## Training

- **Training examples:** 197,323
- **Validation:** _no 2020 data yet_
- **Ensemble weights:** xgboost=0.50 / markov=0.10 / gap=0.30 / venue=0.10

## Test Metrics

Evaluated on temporal holdout (test >= 2021).

| Model | Precision@25 | Recall | F1 | Opener Acc | Pair Match |
|---|---|---|---|---|---|
| Frequency baseline | 11.0% | 14.8% | 12.5% | 0.0% | — |
| Gap-weighted baseline | 13.1% | 19.2% | 15.0% | 0.0% | — |
| XGBoost (solo) | 12.3% | 15.9% | 13.8% | 0.0% | — |
| Markov (solo) | 8.7% | 12.3% | 10.0% | 0.0% | — |
| **Ensemble** | 24.1% | 33.8% | 27.3% | 0.0% | — |

## Hyperparameters

- **XGBoost:** `max_depth=6`, `learning_rate=0.1`, `n_estimators=300`, `scale_pos_weight=12`
- **Markov:** order 2, Laplace smoothing `k=0.01`
- **Calibration:** Platt scaling on validation set (2020)
- **Ensemble:** grid search over `(w_xgb, w_markov, w_gap, w_venue)` optimizing Precision@25

See `models/manifest.json` for the full retrain history.
