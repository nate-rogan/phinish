# Phinish Model Card

**Status:** Trained — last retrained on 2026-05-09T23:27:48 (bootstrap 1983-2000 (Phish only)).

## Dataset

1242 shows. Setlist data is committed so incremental retrains
can merge new years into the existing dataset.

## Training

- **Training examples:** 179,572
- **Validation shows:** 69 (year 1999)
- **Val Precision@25:** 45.9%
- **Ensemble weights:** xgboost=0.33 / markov=0.00 / gap=0.00 / venue=0.67

## Test Metrics

Evaluated on temporal holdout (test >= 2000).

| Model | Precision@25 | Recall | F1 | Opener Acc | Pair Match |
|---|---|---|---|---|---|
| Frequency baseline | 12.1% | 17.0% | 14.1% | 0.0% | — |
| Gap-weighted baseline | 18.4% | 32.2% | 22.1% | 5.2% | — |
| XGBoost (solo) | 25.0% | 42.1% | 29.6% | 3.4% | — |
| Markov (solo) | 8.4% | 11.9% | 9.8% | 0.0% | — |
| **Ensemble** | 46.6% | 75.5% | 55.4% | 6.9% | — |

## Hyperparameters

- **XGBoost:** `max_depth=6`, `learning_rate=0.1`, `n_estimators=300`, `scale_pos_weight=12`
- **Markov:** order 2, Laplace smoothing `k=0.01`
- **Calibration:** Platt scaling on validation set (1999)
- **Ensemble:** grid search over `(w_xgb, w_markov, w_gap, w_venue)` optimizing Precision@25

See `models/manifest.json` for the full retrain history.
