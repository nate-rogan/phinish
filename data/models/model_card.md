# Phinish Model Card

**Status:** Untrained — awaiting first run of `pixi run python scripts/process_year.py` (or end-to-end retrain via the `process` workflow).

## Versions

| Component | Version | Updated |
|---|---|---|
| Dataset | — | — |
| Gap-weighted baseline | — | — |
| XGBoost song selector | — | — |
| Markov chain (order 2) | — | — |
| Ensemble weights | — | — |

## Training Data

Will be populated after first scrape: ~2,100 Phish shows from 1983–present, sourced from the Phish.net API v5.

## Metrics

Populated by `scripts/evaluate.py` on temporal holdout (train < 2024, validate = 2024, test ≥ 2025).

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
- **Calibration:** Platt scaling on validation set
- **Ensemble:** grid search over `(w_xgb, w_markov, w_gap, w_venue)` optimizing Precision@25

## Changelog

- 2026-05-02: Initial structure committed; no models trained yet.
