# 🎸 Phinish

**ML-powered Phish setlist predictions that run themselves from a GitHub repo.**

Issues are the API. Actions are the compute. The repo is the database.

[**→ Try it: Request a Prediction**](../../issues/new?assignees=&labels=predict&projects=&template=predict-show.yml) · [**Browse All Predictions**](../../issues?q=label%3Apredicted) · [**Dashboard**](https://YOUR_USERNAME.github.io/phinish)

---

## How It Works

```
You open a GitHub Issue          →  "Predict: MSG — 2026-12-31"
                                           ↓
GitHub Actions fires             →  Loads trained models, scores 300+ songs
                                           ↓
Ensemble prediction posted       →  Set 1, Set 2, Encore with confidence %
                                           ↓
Issue closed, dashboard updated  →  Results visible on the live dashboard
```

The entire system — data ingestion, feature engineering, model training, inference, and serving — runs from this repository. No servers. No containers. No cloud accounts.

**After submitting a prediction request**, watch it run in real-time on the [Actions tab](../../actions). The prediction typically posts within 30 seconds. Dashboard data refreshes within ~5 minutes (GitHub CDN cache).

---

## Model Performance

Evaluated on held-out shows using temporal train/test split (no future data leakage):

| Model | Precision@25 | Opener Correct | Pair Matches |
|---|---|---|---|
| Frequency Baseline | 42% | 8% | 2% |
| Gap-Weighted Baseline | 58% | 15% | 4% |
| XGBoost (solo) | 61% | 22% | 8% |
| Markov Chain (solo) | — | 18% | 15% |
| **Ensemble** | **64%** | **28%** | **18%** |

The strongest single predictor is the **rotation gap** — Phish almost never repeats a song within 2–3 shows. A song that's 20 shows overdue is significantly more likely to appear. The XGBoost model learns interactions between gap, venue history, tour position, and 25+ other features. The Markov chain captures sequential flow — it knows Mike's Song leads to Hydrogen 65% of the time.

---

## Try It

### Request a Prediction

1. Click [**Request a Prediction**](../../issues/new?assignees=&labels=predict&projects=&template=predict-show.yml)
2. Enter a show date and venue
3. Submit the issue
4. Watch the [Actions tab](../../actions) — prediction posts as a comment in ~30 seconds
5. Issue is automatically labeled `predicted` and closed

### Trigger a Retrain

1. Click [**Process a Year**](../../issues/new?assignees=&labels=process&projects=&template=process-year.yml)
2. Enter a year (e.g., `2025`)
3. Submit — the system scrapes that year's shows from Phish.net, rebuilds all features, retrains every model, and commits the updated artifacts
4. The entire ML pipeline runs in ~2–3 minutes

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    GitHub Repository                  │
│                                                      │
│  data/processed/     Canonical dataset (~2,100 shows)│
│  state/features/     Precomputed feature store       │
│  models/             Trained model artifacts         │
│  scripts/            Pipeline code                   │
│  docs/               Dashboard (GitHub Pages)        │
│                                                      │
└──────┬──────────────────┬────────────────────────────┘
       │                  │
  Issue: predict     Issue: process
       │                  │
       ▼                  ▼
  Load models,       Scrape → features →
  run ensemble,      retrain all models →
  post comment,      commit artifacts,
  close issue        close issue
```

---

## Project Structure

```
phinish/
├── .github/workflows/
│   ├── ci.yml              # Lint + test on push
│   ├── predict.yml         # Issue-triggered inference
│   └── process.yml         # Issue-triggered retrain
├── data/processed/         # Canonical setlist data
├── state/features/         # Precomputed feature store
├── models/                 # Trained model artifacts + model card
├── scripts/
│   ├── scrape.py           # Phish.net data collection
│   ├── build_features.py   # Feature engineering
│   ├── train_baseline.py   # Frequency + gap baselines
│   ├── train_xgboost.py    # Song selection model
│   ├── train_markov.py     # Transition model
│   ├── train_ensemble.py   # Weight optimization
│   ├── predict.py          # Inference pipeline + CLI
│   ├── evaluate.py         # Backtesting + metrics
│   ├── process_issue.py    # Actions: prediction handler
│   └── process_year.py     # Actions: retrain handler
├── docs/index.html         # Dashboard
├── tests/                  # pytest suite
├── SPEC.md                 # Full system design
└── PLAN.md                 # Build plan
```

---

## Design Decisions

**Why not a neural network?**
Setlist prediction is a structured sequence problem with ~2,100 training examples and ~300 candidate items. Classical ML (XGBoost for selection, Markov chains for sequencing) dominates on tabular data at this scale. A transformer would overfit. Gradient boosted trees give interpretable feature importance and train in under a minute.

**Why not a server/API?**
The system has no sustained load — predictions are requested a few times a day. GitHub Actions provides free on-demand compute triggered by events (issues). Running a 24/7 server to handle occasional requests is wasteful. The repo-as-infrastructure pattern eliminates all ops burden.

**Why ensemble over a single model?**
Each model captures a different signal. XGBoost learns multi-feature interactions (gap × venue × season). Markov chains capture sequential flow (song A → song B). The gap heuristic provides a strong rotation prior. Combining them with learned weights consistently outperforms any individual model.

**Why recompute gaps instead of using the API's pre-computed values?**
The API gives the *current* gap. Training requires the gap *at the time of each historical show*. To avoid future data leakage, we recompute all gap values from the raw setlist sequence.

---

## Running Locally

Requires [pixi](https://pixi.sh).

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/phinish.git
cd phinish

# Install (creates .pixi/ with conda-forge env)
pixi install

# Predict
pixi run python scripts/predict.py --date 2026-12-31 --venue "Madison Square Garden"

# Retrain (after adding new data)
pixi run python scripts/train_baseline.py
pixi run python scripts/train_xgboost.py
pixi run python scripts/train_markov.py
pixi run python scripts/train_ensemble.py
pixi run python scripts/evaluate.py
```

---

## Full Documentation

- **[SPEC.md](SPEC.md)** — complete system specification: architecture, data schemas, model details, security model, evaluation framework
- **[PLAN.md](PLAN.md)** — phased build plan
- **[models/model_card.md](models/model_card.md)** — model version, training data, metrics, changelog

---

## Attribution

Setlist data provided by [Phish.net](https://phish.net), a project of the non-profit Mockingbird Foundation. This project is not affiliated with Phish or Phish.net.
