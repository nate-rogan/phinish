# Phinish — Build Plan

> 8-day sprint: May 3–10, 2026
> ML-powered Phish setlist predictions, self-running on GitHub

---

## What Ships

A public GitHub repo containing:

1. **Data pipeline** — httpx scraper pulling ~2,100 shows by year from Phish.net API
2. **Feature store** — precomputed gaps, frequencies, transitions, venue stats as versioned JSON
3. **Trained models** — XGBoost song selector + Markov transition chain + gap-weighted baseline + calibrated ensemble
4. **Evaluation framework** — temporal backtesting with head-to-head model comparison
5. **Prediction CLI** — `python scripts/predict.py --date 2026-12-31 --venue "MSG"` → full setlist
6. **GitHub Actions: predict** — issue with `predict` label → model runs → posts prediction comment → closes issue
7. **GitHub Actions: process** — issue with `process` label → scrapes year → retrains all models → commits artifacts → closes issue
8. **Dashboard** — GitHub Pages static site with prediction form, charts, and live data from repo
9. **CI pipeline** — lint + test on every push
10. **Tests** — 8–10 pytest (unit + integration)
11. **SPEC.md** — full system design document
12. **README.md** — architecture, quickstart, "try it" link, design decisions

---

## Day-by-Day

### Day 1 (May 3) — Scaffolding + Data

- [ ] Create `phinish` repo with full directory structure
- [ ] Apply for Phish.net API key
- [ ] Write `pyproject.toml` (pixi workspace + deps + ruff/pytest config), `.gitignore`, `.gitattributes`
- [ ] Write `scripts/scrape.py`:
  - Pull shows, setlists, songs, venues by year using httpx
  - Group per-song rows into per-show setlists
  - Normalize song names via `canonical_names.json`
  - Flag special shows (Halloween, NYE, festivals)
  - Support `--year YYYY` for single-year pulls (used by process workflow)
- [ ] Run full scrape, validate data
- [ ] Commit `data/processed/setlists.json`, `songs.json`, `venues.json`

**Must ship:** Repo exists + `setlists.json` with ~2,100 shows  
**Can defer:** Incremental mode polish

---

### Day 2 (May 4) — Feature Engineering

- [ ] Write `scripts/build_features.py`:
  - Song gaps (recomputed historically, not from API)
  - Song stats: lifetime_frequency, recent_frequency_50/20, set_distribution, opener/closer frequency
  - Transition matrix: order-1 + order-2 with Laplace smoothing
  - Set opener/closer distributions
  - Venue features: song frequency, common openers/closers per venue
- [ ] Output to `state/features/` as JSON
- [ ] Validate: Mike's Song → Hydrogen transition is high probability

**Must ship:** `song_gaps.json` + `song_stats.json` + `transition_matrix.json` populated  
**Can defer:** Venue features (use empty defaults if needed)

---

### Day 3 (May 5) — Models

- [ ] Write `scripts/train_baseline.py`:
  - Frequency baseline (top 25 most played)
  - Gap-weighted baseline: `score = recent_freq_50 * log(gap + 2)`
- [ ] Write `scripts/evaluate.py`:
  - Temporal train/test split (train < 2024, test ≥ 2024)
  - Precision@25, Recall, F1, Opener accuracy, Pair match rate
  - Model comparison table output
- [ ] Establish baseline numbers
- [ ] Write `scripts/train_xgboost.py`:
  - Build training matrix: (show × song) with features → binary label
  - Hand-tuned params: `max_depth=6, lr=0.1, n_estimators=300, scale_pos_weight=12`
  - Platt scaling calibration
  - Export `models/xgboost_song_selector.pkl` + `models/calibrator.pkl`
- [ ] Write `scripts/train_markov.py`:
  - Order-1 and order-2 transition matrices
  - Set openers, closers, break transitions
  - Export `models/markov_order2.json`
- [ ] Backtest all models, verify XGBoost beats baselines
- [ ] Create `models/model_card.md` with metrics

**Must ship:** Three trained models + evaluation showing XGBoost > baseline  
**Can defer:** Detailed model card polish

---

### Day 4 (May 6) — Ensemble + Predict CLI

- [ ] Write `scripts/train_ensemble.py`:
  - Grid search over weights (w_xgb, w_markov, w_gap, w_venue)
  - Optimize Precision@25 on validation set
  - Export `models/ensemble_weights.json`
- [ ] Write `scripts/predict.py`:
  - `predict(date, venue, city)` function returning structured dict
  - Load all models + features
  - Ensemble scoring
  - Constrained setlist construction (set lengths, no repeats, Markov ordering)
  - Per-song calibrated confidence
  - `if __name__ == "__main__"` with argparse for CLI
- [ ] Write `scripts/utils.py` — shared helpers (JSON I/O, venue matching, sanitization)
- [ ] Test: `python scripts/predict.py --date 2026-07-19 --venue "SPAC"` outputs credible setlist
- [ ] Final backtest: ensemble vs individual models

**Must ship:** `predict.py` working from CLI with credible output  
**Can defer:** Ensemble weight optimization (hand-tune if needed)

---

### Day 5 (May 7) — GitHub Actions

- [ ] Write `.github/ISSUE_TEMPLATE/predict-show.yml`
- [ ] Write `.github/ISSUE_TEMPLATE/process-year.yml`
- [ ] Write `.github/ISSUE_TEMPLATE/config.yml` (disable blank issues)
- [ ] Write `scripts/process_issue.py`:
  - Parse issue body for date + venue
  - Validate inputs
  - Rate limit check (daily + per-user)
  - Call `predict()`
  - Format as markdown table
  - Post comment via GitHub API
  - Update `state/latest_prediction.json`
  - Update `state/predictions_log.json`
  - Append to `reasons.md`
  - Update `state/usage.json`
- [ ] Write `scripts/process_year.py`:
  - Parse year from issue body
  - Call `scrape.py --year YYYY`
  - Call `build_features.py`
  - Call `train_baseline.py`, `train_xgboost.py`, `train_markov.py`, `train_ensemble.py`
  - Call `evaluate.py`
  - Post summary comment
- [ ] Write `.github/workflows/predict.yml`
- [ ] Write `.github/workflows/process.yml`
- [ ] Set up repo secrets: `PHISHNET_API_KEY`
- [ ] Create labels: `predict`, `predicted`, `process`, `processed`, `failed`
- [ ] Test end-to-end: file a predict issue → watch Action → get comment

**Must ship:** Filing a `predict` issue → getting a prediction comment back  
**Can defer:** `process.yml` polish (predict is the priority demo)

---

### Day 6 (May 8) — Dashboard

- [ ] Write `docs/index.html`:
  - Prediction form → pre-filled issue URL
  - Stats bar (shows, precision, prediction count)
  - Latest prediction display with confidence badges
  - Overdue songs bar chart
  - Accuracy over time line chart (Chart.js)
  - "How it works" section
  - Link to Actions tab for live workflow monitoring
- [ ] Enable GitHub Pages (Settings → Pages → main branch → /docs)
- [ ] Test: form creates correct issue URL, dashboard loads live data

**Must ship:** Dashboard with form and latest prediction display  
**Can defer:** Accuracy chart, overdue songs chart (show placeholder if no data yet)

---

### Day 7 (May 9) — Tests + CI + README

- [ ] Write `tests/test_features.py`:
  - Gap calculation correctness
  - Transition probabilities sum to ~1.0
  - Song stats schema validation
  - Edge case: song with 0 plays
- [ ] Write `tests/test_predict.py`:
  - No duplicate songs across sets
  - Set lengths within range
  - Confidences between 0 and 1
- [ ] Write `tests/test_integration.py`:
  - Data → features → predict end-to-end
  - Last show exclusion works
- [ ] Write `.github/workflows/ci.yml` (ruff lint → pytest)
- [ ] Push and verify CI green
- [ ] Write `README.md`:
  - One-liner + dashboard screenshot
  - "How it works" diagram
  - **"Try it"** — direct link to prediction issue template
  - Model performance table
  - Project structure
  - Design decisions
  - Link to SPEC.md
  - Note about raw URL cache delay + link to Actions tab
  - "Browse all predictions" filtered issue link
- [ ] Run `ruff check --fix .`
- [ ] Clean up docstrings and type hints

**Must ship:** CI green + README polished  
**Can defer:** Design decisions section (add after submission if needed)

---

### Day 8 (May 10) — Polish + Submit

- [ ] Review repo as the hiring manager would
- [ ] File a test prediction issue, verify full loop works
- [ ] File a test process issue, verify retrain works
- [ ] Verify dashboard loads and renders
- [ ] Check README renders on GitHub
- [ ] Check SPEC.md renders on GitHub
- [ ] Verify `[Browse all predictions](../../issues?q=label:predicted)` link works
- [ ] Add LICENSE
- [ ] Ensure model_card.md has real metrics
- [ ] Final commit, push, verify CI green
- [ ] Pin repo on GitHub profile
- [ ] Submit

**Must ship:** Everything works. Repo is clean. Submit.

---

## Minimum Bar Per Day

If time runs short, these are non-negotiable:

| Day | Must Ship | Can Defer |
|---|---|---|
| 1 | Repo + setlists.json | Song name edge cases |
| 2 | song_gaps + song_stats + transition_matrix | Venue features |
| 3 | Gap baseline + XGBoost pkl + Markov json | Detailed model card |
| 4 | predict.py working from CLI | Ensemble optimization (hand-tune) |
| 5 | predict issue → prediction comment | process.yml workflow |
| 6 | Dashboard with form + latest prediction | Charts |
| 7 | CI green + README | Design decisions section |
| 8 | Clean push, submit | — |

---

## Dependencies

Managed via [pixi](https://pixi.sh) — declared in `pyproject.toml`, not `requirements.txt`.

Runtime: `numpy`, `scipy`, `scikit-learn`, `xgboost`, `httpx`. Dev: `ruff`, `pytest` in the `dev` feature.

Setup: `pixi install`. Tasks: `pixi run lint`, `pixi run test`, `pixi run check`.
