# Phinish — Project Specification

> ML-powered Phish setlist predictions running entirely from a GitHub repository.  
> Issues are the API. Actions are the compute. The repo is the database.

**Version:** 1.0  
**Last Updated:** 2026-05-02  
**Status:** Pre-build

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Architecture](#2-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Data Pipeline](#4-data-pipeline)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Architecture](#6-model-architecture)
7. [Ensemble Strategy](#7-ensemble-strategy)
8. [Inference Pipeline](#8-inference-pipeline)
9. [GitHub Actions Workflows](#9-github-actions-workflows)
10. [Issue-Based Interface](#10-issue-based-interface)
11. [Security Model](#11-security-model)
12. [Dashboard (GitHub Pages)](#12-dashboard-github-pages)
13. [Evaluation Framework](#13-evaluation-framework)
14. [Testing Strategy](#14-testing-strategy)
15. [CI/CD Pipeline](#15-cicd-pipeline)
16. [Future Work (Part 2)](#16-future-work-part-2)
17. [Risk Register](#17-risk-register)
18. [Appendix](#18-appendix)

---

## 1. Project Summary

### 1.1 What This Is

A self-contained GitHub repository that predicts Phish concert setlists using an ensemble of classical machine learning models. The system is fully serverless — GitHub Actions provides compute, GitHub Issues provide the user interface, Git history provides state persistence, and GitHub Pages hosts a public dashboard. No cloud accounts, no databases, no servers beyond what GitHub provides for free.

### 1.2 Core Principles

- **Repo-as-infrastructure.** The repository is the database, the compute trigger, and the deployment target. Every state change is a git commit. Every prediction is auditable through git history.
- **Classical ML first.** Setlist prediction is a structured sequence problem, not a language generation problem. Markov chains, gradient boosted trees, and heuristics outperform LLMs on this task at a fraction of the cost.
- **GitHub-native.** No external servers, no containers in production, no API endpoints. Issues are the input, Actions are the compute, Pages are the display. Everything runs on GitHub's free tier.
- **KISS and DRY.** One function, two callers, zero duplication. Every component earns its place.

### 1.3 Key Metrics

| Metric | Definition | Baseline Target | Stretch Goal |
|---|---|---|---|
| Precision@25 | Of 25 predicted songs, how many were actually played | 40% | 70%+ |
| Song AUC-PR | Area under precision-recall curve per song | 0.70 | 0.90+ |
| Opener Accuracy | Was the first song of Set 1 predicted correctly | 15% | 40%+ |
| Pair Match Rate | % of actual consecutive song pairs in prediction | 5% | 25%+ |

### 1.4 Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Serving layer | GitHub Actions (not FastAPI) | No server to maintain. Issues are the API. The system runs itself. |
| Deployment | None (not Docker/K8s) | No deployment target needed. Actions is the compute layer. |
| HTTP client | httpx (not pysh-client) | No third-party dependency for core data pipeline. Full control. |
| Data pull strategy | By year (not per show) | ~43 API calls vs ~2,100. Dramatically faster, fewer rate limit risks. |
| Hyperparameter tuning | Hand-tuned defaults (not Optuna) | 90% of quality for 10% of complexity. Optuna deferred to Part 2. |
| Feature importance | Deferred (not SHAP) | Model card with raw metrics sufficient for Part 1. SHAP in early Part 2. |
| Test depth | 8–10 pytest (unit + integration) | Meaningful coverage without ceremony. CI runs real tests. |
| Issue lifecycle | Close + label | Open count stays clean. Filtered label URL for browsing history. |

---

## 2. Architecture

### 2.1 System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                       GitHub Repository                          │
│                                                                  │
│  data/processed/     Canonical setlist dataset (~2,100 shows)    │
│  state/features/     Precomputed feature store (versioned)       │
│  models/             Trained ML artifacts (.pkl, .json)          │
│  scripts/            Data + training + inference pipeline        │
│  docs/               GitHub Pages dashboard                     │
│  .github/workflows/  CI + issue-triggered workflows             │
│  reasons.md          Human-readable prediction history           │
│                                                                  │
└──────┬─────────────────┬─────────────────┬───────────────────────┘
       │                 │                 │
  Issue: predict    Issue: process    GitHub Pages
  (user request)    (retrain trigger) (dashboard)
       │                 │                 │
       ▼                 ▼                 ▼
  Actions Runner    Actions Runner    Static HTML/JS
  ┌────────────┐   ┌──────────────┐  ┌──────────────┐
  │ Parse issue │   │ Scrape year  │  │ Reads JSON   │
  │ Load models │   │ Rebuild      │  │ from repo    │
  │ Run ensemble│   │ features     │  │ via raw URLs │
  │ Post comment│   │ Retrain all  │  │              │
  │ Update state│   │ models       │  │ Form builds  │
  │ Close issue │   │ Commit new   │  │ pre-filled   │
  │             │   │ artifacts    │  │ issue URLs   │
  └────────────┘   └──────────────┘  └──────────────┘
```

### 2.2 Two Workflow Triggers

| Workflow | Label | Trigger | Duration | What It Does |
|---|---|---|---|---|
| `predict.yml` | `predict` | Issue labeled | ~10 sec | Load models, run inference, post prediction comment, update dashboard data, close issue |
| `process.yml` | `process` | Issue labeled | ~2–3 min | Scrape year from API, rebuild features, retrain all models, run evaluation, commit artifacts, close issue |

### 2.3 Data Flow

```
Phish.net API → scrape.py (by year) → data/processed/setlists.json
                                              ↓
                                     build_features.py
                                              ↓
                                     state/features/*.json
                                              ↓
                                     train_*.py
                                              ↓
                                     models/*.pkl, *.json
                                              ↓
                                     predict.py (called by Actions)
                                              ↓
                                     Issue comment + state/ updates
```

---

## 3. Repository Structure

```
phinish/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── predict-show.yml           # Prediction request
│   │   ├── process-year.yml           # Data processing / retrain trigger
│   │   └── config.yml                 # Disable blank issues
│   └── workflows/
│       ├── ci.yml                     # Lint → Test → Validate
│       ├── predict.yml                # Issue-triggered inference
│       └── process.yml                # Issue-triggered retrain
│
├── data/
│   ├── processed/
│   │   ├── setlists.json              # Canonical setlist dataset
│   │   ├── songs.json                 # Song metadata + canonical names
│   │   └── venues.json                # Venue metadata
│   └── canonical_names.json           # Song name normalization map
│
├── state/
│   ├── features/
│   │   ├── song_gaps.json             # Shows since last played per song
│   │   ├── transition_matrix.json     # Song→song transition probabilities
│   │   ├── venue_history.json         # Per-venue song frequencies
│   │   └── song_stats.json            # Frequency, position, set distribution
│   ├── latest_prediction.json         # Most recent prediction (for dashboard)
│   ├── predictions_log.json           # All predictions with metadata
│   └── usage.json                     # Rate limit tracking
│
├── models/
│   ├── xgboost_song_selector.pkl      # Gradient boosted song classifier
│   ├── markov_order2.json             # 2nd-order transition matrix
│   ├── calibrator.pkl                 # Platt scaling calibration
│   ├── ensemble_weights.json          # Learned ensemble weights
│   └── model_card.md                  # Version, metrics, changelog
│
├── scripts/
│   ├── scrape.py                      # Phish.net data collection (httpx)
│   ├── build_features.py              # Feature engineering pipeline
│   ├── train_baseline.py              # Frequency + gap baselines
│   ├── train_xgboost.py               # Song selection model
│   ├── train_markov.py                # Transition model
│   ├── train_ensemble.py              # Ensemble weight optimization
│   ├── predict.py                     # Inference pipeline + CLI
│   ├── evaluate.py                    # Backtesting + metrics
│   ├── process_issue.py               # Actions entrypoint (predict)
│   ├── process_year.py                # Actions entrypoint (retrain)
│   └── utils.py                       # Shared helpers
│
├── docs/
│   └── index.html                     # GitHub Pages dashboard
│
├── tests/
│   ├── test_features.py               # Unit: feature calculations
│   ├── test_predict.py                # Unit: prediction structure
│   └── test_integration.py            # Integration: data → features → predict
│
├── .gitattributes                     # LFS tracking
├── .gitignore
├── LICENSE
├── README.md
├── SPEC.md                            # This document
├── PLAN.md                            # Build plan
└── pyproject.toml                    # Pixi workspace + deps + ruff/pytest config
```

---

## 4. Data Pipeline

### 4.1 Source

Phish.net API v5 — `https://api.phish.net/v5/`

- Free API key (register at phish.net, keys granted in seconds)
- Authentication via `apikey` query parameter
- Responses cached server-side for short periods
- Data storage permitted with periodic refresh

### 4.2 Pull Strategy

Pull by year to minimize API calls:

```python
# ~43 calls for setlists (1983–2026)
for year in range(1983, 2027):
    resp = httpx.get(f"{BASE}/setlists/showyear/{year}.json?apikey={key}")

# 1 call for all shows
resp = httpx.get(f"{BASE}/shows/artist/phish.json?apikey={key}&order_by=showdate")

# 1 call for all songs
resp = httpx.get(f"{BASE}/songs.json?apikey={key}")

# 1 call for all venues
resp = httpx.get(f"{BASE}/venues.json?apikey={key}")
```

Total: ~46 requests. Completes in under 2 minutes.

Raw API responses are cached locally during scraping, processed data committed to `data/processed/`.

### 4.3 API Response Schema

The setlists endpoint returns **one row per song per show**:

```json
{
  "showid": "1252683584",
  "showdate": "1997-11-22",
  "song": "Tweezer",
  "songid": "220",
  "slug": "tweezer",
  "set": "2",
  "position": "1",
  "transition": ">",
  "gap": "3",
  "isjam": "0",
  "isreprise": "0",
  "is_original": "1",
  "tracktime": "923",
  "venueid": "123",
  "venue": "Hampton Coliseum",
  "city": "Hampton",
  "state": "VA",
  "country": "USA",
  "tourid": "45",
  "tourname": "Fall 1997"
}
```

Key fields: `set` (1, 2, e=encore), `position` (order within set), `transition` (`,` = pause, `>` = segue), `gap` (pre-computed by Phish.net, but we recompute for historical accuracy during training).

### 4.4 Processing: Rows → Setlists

The scraper groups per-song rows into per-show setlists:

```python
def group_into_shows(raw_setlist_rows):
    shows = defaultdict(lambda: {"sets": {"1": [], "2": [], "encore": []}})
    for row in raw_setlist_rows:
        show = shows[row["showid"]]
        set_key = "encore" if row["set"] == "e" else row["set"]
        show["sets"][set_key].append({
            "song": canonicalize(row["song"]),
            "position": int(row["position"]),
            "transition": row.get("trans_mark", ","),
        })
        # Populate show-level metadata once
        show.update({
            "show_id": row["showid"],
            "date": row["showdate"],
            "venue_id": row["venueid"],
            "venue_name": row["venue"],
            "city": row["city"],
            "state": row["state"],
            "country": row["country"],
            "tour": row.get("tourname", ""),
        })
    # Sort songs by position within each set
    for show in shows.values():
        for set_songs in show["sets"].values():
            set_songs.sort(key=lambda s: s["position"])
    return list(shows.values())
```

### 4.5 Canonical Dataset Schema

**`data/processed/setlists.json`**
```json
[
  {
    "show_id": "1252683584",
    "date": "1997-11-22",
    "year": 1997,
    "month": 11,
    "day_of_week": "saturday",
    "venue_id": "v_hampton",
    "venue_name": "Hampton Coliseum",
    "city": "Hampton",
    "state": "VA",
    "country": "USA",
    "tour": "Fall 1997",
    "is_festival": false,
    "is_nye": false,
    "is_halloween": false,
    "sets": {
      "1": [
        { "song": "Tweezer", "position": 0, "transition": ">" },
        { "song": "Reba", "position": 1, "transition": "," }
      ],
      "2": [],
      "encore": []
    },
    "total_songs": 23
  }
]
```

### 4.6 Song Name Normalization

`data/canonical_names.json` maps variant names to canonical forms. Applied at scrape time.

### 4.7 Special Show Handling

| Type | Flag | Treatment |
|---|---|---|
| Halloween | `is_halloween: true` | Cover album songs excluded from gap calculations |
| NYE | `is_nye: true` | Separate distribution at prediction time |
| Festival | `is_festival: true` | Longer sets, weighted lower in recent frequency |

---

## 5. Feature Engineering

### 5.1 Feature Store

All features precomputed by `scripts/build_features.py`, stored as JSON in `state/features/`. Versioned through git. Rebuilt by the `process` workflow after new data arrives.

Note: Although the Phish.net API provides a pre-computed `gap` field, we recompute all gaps historically from `setlists.json`. Training requires the gap value *at the time of each show*, not the current gap. The API only gives the current gap.

### 5.2 Song-Level Features (`state/features/song_stats.json`)

| Feature | Type | Description |
|---|---|---|
| `total_plays` | int | Total times played |
| `lifetime_frequency` | float | % of all shows containing this song |
| `recent_frequency_50` | float | % of last 50 shows |
| `recent_frequency_20` | float | % of last 20 shows |
| `avg_set_position` | float | Mean normalized position (0.0–1.0) |
| `typical_set` | int | Mode set number (1, 2, 3=encore) |
| `set_distribution` | dict | Play % per set |
| `opener_frequency` | float | % of plays as set opener |
| `closer_frequency` | float | % of plays as set closer |
| `is_cover` | bool | Cover song flag |
| `debut_year` | int | Year of first performance |

### 5.3 Song Gap Features (`state/features/song_gaps.json`)

```json
{
  "Tweezer": { "gap": 3, "last_played": "2024-07-16" },
  "Harpua": { "gap": 47, "last_played": "2023-12-31" }
}
```

### 5.4 Transition Matrix (`state/features/transition_matrix.json`)

Order-1 and order-2 transition probabilities, plus set openers and closers. Laplace smoothed (k=0.01).

```json
{
  "order_1": {
    "Tweezer": { "Ghost": 0.12, "Piper": 0.08 }
  },
  "order_2": {
    "Mike's Song|Hydrogen": { "Weekapaug Groove": 0.72 }
  },
  "set_openers": {
    "1": { "Buried Alive": 0.04 },
    "2": { "Down with Disease": 0.05 }
  },
  "set_closers": {
    "1": { "David Bowie": 0.04 },
    "2": { "Slave to the Traffic Light": 0.03 }
  }
}
```

### 5.5 Venue Features (`state/features/venue_history.json`)

```json
{
  "v_msg": {
    "total_shows": 67,
    "song_freq": { "Tweezer": 0.45, "YEM": 0.52 },
    "common_openers": ["Buried Alive", "AC/DC Bag"],
    "common_closers": ["Character Zero", "Tweezer Reprise"]
  }
}
```

---

## 6. Model Architecture

### 6.1 Overview

| Model | Purpose | Format | Size |
|---|---|---|---|
| Gap-Weighted Baseline | Floor performance heuristic | Python dict | <1 KB |
| XGBoost Classifier | Song selection | `.pkl` | ~1–5 MB |
| Markov Chain (Order 2) | Song transitions + sequencing | `.json` | ~500 KB |
| Calibration Layer | Probability quality | `.pkl` | <100 KB |

### 6.2 Model A — Gap-Weighted Baseline

**Formula:** `score = recent_frequency_50 * log(gap + 2)`

Songs that are common AND overdue score highest. Every other model must beat this.

**Expected Precision@25:** ~55–65%

### 6.3 Model B — XGBoost Song Selector

Per-song binary classifier: P(song played at this show).

**Training data:** ~2,100 shows × ~300 songs = ~630,000 rows.

**Feature vector per row:**
```
[gap, lifetime_freq, recent_freq_50, recent_freq_20, avg_set_position,
 typical_set, is_cover, is_bustout, opener_freq, closer_freq,
 day_of_week(7), month, is_weekend, is_nye, is_halloween, is_festival,
 tour_position_pct, run_position, venue_total_shows,
 venue_song_freq, venue_is_new, days_since_last_show, played_last_show,
 played_last_3, times_played_this_tour]
```

**Training protocol:**
1. Temporal split: train < 2024, validate = 2024, test ≥ 2025
2. Hand-tuned hyperparameters: `max_depth=6`, `learning_rate=0.1`, `n_estimators=300`, `scale_pos_weight=12`
3. Platt scaling calibration on validation set

**Output:** `models/xgboost_song_selector.pkl` + `models/calibrator.pkl`

Trains in ~30–60 seconds on Actions CPU. No GPU required.

### 6.4 Model C — Markov Chain

Order-2 transition model: P(song_C | song_A, song_B).

Separate distributions for set openers, closers, and set-break transitions. Laplace smoothed.

**Output:** `models/markov_order2.json` — plain JSON, git-diffable.

### 6.5 Calibration

Platt scaling (logistic regression) on XGBoost raw outputs, trained on validation set. Produces honest probability estimates: "Tweezer: 34%" means Tweezer appears ~34% of the time when the model says 34%.

---

## 7. Ensemble Strategy

### 7.1 Score Combination

```
final_score(song, context) =
    w1 * xgboost_prob(song, show_features)
  + w2 * markov_prob(song | previous_songs)
  + w3 * gap_score(song)
  + w4 * venue_affinity(song, venue)
```

### 7.2 Weight Optimization

Grid search on validation set, optimizing Precision@25. Stored as `models/ensemble_weights.json`.

### 7.3 Constrained Setlist Construction

After scoring, build the setlist with greedy selection:

- Set 1: ~10 songs, Set 2: ~8 songs, Encore: ~2 songs
- No song appears twice
- Songs from last show excluded (hard constraint)
- Songs from last 3 shows penalized (score × 0.3)
- Song placement guided by `typical_set` and Markov transitions
- Set openers and closers selected from opener/closer distributions

---

## 8. Inference Pipeline

### 8.1 `scripts/predict.py`

Single function with CLI interface:

```python
def predict(date: str, venue: str, city: str = None) -> dict:
    """Full prediction pipeline. Returns structured setlist with confidences."""

if __name__ == "__main__":
    # argparse: --date YYYY-MM-DD --venue "Venue Name" [--city "City, ST"]
```

Called two ways:
1. Imported by `process_issue.py` in the Action
2. Run directly from CLI for local testing

### 8.2 Pipeline Steps

1. Load models + feature store
2. Resolve venue (exact match or fuzzy)
3. Build show-level features from date + venue
4. Score every song via ensemble
5. Build constrained setlist (greedy selection with set structure)
6. Calibrate confidences
7. Return structured prediction dict

### 8.3 Runtime

- CPU only, ~10 seconds on Actions runner
- ~500 MB memory
- No GPU required

---

## 9. GitHub Actions Workflows

### 9.1 Prediction Workflow (`predict.yml`)

```yaml
name: Predict Setlist
on:
  issues:
    types: [labeled]

permissions:
  contents: write
  issues: write

concurrency:
  group: predict-${{ github.event.issue.number }}
  cancel-in-progress: false

jobs:
  predict:
    runs-on: ubuntu-latest
    if: github.event.label.name == 'predict'
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true
      - uses: prefix-dev/setup-pixi@v0.8.1
        with:
          cache: true
      - name: Run prediction
        run: pixi run python scripts/process_issue.py
        env:
          ISSUE_NUMBER: ${{ github.event.issue.number }}
          ISSUE_BODY: ${{ github.event.issue.body }}
          ISSUE_AUTHOR: ${{ github.event.issue.user.login }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      - name: Commit state
        run: |
          git config user.name "phinish[bot]"
          git config user.email "bot@users.noreply.github.com"
          git add state/ reasons.md
          git diff --staged --quiet || git commit -m "prediction #${{ github.event.issue.number }}"
          git push
      - name: Update labels and close
        run: |
          gh issue edit ${{ github.event.issue.number }} --remove-label "predict" --add-label "predicted"
          gh issue close ${{ github.event.issue.number }}
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### 9.2 Process / Retrain Workflow (`process.yml`)

```yaml
name: Process Year
on:
  issues:
    types: [labeled]

permissions:
  contents: write
  issues: write

concurrency:
  group: process
  cancel-in-progress: false

jobs:
  process:
    runs-on: ubuntu-latest
    if: github.event.label.name == 'process'
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true
      - uses: prefix-dev/setup-pixi@v0.8.1
        with:
          cache: true
      - name: Process year and retrain
        run: pixi run python scripts/process_year.py
        env:
          ISSUE_NUMBER: ${{ github.event.issue.number }}
          ISSUE_BODY: ${{ github.event.issue.body }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          PHISHNET_API_KEY: ${{ secrets.PHISHNET_API_KEY }}
      - name: Commit all artifacts
        run: |
          git config user.name "phinish[bot]"
          git config user.email "bot@users.noreply.github.com"
          git add data/ state/ models/ reasons.md
          git diff --staged --quiet || git commit -m "process + retrain from issue #${{ github.event.issue.number }}"
          git push
      - name: Update labels and close
        run: |
          gh issue edit ${{ github.event.issue.number }} --remove-label "process" --add-label "processed"
          gh issue close ${{ github.event.issue.number }}
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### 9.3 `scripts/process_year.py` Pipeline

```
Parse year from issue body
        ↓
Scrape that year from Phish.net API
        ↓
Merge into data/processed/setlists.json
        ↓
Rebuild all features (build_features.py)
        ↓
Retrain baselines (train_baseline.py)
        ↓
Retrain XGBoost (train_xgboost.py)
        ↓
Retrain Markov (train_markov.py)
        ↓
Re-optimize ensemble weights (train_ensemble.py)
        ↓
Run evaluation (evaluate.py)
        ↓
Update model card with new metrics
        ↓
Post summary comment on issue
```

Total runtime: ~2–3 minutes on Actions CPU.

---

## 10. Issue-Based Interface

### 10.1 Label State Machine

```
Issue created with 'predict' label
        ↓
Action fires → runs prediction
        ↓
Remove 'predict', add 'predicted', close issue
```

```
Issue created with 'process' label
        ↓
Action fires → scrapes, retrains, evaluates
        ↓
Remove 'process', add 'processed', close issue
```

On failure: remove trigger label, add `failed`, close issue, post error comment.

### 10.2 Labels

| Label | Purpose | Applied By |
|---|---|---|
| `predict` | Triggers prediction workflow | Issue template |
| `predicted` | Marks successful prediction | Action |
| `process` | Triggers retrain workflow | Issue template |
| `processed` | Marks successful retrain | Action |
| `failed` | Marks failed processing | Action |

### 10.3 Prediction Request Template

```yaml
# .github/ISSUE_TEMPLATE/predict-show.yml
name: Predict a Show
description: Get a setlist prediction for an upcoming Phish show
labels: ["predict"]
body:
  - type: input
    id: date
    attributes:
      label: Show Date
      description: "Format: YYYY-MM-DD"
      placeholder: "2026-12-31"
    validations:
      required: true
  - type: input
    id: venue
    attributes:
      label: Venue
      placeholder: "Madison Square Garden"
    validations:
      required: true
  - type: input
    id: city
    attributes:
      label: City
      placeholder: "New York, NY"
    validations:
      required: false
```

### 10.4 Process Year Template

```yaml
# .github/ISSUE_TEMPLATE/process-year.yml
name: Process Year
description: Scrape a year's data and retrain all models
labels: ["process"]
body:
  - type: input
    id: year
    attributes:
      label: Year
      description: "Year to scrape and process (1983–2026)"
      placeholder: "2025"
    validations:
      required: true
```

### 10.5 Disable Blank Issues

```yaml
# .github/ISSUE_TEMPLATE/config.yml
blank_issues_enabled: false
```

### 10.6 Prediction Comment Format

```markdown
## 🎸 Phinish Prediction

**Madison Square Garden — December 31, 2026**

### Set 1
| # | Song | Confidence | Gap |
|---|------|-----------|-----|
| 1 | Buried Alive | 🟢 72% | 5 shows |
| 2 | Ocelot | 🟢 58% | 8 shows |

### Set 2
| # | Song | Confidence | Gap |
|---|------|-----------|-----|
| 1 | Tweezer | 🟢 68% | 3 shows |

### Encore
| # | Song | Confidence | Gap |
|---|------|-----------|-----|
| 1 | Character Zero | 🟢 55% | 4 shows |

---
**Model:** v1 • **Avg Confidence:** 48%
_Generated by [Phinish](https://OWNER.github.io/phinish)_
```

---

## 11. Security Model

### 11.1 Secrets

| Secret | Location | Used By |
|---|---|---|
| `PHISHNET_API_KEY` | GitHub Secrets | `process.yml` only |
| `GITHUB_TOKEN` | Auto-provided | All workflows (auto-scoped, auto-expires) |

No external API keys needed in Part 1 beyond Phish.net.

### 11.2 Input Sanitization

All issue body content parsed in Python via environment variables, never shell-interpolated via `${{ }}`.

```python
import re

VALID_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
VALID_YEAR = re.compile(r"^\d{4}$")

def sanitize(raw: str, max_length: int = 500) -> str:
    cleaned = raw.strip()[:max_length]
    cleaned = re.sub(r'[;&|`$(){}]', '', cleaned)
    return cleaned
```

### 11.3 Rate Limiting

```python
MAX_PREDICTIONS_PER_DAY = 20
MAX_PER_USER_PER_DAY = 3
```

Tracked in `state/usage.json`, committed after each run.

### 11.4 Branch Protection

- Require PR reviews for `.github/workflows/` changes
- Disable blank issues
- Only Actions bot and repo owner can push to main

---

## 12. Dashboard (GitHub Pages)

### 12.1 Overview

Static HTML page at `docs/index.html`. Reads JSON from repo via `raw.githubusercontent.com`. Note: raw GitHub URLs are CDN-cached for up to 5 minutes — documented in README.

### 12.2 Sections

1. **Prediction Form** — date, venue, city → builds pre-filled GitHub Issue URL → opens in new tab
2. **Stats Bar** — shows in dataset, Precision@25, predictions made
3. **Latest Prediction** — setlist with per-song confidence badges (green/amber/red)
4. **Overdue Songs** — horizontal bar chart of highest-gap songs
5. **Accuracy Chart** — Precision@25 over time (Chart.js)
6. **How It Works** — 4-step explanation

### 12.3 Pre-Filled Issue URL

```javascript
const issueUrl = `${REPO_URL}/issues/new`
  + `?labels=predict`
  + `&title=${encodeURIComponent(title)}`
  + `&body=${encodeURIComponent(body)}`;
window.open(issueUrl, '_blank');
```

User fills form → clicks submit → lands on GitHub with pre-filled issue → clicks "Submit new issue" → Action fires.

### 12.4 Technology

- Vanilla HTML/CSS/JS (no framework, no build step)
- Chart.js for charts
- Google Fonts for typography
- CSS variables for theming
- Responsive

---

## 13. Evaluation Framework

### 13.1 Temporal Backtesting

Train on shows before cutoff, test on shows after. Features recomputed at each historical point — no future leakage.

### 13.2 Metrics

| Metric | Formula |
|---|---|
| Precision@25 | `\|predicted ∩ actual\| / 25` |
| Recall | `\|predicted ∩ actual\| / \|actual\|` |
| F1 | Harmonic mean |
| Opener Accuracy | Exact match on Set 1 first song |
| Pair Match Rate | Correct consecutive pairs / total pairs |

### 13.3 Model Comparison

Every evaluation compares all models head-to-head. Output stored in `models/model_card.md`.

---

## 14. Testing Strategy

### 14.1 Framework

pytest. 8–10 tests across three files: unit + integration.

### 14.2 Test Allocation

**`tests/test_features.py`** (3–4 unit tests):
- Gap calculation against hand-verified examples
- Transition probabilities sum to ~1.0
- Song stats schema validation
- Edge case: song with 0 plays

**`tests/test_predict.py`** (2–3 unit tests):
- No duplicate songs across sets
- Set lengths within expected range (±2 of target)
- All confidences between 0.0 and 1.0

**`tests/test_integration.py`** (2–3 integration tests):
- Full pipeline: data → features → predict returns valid setlist
- CLI entrypoint exits cleanly with valid args
- Songs from "last show" are excluded from prediction

---

## 15. CI/CD Pipeline

### 15.1 Workflow (`.github/workflows/ci.yml`)

```yaml
name: CI
on: [push, pull_request]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true
      - uses: prefix-dev/setup-pixi@v0.8.1
        with:
          environments: dev
          cache: true
      - run: pixi run -e dev check
```

---

## 16. Future Work (Part 2)

| Feature | Description |
|---|---|
| Optuna hyperparameter tuning | Bayesian search for XGBoost params |
| SHAP feature importance | Explainability analysis + visualizations |
| Song2Vec embeddings | Thematic coherence scoring in ensemble |
| LLM commentary layer | AI-generated prediction analysis |
| Automated scoring | Compare predictions to actuals after shows |
| Scheduled data refresh | Daily cron to pull new shows |
| Prediction tournaments | Users submit and compare their own predictions |
| Cloudflare Worker | Anonymous predictions without GitHub account |

---

## 17. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Phish.net API changes | Medium | High | Cache raw data, pin to v5, monitor |
| API key approval delay | Low | High | Apply immediately, have manual data fallback |
| Model overfits to era | Medium | Medium | Temporal CV, retrain via process workflow |
| Song name inconsistencies | High | Medium | Canonical name map with tests |
| Actions minutes exhausted | Low | High | Rate limits, budget tracking |
| Halloween/NYE unpredictable | High | Low | Flag as special cases |
| Raw GitHub URL cache delay | Certain | Low | Documented in README, link to Actions tab |

---

## 18. Appendix

### 18.1 Known Phish Statistical Patterns

1. **Rotation Rule** — almost never repeats a song within 2–3 shows
2. **Gap Effect** — songs become more likely the longer absent
3. **Venue Signatures** — certain songs tied to specific venues
4. **Tour Arc** — openers safe, mid-tour experimental, closers get rarities
5. **Set Architecture** — Set 1 structured, Set 2 jam-heavy, Encore crowd-pleasers
6. **Song Suites** — Mike's → Hydrogen → Weekapaug always together
7. **NYE Factor** — December 31 is a completely different distribution

### 18.2 Glossary

| Term | Definition |
|---|---|
| Bustout | A song not played in 50+ shows |
| Gap | Number of shows since last played |
| Rotation | The ~80–100 songs currently in regular play |
| Segue (>) | Songs flowing into each other without pause |
| Precision@25 | Of 25 predicted songs, how many were actually played |

### 18.3 Dependencies

Managed by [pixi](https://pixi.sh) via `pyproject.toml`. Single source of truth — no `requirements.txt`.

```toml
[tool.pixi.workspace]
channels = ["conda-forge"]
platforms = ["win-64", "linux-64", "osx-arm64"]

[tool.pixi.dependencies]
python = ">=3.12,<3.13"
numpy = ">=1.26"
scipy = ">=1.12"
scikit-learn = ">=1.4"
xgboost = ">=2.0"
httpx = ">=0.27"

[tool.pixi.feature.dev.dependencies]
ruff = "*"
pytest = "*"
```

Setup: `pixi install`. Tasks: `pixi run lint`, `pixi run test`, `pixi run check`.
