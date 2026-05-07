# Phinish — Working Notes for Claude

## Skills to invoke

When editing this codebase, invoke these skills before writing code (they auto-trigger on description match, but reinforce here):

- **clean-code** — any code change. DRY, SRP, KISS, SOLID, YAGNI as decision rules.
- **python-modern** — any `.py` file. Python-specific idioms (types, pathlib, httpx, structlog, msgspec).
- **pytest-modern** — any file under `tests/`. Use parametrize for multi-case tests; use `tmp_path` and `monkeypatch` instead of manual setup.
- **simplify** — after a meaningful change, run `/simplify` to catch reuse, quality, and efficiency issues across the diff.

## Package layout

```
src/phinish/
├── __init__.py           # dotenv + structlog bootstrap
├── paths.py              # all filesystem path constants (ROOT, DATA_DIR, MODELS_DIR, …)
├── artifacts.py          # typed loaders, shared scoring (load_shows, gap_score_per_song, …)
├── utils/                # generic helpers (re-exports everything for backward compat)
│   ├── __init__.py       # barrel re-export of all sub-modules + paths
│   ├── constants.py      # SET_KEYS, SET_DISPLAY, TOP_K, VALID_DATE, …
│   ├── io.py             # load_json, save_json, post_issue_comment
│   └── helpers.py        # sanitize, venue matching, dates, show utils, rate limiting
├── scrape/               # Phish.net API ingestion
│   ├── __init__.py       # re-exports scrape, cli, types
│   ├── api.py            # scrape() + fetch / group_into_shows / normalize_*
│   └── types.py          # Show, SongEntry, SongCatalogEntry, VenueRecord
├── features/             # raw setlists → aggregate stats
│   ├── __init__.py       # re-exports build_features, main, types
│   ├── build.py          # build_features() + build_song_gaps / _stats / _transition_matrix / _venue_history
│   └── types.py          # SongStats, SongGap, VenueHistory, TransitionMatrix, TransitionDist
├── train/                # ML model fitting
│   ├── __init__.py       # re-exports state primitives + types
│   ├── state.py          # StreamingState, featurize, MIN_PLAYS_FOR_CANDIDATE
│   ├── baseline.py       # train_baseline() + main()
│   ├── markov.py         # train_markov() + main()
│   ├── xgboost.py        # train_xgboost() + main()
│   ├── ensemble.py       # train_ensemble() + main()
│   └── types.py          # EnsembleWeights
├── predict/              # inference
│   ├── __init__.py       # re-exports cli + types
│   ├── pipeline.py       # predict() + helpers
│   └── types.py          # Prediction, PredictionItem, PredictionWeights
├── evaluate/             # backtesting
│   ├── __init__.py       # re-exports evaluate, main
│   └── backtest.py       # evaluate() + main()
├── summarize/            # LLM-powered prediction summaries
│   ├── __init__.py       # re-exports SummaryResult
│   ├── api.py            # summarize() via Claude Haiku (httpx + stamina, no SDK)
│   ├── prompts/          # voice prompt templates (markdown + str.format placeholders)
│   │   ├── full_phan.md  # enthusiastic Phish fan voice
│   │   └── light_fan.md  # measured music journalist voice
│   └── types.py          # SummaryResult
└── process/              # GitHub Actions entry points
    ├── __init__.py
    ├── issue.py          # process_issue (predict-on-issue)
    ├── year.py           # process_year (retrain-on-issue)
    └── types.py          # Usage
```

**Conventions**:
- Types live in their owning subpackage's `types.py`. Cross-stage code imports from `phinish.<stage>.types`, never from `phinish.utils`.
- `__init__.py` re-exports only the subpackage's *public* surface: types + lightweight functions. Heavy entry points (those that import `artifacts` or cross-stage modules) stay in their own modules to avoid circular imports and are referenced directly in `pyproject.toml` entry points.
- `paths.py` owns all filesystem constants (`ROOT`, `DATA_DIR`, `MODELS_DIR`, …). Import paths from `phinish.paths` (or via the `phinish.utils` re-export). All generated data lives under `data/`: `data/source/` (API data, gitignored), `data/state/` (features), `data/models/` (trained artifacts). Only `data/canonical_names.json` is committed.
- `artifacts.py` centralizes repeated `load_json` + `msgspec.convert` patterns and cross-cutting scoring helpers (`markov_score_per_song`, `gap_score_per_song`, `calibrated_predict_proba`).
- `utils/` has three sub-modules: `constants.py` (pure data), `io.py` (disk/network I/O), `helpers.py` (pure transforms). The `__init__.py` re-exports everything, so `from phinish.utils import X` works.
- `train/state.py` exists because both `train` and `predict` need `StreamingState`/`featurize`. When two stages share a non-type primitive, give it its own module rather than picking one stage to "own" it.

## Project rules

- **Run via pixi**: `pixi run -e dev pytest`, `pixi run -e dev ruff check .`. Never invoke a bare `python` or `pytest` — they pick up the wrong env.
- **Editable install** via `[tool.pixi.pypi-dependencies] phinish = { path = ".", editable = true }`. Imports use `from phinish.X import Y`; no `sys.path` bootstrapping.
- **CLI surface** is exposed via `[project.scripts]` (`phinish-predict`, `phinish-scrape`, `phinish-process-issue`, etc.). Each pipeline module has a library function (typed in → typed out, no I/O) and a thin `main()` wrapper that handles load/save. Modules with argparse (`scrape/api.py`, `predict/pipeline.py`) use `cli()` instead. Process entry points use descriptive names (`process_issue`, `process_year`) since they are inherently I/O-bound.
- **Build backend**: hatchling, configured to package `src/phinish`.
- **Verify before claiming done**: `pixi run -e dev ruff check . && pixi run -e dev pytest` must both pass.
- **Environment variables**: `PHISHNET_API_KEY` (required for scraping), `ANTHROPIC_API_KEY` (required for LLM summaries; missing key triggers a static fallback, not a crash).

## No magic dicts, no magic strings

When constructing a value that has a known schema, build it via the matching msgspec Struct:

```python
# ❌ magic dict — keys are strings, no type checking
return {"show_id": sid, "date": d, "year": int(d[:4]), ...}

# ✅ Struct constructor — fields are kwargs, type-checked by ruff/pyright/mypy
return Show(show_id=sid, date=d, year=int(d[:4]), ...)
```

When a string literal appears as a key/identifier in 3+ places, promote it to a module constant in `utils.py` (or the relevant `types.py`). Examples already extracted: `SET_KEYS`, `SET_DISPLAY`, `SET_TO_INT`, `TOP_K`, `FUZZY_VENUE_THRESHOLD`.

The right-side `row.get("showid", "")` calls in scrape code are unavoidable — the API speaks string keys. Confine that string-y access to one named helper per record type (e.g. `_new_show(row)`, `_append_song(...)`) so callers see `Show(...)` and never see the raw API field names.

## Docstring policy

Use **NumPy-style** docstrings (enforced by ruff `D` rules with `convention = "numpy"`).

- **Module**: one-line `"""..."""` at the top describing what the module does. Required for every script.
- **Public functions** (no leading underscore): one-line summary. Required.
- **Non-trivial functions** (>30 lines, multi-step, or non-obvious behavior): full NumPy-style docstring with `Parameters`, `Returns`, and `Raises` sections where applicable. Encouraged.
- **Private helpers** (`_foo`): one-line summary. Required for non-trivial helpers; skip only if the name + typed signature is genuinely self-explanatory.
- **Inline comments** stay rare — WHY only, never WHAT (the existing system-prompt rule).

**NumPy-style example:**

```python
def featurize(state: StreamingState, show: Show, song: str, cover_set: set[str]) -> list[float]:
    """Build the XGBoost feature vector for one (state, show, song) triple.

    Parameters
    ----------
    state
        Streaming statistics accumulated from all shows strictly before ``show``.
    show
        The target show being scored; only its date/venue/tour fields are read.
    song
        Candidate song name.
    cover_set
        Set of song names considered covers (non-Phish originals).

    Returns
    -------
    list[float]
        A 31-element feature vector aligned with ``feature_names()``.
    """
```

Convention rules ruff enforces:
- Summary on the first line (D200, D205), in imperative mood (D401).
- Section headers (`Parameters`, `Returns`, `Raises`, etc.) underlined with `----`.
- Parameter names without type after them — types come from the signature.
- Blank line between summary and the first section.

## Function size

- `main()` should orchestrate, not implement. If `main()` has 5+ phases, extract each as a `_named_phase()` helper.
- Don't decompose single conceptual operations (one feature vector, one streaming accumulator) — splitting by category just scatters context.

## Testing

- Hand-verified expected values for feature engineering live in `tests/conftest.py::tiny_shows`.
- New helpers in `utils.py` should get a unit test before they get a second caller.
- For the predict pipeline, prefer testing the public API (`predict()`, `format_comment()`) over internals.
