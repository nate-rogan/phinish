"""Action entrypoint for the process workflow.

Runs the full pipeline (scrape → features → train → evaluate), writes
models/manifest.json with dataset coverage + retrain history, and posts a
summary comment. Idempotent — safe to re-run anytime to pick up new shows
or late corrections.
"""
from __future__ import annotations

import os
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NoReturn

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import (
    build_features,
    evaluate,
    scrape,
    train_baseline,
    train_ensemble,
    train_markov,
    train_xgboost,
)
from scripts.utils import (
    MANIFEST_PATH,
    MODELS_DIR,
    SETLISTS_PATH,
    VALID_YEAR,
    load_json,
    parse_issue_form,
    post_issue_comment,
    sanitize,
    save_json,
)

MODEL_CARD = MODELS_DIR / "model_card.md"

METRIC_LABELS: tuple[tuple[str, str], ...] = (
    ("Frequency baseline", "frequency"),
    ("Gap-weighted baseline", "gap_weighted"),
    ("XGBoost (solo)", "xgboost"),
    ("Markov (solo)", "markov"),
    ("**Ensemble**", "ensemble"),
)


@dataclass(slots=True)
class _GhContext:
    issue_number: int
    token: str
    repo: str

    @property
    def can_post(self) -> bool:
        return bool(self.token and self.repo and self.issue_number)

    def post(self, body: str) -> None:
        if self.can_post:
            post_issue_comment(self.repo, self.issue_number, body, self.token)


def _metric_rows(summary: dict) -> list[str]:
    rows = []
    for label, key in METRIC_LABELS:
        m = summary.get(key, {})
        if m.get("n_shows", 0) == 0:
            rows.append(f"| {label} | — | — | — | — | — |")
            continue
        rows.append(
            f"| {label} | {m['precision_at_25']:.1%} | {m.get('recall', 0):.1%} | "
            f"{m.get('f1', 0):.1%} | {m.get('opener_accuracy', 0):.1%} | — |"
        )
    return rows


def _render_card(today: str, n_shows: int, scrape_target: str, rows: list[str]) -> str:
    rows_block = "\n".join(rows)
    return f"""# Phinish Model Card

**Status:** Trained — last retrained on {today} ({scrape_target}).

## Dataset

{n_shows} shows scraped fresh per retrain. Raw setlist data is not committed
to the repo (see `.gitignore`); only trained model artifacts are persisted.

## Metrics

Evaluated on temporal holdout (test ≥ 2025).

| Model | Precision@25 | Recall | F1 | Opener Acc | Pair Match |
|---|---|---|---|---|---|
{rows_block}

## Hyperparameters

- **XGBoost:** `max_depth=6`, `learning_rate=0.1`, `n_estimators=300`, `scale_pos_weight=12`
- **Markov:** order 2, Laplace smoothing `k=0.01`
- **Calibration:** Platt scaling on validation set (2024)
- **Ensemble:** grid search over `(w_xgb, w_markov, w_gap, w_venue)` optimizing Precision@25

See `models/manifest.json` for the full retrain history.
"""


def update_model_card(today: str, n_shows: int, scrape_target: str, summary: dict) -> None:
    """Render and persist models/model_card.md from the latest evaluation."""
    body = _render_card(today, n_shows, scrape_target, _metric_rows(summary))
    MODEL_CARD.write_text(body, encoding="utf-8")


def _pipeline_steps(year: int | None) -> list[tuple[str, Callable[[], None]]]:
    scrape_label = f"scrape year {year}" if year else "scrape (full pull)"
    return [
        (scrape_label, lambda: scrape.main(year=year)),
        ("build_features", build_features.main),
        ("train_baseline", train_baseline.main),
        ("train_markov", train_markov.main),
        ("train_xgboost", train_xgboost.main),
        ("train_ensemble", train_ensemble.main),
        ("evaluate", evaluate.main),
    ]


def _run_pipeline(year: int | None) -> None:
    for name, step in _pipeline_steps(year):
        print(f"=== {name} ===", flush=True)
        step()


def _read_year_from_env(gh: _GhContext) -> int | None:
    """Return the year specified in the issue, or None for a full scrape."""
    fields = parse_issue_form(os.environ.get("ISSUE_BODY", ""))
    year_str = sanitize(fields.get("year", ""), 8)
    if not year_str:
        return None
    if not VALID_YEAR.match(year_str):
        _fail(gh, f"❌ Invalid year: `{year_str}`. Expected `YYYY` or empty for full scrape.")
    return int(year_str)


def _fail(gh: _GhContext, msg: str) -> NoReturn:
    gh.post(msg)
    raise SystemExit(msg)


def _read_previous() -> dict | None:
    """Return the previous-run manifest entry, or None on first run."""
    if not MANIFEST_PATH.exists():
        return None
    return load_json(MANIFEST_PATH).get("current")


def _save_manifest(current: dict, previous: dict | None) -> None:
    save_json(MANIFEST_PATH, {"current": current, "previous": previous})


def _diff_line(current_n: int, previous: dict | None) -> str:
    if previous is None:
        return f"Initial training: **{current_n}** shows."
    prev_n = previous.get("n_shows", 0)
    prev_at = previous.get("trained_at", "previous run")
    delta = current_n - prev_n
    if delta == 0:
        return (
            f"Re-trained on the same dataset (**{current_n}** shows; previous run "
            f"{prev_at}). Models refreshed; metrics may shift slightly from training "
            "non-determinism."
        )
    sign = "+" if delta > 0 else ""
    return (
        f"Re-trained: **{current_n}** shows ({sign}{delta} since {prev_at}, "
        f"which had {prev_n})."
    )


def _success_comment(current: dict, previous: dict | None, scrape_target: str) -> str:
    diff = _diff_line(current["n_shows"], previous)
    metrics = current.get("metrics", {})
    return (
        f"## ✅ Retrain Complete — {scrape_target}\n\n"
        f"{diff}\n\n"
        f"- Test shows: **{metrics.get('n_test_shows', 0)}** (since 2025)\n"
        f"- Ensemble Precision@25: **{metrics.get('precision_at_25', 0):.1%}**\n"
        f"- Ensemble Opener Accuracy: **{metrics.get('opener_accuracy', 0):.1%}**\n\n"
        "See [`models/model_card.md`](../blob/main/models/model_card.md) for full "
        "metrics, or [`models/manifest.json`](../blob/main/models/manifest.json) for "
        "retrain history."
    )


def _build_current(today: str, n_shows: int, scrape_target: str, summary: dict) -> dict:
    ensemble = summary.get("ensemble", {})
    return {
        "trained_at": today,
        "n_shows": n_shows,
        "scrape_target": scrape_target,
        "metrics": {
            "precision_at_25": ensemble.get("precision_at_25", 0.0),
            "opener_accuracy": ensemble.get("opener_accuracy", 0.0),
            "n_test_shows": ensemble.get("n_shows", 0),
        },
    }


def main() -> None:
    """Run the full pipeline (scrape → features → train → evaluate) and post results."""
    gh = _GhContext(
        issue_number=int(os.environ.get("ISSUE_NUMBER", "0")),
        token=os.environ.get("GITHUB_TOKEN", ""),
        repo=os.environ.get("GITHUB_REPOSITORY", ""),
    )
    year = _read_year_from_env(gh)
    scrape_target = f"year {year}" if year else "full scrape"
    previous = _read_previous()

    try:
        _run_pipeline(year)
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        gh.post(f"❌ Pipeline failed at step: `{type(e).__name__}: {e}`")
        raise

    eval_path = MODELS_DIR / "evaluation.json"
    summary = load_json(eval_path).get("summary", {}) if eval_path.exists() else {}
    n_shows = len(load_json(SETLISTS_PATH)) if SETLISTS_PATH.exists() else 0
    today = datetime.now().isoformat(timespec="seconds")

    current = _build_current(today, n_shows, scrape_target, summary)
    _save_manifest(current, previous)
    update_model_card(today, n_shows, scrape_target, summary)

    comment = _success_comment(current, previous, scrape_target)
    if gh.can_post:
        gh.post(comment)
    else:
        print(comment)
    print(f"process complete for issue #{gh.issue_number}")


if __name__ == "__main__":
    main()
