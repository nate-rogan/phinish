"""Action entrypoint for the process workflow.

Parses year from issue body, runs the full pipeline:
  scrape -> build_features -> train_baseline -> train_markov ->
  train_xgboost -> train_ensemble -> evaluate
Then updates models/model_card.md and posts a summary comment.
"""
from __future__ import annotations

import os
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
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
    MODELS_DIR,
    SETLISTS_PATH,
    VALID_YEAR,
    load_json,
    parse_issue_form,
    post_issue_comment,
    sanitize,
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


def _render_card(year: int, today: str, n_shows: int, rows: list[str]) -> str:
    rows_block = "\n".join(rows)
    return f"""# Phinish Model Card

**Status:** Trained — last retrained from year `{year}` on {today}.

## Versions

| Component | Version | Updated |
|---|---|---|
| Dataset | `data/processed/setlists.json` | {today} |
| Gap-weighted baseline | `models/baselines.json` | {today} |
| XGBoost song selector | `models/xgboost_song_selector.pkl` | {today} |
| Markov chain (order 2) | `models/markov_order2.json` | {today} |
| Ensemble weights | `models/ensemble_weights.json` | {today} |

## Metrics

Evaluated on temporal holdout (test ≥ 2025), {n_shows} shows.

| Model | Precision@25 | Recall | F1 | Opener Acc | Pair Match |
|---|---|---|---|---|---|
{rows_block}

## Hyperparameters

- **XGBoost:** `max_depth=6`, `learning_rate=0.1`, `n_estimators=300`, `scale_pos_weight=12`
- **Markov:** order 2, Laplace smoothing `k=0.01`
- **Calibration:** Platt scaling on validation set (2024)
- **Ensemble:** grid search over `(w_xgb, w_markov, w_gap, w_venue)` optimizing Precision@25

## Changelog

- {today}: retrained from year `{year}`.
"""


def update_model_card(year: int, summary: dict) -> None:
    """Render and persist models/model_card.md from the latest evaluation."""
    today = date.today().isoformat()
    n_shows = summary.get("ensemble", {}).get("n_shows", 0)
    body = _render_card(year, today, n_shows, _metric_rows(summary))
    MODEL_CARD.write_text(body, encoding="utf-8")


def _pipeline_steps(year: int) -> list[tuple[str, Callable[[], None]]]:
    return [
        (f"scrape year {year}", lambda: scrape.main(year=year)),
        ("build_features", build_features.main),
        ("train_baseline", train_baseline.main),
        ("train_markov", train_markov.main),
        ("train_xgboost", train_xgboost.main),
        ("train_ensemble", train_ensemble.main),
        ("evaluate", evaluate.main),
    ]


def _run_pipeline(year: int) -> None:
    for name, step in _pipeline_steps(year):
        print(f"=== {name} ===", flush=True)
        step()


def _success_comment(year: int, n_shows: int, ensemble: dict) -> str:
    return (
        f"## ✅ Process Complete — year `{year}`\n\n"
        f"- Dataset: **{n_shows}** shows\n"
        f"- Test shows: **{ensemble.get('n_shows', 0)}** (since 2025)\n"
        f"- Ensemble Precision@25: **{ensemble.get('precision_at_25', 0):.1%}**\n"
        f"- Ensemble Opener Accuracy: **{ensemble.get('opener_accuracy', 0):.1%}**\n\n"
        "See [`models/model_card.md`](../blob/main/models/model_card.md) for full metrics."
    )


def _read_year_from_env(gh: _GhContext) -> int:
    fields = parse_issue_form(os.environ.get("ISSUE_BODY", ""))
    year_str = sanitize(fields.get("year", ""), 8)
    if not VALID_YEAR.match(year_str):
        _fail(gh, f"❌ Invalid year: `{year_str}`. Expected `YYYY`.")
    return int(year_str)


def _fail(gh: _GhContext, msg: str) -> NoReturn:
    gh.post(msg)
    raise SystemExit(msg)


def main() -> None:
    """Action entry point: scrape, build features, train all models, evaluate, report."""
    gh = _GhContext(
        issue_number=int(os.environ.get("ISSUE_NUMBER", "0")),
        token=os.environ.get("GITHUB_TOKEN", ""),
        repo=os.environ.get("GITHUB_REPOSITORY", ""),
    )
    year = _read_year_from_env(gh)

    try:
        _run_pipeline(year)
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        gh.post(f"❌ Pipeline failed at step: `{type(e).__name__}: {e}`")
        raise

    eval_path = MODELS_DIR / "evaluation.json"
    summary = load_json(eval_path).get("summary", {}) if eval_path.exists() else {}
    update_model_card(year, summary)

    n_shows = len(load_json(SETLISTS_PATH)) if SETLISTS_PATH.exists() else 0
    comment = _success_comment(year, n_shows, summary.get("ensemble", {}))
    if gh.can_post:
        gh.post(comment)
    else:
        print(comment)
    print(f"process complete for issue #{gh.issue_number}")


if __name__ == "__main__":
    main()
