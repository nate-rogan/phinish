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
from datetime import date as _date
from pathlib import Path

import httpx

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
    sanitize,
)

MODEL_CARD = MODELS_DIR / "model_card.md"


def post_comment(repo: str, issue_number: int, body: str, token: str) -> None:
    r = httpx.post(
        f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={"body": body},
        timeout=30.0,
    )
    r.raise_for_status()


def update_model_card(year: int, summary: dict) -> None:
    today = _date.today().isoformat()
    rows = []
    for label, key in (
        ("Frequency baseline", "frequency"),
        ("Gap-weighted baseline", "gap_weighted"),
        ("XGBoost (solo)", "xgboost"),
        ("Markov (solo)", "markov"),
        ("**Ensemble**", "ensemble"),
    ):
        m = summary.get(key, {})
        if m.get("n_shows", 0) == 0:
            rows.append(f"| {label} | — | — | — | — | — |")
            continue
        rows.append(
            f"| {label} | {m['precision_at_25']:.1%} | {m.get('recall', 0):.1%} | "
            f"{m.get('f1', 0):.1%} | {m.get('opener_accuracy', 0):.1%} | — |"
        )

    n_shows = summary.get("ensemble", {}).get("n_shows", 0)
    body = f"""# Phinish Model Card

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
{chr(10).join(rows)}

## Hyperparameters

- **XGBoost:** `max_depth=6`, `learning_rate=0.1`, `n_estimators=300`, `scale_pos_weight=12`
- **Markov:** order 2, Laplace smoothing `k=0.01`
- **Calibration:** Platt scaling on validation set (2024)
- **Ensemble:** grid search over `(w_xgb, w_markov, w_gap, w_venue)` optimizing Precision@25

## Changelog

- {today}: retrained from year `{year}`.
"""
    MODEL_CARD.write_text(body, encoding="utf-8")


def main() -> None:
    issue_number = int(os.environ.get("ISSUE_NUMBER", "0"))
    body = os.environ.get("ISSUE_BODY", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")

    fields = parse_issue_form(body)
    year_str = sanitize(fields.get("year", ""), 8)
    if not VALID_YEAR.match(year_str):
        msg = f"❌ Invalid year: `{year_str}`. Expected `YYYY`."
        if token and repo and issue_number:
            post_comment(repo, issue_number, msg, token)
        raise SystemExit(msg)
    year = int(year_str)

    try:
        print(f"=== scrape year {year} ===", flush=True)
        scrape.main(year=year)

        print("=== build_features ===", flush=True)
        build_features.main()

        print("=== train_baseline ===", flush=True)
        train_baseline.main()

        print("=== train_markov ===", flush=True)
        train_markov.main()

        print("=== train_xgboost ===", flush=True)
        train_xgboost.main()

        print("=== train_ensemble ===", flush=True)
        train_ensemble.main()

        print("=== evaluate ===", flush=True)
        evaluate.main()
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        if token and repo and issue_number:
            post_comment(repo, issue_number,
                         f"❌ Pipeline failed at step: `{type(e).__name__}: {e}`")
        raise

    eval_path = MODELS_DIR / "evaluation.json"
    summary = load_json(eval_path).get("summary", {}) if eval_path.exists() else {}
    update_model_card(year, summary)

    n_shows = len(load_json(SETLISTS_PATH)) if SETLISTS_PATH.exists() else 0
    ens = summary.get("ensemble", {})
    comment = (
        f"## ✅ Process Complete — year `{year}`\n\n"
        f"- Dataset: **{n_shows}** shows\n"
        f"- Test shows: **{ens.get('n_shows', 0)}** (since 2025)\n"
        f"- Ensemble Precision@25: **{ens.get('precision_at_25', 0):.1%}**\n"
        f"- Ensemble Opener Accuracy: **{ens.get('opener_accuracy', 0):.1%}**\n\n"
        "See [`models/model_card.md`](../blob/main/models/model_card.md) for full metrics."
    )
    if token and repo and issue_number:
        post_comment(repo, issue_number, comment, token)
    else:
        print(comment)
    print(f"process complete for issue #{issue_number}")


if __name__ == "__main__":
    main()
